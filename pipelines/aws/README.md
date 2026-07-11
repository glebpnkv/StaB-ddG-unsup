# AWS / SageMaker pipelines

The AWS counterpart of `pipelines/gcp/`. Same workloads, expressed with **SageMaker Pipelines**
(`ProcessingStep` / `TrainingStep`) instead of Vertex AI / KFP. Region defaults to **us-east-1**.

| Pipeline | GCP equivalent | Status |
|----------|----------------|--------|
| `intact_data_extract` | `pipelines/gcp/intact_data_extract` | ✅ implemented + smoke-tested on SageMaker |
| `intact_pretrain` (multi-GPU training) | `pipelines/gcp/intact_pretrain` | ✅ implemented + run (4× GPU) |
| `skempi_eval` | `pipelines/gcp/skempi_eval` | ⏳ planned |

## One-time setup

1. **Credentials** — an AWS profile/SSO session with SageMaker, S3 and ECR access.
2. **Execution role** — an IAM role SageMaker can assume (e.g. `AmazonSageMakerFullAccess` +
   S3/ECR). Export it:
   ```bash
   export PIPELINE_ROLE_ARN=arn:aws:iam::<acct>:role/<sagemaker-exec-role>
   ```
3. **Container image** — build the project image and push it to ECR:
   ```bash
   ./scripts/build_and_push/sagemaker.sh          # builds containers/Dockerfile.sagemaker
   export PIPELINE_IMAGE_URI=<acct>.dkr.ecr.us-east-1.amazonaws.com/stab-ddg-unsup:latest
   ```

Artifacts default to the bucket `stab-ddg-unsup-<acct>-us-east-1` (override with `PIPELINE_BUCKET`).

## Data-extraction pipeline

```bash
# register / update the pipeline definition in SageMaker
python -m pipelines.aws.intact_data_extract.pipeline upsert

# register and start an execution (use IntactSampleSize<1.0 for a quick smoke run)
python -m pipelines.aws.intact_data_extract.pipeline run
```

Six chained `ProcessingStep`s, each running `processing/run_step.py --step <name>` in the project
image and passing parquet/structure artifacts through S3:

```
PrepareMutations → FetchAlphaFold → FilterMutations → FetchMetadata → SelectAssemblies → FetchAtoms
```

`FetchAtoms` produces the alignment-numbered assembly structures (the fix in
`stabddg/intact/data.py`) under `s3://<bucket>/<prefix>/intact_data_extract/fetch_atoms/`.

### SageMaker features used
- **Pipeline parameters** — bucket, prefix, instance type, sample size, per-step worker counts;
  overridable at start-time without editing code.
- **Step caching** (`CacheConfig`) — re-running skips unchanged upstream steps.
- **Automatic lineage** — steps chain via ProcessingOutput→ProcessingInput, so SageMaker infers the
  DAG and records artifact lineage; the run is visible in Studio / SageMaker Pipelines.

## Pretrain pipeline (multi-GPU)

```bash
# 1. Build the GPU training image (DLC base + project) and push as :gpu
DOCKERFILE=containers/Dockerfile.sagemaker.gpu TAG=gpu ALIAS_TAG=gpu ./scripts/build_and_push/sagemaker.sh
export PIPELINE_IMAGE_TAG=gpu

# 2. Register + run (point --data-uri at an S3 prefix holding the intact/ layout:
#    assemblies/safetensors/, df_intact_mutations_filtered.parquet, df_assemblies_filtered.parquet)
python -m pipelines.aws.intact_pretrain.pipeline run --data-uri s3://<bucket>/<prefix>/intact
```

A single GPU `TrainingStep` on **`ml.g5.12xlarge`** (4× A10G) using SageMaker's managed
`torch_distributed` (torchrun across the 4 GPUs — `jobs/intact_pretrain.py` initialises the process
group and wraps the model in DDP), with TensorBoard synced to S3, model artifacts to
`/opt/ml/model`, then a **`RegisterModel`** step that records the result in the SageMaker **Model
Registry** (group `stab-ddg-intact-pretrain`).

The training data is `DistributedSampler`-sharded across the 4 GPUs: each rank processes ~1/4 of the
train anchors per epoch (≈410 batches/rank at `batch_size=2` for the ~3.3k-anchor set) and metrics
are `all_reduce`-averaged — one full pass per epoch, not one-per-GPU. The `k_pos`/`k_neg` contrastive
pool is sampled per-rank (it class-balances the ~4%-positive anchors), not an extra epoch pass.

### Launch-time overrides + A/B runs
Any pipeline parameter can be overridden at launch with repeatable `--param KEY=VALUE`, so one
upserted definition drives every run. Tag each run so it self-labels in tracking:
```bash
# A/B ablation as two labelled runs
python -m pipelines.aws.intact_pretrain.pipeline run --data-uri s3://<bucket>/<prefix>/intact \
  --param RunTag=b-baseline
python -m pipelines.aws.intact_pretrain.pipeline run --data-uri s3://<bucket>/<prefix>/intact \
  --param RunTag=a-balanced-sampler
# quick smoke: --param Epochs=1 --param RunTag=test-smoke
```

### Viewing metrics (TensorBoard + CloudWatch — no standing cost)
Training logs a clean, tqdm-free `Epoch X/Y: Batch i/N ... metrics: {...}` line per batch, plus
per-epoch summaries. Two free ways to see the curves:

- **TensorBoard** (full step + epoch scalars). SageMaker syncs event files to the
  `TensorBoardOutputConfig` path — note this is **not** the job-output prefix:
  ```bash
  # <job> = pipelines-<execid>-IntactPretrain-<suffix>
  aws s3 sync s3://<bucket>/<prefix>/intact_pretrain/tensorboard/<job>/ /tmp/tb/<job>/
  tensorboard --logdir /tmp/tb          # sync several <job>/ dirs under /tmp/tb to overlay A vs B
  ```
- **Training job Metrics tab / CloudWatch** — `metric_definitions` scrapes the summary lines into
  named metrics (`train:loss_sign`, `train:sign_violation_rate`, …), comparable across jobs.

> **Why Studio's "Experiments" panel looks empty:** that panel is **MLflow-backed** and needs an
> MLflow *tracking server* (an AWS-managed one bills ~$0.64/hr while it exists), which we deliberately
> don't run. `PipelineExperimentConfig` still creates a classic SageMaker Experiment + Trial per
> execution (visible via `aws sagemaker list-trials`, and how the trial name carries `RunTag`), but
> the new Studio UI doesn't surface classic experiments in that nav. Use TensorBoard + the Metrics
> tab above. The `SageMaker Experiments unavailable ...; skipping` log line is expected and benign.

### SageMaker features used
- Managed **multi-GPU** training (`distribution={"torch_distributed": {"enabled": True}}`).
- **TensorBoard → S3** via `TensorBoardOutputConfig`.
- **CloudWatch metrics** via `metric_definitions` (per-epoch + per-batch scalars).
- **Model Registry** versioning with an approval gate (`ModelApprovalStatus`).

> The training data channel expects a single S3 prefix with the `intact/` layout. The
> data-extraction pipeline writes its stages to separate prefixes, so wiring its outputs into this
> channel (a small consolidation step, or per-type channels) is the remaining integration task.

## Roadmap
- `skempi_eval`: evaluation `ProcessingStep`/`TrainingStep` consuming the registered model.
