# AWS / SageMaker pipelines

The AWS counterpart of `pipelines/gcp/`. Same workloads, expressed with **SageMaker Pipelines**
(`ProcessingStep` / `TrainingStep`) instead of Vertex AI / KFP. Region defaults to **us-east-1**.

| Pipeline | GCP equivalent | Status |
|----------|----------------|--------|
| `intact_data_extract` | `pipelines/gcp/intact_data_extract` | ✅ implemented + smoke-tested on SageMaker |
| `intact_pretrain` (multi-GPU training) | `pipelines/gcp/intact_pretrain` | ✅ implemented (not yet run) |
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
#    proteins/safetensors/, assemblies/safetensors/, df_intact_mutations_filtered.parquet, ...)
python -m pipelines.aws.intact_pretrain.pipeline run --data-uri s3://<bucket>/<prefix>/intact
```

A single GPU `TrainingStep` on **`ml.g5.12xlarge`** (4× A10G) using SageMaker's managed
`torch_distributed` (torchrun across the 4 GPUs — `jobs/intact_pretrain.py` initialises the process
group and wraps the model in DDP), with TensorBoard synced to S3, model artifacts to
`/opt/ml/model`, then a **`RegisterModel`** step that records the result in the SageMaker **Model
Registry** (group `stab-ddg-intact-pretrain`).

### SageMaker features used
- Managed **multi-GPU** training (`distribution={"torch_distributed": {"enabled": True}}`).
- **TensorBoard → S3** via `TensorBoardOutputConfig`.
- **Model Registry** versioning with an approval gate (`ModelApprovalStatus`).

> The training data channel expects a single S3 prefix with the `intact/` layout. The
> data-extraction pipeline writes its stages to separate prefixes, so wiring its outputs into this
> channel (a small consolidation step, or per-type channels) is the remaining integration task.

## Roadmap
- `skempi_eval`: evaluation `ProcessingStep`/`TrainingStep` consuming the registered model.
