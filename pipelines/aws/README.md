# AWS / SageMaker pipelines

The AWS counterpart of `pipelines/gcp/`. Same workloads, expressed with **SageMaker Pipelines**
(`ProcessingStep` / `TrainingStep`) instead of Vertex AI / KFP. Region defaults to **us-east-1**.

| Pipeline | GCP equivalent | Status |
|----------|----------------|--------|
| `intact_data_extract` | `pipelines/gcp/intact_data_extract` | ✅ implemented |
| `intact_pretrain` (multi-GPU training) | `pipelines/gcp/intact_pretrain` | ⏳ next |
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

## Roadmap
- `intact_pretrain`: `TrainingStep` on a GPU instance (`ml.g5.12xlarge`), multi-GPU via torchrun,
  TensorBoard → S3, checkpoints to `OutputDataConfig`, model registered in the **Model Registry**.
- `skempi_eval`: evaluation `ProcessingStep`/`TrainingStep` consuming the registered model.
