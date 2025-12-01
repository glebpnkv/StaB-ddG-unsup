import logging
import os
from typing import Optional
from google_cloud_pipeline_components.v1.custom_job import (
    create_custom_training_job_from_component,
)
from kfp import dsl
from kfp.dsl import Output, Dataset, Artifact

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_IMAGE = os.environ.get("PIPELINE_BASE_IMAGE", "YOUR_ARTIFACT_REGISTRY_IMAGE_URI_HERE")

# Hard-coded hardware specs (set once here; not configurable at runtime)
# MACHINE_TYPE = "n2-standard-64"  # testing
MACHINE_TYPE = "g2-standard-12"
ACCELERATOR_TYPE = "NVIDIA_L4"
ACCELERATOR_COUNT = 1

JOB_TIMEOUT = "86400s"   # 1 day
NUM_RETRIES = 0
BACKOFF_DURATION = "600s"   # 10 minutes


@dsl.container_component
def skempi_eval_step(
    # --- GCS inputs ---
    skempi_data_gcs_uri: str,                 # e.g. gs://bucket/path/containing/data+cache
    model_ckpt_gcs_uri: Optional[str] = None, # e.g. gs://bucket/path/to/single_checkpoint.pt
    # --- Evaluation configuration ---
    run_name: str = "skempi-eval",
    output_gcs_uri: Optional[str] = None,     # e.g. gs://bucket/experiments/skempi_eval
    batch_size: int = 10000,
    ensemble: int = 20,
    noise_level: float = 0.1,
    seed: int = 0,
    sample_size: Optional[int] = None,
    # --- Outputs (Vertex artifacts) ---
    predictions_dir: Output[Dataset] = Output[Dataset],
    logs_dir: Output[Artifact] = Output[Artifact],
) -> dsl.ContainerSpec:
    """
    Single-step container that delegates SKEMPI evaluation to /app/scripts/skempi_eval.sh.
    """
    model_ckpt_gcs_uri = model_ckpt_gcs_uri or ""
    output_gcs_uri = output_gcs_uri or ""

    local_root = "/app"  # matches Dockerfile WORKDIR

    command = [
        "/app/scripts/skempi_eval.sh"
    ]
    args = [
        "--run_name", run_name,
        "--skempi_data_gcs_uri", skempi_data_gcs_uri,
        "--model_ckpt_gcs_uri", model_ckpt_gcs_uri,
        "--output_gcs_uri", output_gcs_uri,
        "--local_root", local_root,
        "--batch_size", str(batch_size),
        "--ensemble", str(ensemble),
        "--noise_level", str(noise_level),
        "--seed", str(seed),
        "--use_torchrun", "auto",
    ]

    # Optional: limit evaluation sample size for debugging
    if sample_size is not None:
        args.extend(["--sample_size", str(sample_size)])

    # Map KFP artifacts to directories that skempi_eval.sh writes into.
    #
    # - predictions_dir: where the CSV predictions will end up after the script
    #   (we just expose the local_run_dir from skempi_eval.sh)
    # - logs_dir: can point to the same path (or a subdirectory) if you want logs
    #
    # Note: the shell script itself controls what is placed under these paths; here
    # we simply expose the directory so that downstream pipeline steps can consume it.
    predictions_dir.path = os.path.join(local_root, "runs", "skempi_eval")
    logs_dir.path = os.path.join(local_root, "runs", "skempi_eval")

    return dsl.ContainerSpec(
        image=BASE_IMAGE,
        command=command,
        args=args,
    )

# Wrap component in a Vertex CustomTrainingJob with fixed hardware
skempi_eval_custom_job = create_custom_training_job_from_component(
    skempi_eval_step,
    display_name="Evaluate on SKEMPI data",
    machine_type=MACHINE_TYPE,
    accelerator_type=ACCELERATOR_TYPE,
    accelerator_count=ACCELERATOR_COUNT,
    timeout=JOB_TIMEOUT,
)


@dsl.pipeline(
    name="skempi-eval-pipeline",
    description="Vertex AI pipeline to run StaB-ddG SKEMPI evaluation on multi-GPU.",
)
def skempi_eval_pipeline(
    skempi_data_gcs_uri: str,
    model_ckpt_gcs_uri: Optional[str] = None,
    run_name: str = "skempi-eval",
    output_gcs_uri: Optional[str] = None,
    # evaluation hyperparams (overridable at submission)
    batch_size: int = 10000,
    ensemble: int = 20,
    noise_level: float = 0.1,
    seed: int = 0,
    sample_size: Optional[int] = None,
):
    """
    Single-step pipeline wrapping the multi-GPU SKEMPI evaluation job.
    """
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Starting 'skempi_eval_pipeline'")

    eval_task = skempi_eval_custom_job(
        skempi_data_gcs_uri=skempi_data_gcs_uri,
        model_ckpt_gcs_uri=model_ckpt_gcs_uri,
        run_name=run_name,
        output_gcs_uri=output_gcs_uri,
        batch_size=batch_size,
        ensemble=ensemble,
        noise_level=noise_level,
        seed=seed,
        sample_size=sample_size,
    )
    eval_task.set_display_name("Evaluate on SKEMPI data")

    eval_task.set_retry(
        num_retries=NUM_RETRIES,
        backoff_duration=BACKOFF_DURATION,
    )
