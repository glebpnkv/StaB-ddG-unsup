import logging
import os
from typing import Optional

import google.auth
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

_, project_id = google.auth.default()
DEFAULT_LOCATION = "us-central1"

BASE_IMAGE = os.environ.get("PIPELINE_BASE_IMAGE", "YOUR_ARTIFACT_REGISTRY_IMAGE_URI_HERE")
BASE_OUTPUT_DIR = os.environ.get(
    "PIPELINE_BASE_OUTPUT_DIR",
    f"gs://stab-ddg-unsup-staging"
)

# Hard-coded hardware specs (set once here; not configurable at runtime)
# MACHINE_TYPE = "n2-standard-32"  # testing
MACHINE_TYPE = "g2-standard-24"
ACCELERATOR_TYPE = "NVIDIA_L4"
ACCELERATOR_COUNT = 2

JOB_TIMEOUT = "86400s"   # 1 day
NUM_RETRIES = 0
BACKOFF_DURATION = "600s"   # 10 minutes


@dsl.container_component
def intact_pretrain_training_step(
    # --- GCS inputs ---
    intact_data_gcs_uri: str,
    model_ckpt_gcs_uri: Optional[str] = None,  # e.g. gs://bucket/model_ckpts (contains proteinmpnn.pt)
    # --- Training configuration ---
    run_name: str = "intact-pretrain",
    max_length: int = 400,
    batch_size: int = 2,
    epochs: int = 5,
    lr: float = 1e-3,
    k_neutral: int = 20,
    k_pos: int = 5,
    k_neg: int = 5,
    noise_level: float = 0.1,
    lambda_sign: float = 10.0,
    lambda_neutral: float = 0.0,
    valid_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
    num_dataloader_workers: int = 1,
    model_val_freq: int = 5,
    use_antithetic_variates: bool = True,
    use_wandb: bool = False,
    intact_sample_size: Optional[int] = None,
    # --- Outputs (Vertex artifacts) ---
    model_dir: Output[Artifact] = Output[Artifact],
    metrics_dir: Output[Dataset] = Output[Dataset],
    data_splits_dir: Output[Dataset] = Output[Dataset],
) -> dsl.ContainerSpec:
    """
    Single-step container that delegates training to /app/scripts/intact_pretrain.sh.
    """
    model_ckpt_gcs_uri = model_ckpt_gcs_uri or ""

    local_root = "/app"  # matches Dockerfile WORKDIR

    command = [
        "/app/scripts/intact_pretrain.sh"
    ]
    args = [
        "--run_name", run_name,
        "--intact_data_gcs_uri", intact_data_gcs_uri,
        "--model_ckpt_gcs_uri", model_ckpt_gcs_uri,
        "--local_root", local_root,
        "--max_length", str(max_length),
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--k_neutral", str(k_neutral),
        "--k_pos", str(k_pos),
        "--k_neg", str(k_neg),
        "--noise_level", str(noise_level),
        "--lambda_sign", str(lambda_sign),
        "--lambda_neutral", str(lambda_neutral),
        "--valid_size", str(valid_size),
        "--test_size", str(test_size),
        "--random_state", str(random_state),
        "--num_dataloader_workers", str(num_dataloader_workers),
        "--model_val_freq", str(model_val_freq),
        "--use_wandb", str(use_wandb).lower(),
        "--use_antithetic_variates", str(use_antithetic_variates).lower(),
        "--use_torchrun", "auto",
        # Pass KFP artefact paths into the script so it can write outputs there
        "--vertex_model_dir_path", model_dir.path,
        "--vertex_metrics_dir_path", metrics_dir.path,
        "--vertex_data_splits_dir_path", data_splits_dir.path,
    ]

    if intact_sample_size:
        args.append("--intact_sample_size")
        args.append(str(intact_sample_size))

    return dsl.ContainerSpec(
        image=BASE_IMAGE,
        command=command,
        args=args,
    )


# Wrap component in a Vertex CustomTrainingJob with fixed hardware
intact_pretrain_custom_job = create_custom_training_job_from_component(
    intact_pretrain_training_step,
    display_name="Pretrain on IntAct Mutations",
    machine_type=MACHINE_TYPE,
    accelerator_type=ACCELERATOR_TYPE,
    accelerator_count=ACCELERATOR_COUNT,
    timeout=JOB_TIMEOUT,
    base_output_directory=BASE_OUTPUT_DIR,
)


@dsl.pipeline(
    name="intact-pretrain-pipeline",
    description="Vertex AI pipeline to run StaB-ddG IntAct pretraining on multi-GPU.",
)
def intact_pretrain_pipeline(
    intact_data_gcs_uri: str,
    model_ckpt_gcs_uri: Optional[str] = None,
    run_name: str = "intact-pretrain",
    service_account: str = "",
    # training hyperparams (overridable at submission)
    max_length: int = 400,
    batch_size: int = 2,
    epochs: int = 5,
    lr: float = 1e-3,
    k_neutral: int = 20,
    k_pos: int = 5,
    k_neg: int = 5,
    noise_level: float = 0.1,
    lambda_sign: float = 10.0,
    lambda_neutral: float = 0.0,
    valid_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
    num_dataloader_workers: int = 1,
    model_val_freq: int = 5,
    use_antithetic_variates: bool = True,
    use_wandb: bool = False,
    intact_sample_size: Optional[int] = None,
    tensorboard: Optional[str] = None,
):
    """
    Single-step pipeline wrapping the multi-GPU IntAct pretraining job.
    """
    import logging

    # Configure Cloud Logging
    logger = logging.getLogger(__name__)
    logger.info("Starting 'intact_pretrain_pipeline'")

    train_task = intact_pretrain_custom_job(
        intact_data_gcs_uri=intact_data_gcs_uri,
        model_ckpt_gcs_uri=model_ckpt_gcs_uri,
        run_name=run_name,
        max_length=max_length,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        k_neutral=k_neutral,
        k_pos=k_pos,
        k_neg=k_neg,
        noise_level=noise_level,
        lambda_sign=lambda_sign,
        lambda_neutral=lambda_neutral,
        valid_size=valid_size,
        test_size=test_size,
        random_state=random_state,
        num_dataloader_workers=num_dataloader_workers,
        model_val_freq=model_val_freq,
        use_antithetic_variates=use_antithetic_variates,
        use_wandb=use_wandb,
        intact_sample_size=intact_sample_size,
        tensorboard=tensorboard,
        service_account=service_account
    )

    train_task.set_retry(
        num_retries=NUM_RETRIES,
        backoff_duration=BACKOFF_DURATION,
    )
    train_task.set_display_name("StaB-ddG Intact Pre-Training")
