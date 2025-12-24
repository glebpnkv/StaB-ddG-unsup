import logging
import os
from kfp import dsl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# This is a placeholder. You can replace this string before compilation
# or use a build script to inject the correct URI.
# Example: os.environ.get("PIPELINE_BASE_IMAGE", "your-default-image")
BASE_IMAGE = os.environ.get("PIPELINE_BASE_IMAGE", "YOUR_ARTIFACT_REGISTRY_IMAGE_URI_HERE")


@dsl.component(base_image=BASE_IMAGE)
def download_data_op(
    gcp_bucket: str,
    gcp_prefix: str,
    gcp_region: str = "us-central1",
):
    import os
    import subprocess
    import logging
    from stabddg.utils.gcp import upload_dir_to_gcp

    logger = logging.getLogger(__name__)
    logger.info("Starting 'download_data_op' step")

    # Local temp directory where the script will download data
    temp_data_dir = "/tmp/data"
    model_ckpts_dir = "/app/model_ckpts"
    os.makedirs(temp_data_dir, exist_ok=True)

    # 2) Run the shell script inside the container
    script_path = "/app/scripts/download_stabddg_data.sh"
    logger.info(f"Running {script_path} --data_dir {temp_data_dir}")
    subprocess.run(
        [script_path, "--data_dir", temp_data_dir],
        check=True,
    )

    # 3) Upload the resulting directory to GCS
    logger.info(f"Uploading {temp_data_dir} to gs://{gcp_bucket}/{gcp_prefix}")
    upload_dir_to_gcp(
        bucket_name=gcp_bucket,
        local_dir=temp_data_dir,
        dst_prefix=gcp_prefix,
        region=gcp_region,
    )

    logger.info("'download_data_op' step completed successfully.")

    # 4) Uploading model checkpoints from the local directory to GCS
    logger.info(f"Uploading model checkpoints from {model_ckpts_dir} to gs://{gcp_bucket}/{gcp_prefix}/checkpoints")
    upload_dir_to_gcp(
        bucket_name=gcp_bucket,
        local_dir=model_ckpts_dir,
        dst_prefix=f"{gcp_prefix}/model_ckpts",
        region=gcp_region,
    )

@dsl.pipeline(
    name="stabddg-data-download-pipeline",
    description="Downloads StaB-ddG (Megascale, SKEMPI) data and saves to GCS"
)
def stabddg_data_download_pipeline(
    gcp_bucket: str,
    gcp_prefix: str,
    gcp_region: str = "us-central1",
):
    download_data_task = download_data_op(
        gcp_bucket=gcp_bucket,
        gcp_prefix=gcp_prefix,
        gcp_region=gcp_region
    )

    download_data_task.set_display_name("Download StaB-ddG Data")
