import logging
import os
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Artifact

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
def prepare_mutations_op(
    gcp_bucket: str,
    gcp_prefix: str,
    intact_sample_size: float = 1.0,
    mutations_parquet: Output[Dataset] = Output[Dataset],
):
    import os
    import logging
    import shutil
    from stabddg.jobs.intact_data_extract import prepare_mutations

    # Configure Cloud Logging
    logger = logging.getLogger(__name__)
    logger.info("Starting 'prepare_mutations_op' step")

    # 1. Setup a temporary directory for the existing logic to work in
    temp_data_dir = "/tmp/data"
    os.makedirs(temp_data_dir, exist_ok=True)

    # 2. Run the existing logic
    # This function saves to {temp_data_dir}/intact/df_intact_mutations.parquet
    df_mutations = prepare_mutations(data_dir=temp_data_dir)

    # Taking a sample
    if intact_sample_size < 1.0:
        df_mutations = df_mutations.sample(frac=intact_sample_size, random_state=42)

    local_intact_dir = os.path.join(temp_data_dir, "intact")
    os.makedirs(local_intact_dir, exist_ok=True)
    local_parquet_path = os.path.join(local_intact_dir, "df_intact_mutations.parquet")
    df_mutations.to_parquet(local_parquet_path)

    logger.info(f"Loaded {df_mutations.shape[0]} mutations from Intact dataset.")

    # 3. Move the specific output file to the KFP Output path
    src_path = local_parquet_path
    shutil.move(src_path, mutations_parquet.path)

    # 4. Persist to GCS as an intermediate artifact
    # gs://<bucket>/<prefix>/df_intact_mutations.parquet
    gcs_path = f"gs://{gcp_bucket}/{gcp_prefix}/df_intact_mutations.parquet"
    logger.info(f"Saving IntAct mutations parquet with {df_mutations.shape[0]} rows to {gcs_path}")
    # Reuse df_mutations in memory rather than re-reading from disk
    df_mutations.to_parquet(gcs_path)

    logger.info("'prepare_mutations_op' step completed successfully.")


@dsl.component(base_image=BASE_IMAGE)
def fetch_alphafold_op(
    mutations_parquet: Input[Dataset],
    proteins_dir: Output[Dataset],
    step_outcome_json: Output[Artifact],
    gcp_bucket: str,
    gcp_prefix: str,
    gcp_region: str,
    max_workers: int = 128,
):
    import os
    import logging
    import json
    import pandas as pd
    from stabddg.jobs.intact_data_extract import fetch_alphafold_for_uniprots
    from stabddg.utils.gcp import upload_dir_to_gcp

    # Configure Cloud Logging
    logger = logging.getLogger(__name__)
    logger.info("Starting 'fetch_alphafold_op' step")

    # 1. Load input
    df_mutations = pd.read_parquet(mutations_parquet.path)

    # 2. Prepare output directory
    os.makedirs(proteins_dir.path, exist_ok=True)

    # 3. Run logic
    outcome = fetch_alphafold_for_uniprots(
        df_mutations=df_mutations,
        proteins_dir=proteins_dir.path,
        max_workers=max_workers
    )
    logger.info(f"Fetched AlphaFold for {len(outcome)} UniProts.")

    # 4. Save metadata (step outcome) for the next step
    with open(step_outcome_json.path, 'w') as f:
        json.dump(outcome, f)

    # 5. Upload proteins directory (parquet + safetensors) to GCS
    # Layout: gs://<bucket>/<prefix>/proteins/...
    logger.info(
        f"Uploading AlphaFold proteins directory to "
        f"gs://{gcp_bucket}/{gcp_prefix}/proteins"
    )
    upload_dir_to_gcp(
        bucket_name=gcp_bucket,
        local_dir=proteins_dir.path + "/",
        dst_prefix=f"{gcp_prefix}/proteins",
        region=gcp_region,
    )

    logger.info("'fetch_alphafold_op' step completed successfully.")


@dsl.component(base_image=BASE_IMAGE)
def filter_mutations_op(
    mutations_parquet: Input[Dataset],
    step_outcome_json: Input[Artifact],
    filtered_mutations_parquet: Output[Dataset],
    gcp_bucket: str,
    gcp_prefix: str,
):
    import json
    import pandas as pd
    from stabddg.jobs.intact_data_extract import filter_mutations_by_available_structures

    # 1. Load inputs
    df_mutations = pd.read_parquet(mutations_parquet.path)
    with open(step_outcome_json.path, 'r') as f:
        step_outcome = json.load(f)

    # 2. Run logic
    df_filtered = filter_mutations_by_available_structures(
        df_mutations=df_mutations,
        step_outcome=step_outcome
    )

    # 3. Save output to KFP artifact
    df_filtered.to_parquet(filtered_mutations_parquet.path)

    # 4. Persist to GCS as intermediate
    gcs_path = f"gs://{gcp_bucket}/{gcp_prefix}/df_intact_mutations_filtered.parquet"
    df_filtered.to_parquet(gcs_path)


@dsl.component(base_image=BASE_IMAGE)
def fetch_metadata_op(
    filtered_mutations_parquet: Input[Dataset],
    assemblies_raw_parquet: Output[Dataset],
    gcp_bucket: str,
    gcp_prefix: str,
    max_workers: int = 32,
):
    import pandas as pd
    from stabddg.jobs.intact_data_extract import get_uniprot_pairs, fetch_assemblies_metadata

    # 1. Load input
    df_filtered = pd.read_parquet(filtered_mutations_parquet.path)

    # 2. Run logic
    pairs = get_uniprot_pairs(df_filtered)
    df_raw = fetch_assemblies_metadata(uniprots_pairs=pairs, max_workers=max_workers)

    # 3. Save output to KFP artifact
    df_out = df_raw.drop(columns=["mutations"], errors="ignore")
    df_out.to_parquet(assemblies_raw_parquet.path)

    # 4. Persist to GCS as intermediate
    gcs_path = f"gs://{gcp_bucket}/{gcp_prefix}/df_assemblies_raw.parquet"
    df_out.to_parquet(gcs_path)


@dsl.component(base_image=BASE_IMAGE)
def select_assemblies_op(
    assemblies_raw_parquet: Input[Dataset],
    assemblies_parquet: Output[Dataset],
    gcp_bucket: str,
    gcp_prefix: str,
):
    import pandas as pd
    from stabddg.jobs.intact_data_extract import select_and_normalize_assemblies

    df_raw = pd.read_parquet(assemblies_raw_parquet.path)
    df_assemblies = select_and_normalize_assemblies(df_raw)

    # Save to KFP artifact
    df_assemblies.to_parquet(assemblies_parquet.path)

    # Persist to GCS
    gcs_path = f"gs://{gcp_bucket}/{gcp_prefix}/df_assemblies.parquet"
    df_assemblies.to_parquet(gcs_path)


@dsl.component(base_image=BASE_IMAGE)
def fetch_assemblies_atoms_op(
    assemblies_parquet: Input[Dataset],
    assemblies_dir: Output[Dataset],
    assemblies_filtered_parquet: Output[Dataset],
    gcp_bucket: str,
    gcp_prefix: str,
    gcp_region: str,
    max_workers: int = 128,
):
    import os
    import pandas as pd
    from stabddg.jobs.intact_data_extract import fetch_and_summarize_assemblies_atoms
    from stabddg.utils.gcp import upload_dir_to_gcp

    df_assemblies = pd.read_parquet(assemblies_parquet.path)
    os.makedirs(assemblies_dir.path, exist_ok=True)

    # This downloads atoms to `assemblies_dir.path` and returns the filtered DF
    df_filtered = fetch_and_summarize_assemblies_atoms(
        df_assemblies=df_assemblies,
        assemblies_dir=assemblies_dir.path,
        max_workers=max_workers
    )

    # Save filtered assemblies DF to KFP artifact
    df_filtered.to_parquet(assemblies_filtered_parquet.path)

    # Persist filtered assemblies DF to GCS
    gcs_df_path = f"gs://{gcp_bucket}/{gcp_prefix}/df_assemblies_filtered.parquet"
    df_filtered.to_parquet(gcs_df_path)

    # Upload assemblies atoms directory to GCS
    # Layout: gs://<bucket>/<prefix>/assemblies/...
    upload_dir_to_gcp(
        bucket_name=gcp_bucket,
        local_dir=assemblies_dir.path + "/",
        dst_prefix=f"{gcp_prefix}/assemblies",
        region=gcp_region,
    )


@dsl.pipeline(
    name="intact-data-extract-pipeline",
    description="Extracts IntAct pretraining data and saves to GCS"
)
def intact_data_extract_pipeline(
    gcp_bucket: str,
    gcp_prefix: str,
    gcp_region: str = "europe-west4",
    intact_sample_size: float = 1.0,
    alphafold_workers: int = 128,
    metadata_workers: int = 32,
    assemblies_workers: int = 128,
):
    # 1. Prepare Mutations
    prepare_task = prepare_mutations_op(
        gcp_bucket=gcp_bucket,
        gcp_prefix=gcp_prefix,
        intact_sample_size=intact_sample_size,
    )
    prepare_task.set_display_name("Prepare IntAct Mutations")

    # 2. Fetch AlphaFold
    fetch_af_task = (
        fetch_alphafold_op(
            mutations_parquet=prepare_task.outputs["mutations_parquet"],
            gcp_bucket=gcp_bucket,
            gcp_prefix=gcp_prefix,
            gcp_region=gcp_region,
            max_workers=alphafold_workers,
        )
        .set_memory_limit('16G')
        .set_cpu_limit('4')
    )
    fetch_af_task.set_display_name("Fetch AlphaFold Structures")

    # 3. Filter Mutations
    filter_task = filter_mutations_op(
        mutations_parquet=prepare_task.outputs["mutations_parquet"],
        step_outcome_json=fetch_af_task.outputs["step_outcome_json"],
        gcp_bucket=gcp_bucket,
        gcp_prefix=gcp_prefix,
    )
    filter_task.set_display_name("Filter Mutations")

    # 4. Fetch Assemblies Metadata
    fetch_meta_task = fetch_metadata_op(
        filtered_mutations_parquet=filter_task.outputs["filtered_mutations_parquet"],
        gcp_bucket=gcp_bucket,
        gcp_prefix=gcp_prefix,
        max_workers=metadata_workers,
    )
    fetch_meta_task.set_display_name("Fetch Assemblies Metadata")

    # 5. Select Assemblies
    select_task = select_assemblies_op(
        assemblies_raw_parquet=fetch_meta_task.outputs["assemblies_raw_parquet"],
        gcp_bucket=gcp_bucket,
        gcp_prefix=gcp_prefix,
    )
    select_task.set_display_name("Select Best Assemblies")

    # 6. Fetch Assemblies Atoms
    fetch_atoms_task = (
        fetch_assemblies_atoms_op(
            assemblies_parquet=select_task.outputs["assemblies_parquet"],
            gcp_bucket=gcp_bucket,
            gcp_prefix=gcp_prefix,
            gcp_region=gcp_region,
            max_workers=assemblies_workers,
        )
        .set_memory_limit('16G')
        .set_cpu_limit('4')
    )
    fetch_atoms_task.set_display_name("Fetch Assemblies Atoms")
