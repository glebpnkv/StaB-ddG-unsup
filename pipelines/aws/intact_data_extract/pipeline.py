"""SageMaker Pipeline for IntAct data extraction.

The AWS counterpart of pipelines/gcp/intact_data_extract/pipeline.py: the same six stages, expressed
as chained SageMaker ``ProcessingStep``s. Each step runs ``processing/run_step.py --step <name>`` in
the project's ECR image; steps chain purely through S3 (one step's ProcessingOutput becomes the
next's ProcessingInput, which also makes SageMaker infer the DAG edges and enables step caching).

Build / run:
    python -m pipelines.aws.intact_data_extract.pipeline upsert         # register/update the pipeline
    python -m pipelines.aws.intact_data_extract.pipeline run            # register + start an execution
Requires PIPELINE_ROLE_ARN (and a built image, see scripts/build_and_push/sagemaker.sh).
"""
from __future__ import annotations

import argparse
import os

from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.workflow.functions import Join
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import CacheConfig, ProcessingStep

from pipelines.aws import config

PIPELINE_NAME = "stab-ddg-intact-data-extract"
RUN_STEP = os.path.join(os.path.dirname(__file__), "processing", "run_step.py")
_CONTAINER_IN = "/opt/ml/processing/input"
_CONTAINER_OUT = "/opt/ml/processing/output"


def build_pipeline() -> Pipeline:
    session = config.pipeline_session()
    role = config.execution_role()
    image = config.image_uri()

    # ---- Pipeline parameters (overridable at start-time) ---------------------------------------
    bucket = ParameterString("ArtifactBucket", default_value=config.default_bucket())
    prefix = ParameterString("S3Prefix", default_value=config.DEFAULT_S3_PREFIX)
    intact_sample_size = ParameterFloat("IntactSampleSize", default_value=1.0)
    alphafold_workers = ParameterInteger("AlphafoldWorkers", default_value=16)
    metadata_workers = ParameterInteger("MetadataWorkers", default_value=16)
    assemblies_workers = ParameterInteger("AssembliesWorkers", default_value=16)

    # Per-step instance types: the pandas-only steps are tiny; the network-bound fetch steps want
    # more memory and (for AlphaFold/atoms) a much bigger volume to hold thousands of structure files.
    prepare_instance = ParameterString("PrepareInstanceType", default_value="ml.m5.xlarge")
    alphafold_instance = ParameterString("FetchAlphafoldInstanceType", default_value="ml.m5.2xlarge")
    filter_instance = ParameterString("FilterInstanceType", default_value="ml.m5.xlarge")
    metadata_instance = ParameterString("FetchMetadataInstanceType", default_value="ml.m5.2xlarge")
    select_instance = ParameterString("SelectInstanceType", default_value="ml.m5.xlarge")
    atoms_instance = ParameterString("FetchAtomsInstanceType", default_value="ml.m5.2xlarge")

    cache = CacheConfig(enable_caching=True, expire_after="30d")

    def processor(name: str, instance_type, volume_size_gb: int = 30) -> ScriptProcessor:
        return ScriptProcessor(
            image_uri=image,
            command=["python3"],
            role=role,
            instance_type=instance_type,
            instance_count=1,
            volume_size_in_gb=volume_size_gb,
            sagemaker_session=session,
            base_job_name=f"intact-{name}",
            tags=config.DEFAULT_TAGS,
        )

    def dest(step: str, channel: str):
        # s3://<bucket>/<prefix>/intact_data_extract/<step>/<channel>
        return Join(on="/", values=["s3:/", bucket, prefix, "intact_data_extract", step, channel])

    def out(channel: str, step: str) -> ProcessingOutput:
        return ProcessingOutput(
            output_name=channel,
            source=f"{_CONTAINER_OUT}/{channel}",
            destination=dest(step, channel),
        )

    def published(*subparts: str):
        # s3://<bucket>/<prefix>/training_data[/<subpart>...] — the consolidated, training-ready
        # layout consumed directly by the pretrain pipeline's data channel (--data-uri).
        return Join(on="/", values=["s3:/", bucket, prefix, "training_data", *subparts])

    def out_published(channel: str, destination) -> ProcessingOutput:
        return ProcessingOutput(
            output_name=channel, source=f"{_CONTAINER_OUT}/{channel}", destination=destination
        )

    def inp(channel: str, source) -> ProcessingInput:
        return ProcessingInput(
            input_name=channel, source=source, destination=f"{_CONTAINER_IN}/{channel}"
        )

    def s3_out(step: ProcessingStep, channel: str):
        return step.properties.ProcessingOutputConfig.Outputs[channel].S3Output.S3Uri

    # ---- 1. Prepare IntAct mutations -----------------------------------------------------------
    prepare = ProcessingStep(
        name="PrepareMutations",
        step_args=processor("prepare", prepare_instance).run(
            code=RUN_STEP,
            outputs=[out("mutations", "prepare_mutations")],
            arguments=["--step", "prepare_mutations", "--intact-sample-size", intact_sample_size.to_string()],
        ),
        cache_config=cache,
    )

    # ---- 2. Fetch AlphaFold structures ---------------------------------------------------------
    fetch_af = ProcessingStep(
        name="FetchAlphaFold",
        step_args=processor("fetch-alphafold", alphafold_instance, volume_size_gb=200).run(
            code=RUN_STEP,
            inputs=[inp("mutations", s3_out(prepare, "mutations"))],
            outputs=[out("proteins", "fetch_alphafold"), out("step_outcome", "fetch_alphafold")],
            arguments=["--step", "fetch_alphafold", "--max-workers", alphafold_workers.to_string()],
        ),
        cache_config=cache,
    )

    # ---- 3. Filter mutations by available structures -------------------------------------------
    filter_step = ProcessingStep(
        name="FilterMutations",
        step_args=processor("filter", filter_instance).run(
            code=RUN_STEP,
            inputs=[
                inp("mutations", s3_out(prepare, "mutations")),
                inp("step_outcome", s3_out(fetch_af, "step_outcome")),
            ],
            outputs=[out_published("filtered", published())],
            arguments=["--step", "filter_mutations"],
        ),
        cache_config=cache,
    )

    # ---- 4. Fetch assembly metadata ------------------------------------------------------------
    fetch_meta = ProcessingStep(
        name="FetchMetadata",
        step_args=processor("fetch-metadata", metadata_instance).run(
            code=RUN_STEP,
            inputs=[inp("filtered", s3_out(filter_step, "filtered"))],
            outputs=[out("assemblies_raw", "fetch_metadata")],
            arguments=["--step", "fetch_metadata", "--max-workers", metadata_workers.to_string()],
        ),
        cache_config=cache,
    )

    # ---- 5. Select best assemblies -------------------------------------------------------------
    select = ProcessingStep(
        name="SelectAssemblies",
        step_args=processor("select", select_instance).run(
            code=RUN_STEP,
            inputs=[inp("assemblies_raw", s3_out(fetch_meta, "assemblies_raw"))],
            outputs=[out("assemblies", "select_assemblies")],
            arguments=["--step", "select_assemblies"],
        ),
        cache_config=cache,
    )

    # ---- 6. Fetch assembly atoms (the alignment-numbered structures) ---------------------------
    fetch_atoms = ProcessingStep(
        name="FetchAtoms",
        step_args=processor("fetch-atoms", atoms_instance, volume_size_gb=100).run(
            code=RUN_STEP,
            inputs=[inp("assemblies", s3_out(select, "assemblies"))],
            outputs=[
                out_published("assemblies_atoms", published("assemblies")),
                out_published("assemblies_filtered", published()),
            ],
            arguments=["--step", "fetch_atoms", "--max-workers", assemblies_workers.to_string()],
        ),
        cache_config=cache,
    )

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[
            bucket, prefix, intact_sample_size,
            alphafold_workers, metadata_workers, assemblies_workers,
            prepare_instance, alphafold_instance, filter_instance,
            metadata_instance, select_instance, atoms_instance,
        ],
        steps=[prepare, fetch_af, filter_step, fetch_meta, select, fetch_atoms],
        sagemaker_session=session,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["upsert", "run"], help="register the pipeline, or register + start")
    args = parser.parse_args()

    pipeline = build_pipeline()
    pipeline.upsert(role_arn=config.execution_role(), tags=config.DEFAULT_TAGS)
    print(f"Upserted pipeline: {PIPELINE_NAME}")
    if args.action == "run":
        execution = pipeline.start()
        print(f"Started execution: {execution.arn}")


if __name__ == "__main__":
    main()
