"""SageMaker Pipeline for standalone SKEMPI v2 evaluation.

AWS counterpart of pipelines/gcp/skempi_eval. A single-GPU ``ProcessingStep`` that scores a pretrained
ProteinMPNN checkpoint on the SKEMPI test split and writes two S3 outputs — the per-mutation
predictions (the asset) and a summary of the paper's metrics (pooled Spearman etc.).

Deliberately **standalone and parameterized** (not chained onto pretrain): you evaluate *any*
checkpoint on demand — a fresh pretrain run, an old one, or the paper's own baseline — and compare by
``RunTag``. Single GPU by design (the eval doesn't need DDP); it reuses the ``:gpu`` image.

Build the GPU image, then:
    export PIPELINE_IMAGE_TAG=gpu
    python -m pipelines.aws.skempi_eval.pipeline run \
        --model-uri s3://<bucket>/pipelines-<execid>-IntactPretrain-<suffix>/output/model.tar.gz \
        --param RunTag=b-baseline
"""
from __future__ import annotations

import argparse
import os

from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_experiment_config import PipelineExperimentConfig
from sagemaker.workflow.steps import ProcessingStep

from pipelines.aws import config

PIPELINE_NAME = "stab-ddg-skempi-eval"
RUN_EVAL = os.path.join(os.path.dirname(__file__), "processing", "run_eval.py")
_CONTAINER_IN = "/opt/ml/processing/input"
_CONTAINER_OUT = "/opt/ml/processing/output"


def build_pipeline() -> Pipeline:
    session = config.pipeline_session()
    role = config.execution_role()
    image = config.image_uri(tag=os.environ.get("PIPELINE_IMAGE_TAG", "gpu"))  # needs CUDA torch

    # ---- Parameters ----------------------------------------------------------------------------
    bucket = ParameterString("ArtifactBucket", default_value=config.default_bucket())
    prefix = ParameterString("S3Prefix", default_value=config.DEFAULT_S3_PREFIX)
    # The pretrain artifact to score: a model.tar.gz (holds model/epoch_N.pt + metrics/metrics.csv).
    model_uri = ParameterString("ModelArtifactS3Uri")
    # SKEMPI eval data (filtered_skempi.csv, test_pdb.pkl, skempi_full_mask_pdb_dict.pkl). The PDB
    # cache is authoritative, so raw PDBs are not needed.
    skempi_uri = ParameterString(
        "SkempiDataS3Uri",
        default_value=config.s3_uri("data", "SKEMPI", bucket=config.default_bucket()),
    )
    run_tag = ParameterString("RunTag", default_value="run")
    # The eval is inference (no gradients), so it runs fine on CPU — and *all* GPU processing-job
    # quotas are 0 in this account (GPU is gated to training jobs). ml.m5.4xlarge (16 vCPU) has ample
    # processing quota and the :gpu image's torch runs on CPU; the SKEMPI test set is small (~1.5k
    # mutations x ensemble), so a full run is well under an hour. Override with a GPU type if/when a
    # GPU processing quota is granted (run_eval.py auto-uses cuda when available).
    instance_type = ParameterString("EvalInstanceType", default_value="ml.m5.4xlarge")
    # Which checkpoint inside the artifact: best_val (min validation sign_violation_rate) | last |
    # initial | epoch_<N>. best_val is the principled default for a "reference point".
    checkpoint_select = ParameterString("CheckpointSelect", default_value="best_val")
    ensemble = ParameterInteger("Ensemble", default_value=20)  # paper uses 20 noisy passes
    noise_level = ParameterFloat("NoiseLevel", default_value=0.1)
    seed = ParameterInteger("Seed", default_value=0)
    sample_size = ParameterInteger("SampleSize", default_value=0)  # 0 = full test set (debug knob)

    processor = ScriptProcessor(
        image_uri=image,
        command=["python3"],
        role=role,
        instance_type=instance_type,
        instance_count=1,
        volume_size_in_gb=30,
        sagemaker_session=session,
        base_job_name="skempi-eval",
        env={"TQDM_DISABLE": "1"},
        tags=config.DEFAULT_TAGS,
    )

    # Outputs land under skempi_eval/<run_tag>/ so runs self-organise and comparisons are a plain
    # `aws s3 cp .../summary/summary_metrics.json`. Reusing a RunTag overwrites — name them per model.
    out_root = Join(on="/", values=["s3:/", bucket, prefix, "skempi_eval", run_tag])

    step = ProcessingStep(
        name="SkempiEval",
        step_args=processor.run(
            code=RUN_EVAL,
            inputs=[
                ProcessingInput(input_name="model", source=model_uri,
                                destination=f"{_CONTAINER_IN}/model"),
                ProcessingInput(input_name="skempi", source=skempi_uri,
                                destination=f"{_CONTAINER_IN}/skempi"),
            ],
            outputs=[
                ProcessingOutput(output_name="predictions", source=f"{_CONTAINER_OUT}/predictions",
                                 destination=Join(on="/", values=[out_root, "predictions"])),
                ProcessingOutput(output_name="summary", source=f"{_CONTAINER_OUT}/summary",
                                 destination=Join(on="/", values=[out_root, "summary"])),
            ],
            arguments=[
                "--run-tag", run_tag,
                "--checkpoint-select", checkpoint_select,
                "--ensemble", ensemble.to_string(),
                "--noise-level", noise_level.to_string(),
                "--seed", seed.to_string(),
                "--sample-size", sample_size.to_string(),
            ],
        ),
    )

    experiment_config = PipelineExperimentConfig(
        experiment_name=PIPELINE_NAME,
        trial_name=Join(on="-", values=[PIPELINE_NAME, run_tag, ExecutionVariables.PIPELINE_EXECUTION_ID]),
    )

    return Pipeline(
        name=PIPELINE_NAME,
        pipeline_experiment_config=experiment_config,
        parameters=[
            bucket, prefix, model_uri, skempi_uri, run_tag, instance_type,
            checkpoint_select, ensemble, noise_level, seed, sample_size,
        ],
        steps=[step],
        sagemaker_session=session,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["upsert", "run"])
    parser.add_argument("--model-uri", help="S3 URI of the pretrain model.tar.gz (required for run)")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a pipeline parameter (repeatable), e.g. --param RunTag=b-baseline.")
    args = parser.parse_args()

    pipeline = build_pipeline()
    pipeline.upsert(role_arn=config.execution_role(), tags=config.DEFAULT_TAGS)
    print(f"Upserted pipeline: {PIPELINE_NAME}")
    if args.action == "run":
        if not args.model_uri:
            parser.error("--model-uri is required for run")
        overrides = {}
        for item in args.param:
            if "=" not in item:
                parser.error(f"--param must be KEY=VALUE, got {item!r}")
            key, value = item.split("=", 1)
            overrides[key.strip()] = value.strip()
        parameters = {"ModelArtifactS3Uri": args.model_uri, **overrides}
        execution = pipeline.start(parameters=parameters)
        print(f"Started execution: {execution.arn}")
        if overrides:
            print(f"Parameter overrides: {overrides}")


if __name__ == "__main__":
    main()
