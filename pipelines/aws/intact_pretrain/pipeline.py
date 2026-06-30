"""SageMaker Pipeline for IntAct contrastive pretraining (multi-GPU).

AWS counterpart of pipelines/gcp/intact_pretrain. A single GPU ``TrainingStep`` — using SageMaker's
managed ``torch_distributed`` to run torchrun across the node's GPUs — followed by registration in
the **SageMaker Model Registry**.

The training code (jobs/intact_pretrain.py) is reused unchanged; ``stabddg`` comes from the baked
GPU image (containers/Dockerfile.sagemaker.gpu), so only the tiny ``jobs/`` dir is uploaded as
source — the 6.8 GB ``data/`` tree is never packaged. Training data is supplied as an S3 input
channel mounted at /opt/ml/input/data/intact.

Build the GPU image, then:
    DOCKERFILE=containers/Dockerfile.sagemaker.gpu TAG=gpu ./scripts/build_and_push/sagemaker.sh
    export PIPELINE_IMAGE_TAG=gpu  # or PIPELINE_IMAGE_URI=...:gpu
    python -m pipelines.aws.intact_pretrain.pipeline upsert
    python -m pipelines.aws.intact_pretrain.pipeline run --data-uri s3://<bucket>/<prefix>/intact
"""
from __future__ import annotations

import argparse
import os

from sagemaker.debugger import TensorBoardOutputConfig
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.pytorch import PyTorch
from sagemaker.workflow.functions import Join
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep

from pipelines.aws import config

PIPELINE_NAME = "stab-ddg-intact-pretrain"
MODEL_PACKAGE_GROUP = "stab-ddg-intact-pretrain"
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def build_pipeline() -> Pipeline:
    session = config.pipeline_session()
    role = config.execution_role()
    image = config.image_uri(tag=os.environ.get("PIPELINE_IMAGE_TAG", "gpu"))  # the GPU training image

    # ---- Parameters ----------------------------------------------------------------------------
    bucket = ParameterString("ArtifactBucket", default_value=config.default_bucket())
    prefix = ParameterString("S3Prefix", default_value=config.DEFAULT_S3_PREFIX)
    data_uri = ParameterString("IntactDataS3Uri")  # s3 prefix with the intact/ layout
    instance_type = ParameterString("TrainingInstanceType", default_value="ml.g5.12xlarge")  # 4x A10G
    run_name = ParameterString("RunName", default_value="intact-pretrain")
    epochs = ParameterInteger("Epochs", default_value=5)
    batch_size = ParameterInteger("BatchSize", default_value=2)
    max_length = ParameterInteger("MaxLength", default_value=400)
    lr = ParameterFloat("LearningRate", default_value=1e-3)
    k_neutral = ParameterInteger("KNeutral", default_value=20)
    k_pos = ParameterInteger("KPos", default_value=5)
    k_neg = ParameterInteger("KNeg", default_value=5)
    lambda_sign = ParameterFloat("LambdaSign", default_value=0.0)
    lambda_neutral = ParameterFloat("LambdaNeutral", default_value=0.0)
    model_val_freq = ParameterInteger("ModelValFreq", default_value=5)
    approval = ParameterString("ModelApprovalStatus", default_value="PendingManualApproval")

    # TensorBoard event files written by training are synced to S3 (the train loop already honours
    # AIP_TENSORBOARD_LOG_DIR, which we point at the SageMaker TensorBoard output dir).
    tb_output = TensorBoardOutputConfig(
        s3_output_path=Join(on="/", values=["s3:/", bucket, prefix, "intact_pretrain", "tensorboard"]),
        container_local_output_path="/opt/ml/output/tensorboard",
    )

    estimator = PyTorch(
        image_uri=image,
        entry_point="intact_pretrain.py",
        source_dir=os.path.join(_REPO_ROOT, "jobs"),  # tiny upload; stabddg comes from the image
        role=role,
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=session,
        base_job_name="intact-pretrain",
        # Managed torchrun: launches one process per GPU on the node and sets RANK/WORLD_SIZE/LOCAL_RANK,
        # which jobs/intact_pretrain.py now uses to init the process group + DDP.
        distribution={"torch_distributed": {"enabled": True}},
        tensorboard_output_config=tb_output,
        environment={"AIP_TENSORBOARD_LOG_DIR": "/opt/ml/output/tensorboard"},
        hyperparameters={
            "run_name": run_name,
            "data_dir": "/opt/ml/input/data/intact",
            "proteins_dir": "/opt/ml/input/data/intact/proteins/safetensors",
            "assemblies_dir": "/opt/ml/input/data/intact/assemblies/safetensors",
            "model_save_dir": "/opt/ml/model",
            "model_existing_checkpoint": "/app/model_ckpts/proteinmpnn.pt",
            "max_length": max_length,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "k_neutral": k_neutral,
            "k_pos": k_pos,
            "k_neg": k_neg,
            "lambda_sign": lambda_sign,
            "lambda_neutral": lambda_neutral,
            "model_val_freq": model_val_freq,
        },
        tags=config.DEFAULT_TAGS,
    )

    train_step = TrainingStep(
        name="IntactPretrain",
        step_args=estimator.fit(inputs={"intact": TrainingInput(s3_data=data_uri)}),
    )

    # ---- Register the trained model in the Model Registry ---------------------------------------
    model = Model(
        image_uri=image,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        sagemaker_session=session,
    )
    register_step = ModelStep(
        name="RegisterModel",
        step_args=model.register(
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.g5.xlarge"],
            transform_instances=["ml.g5.xlarge"],
            model_package_group_name=MODEL_PACKAGE_GROUP,
            approval_status=approval,
            description="StaB-ddG ProteinMPNN pretrained on IntAct mutation-effect contrastive signal.",
        ),
    )

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[
            bucket, prefix, data_uri, instance_type, run_name, epochs, batch_size, max_length, lr,
            k_neutral, k_pos, k_neg, lambda_sign, lambda_neutral, model_val_freq, approval,
        ],
        steps=[train_step, register_step],
        sagemaker_session=session,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["upsert", "run"])
    parser.add_argument("--data-uri", help="S3 prefix with the intact/ data layout (required for run)")
    args = parser.parse_args()

    pipeline = build_pipeline()
    pipeline.upsert(role_arn=config.execution_role(), tags=config.DEFAULT_TAGS)
    print(f"Upserted pipeline: {PIPELINE_NAME}")
    if args.action == "run":
        if not args.data_uri:
            parser.error("--data-uri is required for run")
        execution = pipeline.start(parameters={"IntactDataS3Uri": args.data_uri})
        print(f"Started execution: {execution.arn}")


if __name__ == "__main__":
    main()
