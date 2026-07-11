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
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_experiment_config import PipelineExperimentConfig
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
    # Short human label baked into the experiment trial name so runs are self-identifying in Studio
    # Experiments (e.g. RunTag=b-baseline vs RunTag=a-balanced-sampler for the A/B comparison).
    run_tag = ParameterString("RunTag", default_value="run")
    epochs = ParameterInteger("Epochs", default_value=10)
    # Kept at the last known-good value. NB raising batch_size dilutes the pool's class-balancing
    # (more imbalanced anchors per fixed k_pos/k_neg), so scale k_pos with it or balance the sampler.
    batch_size = ParameterInteger("BatchSize", default_value=2)
    # DataLoader workers PER RANK. The contrastive stream builds k_pos+k_neg(+k_neutral) datapoints on
    # CPU every step; with 1 worker the GPUs starve. g5.12xlarge has 48 vCPUs — feed them.
    num_dataloader_workers = ParameterInteger("NumDataloaderWorkers", default_value=16)
    # If >0, forward each contrast pool in chunks of this many datapoints (concatenating outputs) to
    # cap peak activation memory — lets longer sequences / larger k_* fit a fixed GPU. 0 = whole pool.
    micro_batch_size = ParameterInteger("MicroBatchSize", default_value=2)
    # If 1, gradient-checkpoint the ProteinMPNN forward (recompute in backward) to also cap the
    # backward-pass peak (~30% slower). Off by default now that the neutral pool is gone (much lower
    # memory); flip to 1 (with micro_batch_size) if a longer max_length OOMs.
    grad_checkpoint = ParameterInteger("GradCheckpoint", default_value=0)
    # 768 keeps ~73% of SKEMPI eval mutations' length regime + recovers long IntAct positives.
    # Overridable at start-time (e.g. 1024 for ~98% SKEMPI coverage, with micro_batch_size=1).
    max_length = ParameterInteger("MaxLength", default_value=768)
    lr = ParameterFloat("LearningRate", default_value=1e-3)
    # Contrast pools: each pool item is a ProteinMPNN forward PER RANK. The pos/neg pools are NOT
    # redundant even at lambda_supcon=0 — they CLASS-BALANCE the sign loss: anchors are only ~4% positive
    # (139 vs 3146), so k_pos guarantees positive-direction signal every step (else ~half of batches see
    # zero positives). Keep k_pos≈k_neg>0. Neutrals feed only the (off) normaliser, so k_neutral stays 0.
    k_neutral = ParameterInteger("KNeutral", default_value=0)
    k_pos = ParameterInteger("KPos", default_value=5)
    k_neg = ParameterInteger("KNeg", default_value=5)
    # Loss weights. lambda_sign>0 anchors the ΔΔG sign to the weak labels — the meaningful signal (the
    # distance-based SupCon term is sign-invariant on scalar z). lambda_supcon defaults to 0: start
    # with a pure sign-direction objective (SupCon on 1-D ΔΔG with a ~2 valid-anchor batch was finicky
    # and, with the normaliser, exploded — run 3mfdr02r6yg4). Add a small SupCon back once sign works.
    lambda_supcon = ParameterFloat("LambdaSupcon", default_value=0.0)
    lambda_sign = ParameterFloat("LambdaSign", default_value=1.0)
    lambda_neutral = ParameterFloat("LambdaNeutral", default_value=0.0)
    # Neutral normaliser OFF by default: it divides z by the neutral MAD (~0.13, neutrals cluster near
    # 0), inflating z_hat ~7x and blowing up / oscillating the SupCon loss. See run 3mfdr02r6yg4.
    use_neutral_normalizer = ParameterInteger("UseNeutralNormalizer", default_value=0)
    model_val_freq = ParameterInteger("ModelValFreq", default_value=5)
    # Per-batch metrics to stdout every N batches (1 = every batch); per-epoch summaries always log.
    log_every_batches = ParameterInteger("LogEveryBatches", default_value=1)
    approval = ParameterString("ModelApprovalStatus", default_value="PendingManualApproval")

    # TensorBoard event files written by training are synced to S3 (the train loop already honours
    # AIP_TENSORBOARD_LOG_DIR, which we point at the SageMaker TensorBoard output dir).
    tb_output = TensorBoardOutputConfig(
        s3_output_path=Join(on="/", values=["s3:/", bucket, prefix, "intact_pretrain", "tensorboard"]),
        container_local_output_path="/opt/ml/output/tensorboard",
    )

    # Scrape the per-epoch summary lines (training.py logs these via the module logger -> stdout ->
    # CloudWatch; the per-batch lines only go to logs.txt). These surface in the training job's
    # "Metrics" tab and as CloudWatch metrics. Format is e.g.
    #   Epoch 3: Train metrics: {'loss_total': '0.1234', 'loss_supcon': '0.0000', ...}
    metric_definitions = [
        {"Name": "train:loss_total", "Regex": r"Train metrics: \{'loss_total': '([-0-9.]+)'"},
        {"Name": "train:loss_supcon", "Regex": r"Train metrics: \{[^}]*'loss_supcon': '([-0-9.]+)'"},
        {"Name": "train:loss_sign", "Regex": r"Train metrics: \{[^}]*'loss_sign': '([-0-9.]+)'"},
        {"Name": "train:cos_pos_mean", "Regex": r"Train metrics: \{[^}]*'cos_pos_mean': '([-0-9.]+)'"},
        {"Name": "train:cos_neg_mean", "Regex": r"Train metrics: \{[^}]*'cos_neg_mean': '([-0-9.]+)'"},
        {"Name": "train:sign_violation_rate", "Regex": r"Train metrics: \{[^}]*'sign_violation_rate': '([-0-9.]+)'"},
        {"Name": "validation:loss_total", "Regex": r"Validation metrics: \{'loss_total': '([-0-9.]+)'"},
        {"Name": "validation:loss_supcon", "Regex": r"Validation metrics: \{[^}]*'loss_supcon': '([-0-9.]+)'"},
        {"Name": "validation:sign_violation_rate", "Regex": r"Validation metrics: \{[^}]*'sign_violation_rate': '([-0-9.]+)'"},
    ]

    estimator = PyTorch(
        image_uri=image,
        entry_point="intact_pretrain.py",
        source_dir=os.path.join(_REPO_ROOT, "jobs"),  # tiny upload; stabddg comes from the image
        role=role,
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=session,
        base_job_name="intact-pretrain",
        # Hard wall-clock cap: SageMaker stops the job at this many seconds (model dir uploaded on
        # stop). 9000s (~2.5h, ~$14 on g6) covers 10 epochs of the known-good bs=2/k=5/5 config; the
        # faster balanced-sampler variant finishes well under it and stops on its own.
        max_run=9000,
        metric_definitions=metric_definitions,
        # Managed torchrun: launches one process per GPU on the node and sets RANK/WORLD_SIZE/LOCAL_RANK,
        # which jobs/intact_pretrain.py now uses to init the process group + DDP.
        distribution={"torch_distributed": {"enabled": True}},
        tensorboard_output_config=tb_output,
        environment={"AIP_TENSORBOARD_LOG_DIR": "/opt/ml/output/tensorboard", "TQDM_DISABLE": "1"},
        hyperparameters={
            "run_name": run_name,
            "data_dir": "/opt/ml/input/data/intact",
            "assemblies_dir": "/opt/ml/input/data/intact/assemblies/safetensors",
            "model_save_dir": "/opt/ml/model",
            "model_existing_checkpoint": "/app/model_ckpts/proteinmpnn.pt",
            "max_length": max_length,
            "batch_size": batch_size,
            "num_dataloader_workers": num_dataloader_workers,
            "micro_batch_size": micro_batch_size,
            "grad_checkpoint": grad_checkpoint,
            "epochs": epochs,
            "lr": lr,
            "k_neutral": k_neutral,
            "k_pos": k_pos,
            "k_neg": k_neg,
            "lambda_supcon": lambda_supcon,
            "lambda_sign": lambda_sign,
            "lambda_neutral": lambda_neutral,
            "use_neutral_normalizer": use_neutral_normalizer,
            "model_val_freq": model_val_freq,
            "log_every_batches": log_every_batches,
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

    # Auto-create a SageMaker Experiment for the pipeline and a Trial per execution, so each run's
    # jobs are associated and load_run() inside training attaches to them (-> Studio Experiments).
    experiment_config = PipelineExperimentConfig(
        experiment_name=PIPELINE_NAME,
        trial_name=Join(on="-", values=[PIPELINE_NAME, run_tag, ExecutionVariables.PIPELINE_EXECUTION_ID]),
    )

    return Pipeline(
        name=PIPELINE_NAME,
        pipeline_experiment_config=experiment_config,
        parameters=[
            bucket, prefix, data_uri, instance_type, run_name, run_tag, epochs, batch_size,
            num_dataloader_workers, micro_batch_size, grad_checkpoint, max_length, lr, k_neutral,
            k_pos, k_neg, lambda_supcon, lambda_sign, lambda_neutral, use_neutral_normalizer,
            model_val_freq, log_every_batches, approval,
        ],
        steps=[train_step, register_step],
        sagemaker_session=session,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["upsert", "run"])
    parser.add_argument("--data-uri", help="S3 prefix with the intact/ data layout (required for run)")
    # Override any pipeline parameter at launch, repeatable, e.g. `--param Epochs=1 --param
    # RunTag=b-baseline`. Values go on the wire as strings (SageMaker coerces to the param's type), so a
    # bare `--param Epochs=1` is fine. Lets one upserted definition drive the A/B runs (and this smoke)
    # without editing defaults.
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a pipeline parameter (repeatable).")
    args = parser.parse_args()

    pipeline = build_pipeline()
    pipeline.upsert(role_arn=config.execution_role(), tags=config.DEFAULT_TAGS)
    print(f"Upserted pipeline: {PIPELINE_NAME}")
    if args.action == "run":
        if not args.data_uri:
            parser.error("--data-uri is required for run")
        overrides = {}
        for item in args.param:
            if "=" not in item:
                parser.error(f"--param must be KEY=VALUE, got {item!r}")
            key, value = item.split("=", 1)
            overrides[key.strip()] = value.strip()
        parameters = {"IntactDataS3Uri": args.data_uri, **overrides}
        execution = pipeline.start(parameters=parameters)
        print(f"Started execution: {execution.arn}")
        if overrides:
            print(f"Parameter overrides: {overrides}")


if __name__ == "__main__":
    main()
