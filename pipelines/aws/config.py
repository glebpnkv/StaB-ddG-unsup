"""Shared configuration for the AWS / SageMaker pipelines.

Mirrors the role that the ``PIPELINE_*`` environment variables play on the GCP side, but resolves
SageMaker-native concepts: the execution role, the default S3 bucket, the ECR image URI and a
``PipelineSession`` (which records ``.run()`` calls as pipeline steps instead of launching them).

Everything is overridable by environment variable so the same code works locally, in CI and inside a
SageMaker notebook/Studio:

    PIPELINE_IMAGE_URI   full ECR image URI (default: <acct>.dkr.ecr.<region>.amazonaws.com/<project>:latest)
    PIPELINE_ROLE_ARN    SageMaker execution role ARN
    PIPELINE_BUCKET      S3 bucket for artifacts (default: <project>-<acct>-<region>)
    PIPELINE_S3_PREFIX   key prefix within the bucket (default: data/intact)
    AWS_REGION           region (default: us-east-1)
"""
from __future__ import annotations

import functools
import os

import boto3
import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession

PROJECT = "stab-ddg-unsup"
REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
DEFAULT_S3_PREFIX = os.environ.get("PIPELINE_S3_PREFIX", "data/intact")

# Tags propagated to every pipeline / job for cost attribution and lineage filtering.
DEFAULT_TAGS = [{"Key": "project", "Value": PROJECT}]


@functools.lru_cache(maxsize=1)
def boto_session() -> boto3.Session:
    return boto3.Session(region_name=REGION)


@functools.lru_cache(maxsize=1)
def account_id() -> str:
    return boto_session().client("sts").get_caller_identity()["Account"]


@functools.lru_cache(maxsize=1)
def pipeline_session() -> PipelineSession:
    """Session used when *defining* a pipeline: processor.run(...) is recorded, not executed."""
    return PipelineSession(boto_session=boto_session())


@functools.lru_cache(maxsize=1)
def sagemaker_session() -> sagemaker.Session:
    """Session used for direct (non-pipeline) SDK calls, e.g. resolving the default bucket."""
    return sagemaker.Session(boto_session=boto_session())


def default_bucket() -> str:
    """S3 bucket for pipeline artifacts. Defaults to SageMaker's managed bucket
    (``sagemaker-<region>-<account>``), which is created automatically if absent."""
    override = os.environ.get("PIPELINE_BUCKET")
    if override:
        return override
    return sagemaker_session().default_bucket()


def execution_role() -> str:
    """SageMaker execution role ARN. Inside SageMaker this is auto-detected; elsewhere set
    PIPELINE_ROLE_ARN (an IAM role that SageMaker can assume with S3 + ECR + SageMaker access)."""
    role = os.environ.get("PIPELINE_ROLE_ARN")
    if not role:
        try:
            role = sagemaker.get_execution_role(sagemaker_session())
        except Exception as exc:  # not running inside SageMaker and no env override
            raise RuntimeError(
                "Could not resolve a SageMaker execution role. Set PIPELINE_ROLE_ARN to an IAM role "
                "ARN that SageMaker can assume (with AmazonSageMakerFullAccess + S3/ECR access)."
            ) from exc

    # Guard the most common footgun: with PIPELINE_ROLE_ARN unset and an IAM Identity Center (SSO)
    # login, get_execution_role() returns the *caller's* SSO "reserved" role. That is not a role
    # SageMaker can assume (it doesn't trust sagemaker.amazonaws.com), so using it as the pipeline
    # execution role fails only later, at upsert/start, with a cryptic "does not exist or does not
    # trust 'sagemaker.amazonaws.com'". Fail here instead, with the fix.
    if "aws-reserved/sso.amazonaws.com" in role or ":role/aws-reserved/" in role:
        raise RuntimeError(
            f"Resolved execution role is an AWS SSO reserved role that SageMaker cannot assume:\n"
            f"    {role}\n"
            "This happens when PIPELINE_ROLE_ARN is unset and you are logged in via SSO. Point it at a "
            "dedicated SageMaker execution role instead, e.g.:\n"
            "    export PIPELINE_ROLE_ARN=arn:aws:iam::<acct>:role/service-role/AmazonSageMaker-ExecutionRole-...\n"
            "Find one in the SageMaker console (Studio -> Domain -> user profile -> 'Execution role'), or:\n"
            "    aws iam list-roles --query \"Roles[?contains(RoleName,'SageMaker-ExecutionRole')].Arn\" --output text"
        )
    return role


def image_uri(tag: str | None = None) -> str:
    """ECR URI of the project image (built by scripts/build_and_push/sagemaker.sh)."""
    override = os.environ.get("PIPELINE_IMAGE_URI")
    if override:
        return override
    tag = tag or os.environ.get("PIPELINE_IMAGE_TAG", "latest")
    return f"{account_id()}.dkr.ecr.{REGION}.amazonaws.com/{PROJECT}:{tag}"


def s3_uri(*parts: str, bucket: str | None = None) -> str:
    bucket = bucket or default_bucket()
    key = "/".join(p.strip("/") for p in parts if p)
    return f"s3://{bucket}/{key}"
