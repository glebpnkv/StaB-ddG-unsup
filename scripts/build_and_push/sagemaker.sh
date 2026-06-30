#!/bin/bash
# Build the project image and push it to Amazon ECR (AWS counterpart of gcp_training.sh).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

REGION="${AWS_REGION:-us-east-1}"
REPO_NAME="${REPO_NAME:-stab-ddg-unsup}"
DOCKERFILE="${DOCKERFILE:-containers/Dockerfile.sagemaker}"   # set to .gpu for the training image
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"

# AWS Deep Learning Container registry (base image for the GPU build); needs its own ECR login.
DLC_REGISTRY="763104351884.dkr.ecr.${REGION}.amazonaws.com"
if [ -z "$ACCOUNT_ID" ]; then
    echo "Error: could not resolve AWS account id (is the AWS CLI configured?)." >&2
    exit 1
fi

GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo local)"
TAG="${TAG:-$GIT_COMMIT}"
# Moving alias tag pushed alongside $TAG. Defaults to "latest" for the CPU/processing image; pass
# ALIAS_TAG=gpu for the GPU training image so it doesn't clobber the processing image's :latest.
ALIAS_TAG="${ALIAS_TAG:-latest}"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${REPO_NAME}:${TAG}"
IMAGE_URI_LATEST="${REGISTRY}/${REPO_NAME}:${ALIAS_TAG}"

echo "=========================================="
echo "  Account:     $ACCOUNT_ID"
echo "  Region:      $REGION"
echo "  Repository:  $REPO_NAME"
echo "  Image URI:   $IMAGE_URI"
echo "=========================================="

# Ensure the ECR repository exists.
aws ecr describe-repositories --repository-names "$REPO_NAME" --region "$REGION" >/dev/null 2>&1 \
    || aws ecr create-repository --repository-name "$REPO_NAME" --region "$REGION" >/dev/null

# Authenticate Docker to our ECR (push target) and to the AWS DLC registry (GPU base image pull).
aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$REGISTRY"
aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$DLC_REGISTRY"

# Build (linux/amd64 so it runs on SageMaker regardless of local arch) and push.
echo "  Dockerfile:  $DOCKERFILE"
docker build --platform=linux/amd64 \
    -t "$IMAGE_URI" -t "$IMAGE_URI_LATEST" \
    -f "$DOCKERFILE" .

docker push "$IMAGE_URI"
docker push "$IMAGE_URI_LATEST"

echo "=========================================="
echo "Pushed: $IMAGE_URI"
echo "Use in pipelines via:  export PIPELINE_IMAGE_URI=$IMAGE_URI"
echo "=========================================="
