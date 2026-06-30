#!/bin/bash
# Build the project image and push it to Amazon ECR (AWS counterpart of gcp_training.sh).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

REGION="${AWS_REGION:-us-east-1}"
REPO_NAME="${REPO_NAME:-stab-ddg-unsup}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
if [ -z "$ACCOUNT_ID" ]; then
    echo "Error: could not resolve AWS account id (is the AWS CLI configured?)." >&2
    exit 1
fi

GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo local)"
TAG="${TAG:-$GIT_COMMIT}"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${REPO_NAME}:${TAG}"
IMAGE_URI_LATEST="${REGISTRY}/${REPO_NAME}:latest"

echo "=========================================="
echo "  Account:     $ACCOUNT_ID"
echo "  Region:      $REGION"
echo "  Repository:  $REPO_NAME"
echo "  Image URI:   $IMAGE_URI"
echo "=========================================="

# Ensure the ECR repository exists.
aws ecr describe-repositories --repository-names "$REPO_NAME" --region "$REGION" >/dev/null 2>&1 \
    || aws ecr create-repository --repository-name "$REPO_NAME" --region "$REGION" >/dev/null

# Authenticate Docker to ECR.
aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$REGISTRY"

# Build (linux/amd64 so it runs on SageMaker regardless of local arch) and push.
docker build --platform=linux/amd64 \
    -t "$IMAGE_URI" -t "$IMAGE_URI_LATEST" \
    -f containers/Dockerfile.sagemaker .

docker push "$IMAGE_URI"
docker push "$IMAGE_URI_LATEST"

echo "=========================================="
echo "Pushed: $IMAGE_URI"
echo "Use in pipelines via:  export PIPELINE_IMAGE_URI=$IMAGE_URI"
echo "=========================================="
