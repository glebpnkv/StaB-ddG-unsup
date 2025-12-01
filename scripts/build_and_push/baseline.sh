#!/bin/bash
# build_and_push.sh

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root (parent of scripts/)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Change to project root to ensure relative paths work
cd "$PROJECT_ROOT"

# Configuration
PROJECT_ID=$(gcloud config get-value project)
if [ -z "$PROJECT_ID" ]; then
    echo "Error: No GCP project set. Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

REGION="${REGION:-us-central1}"
REPO_NAME="${REPO_NAME:-stab-ddg-unsup}"
IMAGE_NAME="${IMAGE_NAME:-baseline}"
PYTHON_VERSION=$(cat .python-version | tr -d '\n' | sed 's/\s//g')

# Add patch version if not present (3.10 -> 3.10.16)
if [[ $PYTHON_VERSION =~ ^[0-9]+\.[0-9]+$ ]]; then
    PYTHON_VERSION="${PYTHON_VERSION}.16"
fi

# Use git commit hash as tag, or allow override
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "local")
TAG="${TAG:-$GIT_COMMIT}"

# Full image path
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TAG}"
IMAGE_URI_LATEST="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"

echo "=========================================="
echo "Build Configuration:"
echo "  Project ID:     $PROJECT_ID"
echo "  Region:         $REGION"
echo "  Repository:     $REPO_NAME"
echo "  Image Name:     $IMAGE_NAME"
echo "  Python Version: $PYTHON_VERSION"
echo "  Tag:            $TAG"
echo "  Image URI:      $IMAGE_URI"
echo "=========================================="

# Ensure Artifact Registry repository exists
echo "Ensuring Artifact Registry repository exists..."
gcloud artifacts repositories describe $REPO_NAME \
    --location=$REGION 2>/dev/null || \
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION \
    --description="ML container images for StaB-ddG project"

# Configure Docker to authenticate with Artifact Registry
echo "Configuring Docker authentication..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# Build the image
echo "Building Docker image..."
docker build \
    --platform=linux/amd64 \
    --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
    -t ${IMAGE_URI} \
    -t ${IMAGE_URI_LATEST} \
    -f containers/Dockerfile \
    .

# Push to Artifact Registry
echo "Pushing image to Artifact Registry..."
docker push ${IMAGE_URI}
docker push ${IMAGE_URI_LATEST}

echo "=========================================="
echo "✓ Successfully built and pushed:"
echo "  ${IMAGE_URI}"
echo "  ${IMAGE_URI_LATEST}"
echo "=========================================="

# Output for use in pipelines
echo ""
echo "Use this image URI in your Vertex AI pipelines:"
echo "  ${IMAGE_URI}"