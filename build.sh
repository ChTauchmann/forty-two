#!/bin/bash
# Build script for hessian-ml-base Docker image
#
# Usage: ./build.sh [tag]
# Example: ./build.sh latest
#          ./build.sh py311-pt251-cu124

set -e

IMAGE_NAME="ctctctct/hessian-ml-base"
TAG="${1:-latest}"
DATE_TAG=$(date +%Y%m%d)

echo "Building ${IMAGE_NAME}:${TAG}"
echo "========================================"

# Build the image
docker build \
    --progress=plain \
    --tag "${IMAGE_NAME}:${TAG}" \
    --tag "${IMAGE_NAME}:${DATE_TAG}" \
    --tag "${IMAGE_NAME}:py311-pt251-cu124" \
    .

echo ""
echo "Build complete!"
echo "========================================"
echo "Tags created:"
echo "  - ${IMAGE_NAME}:${TAG}"
echo "  - ${IMAGE_NAME}:${DATE_TAG}"
echo "  - ${IMAGE_NAME}:py311-pt251-cu124"
echo ""
echo "To push to Docker Hub:"
echo "  docker push ${IMAGE_NAME}:${TAG}"
echo "  docker push ${IMAGE_NAME}:${DATE_TAG}"
echo "  docker push ${IMAGE_NAME}:py311-pt251-cu124"
echo ""
echo "To test locally:"
echo "  docker run --gpus all -it ${IMAGE_NAME}:${TAG} python -c 'import flash_attn; print(flash_attn.__version__)'"
