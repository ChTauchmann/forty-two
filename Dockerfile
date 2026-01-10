# Forty-Two ML Base Image
# Based on mbrack/forty-two - MINIMAL modifications
# The base image already has: PyTorch 2.2, Flash Attention 2.5.9, transformers, peft, trl, etc.
#
# Build: docker build -t ctctctct/forty-two:latest .
# Push:  docker push ctctctct/forty-two:latest

FROM mbrack/forty-two:cuda-11.8-pytorch-2.2-gpu-mpi-748dda4

LABEL maintainer="tauchmann"
LABEL description="Forty-Two ML image with system utilities (LLM-optimized)"
LABEL version="1.2"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

# Python environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system utilities only - DO NOT touch Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    htop \
    nvtop \
    iotop \
    tmux \
    screen \
    tree \
    vim \
    nano \
    less \
    jq \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Remove torchvision to avoid nms operator conflict with HF packages
# The base image has torchvision 0.17.2+cu118 which causes issues
RUN pip uninstall -y torchvision || true

# Skip Python imports during build - they require CUDA runtime
# Verify packages at runtime instead
RUN echo "Build complete - verify packages at runtime with CUDA"

# Default command
CMD ["/bin/bash"]
