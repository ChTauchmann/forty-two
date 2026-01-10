# Forty-Two ML Base Image
# Based on mbrack/forty-two with Flash Attention added
# Optimized for LLM/language modeling work
#
# Build: docker build -t ctctctct/forty-two:latest .
# Push:  docker push ctctctct/forty-two:latest

FROM mbrack/forty-two:cuda-11.8-pytorch-2.2-gpu-mpi-748dda4

LABEL maintainer="tauchmann"
LABEL description="Forty-Two ML training image with Flash Attention (LLM-optimized)"
LABEL version="1.1"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

# Python environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Install additional system utilities
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

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Remove torchvision to avoid compatibility issues (not needed for LLM work)
# The base image has torchvision 0.17.2+cu118 which conflicts with upgraded packages
RUN pip uninstall -y torchvision || true

# Flash Attention 2.5.9.post1 is already in base image (mbrack/forty-two)
# No need to rebuild - just install ninja for any future compilation needs
RUN pip install ninja packaging

# Upgrade HuggingFace ecosystem (but don't upgrade torch to avoid breaking compatibility)
RUN pip install --upgrade --no-deps \
    transformers \
    accelerate \
    datasets \
    tokenizers \
    safetensors \
    huggingface_hub

# Install missing dependencies for upgraded packages (without touching torch)
RUN pip install --upgrade \
    filelock \
    fsspec \
    pyyaml \
    regex \
    requests \
    tqdm \
    packaging \
    numpy

# Upgrade/install training libraries WITHOUT touching torch
# Use --no-deps to prevent torch upgrade, then install their deps separately
RUN pip install --upgrade --no-deps \
    peft \
    trl \
    bitsandbytes

# Install deepspeed (needs special handling - don't upgrade, use base version if present)
RUN pip install --upgrade --no-deps deepspeed || true

# Install evaluation and utilities (--no-deps for torch-dependent packages)
RUN pip install --upgrade --no-deps \
    evaluate \
    lm-eval

RUN pip install --upgrade \
    wandb \
    tensorboard

# Install additional utilities (these don't touch torch)
RUN pip install --upgrade \
    rich \
    typer \
    python-dotenv \
    httpx

# Install missing dependencies for training libs (excluding torch)
RUN pip install --upgrade \
    sentencepiece \
    protobuf \
    scipy \
    scikit-learn \
    rouge-score \
    nltk \
    py7zr

# Clean up pip cache (ignore errors if cache is empty)
RUN pip cache purge || true

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Health check - verify key packages and torch version hasn't changed
RUN python -c "import torch; v=torch.__version__; print(f'PyTorch: {v}'); assert v.startswith('2.2'), f'ERROR: torch upgraded to {v}, expected 2.2.x'" && \
    python -c "import transformers; print(f'Transformers: {transformers.__version__}')" && \
    python -c "import peft; print(f'PEFT: {peft.__version__}')" && \
    python -c "import trl; print(f'TRL: {trl.__version__}')" && \
    python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')" && \
    python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')"

# Default command
CMD ["/bin/bash"]
