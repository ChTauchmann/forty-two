# Forty-Two ML Base Image
# Based on mbrack/forty-two with Flash Attention added
#
# Build: docker build -t ctctctct/forty-two:latest .
# Push:  docker push ctctctct/forty-two:latest

FROM mbrack/forty-two:cuda-11.8-pytorch-2.2-gpu-mpi-748dda4

LABEL maintainer="tauchmann"
LABEL description="Forty-Two ML training image with Flash Attention"
LABEL version="1.0"

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

# Install Flash Attention 2 build dependencies
RUN pip install ninja packaging

# Install Flash Attention 2 (build from source if no pre-built wheel)
RUN MAX_JOBS=4 pip install flash-attn --no-build-isolation || \
    echo "Flash Attention installation failed, trying alternative method" && \
    pip install flash-attn==2.5.9.post1 --no-build-isolation || \
    echo "Flash Attention not available for this CUDA/PyTorch combination"

# Upgrade HuggingFace ecosystem to latest
RUN pip install --upgrade \
    transformers \
    accelerate \
    datasets \
    tokenizers \
    safetensors \
    huggingface_hub

# Upgrade/install training libraries
RUN pip install --upgrade \
    peft \
    trl \
    bitsandbytes \
    deepspeed

# Install evaluation and utilities
RUN pip install --upgrade \
    evaluate \
    lm-eval \
    wandb \
    tensorboard

# Install additional utilities
RUN pip install --upgrade \
    rich \
    typer \
    python-dotenv \
    httpx

# Clean up pip cache (ignore errors if cache is empty)
RUN pip cache purge || true

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Health check - verify key packages
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python -c "import transformers; print(f'Transformers: {transformers.__version__}')" && \
    (python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')" || echo "Flash Attention not installed")

# Default command
CMD ["/bin/bash"]
