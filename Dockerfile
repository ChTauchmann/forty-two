# HessianLLM ML Base Image
# Python 3.11 + PyTorch 2.5.1 + CUDA 12.4 + Flash Attention 2.7.3
#
# Build: docker build -t ctctctct/hessian-ml-base:latest .
# Push:  docker push ctctctct/hessian-ml-base:latest

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

LABEL maintainer="tauchmann"
LABEL description="HessianLLM ML training base image with Flash Attention"
LABEL version="2.0"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

# CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Python environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python and build tools
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    cmake \
    ninja-build \
    git \
    git-lfs \
    curl \
    wget \
    # Networking and utilities
    openssh-client \
    openssh-server \
    rsync \
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
    # Libraries
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    liblzma-dev \
    libsqlite3-dev \
    libreadline-dev \
    libncurses5-dev \
    libncursesw5-dev \
    zlib1g-dev \
    # InfiniBand support
    libibverbs-dev \
    ibverbs-utils \
    rdma-core \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.5.1 with CUDA 12.4
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install Flash Attention 2 (pre-built wheel for CUDA 12.4 + PyTorch 2.5)
# Using the official flash-attn package which has pre-built wheels
RUN pip install flash-attn==2.7.3 --no-build-isolation

# Install HuggingFace ecosystem
RUN pip install \
    transformers>=4.47.0 \
    accelerate>=1.2.0 \
    datasets>=3.2.0 \
    tokenizers>=0.21.0 \
    safetensors>=0.4.0 \
    huggingface_hub>=0.27.0

# Install training libraries
RUN pip install \
    peft>=0.14.0 \
    trl>=0.13.0 \
    bitsandbytes>=0.45.0 \
    deepspeed>=0.16.0

# Install evaluation and utilities
RUN pip install \
    evaluate>=0.4.0 \
    lm-eval>=0.4.0 \
    wandb>=0.19.0 \
    tensorboard>=2.18.0

# Install additional ML utilities
RUN pip install \
    scipy \
    scikit-learn \
    pandas \
    pyarrow \
    matplotlib \
    seaborn \
    tqdm \
    rich \
    typer \
    pyyaml \
    python-dotenv \
    httpx \
    aiohttp

# Install vLLM for inference (optional but useful)
RUN pip install vllm>=0.6.0 || echo "vLLM install failed, skipping"

# Clean up pip cache
RUN pip cache purge

# Create workspace directory
RUN mkdir -p /workspace
WORKDIR /workspace

# SSH configuration for distributed training
RUN mkdir -p /var/run/sshd && \
    echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config && \
    echo 'PasswordAuthentication no' >> /etc/ssh/sshd_config && \
    echo 'StrictHostKeyChecking no' >> /etc/ssh/ssh_config

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Health check - verify key packages
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')" && \
    python -c "import transformers; print(f'Transformers: {transformers.__version__}')" && \
    python -c "print('CUDA available:', torch.cuda.is_available())"

# Default command
CMD ["/bin/bash"]
