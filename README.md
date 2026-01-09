# HessianLLM ML Base Image

Docker image for ML training with Flash Attention support.

## Specifications

| Component | Version |
|-----------|---------|
| Base | NVIDIA CUDA 12.4.1 + cuDNN (Ubuntu 22.04) |
| Python | 3.11 |
| PyTorch | 2.5.1 |
| Flash Attention | 2.7.3 |
| Transformers | 4.47+ |
| DeepSpeed | 0.16+ |

## Included Packages

### Core ML
- torch, torchvision, torchaudio
- flash-attn (pre-built, no compilation needed)
- transformers, accelerate, datasets, tokenizers
- peft, trl, bitsandbytes
- deepspeed
- vllm

### Evaluation & Tracking
- evaluate, lm-eval
- wandb, tensorboard

### System Tools
- tmux, screen, htop, nvtop, iotop
- vim, nano, tree, git, git-lfs
- SSH server for distributed training

## Building

From a machine with Docker access:

```bash
cd /pfss/mlde/workspaces/mlde_wsp_HessianLLM/tauchmann/docker/hessian-ml-base
./build.sh latest
```

Or manually:
```bash
docker build -t ctctctct/hessian-ml-base:latest .
docker push ctctctct/hessian-ml-base:latest
```

## Usage with Determined AI

In your shell-config.yaml:
```yaml
environment:
  force_pull_image: true
  image:
    cuda: ctctctct/hessian-ml-base:latest
```

Then start a shell:
```bash
det shell start --config-file shell-config.yaml -w HessianLLM
```

## Verifying Installation

```python
import torch
import flash_attn
import transformers

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Flash Attention: {flash_attn.__version__}")
print(f"Transformers: {transformers.__version__}")
```

## Tags

- `latest` - Most recent build
- `py311-pt251-cu124` - Explicit version tag
- `YYYYMMDD` - Date-based tags for reproducibility
