# Forty-Two ML Image with Flash Attention

Docker image based on `mbrack/forty-two` with Flash Attention 2 added.

## Base Image

Built on top of `mbrack/forty-two:cuda-11.8-pytorch-2.2-gpu-mpi-748dda4` which includes:
- CUDA 11.8
- PyTorch 2.2
- MPI support for distributed training

## Additions

| Component | Description |
|-----------|-------------|
| Flash Attention 2 | GPU-optimized attention mechanism |
| tmux, screen | Terminal multiplexers |
| htop, nvtop, iotop | System monitoring |
| vim, nano | Text editors |
| Latest HuggingFace | transformers, accelerate, peft, trl |
| lm-eval | Language model evaluation |

## Building

Builds are automated via GitHub Actions on push to main branch.

For manual builds:
```bash
./build.sh latest
```

## Usage with Determined AI

In your shell-config.yaml:
```yaml
environment:
  force_pull_image: true
  image:
    cuda: ctctctct/forty-two:latest
```

Then start a shell:
```bash
det shell start --config-file shell-config.yaml -w HessianLLM
```

## Verifying Flash Attention

```python
import torch
import flash_attn

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Flash Attention: {flash_attn.__version__}")
```

## Tags

- `latest` - Most recent build
- `flash-attn` - Explicit Flash Attention tag
- `YYYYMMDD` - Date-based tags for reproducibility
