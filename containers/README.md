# Containers: Nemotron Training on ROCm (MI210)

## TL;DR

```bash
# On KU-HPC cluster
apptainer exec --rocm /home/a474r867/scratch/nemotron-eco-reasoner/nemotron-rocm.sif \
  python3 -c "import torch; print(torch.cuda.is_available())"  # True on MI210
```

**One container, zero compilation, zero dependency hell.** Works on any MI210 node.

---

## Available Container

| Property | Value |
|----------|-------|
| **File** | `nemotron-rocm.sif` |
| **Size** | ~21 GB |
| **Base image** | `rocm/pytorch:rocm6.4.2_ubuntu22.04_py3.10_pytorch_release_2.6.0` |
| **PyTorch** | 2.6.0 (ROCm build: `+git5f46534`) |
| **Python** | 3.10 |
| **transformers** | 4.57.6 |
| **peft** | 0.17.1 |
| **trl** | 0.24.0 |
| **datasets** | 4.5.0 |
| **accelerate** | 1.10.1 |
| **sentencepiece** | 0.2.1 |
| **wandb / tensorboard** | included |

### Locations

| Machine | Path |
|---------|------|
| reumanlab (build host) | `/home/reumanlab/work/Github/nemotron-eco-reasoner/containers/nemotron-rocm.sif` |
| KU-HPC cluster | `/home/a474r867/scratch/nemotron-eco-reasoner/nemotron-rocm.sif` |

---

## Verified (June 2, 2026)

Tested on KU-HPC MI210 (job 22284212):

```
PyTorch 2.6.0+git5f46534
ROCm available: True
GPU: AMD Instinct MI210
VRAM: 63.98 GB
transformers+peft+trl: OK
```

---

## How to Use

### Single GPU training (MI210)

```bash
#!/bin/bash
#SBATCH --job-name=nem-train
#SBATCH --partition=sixhour
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/$USER/scratch/nemotron-eco-reasoner/train_%j.out

CONTAINER=/home/$USER/scratch/nemotron-eco-reasoner/nemotron-rocm.sif

apptainer exec --rocm $CONTAINER python3 \
  /home/$USER/scratch/nemotron-eco-reasoner/scripts/train_bf16_lora.py \
  --dataset /home/$USER/scratch/nemotron-eco-reasoner/data/combined_dataset.jsonl \
  --output /home/$USER/scratch/nemotron-eco-reasoner/outputs/run1
```

### Quick verification

```bash
apptainer exec --rocm nemotron-rocm.sif python3 -c "
import torch
print('PyTorch', torch.__version__)
print('ROCm:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"
```

### Interactive shell

```bash
apptainer shell --rocm nemotron-rocm.sif
```

---

## Build Recipe: `nemotron.def`

The container is built from `nemotron.def` (32 lines, zero compilation):

```def
Bootstrap: docker
From: rocm/pytorch:rocm6.4.2_ubuntu22.04_py3.10_pytorch_release_2.6.0

%post
    pip install --no-cache-dir \
        transformers==4.57.6 \
        peft==0.17.1 \
        trl==0.24.0 \
        datasets==4.5.0 \
        accelerate==1.10.1 \
        sentencepiece==0.2.1 \
        wandb \
        tensorboard

    python -c "import torch; print(f'PyTorch {torch.__version__}, ROCm: {torch.cuda.is_available()}')"
    python -c "import transformers, peft, trl; print('Training stack OK')"

%runscript
    exec python "$@"
```

### Build command

```bash
cd containers/
apptainer build --force nemotron-rocm.sif nemotron.def
```

### Push to cluster

```bash
scp nemotron-rocm.sif a474r867@hpc.crc.ku.edu:/home/a474r867/scratch/nemotron-eco-reasoner/
```

---

## Key Design Decision: Why No mamba-ssm Compilation?

**PyTorch 2.6.0+ has a pure-Python Mamba fallback.** Nemotron-3-Nano's Mamba layers work
without compiled `mamba-ssm` or `causal-conv1d` kernels on ROCm. This eliminates:

- The need for `hipcc` and ROCm SDK during build
- GPU detection during `%post` (which Apptainer doesn't support)
- The `amdgpu-arch` / `--offload-arch` compilation nightmare

The pure-Python fallback is slower than compiled kernels, but it **works correctly**
and training throughput is still dominated by the Transformer/MLP layers.

---

## The `nemotron-train.def` Mistake (DO NOT USE)

`nemotron-train.def` attempts to compile `causal-conv1d` + `mamba-ssm` from source
inside the Apptainer `%post` section. This fails for two reasons:

1. **Apptainer `%post` has no GPU access** — `amdgpu-arch` can't detect the GPU
   even on a node with MI210s. `--rocm` only works at runtime (`exec`/`run`).
2. **The precompiled wheel is 404** — `causal-conv1d` setup.py tries to download
   a precompiled wheel from GitHub releases and gets HTTP 404.

Attempts on both reumanlab (NVIDIA GPU) and KU-HPC (MI210 GPU) failed identically.
**Not necessary anymore** with PyTorch 2.6.0+.

---

## Common Pitfalls

### "ROCm: False" on reumanlab
This is **expected**. The container has ROCm libraries; reumanlab has an NVIDIA GPU.
ROCm can't talk to NVIDIA hardware. On the cluster with `--rocm` flag it works.

### "rootlesscontainers" warnings during build
Harmless. Non-root Apptainer builds can't set extended attributes. Ignore them.

### Parallel builds compete for Docker Hub
Don't run two `apptainer build` simultaneously — they share the cache and compete
for Docker Hub bandwidth. Observed: 3 concurrent builds caused significant slowdown.

### Version check: `+git5f46534` not `+rocm`
The AMD Docker image uses a custom PyTorch build identified by git hash, not the
`+rocm` suffix. Check with `torch.cuda.is_available()` and `torch.cuda.get_device_name(0)`,
not by string-matching the version.

### --no-deps is NOT needed
Unlike the old `/tmp` venv pattern, the container has full dependency resolution.
`pip install` inside the `%post` resolves all transitive deps correctly.
No `ModuleNotFoundError` cascades.

---

## Related

- Main README: `../README.md`
- Training scripts: `../scripts/train_bf16_lora.py`
- HPC templates: `../hpc/`
- Skill: `hpc-lora-fine-tuning` (Hermes skill)
