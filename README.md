# Nemotron Eco-Reasoner

**🟢 TRAINING ACTIVE — Jun 5, 2026 | Step 10, Loss: 18.29, ~96s/it on A100 40GB**  
First successful Nemotron-3-Nano-30B-A3B fine-tuning after 15+ iterations.  
Using **Unsloth 4-bit QLoRA** on A100 40GB with monkey-patched MoE forward
(dtype-safe index_add_ + top-k aggregation). See [CHANGELOG.md](CHANGELOG.md).

Dual-purpose fine-tuning for:

1. **Reasoning puzzles** — Kaggle [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/) (deadline: June 15, 2026)
2. **Ecological agent tasks** — ecoSeek scientific assistant (tool-calling, taxonomy, host-parasite extraction)

One model, two capabilities. Trained on a combined dataset: 50% Kaggle logic puzzles + 50% ecology synthetic data.

## Architecture

- **Base model:** Nemotron-3-Nano-30B-A3B-BF16 (30B total, ~3B active per token)
- **Method:** Unsloth 4-bit QLoRA, rank 32, BF16 compute — the only approach that works
- **MoE fix:** Custom monkey-patch for dtype-safe `index_add_` + top-k aggregation (Nemotron routes to 6 experts per token; vanilla bitsandbytes fails with shape/dtype mismatches)
- **Hardware:** A100 40GB (single GPU) — 4-bit model fits in ~16-20GB VRAM
- **Framework:** Unsloth 2026.6.1 + transformers 4.57 + TRL + PyTorch 2.5.1 (CUDA 12.4)
- **Container:** Pre-built Apptainer SIF (`nemotron-cuda.sif`, 7GB) with isolated PYTHONUSERBASE for deps

## Quick Start

```bash
# Clone
gh repo clone alrobles/nemotron-eco-reasoner
cd nemotron-eco-reasoner

# Prepare dataset (Kaggle CSV required in data/train.csv)
python scripts/prepare_dataset.py --kaggle-csv data/train.csv --output data/combined_dataset.jsonl

# Generate ecology dataset from PubMed papers
python scripts/generate_eco_dataset.py --output data/eco_dataset.jsonl
```

### Local training (reumanlab, RTX 2000 Ada)

```bash
python scripts/train_m1_container.py \
    --model /path/to/nemotron-model \
    --data data/kaggle_5k_train.jsonl \
    --output outputs/m1_run1 \
    --max_steps 500
```

### HPC training (KU CRC, MI210)

Pre-built Apptainer container with PyTorch 2.6.0 ROCm, transformers, peft, trl — zero setup. `mamba-ssm` installed at runtime (required for Nemotron hybrid architecture).

```bash
# Submit to cluster
sbatch hpc/train_m1_mi210.slurm

# A100 fallback
sbatch hpc/train_a100.slurm

# Multi-node scaling (3 nodes × 2 MI210s)
sbatch hpc/train_multi_node.slurm
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train_m1_container.py` | **Canonical** — DoRA training via Apptainer container |
| `scripts/train_bf16_lora.py` | BF16 LoRA training (A100, no container needed) |
| `scripts/prepare_dataset.py` | Generate combined Kaggle+ecology JSONL dataset |
| `scripts/generate_eco_dataset.py` | PubMed-based ecological reasoning examples |
| `scripts/cluster_api.py` | HTTP API for remote training orchestration |
| `scripts/evaluate.py` | Dual-benchmark evaluation |
| `scripts/submit_kaggle.py` | Package adapter for Kaggle submission |

## Project Structure

```
nemotron-eco-reasoner/
├── scripts/           # Training, dataset prep, eval, submission
├── hpc/               # Active Slurm templates (3)
│   └── archive/       # Deprecated templates (9, kept for reference)
├── containers/        # Apptainer definition + docs
├── data/              # Training JSONL datasets (gitignored)
├── outputs/           # Checkpoints and adapters (gitignored)
└── logs/              # Job output logs (gitignored)
```

## Dataset Composition

| Source | Examples | Description |
|--------|----------|-------------|
| Kaggle reasoning | ~15K | Logic puzzles (bit manipulation, algebra, truth tables) |
| EcoAgent CoFID | ~5K | Host-parasite Q&A |
| EcoAgent tool-calling | ~5K | Ecological tool selection and synthesis |
| EcoAgent triplets | ~3K | Host-parasite relationship extraction |
| EcoAgent taxonomy | ~2K | Taxonomic resolution |
| PubMed ecology | var | Paper-based ecological reasoning |
| **Total** | **~30K** | ShareGPT format |

## Evaluation

Dual benchmark:
- **Reasoning accuracy:** Puzzle answer match on Kaggle test split
- **Ecology accuracy:** Tool-call accuracy, triplet extraction F1, abstract classification, taxonomy resolution, report quality

## License

Apache 2.0 — see [LICENSE](LICENSE)
