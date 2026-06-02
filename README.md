# Nemotron Eco-Reasoner

Dual-purpose LoRA fine-tuning of NVIDIA Nemotron-3-Nano-30B-A3B for:

1. **Reasoning puzzles** — Kaggle [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/) (deadline: June 15, 2026)
2. **Ecological agent tasks** — ecoSeek scientific assistant (tool-calling, taxonomy, host-parasite extraction)

One model, two capabilities. Trained on a combined dataset: 50% Kaggle logic puzzles + 50% ecology synthetic data from [ecoagent](https://github.com/alrobles/ecoagent).

## Architecture

- **Base model:** Nemotron-3-Nano-30B-A3B-BF16 (30B total, ~3B active per token)
- **Method:** 4-bit QLoRA (rank 32, alpha 64)
- **Hardware:** A100 80GB or MI210 64GB (single GPU), or multi-node MI210 (FSDP+QLoRA)
- **Framework:** Hugging Face PEFT + bitsandbytes

## Quick Start

```bash
# Clone
git clone https://github.com/alrobles/nemotron-eco-reasoner.git
cd nemotron-eco-reasoner

# Install
pip install -r requirements.txt

# Prepare dual dataset
python scripts/prepare_dataset.py --kaggle-csv data/train.csv --output data/combined_dataset.jsonl

# Train (single GPU)
python scripts/train_qlora.py --dataset data/combined_dataset.jsonl --output checkpoints/

# Evaluate
python scripts/evaluate.py --adapter checkpoints/final/

# Package for Kaggle submission
python scripts/submit_kaggle.py --adapter checkpoints/final/ --output submission.zip
```

## HPC (KU CRC)

### Container (recommended for MI210)

Pre-built Apptainer container with PyTorch 2.6.0 ROCm, transformers, peft, trl — zero setup.
See [containers/README.md](containers/README.md) for full docs.

```bash
# On KU-HPC cluster
CONTAINER=/home/a474r867/scratch/nemotron-eco-reasoner/nemotron-rocm.sif

# Verify
apptainer exec --rocm $CONTAINER python3 -c "import torch; print(torch.cuda.is_available())"

# Train
apptainer exec --rocm $CONTAINER python3 scripts/train_bf16_lora.py ...
```

### Slurm templates (legacy venv approach)

```bash
# Submit to cluster
sbatch hpc/train_a100.slurm
# or
sbatch hpc/train_mi210.slurm
# or multi-node (3 nodes × 2 MI210s)
sbatch hpc/train_multi_node.slurm
```

## Dataset Composition

| Source | Examples | Description |
|--------|----------|-------------|
| Kaggle reasoning | ~15K | Logic puzzles (bit manipulation, algebra, truth tables) |
| EcoAgent CoFID | ~5K | Host-parasite Q&A |
| EcoAgent tool-calling | ~5K | Ecological tool selection and synthesis |
| EcoAgent triplets | ~3K | Host-parasite relationship extraction |
| EcoAgent taxonomy | ~2K | Taxonomic resolution (WoRMS) |
| **Total** | **~30K** | ShareGPT format |

## Evaluation

Dual benchmark:
- **Reasoning accuracy:** Puzzle answer match (\boxed{...}) on Kaggle test split
- **Ecology accuracy:** Tool-call accuracy, triplet extraction F1, abstract classification, taxonomy resolution, report quality

## License

Apache 2.0 — see [LICENSE](LICENSE)
