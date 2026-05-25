#!/bin/bash
#SBATCH --job-name=nemotron-a100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/nemotron_a100_%j.out
#SBATCH --error=logs/nemotron_a100_%j.err
#SBATCH --signal=B:SIGUSR1@300

# Nemotron-3-Nano-30B QLoRA training — single A100 80GB
# 4-bit QLoRA with rank 32 fits in ~35 GB VRAM on A100

set -euo pipefail

PROJECT_DIR="$HOME/nemotron-eco-reasoner"
DATASET_DIR="$PROJECT_DIR/data"
OUTPUT_DIR="$HOME/scratch/nemotron-training/$(date +%Y%m%d_%H%M)"
KAGGLE_CSV="$DATASET_DIR/train.csv"

mkdir -p "$DATASET_DIR" "$OUTPUT_DIR" logs

# ── Environment ────────────────────────────────────────────────────
export NEMOTRON_BASE_MODEL="nvidia/Nemotron-3-Nano-30B-A3B-BF16"
export NEMOTRON_LORA_RANK=32
export NEMOTRON_LORA_ALPHA=64
export NEMOTRON_EPOCHS=3
export NEMOTRON_BATCH_SIZE=2
export NEMOTRON_GRAD_ACCUM=8
export NEMOTRON_LEARNING_RATE="2e-4"
export NEMOTRON_MAX_LENGTH=2048
export NEMOTRON_SAVE_STEPS=200
export NEMOTRON_WANDB_PROJECT="nemotron-eco-reasoner"

# ── Activate environment (Apptainer or conda) ──────────────────────
if command -v apptainer &>/dev/null && [ -f "$HOME/images/pytorch_cuda.sif" ]; then
    CONTAINER="$HOME/images/pytorch_cuda.sif"
    echo "Using Apptainer: $CONTAINER"
    RUN="apptainer exec --nv --bind $HOME:$HOME $CONTAINER"
else
    echo "Using conda environment"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate nemotron
    RUN=""
fi

# ── Prepare dataset ────────────────────────────────────────────────
if [ ! -f "$DATASET_DIR/combined_dataset.jsonl" ]; then
    echo "Generating combined dataset..."
    $RUN python "$PROJECT_DIR/scripts/prepare_dataset.py" \
        --kaggle-csv "$KAGGLE_CSV" \
        --ecoagent-dir "$HOME/ecoagent" \
        --output "$DATASET_DIR/combined_dataset.jsonl" \
        --kaggle-ratio 0.5 \
        --total 30000
fi

# ── Train ──────────────────────────────────────────────────────────
echo "Starting training..."
echo "Output: $OUTPUT_DIR"
nvidia-smi 2>/dev/null || true

$RUN python "$PROJECT_DIR/scripts/train_qlora.py" \
    --dataset "$DATASET_DIR/combined_dataset.jsonl" \
    --output "$OUTPUT_DIR"

echo "Training complete!"
echo "Adapter saved to: $OUTPUT_DIR/final/"

# ── Package submission ─────────────────────────────────────────────
$RUN python "$PROJECT_DIR/scripts/submit_kaggle.py" \
    --adapter "$OUTPUT_DIR/final/" \
    --output "$OUTPUT_DIR/submission.zip"

echo "Submission packaged: $OUTPUT_DIR/submission.zip"
