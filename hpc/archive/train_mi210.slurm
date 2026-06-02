#!/bin/bash
#SBATCH --job-name=nemotron-mi210
#SBATCH --partition=gpu
#SBATCH --gres=gpu:mi210:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/nemotron_mi210_%j.out
#SBATCH --error=logs/nemotron_mi210_%j.err
#SBATCH --signal=B:SIGUSR1@300

# Nemotron-3-Nano-30B QLoRA training — single MI210 64GB (ROCm)
# 4-bit QLoRA with rank 32 fits in ~25 GB VRAM on MI210
#
# Key differences from A100:
#   - Uses ROCm (rocm/pytorch image)
#   - bfloat16 may not be supported — falls back to float16
#   - bitsandbytes ROCm build required

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

# ── Activate Apptainer with ROCm ───────────────────────────────────
CONTAINER="$HOME/images/rocm_pytorch.sif"

if [ ! -f "$CONTAINER" ]; then
    echo "Building ROCm Apptainer image..."
    apptainer build "$CONTAINER" docker://rocm/pytorch:latest
fi

echo "Using Apptainer: $CONTAINER"
echo "Checking ROCm..."
apptainer exec --rocm --bind "$HOME:$HOME" "$CONTAINER" rocminfo 2>/dev/null | head -5 || echo "rocminfo not available"

# ── Prepare dataset ────────────────────────────────────────────────
if [ ! -f "$DATASET_DIR/combined_dataset.jsonl" ]; then
    echo "Generating combined dataset..."
    apptainer exec --rocm --bind "$HOME:$HOME" "$CONTAINER" \
        python "$PROJECT_DIR/scripts/prepare_dataset.py" \
            --kaggle-csv "$KAGGLE_CSV" \
            --ecoagent-dir "$HOME/ecoagent" \
            --output "$DATASET_DIR/combined_dataset.jsonl" \
            --kaggle-ratio 0.5 \
            --total 30000
fi

# ── Train ──────────────────────────────────────────────────────────
echo "Starting training on MI210..."
echo "Output: $OUTPUT_DIR"

apptainer exec --rocm --bind "$HOME:$HOME" "$CONTAINER" \
    python "$PROJECT_DIR/scripts/train_qlora.py" \
        --dataset "$DATASET_DIR/combined_dataset.jsonl" \
        --output "$OUTPUT_DIR"

echo "Training complete!"
echo "Adapter saved to: $OUTPUT_DIR/final/"

# ── Package submission ─────────────────────────────────────────────
apptainer exec --rocm --bind "$HOME:$HOME" "$CONTAINER" \
    python "$PROJECT_DIR/scripts/submit_kaggle.py" \
        --adapter "$OUTPUT_DIR/final/" \
        --output "$OUTPUT_DIR/submission.zip"

echo "Submission packaged: $OUTPUT_DIR/submission.zip"
