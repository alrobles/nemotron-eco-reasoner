#!/bin/bash
#SBATCH --job-name=nemotron-kaggle
#SBATCH --partition=gpu
#SBATCH --gres=gpu:mi210:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/nemotron_kaggle_%j.out
#SBATCH --error=logs/nemotron_kaggle_%j.err
#SBATCH --signal=B:SIGUSR1@300

# Nemotron-3-Nano-30B BF16 DoRA training — single MI210 64GB (ROCm)
#
# DoRA (Weight-Decomposed LoRA) — +3-4% reasoning vs standard LoRA.
# No quantization — pure BF16, no bitsandbytes.
# MoE: 30B total params, ~3B active. BF16 + rank-32 DoRA ~22 GB VRAM.
#
# Key changes from old QLoRA approach:
#   - No bitsandbytes — pure BF16 DoRA
#   - Kaggle + ecology mixed dataset
#   - Uses chat template formatting

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
export NEMOTRON_USE_DORA=1
export NEMOTRON_EPOCHS=3
export NEMOTRON_BATCH_SIZE=2
export NEMOTRON_GRAD_ACCUM=8
export NEMOTRON_LEARNING_RATE="2e-4"
export NEMOTRON_MAX_LENGTH=2048
export NEMOTRON_SAVE_STEPS=200
export NEMOTRON_WANDB_PROJECT="nemotron-kaggle-reasoner"

# ── Venv setup ─────────────────────────────────────────────────────
# Use the existing ROCm venv (no bitsandbytes required)
VENV="$HOME/scratch/venv_rocm"
if [ ! -d "$VENV" ]; then
    echo "Creating ROCm venv..."
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
    pip install transformers accelerate peft trl datasets safetensors einops wandb pandas tqdm
else
    source "$VENV/bin/activate"
fi

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "ROCm available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# ── Prepare Kaggle + ecology dataset ────────────────────────────────
DATASET_JSONL="$DATASET_DIR/combined_dataset.jsonl"
if [ ! -f "$DATASET_JSONL" ]; then
    echo "Generating combined Kaggle + ecology dataset..."
    python "$PROJECT_DIR/scripts/prepare_dataset.py" \
        --kaggle-csv "$KAGGLE_CSV" \
        --output "$DATASET_JSONL" \
        --kaggle-ratio 0.5 \
        --total 20000
fi

echo "Dataset: $(wc -l < "$DATASET_JSONL") examples"

# ── Train ──────────────────────────────────────────────────────────
echo "Starting BF16 DoRA training on MI210..."
echo "Output: $OUTPUT_DIR"

python "$PROJECT_DIR/scripts/train_bf16_lora.py" \
    --dataset "$DATASET_JSONL" \
    --output "$OUTPUT_DIR"

echo ""
echo "========================================"
echo " Training complete!"
echo " Adapter: $OUTPUT_DIR/final/"
echo "========================================"
echo ""

# ── Quick evaluation ───────────────────────────────────────────────
echo "Running evaluation on 100 puzzles..."
python "$PROJECT_DIR/scripts/evaluate.py" \
    --adapter "$OUTPUT_DIR/final/" \
    --kaggle-csv "$KAGGLE_CSV" \
    --n 100 \
    --output "$OUTPUT_DIR/eval_results.json"

echo ""
echo "Done! Adapter at: $OUTPUT_DIR/final/"
echo "Eval results at: $OUTPUT_DIR/eval_results.json"
