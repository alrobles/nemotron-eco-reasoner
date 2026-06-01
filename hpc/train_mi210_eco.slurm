#!/bin/bash
#SBATCH --job-name=nemotron-eco
#SBATCH --partition=gpu
#SBATCH --gres=gpu:mi210:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/nemotron_eco_%j.out
#SBATCH --error=logs/nemotron_eco_%j.err
#SBATCH --signal=B:SIGUSR1@600

# Nemotron-3-Nano-30B BF16 LoRA — Kaggle reasoning + ecology
# Milestone 1: 10K Kaggle puzzles + ecology mix, BF16 LoRA rank=32
#
# No bitsandbytes — pure BF16. MoE: 30B total, ~3B active.
# Single MI210 (64 GB), ~20 GB VRAM with rank-32 LoRA.

set -euo pipefail

REPO_DIR="$HOME/scratch/nemotron-eco-reasoner"
DATASET_DIR="$REPO_DIR/data"
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

# ── Venv ───────────────────────────────────────────────────────────
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
echo "ROCm: $(python -c 'import torch; print(torch.cuda.is_available())')"

# ── Generate combined dataset (Kaggle 10K + ecology) ──────────────
COMBINED_JSONL="$DATASET_DIR/combined_dataset.jsonl"
if [ ! -f "$COMBINED_JSONL" ]; then
    echo "Generating combined Kaggle + ecology dataset..."
    python "$REPO_DIR/scripts/prepare_dataset.py" \
        --kaggle-csv "$KAGGLE_CSV" \
        --output "$COMBINED_JSONL" \
        --kaggle-ratio 0.5 \
        --total 20000
    echo "  -> $(wc -l < "$COMBINED_JSONL") examples"
else
    echo "Dataset exists: $(wc -l < "$COMBINED_JSONL") examples"
fi

# ── Train ──────────────────────────────────────────────────────────
echo "Starting BF16 LoRA training on MI210..."
echo "Output: $OUTPUT_DIR"

python "$REPO_DIR/scripts/train_bf16_lora.py" \
    --dataset "$COMBINED_JSONL" \
    --output "$OUTPUT_DIR"

echo ""
echo "========================================"
echo " Training complete!"
echo " Adapter: $OUTPUT_DIR/final/"
echo "========================================"
echo ""

# ── Evaluate ───────────────────────────────────────────────────────
echo "Evaluating on 100 Kaggle puzzles..."
python "$REPO_DIR/scripts/evaluate.py" \
    --adapter "$OUTPUT_DIR/final/" \
    --kaggle-csv "$KAGGLE_CSV" \
    --n 100 \
    --output "$OUTPUT_DIR/eval_results.json"

echo ""
echo "Done! Results: $OUTPUT_DIR/eval_results.json"
