#!/bin/bash
#SBATCH --job-name=nem-dora-m1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:mi210:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/nem_dora_mi210_%j.out
#SBATCH --error=logs/nem_dora_mi210_%j.err
#SBATCH --signal=B:SIGUSR1@300

# ── Milestone 1: BF16 DoRA on MI210 ──────────────────────────────
# Nemotron-3-Nano-30B-A3B-BF16 → DoRA rank=32 → ~35 GB VRAM
# Dataset: 10K total (5K Kaggle + 5K ecology) — proof of concept
# 3 epochs, batch=2, grad_accum=8 = effective batch 16
#
# If successful: scale to multi-GPU, more data, higher rank.
# If SIGUSR1: saves checkpoint, exits 42 for Slurm requeue.

set -euo pipefail

PROJECT_DIR="$HOME/work/Github/nemotron-eco-reasoner"
DATASET_DIR="$HOME/scratch/nemotron-eco-reasoner/data"
OUTPUT_DIR="$HOME/scratch/nemotron-eco-reasoner/outputs/m1_dora_$(date +%Y%m%d_%H%M)"
KAGGLE_CSV="$HOME/scratch/nemotron-kaggle/train.csv"
ECOAGENT_DIR="$HOME/work/Github/ecoagent"
VENV_DIR="$HOME/nemotron-venv"

mkdir -p "$DATASET_DIR" "$OUTPUT_DIR" logs

# ── Activate venv ────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo "Activated nemotron-venv: $(which python)"
else
    echo "ERROR: nemotron-venv not found at $VENV_DIR"
    echo "Run nem-venv-setup job first (job 22274648)"
    exit 1
fi

# ── Environment ──────────────────────────────────────────────────
export NEMOTRON_BASE_MODEL="nvidia/Nemotron-3-Nano-30B-A3B-BF16"
export NEMOTRON_LORA_RANK=32
export NEMOTRON_LORA_ALPHA=64
export NEMOTRON_USE_DORA=1           # DoRA: +3-4% reasoning
export NEMOTRON_EPOCHS=3
export NEMOTRON_BATCH_SIZE=2
export NEMOTRON_GRAD_ACCUM=8
export NEMOTRON_LEARNING_RATE="2e-4"
export NEMOTRON_MAX_LENGTH=2048
export NEMOTRON_SAVE_STEPS=100
export NEMOTRON_WANDB_PROJECT="nemotron-eco-reasoner"

# ── Check GPU ────────────────────────────────────────────────────
echo "=== GPU Info ==="
rocm-smi --showproductname 2>/dev/null || echo "rocm-smi unavailable"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA/ROCm: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# ── Prepare dataset (if not exists) ──────────────────────────────
COMBINED_JSONL="$DATASET_DIR/combined_kaggle_eco_10k.jsonl"
if [ ! -f "$COMBINED_JSONL" ]; then
    echo "=== Preparing dataset (10K: 5K Kaggle + 5K ecology) ==="
    python "$PROJECT_DIR/scripts/prepare_dataset.py" \
        --kaggle-csv "$KAGGLE_CSV" \
        --ecoagent-dir "$ECOAGENT_DIR" \
        --output "$COMBINED_JSONL" \
        --kaggle-ratio 0.5 \
        --total 10000
    echo "Dataset written: $(wc -l < "$COMBINED_JSONL") lines"
else
    echo "Dataset exists: $COMBINED_JSONL ($(wc -l < "$COMBINED_JSONL") lines)"
fi

# ── Train ────────────────────────────────────────────────────────
echo "=== Training: BF16 DoRA rank=32 on MI210 ==="
echo "Output: $OUTPUT_DIR"
echo "Start: $(date)"

python "$PROJECT_DIR/scripts/train_bf16_lora.py" \
    --dataset "$COMBINED_JSONL" \
    --output "$OUTPUT_DIR"

EXIT_CODE=$?

echo "=== Done: $(date) ==="
echo "Exit code: $EXIT_CODE"
echo "Adapter: $OUTPUT_DIR/final/"

exit $EXIT_CODE
