#!/bin/bash
# === train_mi210_dora_chain.slurm — MI210 DoRA checkpoint chain ===
#
# Designed for MI210 (68.7GB) — big VRAM, no Unsloth hacks needed.
# Strategy: 5-job checkpoint chain @ 5h each = ~25h continuous training.
# Uses venv-on-SSD (fast pip) + standard PEFT DoRA + BF16.
#
# WHY THIS BEATS Q6000:
#   - 68.7GB vs 24GB VRAM → 3× larger batch/seq
#   - seq=2048 (vs seq=48 on Q6000) → actual CoT reasoning
#   - DoRA rank=64 (vs rank=4) → better knowledge capture
#   - BF16 native (no quantization) → better gradient quality
#
# Usage:
#   sbatch hpc/train_mi210_dora_chain.slurm
#   DATA_PATH=/path/to/data.jsonl sbatch hpc/train_mi210_dora_chain.slurm
#   # Array: sbatch --array=0-4 hpc/train_mi210_dora_chain.slurm

#SBATCH --job-name=mi210_dora
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/a474r867/scratch/nemotron-eco-reasoner/logs/mi210_dora_%j.out
#SBATCH --error=/home/a474r867/scratch/nemotron-eco-reasoner/logs/mi210_dora_%j.err
#SBATCH --signal=B:SIGUSR1@120

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export TORCH_COMPILE_DISABLE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_DATASETS_OFFLINE=1

# ── Paths ─────────────────────────────────────────────────────────────
REPO_DIR="/home/a474r867/scratch/nemotron-eco-reasoner"
MODEL_DIR="/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848"
DATA="${DATA_PATH:-$REPO_DIR/data/train_cot_unified.jsonl}"
OUTPUT="$REPO_DIR/outputs/mi210_dora_v1"
VENV_DIR="/tmp/nemotron_mi210_$$"

# ── Config (override via env) ─────────────────────────────────────────
SEQ_LEN="${SEQ_LEN:-2048}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
PER_JOB_STEPS="${PER_JOB_STEPS:-200}"
LR="${LR:-1e-4}"
BATCH="${BATCH:-2}"
GRAD="${GRAD_ACCUM:-4}"

echo "=========================================="
echo "MI210 DoRA Training — Checkpoint Chain"
echo "=========================================="
echo "Model: $MODEL_DIR"
echo "Data:  $DATA ($(wc -l < "$DATA" 2>/dev/null || echo '?') lines)"
echo "GPU:   MI210 (68.7GB)"
echo "Config: seq=$SEQ_LEN rank=$LORA_RANK steps=$PER_JOB_STEPS lr=$LR"
echo "Batch:  $BATCH × $GRAD = $((BATCH * GRAD)) effective"
echo "Output: $OUTPUT"
echo "Job:    $SLURM_JOB_ID"
echo ""

mkdir -p "$OUTPUT" "$(dirname "$OUTPUT")"

# ── Step 1: Create venv on fast local SSD ─────────────────────────────
echo "=== Step 1: Python venv ==="
python3 -m venv --without-pip "$VENV_DIR" 2>/dev/null || python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install -q --upgrade pip 2>&1 | tail -1

# ── Step 2: PyTorch ROCm ──────────────────────────────────────────────
echo "=== Step 2: PyTorch ROCm 6.2 ==="
pip install --no-deps torch==2.6.0 torchvision==0.21.0 2>&1 | tail -2
pip install --index-url https://download.pytorch.org/whl/rocm6.2 \
    torch==2.6.0 torchvision==0.21.0 2>&1 | tail -3

# ── Step 3: Core deps ─────────────────────────────────────────────────
echo "=== Step 3: Core ML deps ==="
pip install -q huggingface_hub 2>&1 | tail -1
for pkg in \
    "transformers==4.57.6" \
    "peft==0.17.1" \
    "accelerate==1.10.1" \
    "datasets==4.5.0" \
    "trl==0.24.0" \
    "sentencepiece" \
    "numpy"; do
    pip install -q --no-deps "$pkg" 2>&1 | tail -1
done

# ── Step 4: Verify GPU ────────────────────────────────────────────────
echo "=== Step 4: GPU Check ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA/ROCm available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'BF16: {torch.cuda.is_bf16_supported()}')
"

# ── Step 5: Train ─────────────────────────────────────────────────────
echo "=== Step 5: Training ==="

cd "$REPO_DIR"

python scripts/train_mi210_dora.py \
    --model "$MODEL_DIR" \
    --data "$DATA" \
    --output "$OUTPUT" \
    --max_steps "$PER_JOB_STEPS" \
    --batch_size "$BATCH" \
    --grad_accum "$GRAD" \
    --lr "$LR" \
    --seq_len "$SEQ_LEN" \
    --use_dora 1 \
    --fsdp 0 \
    --packing 1

EXIT=$?

# ── Cleanup ───────────────────────────────────────────────────────────
rm -rf "$VENV_DIR"

echo ""
echo "Exit: $EXIT"
if [ $EXIT -eq 42 ]; then
    echo "SIGUSR1 — checkpoint saved, chain continues"
elif [ $EXIT -eq 0 ]; then
    echo "SUCCESS!"
    find "$OUTPUT" -name "*.safetensors" -type f 2>/dev/null | sort
fi

exit $EXIT
