#!/bin/bash
#SBATCH --job-name=nem-us
#SBATCH --partition=sixhour
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/unsloth/train_%j.out
#SBATCH --error=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/unsloth/train_%j.err

# === Unsloth 4-bit QLoRA — Nemotron-3-Nano-30B-A3B ===
# Unsloth's FastLanguageModel handles hybrid Mamba2+MoE+Attention,
# including num_logits_to_keep=1 that breaks vanilla bitsandbytes.

set -euo pipefail

export TORCH_COMPILE_DISABLE=1

CONTAINER=/home/a474r867/scratch/nemotron-eco-reasoner/nemotron-cuda.sif
MODEL_PATH=/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848
DATA=/home/a474r867/scratch/nemotron-eco-reasoner/data/train_merged.jsonl
OUT=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/unsloth
SCRIPT=/home/a474r867/scratch/nemotron-eco-reasoner/scripts/train_unsloth.py
TMPDIR=/tmp/nem_us_$$

mkdir -p "$OUT" "$TMPDIR"

echo "===== Unsloth Training $(date) ====="
echo "Model:  $MODEL_PATH"
echo "Data:   $DATA"
echo "Output: $OUT"
echo "====================================="

# ── GPU check ────────────────────────────────────────────────────
echo ""
echo "--- GPU check ---"
apptainer exec --nv --bind "$TMPDIR:/tmp" "$CONTAINER" python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
print(f'GPU count: {torch.cuda.device_count()}')
"

# ── Install deps ─────────────────────────────────────────────────
echo ""
echo "--- Installing mamba-ssm + unsloth ---"
apptainer exec --nv --bind "$TMPDIR:/tmp" "$CONTAINER" bash -c "
    set -euo pipefail
    pip install --no-cache-dir -q mamba-ssm causal-conv1d 2>&1 | tail -3
    python3 -c 'import mamba_ssm; print(\"mamba-ssm OK\")'
    pip install --no-cache-dir -q --no-deps unsloth 2>&1 | tail -3
    python3 -c 'import unsloth; print(\"unsloth OK (torch untouched)\")'
    python3 -c 'import torch; print(f\"torch: {torch.__version__}\")'
"

# ── Train ────────────────────────────────────────────────────────
echo ""
echo "--- Training ---"
apptainer exec --nv --bind "$TMPDIR:/tmp" \
    --env TORCH_COMPILE_DISABLE=1 \
    --env MODEL_PATH="$MODEL_PATH" \
    --env DATA_PATH="$DATA" \
    --env OUT_PATH="$OUT" \
    "$CONTAINER" python3 -u "$SCRIPT"

EXIT=$?
echo ""
echo "Exit code: $EXIT"

if [ $EXIT -eq 0 ]; then
    echo "SUCCESS!"
    echo "Adapter: $OUT/"
    find "$OUT" -name "*.safetensors" -ls 2>/dev/null || true
    find "$OUT" -name "manifest.json" -exec cat {} \; 2>/dev/null || true
else
    echo "FAILED with exit $EXIT"
fi

rm -rf "$TMPDIR"
