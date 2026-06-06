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
# FIX v7: ALL deps in one shot — datasets, mamba-ssm, bitsandbytes, unsloth+zoo
# Isolated PYTHONUSERBASE, torch untouched at 2.5.1

set -euo pipefail
export TORCH_COMPILE_DISABLE=1

CONTAINER=/home/a474r867/scratch/nemotron-eco-reasoner/nemotron-cuda.sif
MODEL_PATH=/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848
DATA=/home/a474r867/scratch/nemotron-eco-reasoner/data/train_merged.jsonl
OUT=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/unsloth
SCRIPT=/home/a474r867/scratch/nemotron-eco-reasoner/scripts/train_unsloth.py
TMPDIR=/tmp/nem_us_$$
PIPBASE=/tmp/pip_$$

mkdir -p "$OUT" "$TMPDIR" "$PIPBASE"

echo "===== Unsloth Training $(date) ====="

# ── GPU check ────────────────────────────────────────
echo "--- GPU check ---"
apptainer exec --nv --bind "$TMPDIR:/tmp,$PIPBASE:$PIPBASE" \
    --env PYTHONUSERBASE="$PIPBASE" \
    "$CONTAINER" python3 -c "
import torch; print(f'PyTorch {torch.__version__} | GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)')
"

# ── Install ALL deps (isolated, torch untouched) ────
echo "--- Installing deps ---"
apptainer exec --nv --bind "$TMPDIR:/tmp,$PIPBASE:$PIPBASE" \
    --env PYTHONUSERBASE="$PIPBASE" \
    "$CONTAINER" bash -c '
set -euo pipefail
pip install --user --no-cache-dir -q \
    "datasets==4.3.0" \
    mamba-ssm causal-conv1d \
    bitsandbytes \
    unsloth_zoo \
    2>&1 | tail -3

# unsloth must be --no-deps to prevent torch upgrade
pip install --user --no-cache-dir -q --no-deps unsloth 2>&1 | tail -1

echo "--- Verifying imports ---"
python3 -c "import mamba_ssm; print(\"mamba-ssm OK\")"
python3 -c "import bitsandbytes; print(\"bitsandbytes OK\")"
python3 -c "import unsloth; print(\"unsloth OK\")"
python3 -c "import torch; print(f\"torch: {torch.__version__}\")"
'

# ── Train ────────────────────────────────────────────
echo "--- Training ---"
apptainer exec --nv --bind "$TMPDIR:/tmp,$PIPBASE:$PIPBASE" \
    --env TORCH_COMPILE_DISABLE=1 \
    --env PYTHONUSERBASE="$PIPBASE" \
    --env MODEL_PATH="$MODEL_PATH" \
    --env DATA_PATH="$DATA" \
    --env OUT_PATH="$OUT" \
    "$CONTAINER" python3 -u "$SCRIPT"

EXIT=$?
echo "Exit code: $EXIT"

if [ $EXIT -eq 0 ]; then
    echo "SUCCESS! Adapter: $OUT/"
    find "$OUT" -name "*.safetensors" -ls 2>/dev/null || true
else
    echo "FAILED with exit $EXIT"
fi
rm -rf "$TMPDIR" "$PIPBASE"
