#!/bin/bash
# === Q6000 Checkpoint Chain — Nemotron-3-Nano-30B-A3B ===
# Chains 5h jobs on sixhour partition via Slurm array.
# Q6000 recipe: seq=128, rank=16, fp16 (Turing limitation).
# Each array task auto-resumes from the latest checkpoint.
#
# Usage:
#   sbatch hpc/train_q6k_chain.slurm          # Start chain
#   scancel <jobid>                             # Stop chain

#SBATCH --job-name=nem-q6k
#SBATCH --array=0-4%1
#SBATCH --partition=sixhour
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:q6000:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/q6k_chain/chain_%A_%a.out
#SBATCH --error=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/q6k_chain/chain_%A_%a.err
#SBATCH --signal=B:SIGUSR1@120

set -euo pipefail
export TORCH_COMPILE_DISABLE=1

CONTAINER=/home/a474r867/scratch/nemotron-eco-reasoner/nemotron-cuda.sif
MODEL_PATH=/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848
DATA=/home/a474r867/scratch/nemotron-eco-reasoner/data/train_final_v2.jsonl
OUT=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/q6k_chain
SCRIPT=/home/a474r867/scratch/nemotron-eco-reasoner/scripts/train_unsloth.py
TMPDIR=/tmp/nem_q6k_$$
PIPBASE=/tmp/pip_q6k_$$

ARRAY_TASK=${SLURM_ARRAY_TASK_ID:-0}

mkdir -p "$OUT" "$TMPDIR" "$PIPBASE"

echo "============================================="
echo "  Q6000 Chain — Job $ARRAY_TASK / $(date)"
echo "  Recipe: seq=128, rank=16, fp16"
echo "  Output: $OUT"
echo "============================================="

# Determine resume mode
RESUME_ENV=""
if [ "$ARRAY_TASK" -gt 0 ]; then
    RESUME_ENV="--env RESUME_CHECKPOINT=True"
    echo "  Mode: RESUME from latest checkpoint"
else
    echo "  Mode: FRESH training"
fi

# GPU info
echo "--- GPU ---"
export APPTAINER_WRITABLE_TMPFS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

apptainer exec --nv --bind "$TMPDIR:/tmp,$PIPBASE:$PIPBASE" \
    --env PYTHONUSERBASE="$PIPBASE" \
    "$CONTAINER" python3 -c "
import torch; print(f'PyTorch {torch.__version__} | GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)')
"

# Install deps
echo "--- Installing deps ---"
apptainer exec --nv --bind "$TMPDIR:/tmp,$PIPBASE:$PIPBASE" \
    --env PYTHONUSERBASE="$PIPBASE" \
    "$CONTAINER" bash << 'INSTALL_EOF'
set -euo pipefail
pip install --user --no-cache-dir -q --no-deps "torch==2.5.1" 2>&1 | tail -1
pip install --user --no-cache-dir -q mamba-ssm causal-conv1d 2>&1 | tail -1
pip install --user --no-cache-dir -q --no-deps "datasets==4.3.0" bitsandbytes unsloth_zoo 2>&1 | tail -1
pip install --user --no-cache-dir -q --no-deps unsloth 2>&1 | tail -1
python3 -c 'import unsloth; print("unsloth OK")'
INSTALL_EOF

# Train (Q6000 specific: seq=128, rank=16, fp16)
echo "--- Training (Q6000, seq=128, rank=16, fp16) ---"
apptainer exec --nv --bind "$TMPDIR:/tmp,$PIPBASE:$PIPBASE" \
    --env TORCH_COMPILE_DISABLE=1 \
    --env PYTHONUSERBASE="$PIPBASE" \
    --env APPTAINER_WRITABLE_TMPFS=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env MODEL_PATH="$MODEL_PATH" \
    --env DATA_PATH="$DATA" \
    --env OUT_PATH="$OUT" \
    --env MAX_SEQ_LEN=128 \
    --env LORA_RANK=16 \
    --env LORA_ALPHA=16 \
    --env MAX_STEPS=500 \
    --env SAVE_STEPS=100 \
    --env USE_FP16=1 \
    $RESUME_ENV \
    "$CONTAINER" env PYTHONUNBUFFERED=1 python3 -u "$SCRIPT"

EXIT=$?
echo ""
echo "Chain job $ARRAY_TASK exit: $EXIT"
if [ $EXIT -eq 0 ]; then
    echo "SUCCESS!"
    find "$OUT" -name "*.safetensors" -ls 2>/dev/null | tail -3 || true
    find "$OUT" -name "manifest.json" -exec cat {} \; 2>/dev/null || true
else
    echo "Exited with $EXIT — next array task will resume from checkpoint"
fi
rm -rf "$TMPDIR" "$PIPBASE"
