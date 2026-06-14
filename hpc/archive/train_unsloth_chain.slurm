#!/bin/bash
# === Continuous Multi-node Training Chain ===
# Chains 6h jobs via Slurm job dependency. Each job auto-resumes
# from the previous checkpoint. Runs until manually cancelled.
#
# Usage:
#   sbatch hpc/train_unsloth_chain.slurm          # Start chain
#   scancel <jobid>                                # Stop chain
#
# Job 1 (array task 0): Fresh training, 0-6h
# Job 2 (array task 1): Resume from job 1 checkpoint, 6-12h
# Job 3 (array task 2): Resume from job 2 checkpoint, 12-18h
# ... continues indefinitely until cancelled

#SBATCH --job-name=nem-chain
#SBATCH --array=0-9%1           # 10 jobs max, one at a time
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=05:50:00
#SBATCH --partition=sixhour
#SBATCH --output=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/multi/chain_%A_%a.out
#SBATCH --error=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/multi/chain_%A_%a.err
#SBATCH --signal=B:SIGUSR1@120

set -euo pipefail
export TORCH_COMPILE_DISABLE=1

CONTAINER=/home/a474r867/scratch/nemotron-eco-reasoner/nemotron-cuda.sif
MODEL_PATH=/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848
DATA=/home/a474r867/scratch/nemotron-eco-reasoner/data/train_merged.jsonl
OUT=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/multi
SCRIPT=/home/a474r867/scratch/nemotron-eco-reasoner/scripts/train_unsloth_dist.py
TMPDIR=/tmp/nem_chain_$$
PIPBASE=/tmp/pip_$$

ARRAY_TASK=${SLURM_ARRAY_TASK_ID:-0}

# Determine resume mode
if [ "$ARRAY_TASK" -eq 0 ]; then
    RESUME_MODE=""
    echo "=== Chain Job 0: Fresh training ==="
else
    # Resume from latest checkpoint
    LATEST=$(ls -dt "$OUT"/checkpoint-* 2>/dev/null | head -1 || echo "")
    if [ -n "$LATEST" ]; then
        RESUME_MODE="--env RESUME_CHECKPOINT=$LATEST"
        echo "=== Chain Job $ARRAY_TASK: Resuming from $LATEST ==="
    else
        RESUME_MODE="--env RESUME_CHECKPOINT=True"
        echo "=== Chain Job $ARRAY_TASK: Auto-resume (no specific checkpoint found) ==="
    fi
fi

mkdir -p "$OUT" "$TMPDIR" "$PIPBASE"
MASTER_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -1)
TOTAL_GPUS=$((SLURM_NTASKS_PER_NODE * SLURM_NNODES))

echo "Nodes: $SLURM_NNODES × $SLURM_NTASKS_PER_NODE = $TOTAL_GPUS GPUs"
echo "Master: $MASTER_NODE | Array: $ARRAY_TASK"

# ── Install deps ─────────────────────────────────────────────────
srun --ntasks=1 --nodes=$SLURM_NNODES bash -c "
    mkdir -p $PIPBASE
    apptainer exec --nv --bind $TMPDIR:/tmp,$PIPBASE:$PIPBASE \
        --env PYTHONUSERBASE=$PIPBASE \
        $CONTAINER bash -c '
        pip install --user --no-cache-dir -q datasets==4.3.0 mamba-ssm causal-conv1d bitsandbytes unsloth_zoo 2>&1 | tail -1
        pip install --user --no-cache-dir -q --no-deps unsloth 2>&1 | tail -1
    '
"

# ── Launch ────────────────────────────────────────────────────────
echo "--- torchrun $TOTAL_GPUS processes ---"
srun --ntasks=$TOTAL_GPUS --nodes=$SLURM_NNODES \
    apptainer exec --nv --bind "$TMPDIR:/tmp,$PIPBASE:$PIPBASE" \
        --env PYTHONUSERBASE="$PIPBASE" \
        --env TORCH_COMPILE_DISABLE=1 \
        --env MODEL_PATH="$MODEL_PATH" \
        --env DATA_PATH="$DATA" \
        --env OUT_PATH="$OUT" \
        $RESUME_MODE \
        "$CONTAINER" env PYTHONUNBUFFERED=1 \
                torchrun \
            --nnodes=$SLURM_NNODES \
            --nproc_per_node=$SLURM_NTASKS_PER_NODE \
            --master_addr=$MASTER_NODE \
            --master_port=29501 \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$MASTER_NODE:29501 \
            "$SCRIPT"

EXIT=$?
echo "Chain job $ARRAY_TASK exit: $EXIT"

if [ $EXIT -eq 0 ]; then
    echo "SUCCESS! Training complete after $ARRAY_TASK jobs ($((ARRAY_TASK * 6))h)"
    # Exit successfully — Slurm won't spawn more array tasks if we use singleton
    # For continuous: just let it exit, next array task auto-starts
else
    echo "Chain stopped with exit $EXIT"
fi

rm -rf "$TMPDIR"
