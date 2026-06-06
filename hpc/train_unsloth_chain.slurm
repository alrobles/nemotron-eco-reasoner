#!/bin/bash
# === Continuous Multi-node Training Chain ===
# Chains 6h jobs via Slurm array. Each auto-resumes from prior checkpoint.
# Usage: sbatch hpc/train_unsloth_chain.slurm   |   scancel <jobid> to stop

#SBATCH --job-name=nem-chain
#SBATCH --array=0-9%1
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

if [ "$ARRAY_TASK" -eq 0 ]; then
    RESUME_MODE=""
    echo "=== Chain Job 0: Fresh training ==="
else
    LATEST=$(ls -dt "$OUT"/checkpoint-* 2>/dev/null | head -1 || echo "")
    if [ -n "$LATEST" ]; then
        RESUME_MODE="--env RESUME_CHECKPOINT=$LATEST"
        echo "=== Chain Job $ARRAY_TASK: Resume from $LATEST ==="
    else
        RESUME_MODE="--env RESUME_CHECKPOINT=True"
        echo "=== Chain Job $ARRAY_TASK: Auto-resume ==="
    fi
fi

mkdir -p "$OUT" "$TMPDIR" "$PIPBASE"
MASTER_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -1)
TOTAL_GPUS=$((SLURM_NTASKS_PER_NODE * SLURM_NNODES))
echo "Nodes: $SLURM_NNODES x $SLURM_NTASKS_PER_NODE = $TOTAL_GPUS GPUs | Master: $MASTER_NODE"

srun --ntasks=1 --nodes=$SLURM_NNODES bash -c "
    mkdir -p $PIPBASE
    apptainer exec --nv --bind $TMPDIR:/tmp,$PIPBASE:$PIPBASE \
        --env PYTHONUSERBASE=$PIPBASE \
        $CONTAINER bash -c '
        pip install --user --no-cache-dir -q datasets==4.3.0 mamba-ssm causal-conv1d bitsandbytes unsloth_zoo 2>&1 | tail -1
        pip install --user --no-cache-dir -q --no-deps unsloth 2>&1 | tail -1
    '
"

srun --ntasks=$TOTAL_GPUS --nodes=$SLURM_NNODES \
    apptainer exec --nv --bind "$TMPDIR:/tmp,$PIPBASE:$PIPBASE" \
        --env PYTHONUSERBASE="$PIPBASE" \
        --env TORCH_COMPILE_DISABLE=1 \
        --env MODEL_PATH="$MODEL_PATH" \
        --env DATA_PATH="$DATA" \
        --env OUT_PATH="$OUT" \
        $RESUME_MODE \
        "$CONTAINER" \
        torchrun \
            --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_NTASKS_PER_NODE \
            --master_addr=$MASTER_NODE --master_port=29501 \
            --rdzv_backend=c10d --rdzv_endpoint=$MASTER_NODE:29501 \
            "$SCRIPT"

EXIT=$?
echo "Chain job $ARRAY_TASK exit: $EXIT"
rm -rf "$TMPDIR"
