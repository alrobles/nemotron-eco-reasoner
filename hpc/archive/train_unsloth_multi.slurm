#!/bin/bash
# === Multi-node Unsloth 4-bit QLoRA — Nemotron-3-Nano-30B-A3B ===
# 4 nodes × 2 A100 80GB = 8 GPUs, NCCL backend (10-16 GB/s)
# SIGUSR1 handler saves checkpoint 120s before 6h walltime kill
#
# Usage:
#   sbatch hpc/train_unsloth_multi.slurm
#   RESUME=1 sbatch hpc/train_unsloth_multi.slurm   # resume from latest
#   RESUME=/path/to/checkpoint sbatch ...            # resume from specific

#SBATCH --job-name=nem-multi
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:a100:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=05:50:00
#SBATCH --nodelist=r31r05n01,r31r10n01,r31r15n01,r31r20n01,r31r25n01,r31r35n01
#SBATCH --partition=sixhour
#SBATCH --output=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/multi/train_%j.out
#SBATCH --error=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/multi/train_%j.err
#SBATCH --signal=B:SIGUSR1@120

set -euo pipefail
export TORCH_COMPILE_DISABLE=1

CONTAINER=/home/a474r867/scratch/nemotron-eco-reasoner/nemotron-cuda.sif
MODEL_PATH=/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848
DATA=/home/a474r867/scratch/nemotron-eco-reasoner/data/train_merged.jsonl
OUT=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/multi
SCRIPT=/home/a474r867/scratch/nemotron-eco-reasoner/scripts/train_unsloth_dist.py
TMPDIR=/tmp/nem_multi_$$
PIPBASE=/tmp/pip_$$

# Handle resume
RESUME_FLAG=""
if [ "${RESUME:-}" = "1" ]; then
    RESUME_FLAG="--env RESUME_CHECKPOINT=True"
    echo "Resume mode: auto-find latest checkpoint in $OUT"
elif [ -n "${RESUME:-}" ]; then
    RESUME_FLAG="--env RESUME_CHECKPOINT=$RESUME"
    echo "Resume mode: specific checkpoint $RESUME"
fi

mkdir -p "$OUT" "$TMPDIR" "$PIPBASE"

MASTER_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -1)
TOTAL_GPUS=$((SLURM_NTASKS_PER_NODE * SLURM_NNODES))

echo "============================================="
echo "  Nemotron Multi-node Training"
echo "  Nodes: $SLURM_NNODES × $SLURM_NTASKS_PER_NODE GPUs = $TOTAL_GPUS GPUs"
echo "  Master: $MASTER_NODE"
echo "  Backend: NCCL (auto-configure)"
echo "  Output:  $OUT"
echo "  Resume:  ${RESUME:-no}"
echo "============================================="

# ── GPU check (rank 0 only) ──────────────────────────────────────
HEAD_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -1)
if [ "$(hostname -s)" = "$HEAD_NODE" ]; then
    echo ""
    echo "--- GPU check (head node) ---"
    apptainer exec --nv --bind "$TMPDIR:/tmp,$PIPBASE:$PIPBASE" \
        --env PYTHONUSERBASE="$PIPBASE" \
        "$CONTAINER" python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB)')
"
fi

# ── Install deps (all nodes) ─────────────────────────────────────
srun --ntasks=1 --nodes=$SLURM_NNODES bash -c "
    mkdir -p $PIPBASE
    apptainer exec --nv --bind $TMPDIR:/tmp,$PIPBASE:$PIPBASE \
        --env PYTHONUSERBASE=$PIPBASE \
        $CONTAINER bash -c '
        pip install --user --no-cache-dir -q datasets==4.3.0 mamba-ssm causal-conv1d bitsandbytes unsloth_zoo 2>&1 | tail -1
        pip install --user --no-cache-dir -q --no-deps unsloth 2>&1 | tail -1
    '
"
echo "Deps installed on all nodes"

# ── Launch distributed training ──────────────────────────────────
echo ""
echo "--- Launching torchrun ($TOTAL_GPUS processes) ---"

srun --ntasks=$TOTAL_GPUS --nodes=$SLURM_NNODES \
    apptainer exec --nv --bind "$TMPDIR:/tmp,$PIPBASE:$PIPBASE" \
        --env PYTHONUSERBASE="$PIPBASE" \
        --env TORCH_COMPILE_DISABLE=1 \
        --env MODEL_PATH="$MODEL_PATH" \
        --env DATA_PATH="$DATA" \
        --env OUT_PATH="$OUT" \
        $RESUME_FLAG \
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
echo ""
echo "Exit code: $EXIT"

if [ $EXIT -eq 0 ]; then
    echo "SUCCESS! Adapter: $OUT/"
    find "$OUT" -name "*.safetensors" -ls 2>/dev/null | head -3 || true
    find "$OUT" -name "manifest.json" -exec cat {} \; 2>/dev/null || true
else
    echo "FAILED with exit $EXIT"
    echo "Checkpoint should be at $OUT/checkpoint-latest"
fi

rm -rf "$TMPDIR"
