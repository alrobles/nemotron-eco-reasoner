#!/bin/bash
# NEM_MI210_MULTINODE — Multi-node DDP training on MI210 (ROCm)
# Uses torchrun for DDP across nodes, 1 GPU per process, full model per GPU.
# 2 nodes × 2 GPUs = 4 total processes.
#
# Usage:
#   sbatch --export=ALL,DATA_PATH=...,OUT_PATH=...,TARGET_TOTAL=500,STEPS_PER_JOB=200 \
#          hpc/nem_mi210_multinode.slurm
#
#SBATCH --job-name=nem-m210-mn
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:mi210:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=6
#SBATCH --output=/home/a474r867/scratch/nem_mi210_mn_%j.out
#SBATCH --error=/home/a474r867/scratch/nem_mi210_mn_%j.err

export PYTHONUNBUFFERED=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

R=/home/a474r867/scratch/nemotron-eco-reasoner
C=$R/nemotron-rocm.sif
M=/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848
DATA=${DATA_PATH:-$R/data/train_balanced_v10.jsonl}
OUT=${OUT_PATH:-$R/outputs/mi210_v10_mn}
TARGET_TOTAL=${TARGET_TOTAL:-500}
STEPS_PER_JOB=${STEPS_PER_JOB:-200}
SEQ_LEN=${SEQ_LEN:-2048}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-1e-4}
JOBID=${SLURM_JOB_ID:-$$}

mkdir -p "$OUT"

# ── Multi-node setup ─────────────────────────────────────
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=$((29500 + RANDOM % 1000))
NNODES=$SLURM_NNODES
NPROC=$SLURM_NTASKS

echo "=== MI210 MULTINODE job=$JOBID $(date) ==="
echo "  nodes=$NNODES gpus_per_node=1 total_gpus=$NPROC"
echo "  master=$MASTER_ADDR:$MASTER_PORT"
echo "  data=$DATA out=$OUT target=$TARGET_TOTAL batch=$STEPS_PER_JOB"
echo "  seq=$SEQ_LEN grad_accum=$GRAD_ACCUM lr=$LR"
echo "  node_list=$SLURM_JOB_NODELIST"

# ── Launch with torchrun ─────────────────────────────────
apptainer exec --rocm --bind "/dev/shm,/tmp,$M,$R" \
    --env PYTHONUNBUFFERED=1 \
    --env TORCH_COMPILE_DISABLE=1 \
    --env PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env OMP_NUM_THREADS=1 \
    --env DATA="$DATA" --env OUT="$OUT" --env M="$M" \
    --env JOBID=$JOBID --env TARGET_TOTAL=$TARGET_TOTAL \
    --env STEPS_PER_JOB=$STEPS_PER_JOB --env SEQ_LEN=$SEQ_LEN \
    --env GRAD_ACCUM=$GRAD_ACCUM --env LR=$LR \
    --env MASTER_ADDR=$MASTER_ADDR --env MASTER_PORT=$MASTER_PORT \
    --env NNODES=$NNODES --env NPROC=$NPROC \
    $C torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=mi210_mn_$JOBID \
    $R/scripts/train_mi210_multinode.py

PY_EXIT=$?
echo "Exit: $PY_EXIT"
exit $PY_EXIT
