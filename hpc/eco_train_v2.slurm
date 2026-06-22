#!/bin/bash
#SBATCH --job-name=eco-v2-train
#SBATCH --partition=sixhour
#SBATCH --time=05:55:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:mi210:2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --output=/home/a474r867/scratch/eco_train_v2_%j.out
#SBATCH --error=/home/a474r867/scratch/eco_train_v2_%j.err

# eco_train_v2.slurm — Two-phase ecological reasoning trainer
# Lessons from Kaggle Nemotron silver solutions:
#   - Two-phase (Train→Nudge)
#   - NEFTune embedding noise
#   - Stratified sampling by ecological method
#   - Pure BF16 (MI210)
#   - MoE weight tying
#
# Usage:
#   sbatch hpc/eco_train_v2.slurm
#   sbatch --export=ALL,DATA_PATH=...,OUT_PATH=... hpc/eco_train_v2.slurm
#   # PRO6000 (single GPU, QLoRA not needed for BF16):
#   sbatch --gres=gpu:pro6000:1 --ntasks=1 --ntasks-per-node=1 hpc/eco_train_v2.slurm

set -euo pipefail

# Repo root
R=/home/a474r867/scratch/nemotron-eco-reasoner

# Model cache
M=/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848

# Defaults (overridable via --export)
DATA_PATH=${DATA_PATH:-$R/data/ecoreasoner_train.jsonl}
OUT_PATH=${OUT_PATH:-$R/outputs/eco_v2_twophase}
TARGET_TOTAL=${TARGET_TOTAL:-500}
STEPS_PER_JOB=${STEPS_PER_JOB:-250}
SEQ_LEN=${SEQ_LEN:-4096}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-2e-4}
NUDGE_LR=${NUDGE_LR:-5e-6}
NUDGE_FRAC=${NUDGE_FRAC:-0.3}
NEFTUNE_ALPHA=${NEFTUNE_ALPHA:-5.0}

echo "== ECO TRAIN V2 =="
echo "  Job: $SLURM_JOB_ID on $(hostname)"
echo "  GPUs: $(echo $CUDA_VISIBLE_DEVICES 2>/dev/null || echo 'N/A')"
echo "  Data: $DATA_PATH"
echo "  Output: $OUT_PATH"
echo "  Target: $TARGET_TOTAL steps"
echo "  Phase1: lr=$LR (no clipping)"
echo "  Phase2: lr=$NUDGE_LR (frac=$NUDGE_FRAC)"
echo "  NEFTune: alpha=$NEFTUNE_ALPHA"
echo "  SeqLen: $SEQ_LEN"
date

# Activate venv
source /home/a474r867/scratch/nemotron-venv/bin/activate 2>/dev/null || true
set -euo pipefail

# AMD/ROCm settings
if command -v rocminfo &>/dev/null; then
    export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
    export HSA_FORCE_FINE_GRAIN_PCIE=1
    echo "  ROCm detected: PYTORCH_HIP_ALLOC_CONF=$PYTORCH_HIP_ALLOC_CONF"
fi

# CUDA wheels (Mamba SSM kernels)
export LD_LIBRARY_PATH="/home/a474r867/scratch/wheels:${LD_LIBRARY_PATH:-}"

# Git pull latest code
cd $R && git pull --ff-only 2>/dev/null || true

# Check data exists; if not, run filter script
if [ ! -f "$DATA_PATH" ]; then
    echo "Data not found, running filter_dataset.py..."
    python3 $R/scripts/filter_dataset.py --output "$DATA_PATH"
fi

# DDP env
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Export training vars
export M DATA="$DATA_PATH" OUT="$OUT_PATH"
export TARGET_TOTAL STEPS_PER_JOB SEQ_LEN GRAD_ACCUM
export LR NUDGE_LR NUDGE_FRAC NEFTUNE_ALPHA

mkdir -p "$OUT_PATH"

echo "Starting training..."
srun python3 $R/hpc/eco_train_v2.py

EXIT=$?
echo "== DONE: exit=$EXIT at $(date) =="

# Auto-resubmit if not done
if grep -q "NEED_RESUBMIT" /home/a474r867/scratch/eco_train_v2_${SLURM_JOB_ID}.out 2>/dev/null; then
    echo "Auto-resubmitting..."
    sbatch --export=ALL,DATA_PATH=$DATA_PATH,OUT_PATH=$OUT_PATH,TARGET_TOTAL=$TARGET_TOTAL,STEPS_PER_JOB=$STEPS_PER_JOB $R/hpc/eco_train_v2.slurm
    echo "RESUBMITTED"
fi

exit $EXIT
