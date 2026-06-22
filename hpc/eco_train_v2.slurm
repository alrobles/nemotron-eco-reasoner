#!/bin/bash
#SBATCH --job-name=eco-v2-train
#SBATCH --partition=sixhour
#SBATCH --time=05:55:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/a474r867/scratch/eco_train_v2_%j.out
#SBATCH --error=/home/a474r867/scratch/eco_train_v2_%j.err

# eco_train_v2.slurm — Two-phase ecological reasoning trainer
# Incorporates 3rd place (0.900) + VCDAD silver techniques
#
# Default: 1x PRO6000 (96GB Blackwell) — BF16 native, torch nightly+cu128
#
# GPU overrides:
#   A100:       sbatch --gres=gpu:a100:1 --export=ALL,VENV_PATH=/home/a474r867/scratch/ecocoder-training/venv hpc/eco_train_v2.slurm
#   MI210 2x:   sbatch --gres=gpu:mi210:2 --ntasks=2 --ntasks-per-node=2 --export=ALL,VENV_PATH=.../rocm-venv hpc/eco_train_v2.slurm

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
SEQ_LEN=${SEQ_LEN:-2048}  # 4096 OOMs on PRO6000 96GB; 2048 fits safely
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-2e-4}
NUDGE_LR=${NUDGE_LR:-5e-6}
NUDGE_FRAC=${NUDGE_FRAC:-0.3}
NEFTUNE_ALPHA=${NEFTUNE_ALPHA:-5.0}
VENV_PATH=${VENV_PATH:-/home/a474r867/scratch/nemotron-blackwell-venv}

echo "== ECO TRAIN V2 =="
echo "  Job: $SLURM_JOB_ID on $(hostname)"
echo "  GPUs: $(echo $CUDA_VISIBLE_DEVICES 2>/dev/null || echo 'N/A')"
echo "  Data: $DATA_PATH"
echo "  Output: $OUT_PATH"
echo "  Target: $TARGET_TOTAL steps ($STEPS_PER_JOB per job)"
echo "  Phase1: lr=$LR (no clipping)"
echo "  Phase2: lr=$NUDGE_LR (frac=$NUDGE_FRAC)"
echo "  NEFTune: alpha=$NEFTUNE_ALPHA"
echo "  SeqLen: $SEQ_LEN"
echo "  Venv: $VENV_PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || \
  rocm-smi --showproductname 2>/dev/null || echo "  GPU info unavailable"
date

# Load newer Python for Blackwell venv (HPC default is 3.9)
module load python/3.12 2>/dev/null || module load python/3.11 2>/dev/null || true

# Offline mode (compute nodes may lack internet)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Activate venv
source "$VENV_PATH/bin/activate" 2>/dev/null || {
    echo "ERROR: venv not found at $VENV_PATH"
    exit 1
}

echo "Python: $(python3 --version)"
echo "Torch: $(python3 -c 'import torch; print(torch.__version__, "CUDA" if torch.cuda.is_available() else "CPU")' 2>/dev/null || echo 'MISSING')"

# AMD/ROCm settings (for MI210 runs)
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

echo "Data records: $(wc -l < "$DATA_PATH")"

# DDP env — randomize port to avoid collisions with other jobs on same node
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((10000 + RANDOM % 50000))

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
    sbatch --export=ALL,DATA_PATH=$DATA_PATH,OUT_PATH=$OUT_PATH,TARGET_TOTAL=$TARGET_TOTAL,STEPS_PER_JOB=$STEPS_PER_JOB,VENV_PATH=$VENV_PATH $R/hpc/eco_train_v2.slurm
    echo "RESUBMITTED"
fi

exit $EXIT
