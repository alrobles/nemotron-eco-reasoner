#!/bin/bash
#SBATCH --partition=sixhour
#SBATCH --account=kbs
#SBATCH --job-name=train_stage1_tas
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --time=05:55:00
#SBATCH --output=/home/a474r867/beegfs/stage1_conus_2015/models/train_stage1_%j.out
#SBATCH --error=/home/a474r867/beegfs/stage1_conus_2015/models/train_stage1_%j.err

set -euo pipefail
VENV="/home/a474r867/beegfs/venv_dl"
OUTDIR="/home/a474r867/beegfs/stage1_conus_2015/models"
SCRIPT="/home/a474r867/beegfs/stage1_conus_2015/train_stage1.py"
mkdir -p "$OUTDIR"

echo "[$(date)] Training stage 1 on $SLURM_JOB_NODELIST (GPU=$CUDA_VISIBLE_DEVICES)"
"$VENV/bin/python" "$SCRIPT" \
  --lr /home/a474r867/beegfs/stage1_conus_2015/era5_tas_2015_1-120_c.tif \
  --dem /home/a474r867/beegfs/stage1_conus_2015/dem_1-120.tif \
  --hr /home/a474r867/beegfs/stage1_conus_2015/chelsa_tas_2015_1-120.tif \
  --out-dir "$OUTDIR" \
  --patch-size 128 \
  --batch-size 32 \
  --n-resblocks 8 \
  --n-feats 64 \
  --epochs 100 \
  --lr-rate 1e-3 \
  --num-workers 4
