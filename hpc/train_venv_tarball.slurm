#!/bin/bash
#SBATCH --job-name=nem-dora-venv
#SBATCH --partition=sixhour
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:mi210:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/a474r867/scratch/nemotron-eco-reasoner/logs/nem_dora_venv_%j.out
#SBATCH --error=/home/a474r867/scratch/nemotron-eco-reasoner/logs/nem_dora_venv_%j.err

# ============================================================================
# Nemotron DoRA Training - Pre-built venv tarball mode
# ZERO pip installs. ZERO Lustre for dependencies.
# ============================================================================

set -e

echo "==========================================="
echo "NEMOTRON DoRA TRAINING START: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $SLURM_GPUS"
echo "==========================================="

# Paths (all on Lustre/scratch - read only)
VENV_TARBALL="/home/a474r867/scratch/nemotron-venv-rocm61.tar.gz"
MODEL_PATH="/home/a474r867/scratch/nemotron-model"
DATASET_PATH="/home/a474r867/scratch/nemotron-eco-reasoner/data/kaggle_5k_train.jsonl"
TRAIN_SCRIPT="/home/a474r867/scratch/nemotron-eco-reasoner/scripts/train_with_venv.py"
OUTPUT_DIR="/home/a474r867/scratch/nemotron-eco-reasoner/outputs/nem_dora_$(date +%Y%m%d_%H%M)"

mkdir -p "$(dirname $DATASET_PATH)"
mkdir -p "$OUTPUT_DIR"
mkdir -p /home/a474r867/scratch/nemotron-eco-reasoner/logs

echo ""
echo "Inputs:"
echo "  venv tarball: $VENV_TARBALL"
echo "  model: $MODEL_PATH"
echo "  dataset: $DATASET_PATH"
echo "  output: $OUTPUT_DIR"
echo ""

# Verify inputs exist
for f in "$VENV_TARBALL" "$TRAIN_SCRIPT"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found"
        exit 1
    fi
done
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: model dir $MODEL_PATH not found"
    exit 1
fi
if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: dataset $DATASET_PATH not found"
    exit 1
fi

echo "All inputs verified."
echo ""

# Run training with python3.11 (available on cluster)
python3.11 "$TRAIN_SCRIPT" \
    --venv-tarball "$VENV_TARBALL" \
    --model-path "$MODEL_PATH" \
    --dataset "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --max-steps 500 \
    --batch-size 1 \
    --gradient-accumulation 8 \
    --learning-rate 2e-4 \
    --lora-rank 32

echo ""
echo "==========================================="
echo "TRAINING COMPLETE: $(date)"
echo "Output: $OUTPUT_DIR"
echo "==========================================="
