#!/bin/bash
#SBATCH --job-name=nem-m1-a100
#SBATCH --partition=sixhour
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/m1_run1/train_a100_%j.out
#SBATCH --error=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/m1_run1/train_a100_%j.err

# M1 Training — Nemotron-3-Nano + DoRA on single A100
# Container has CUDA torch natively — works on A100 without ROCm fix

set -euo pipefail

CONTAINER=/home/a474r867/scratch/nemotron-rocm.sif
MODEL_DIR=/home/a474r867/scratch/nemotron-model-cache
DATA=/home/a474r867/scratch/nemotron-eco-reasoner/data/kaggle_5k_train.jsonl
OUTPUT=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/m1_run1
SCRIPTS=/home/a474r867/scratch/nemotron-eco-reasoner/scripts
TMPDIR=/tmp/nem_m1_a100_$$

mkdir -p $OUTPUT $TMPDIR

echo "=== M1 Training (A100) ==="
echo "Container: $CONTAINER"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "==================="

# Find model path
MODEL_PATH=$(find $MODEL_DIR -name "config.json" -path "*/snapshots/*" 2>/dev/null | head -1 | xargs dirname)
if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: Model not found in $MODEL_DIR"
    exit 1
fi
echo "Model path: $MODEL_PATH"

# Verify GPU is visible with CUDA torch
echo "=== GPU verification ==="
apptainer exec --nv --bind $TMPDIR:/tmp $CONTAINER python3 -c "
import torch
print(f'torch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
assert torch.cuda.is_available(), 'GPU not available!'
"

# Install mamba-ssm with CUDA support
echo "=== Installing mamba-ssm ==="
apptainer exec --nv --bind $TMPDIR:/tmp $CONTAINER bash -c "
    set -euo pipefail
    pip install --no-cache-dir mamba-ssm causal-conv1d -q 2>&1 | tail -3
    python3 -c 'import mamba_ssm; print(\"mamba-ssm OK\")'
"

# Train
echo "=== Starting training ==="
apptainer exec --nv --bind $TMPDIR:/tmp $CONTAINER python3 -u $SCRIPTS/train_m1_container.py \
    --model "$MODEL_PATH" \
    --data "$DATA" \
    --output "$OUTPUT" \
    --max_steps 500 \
    --batch_size 1 \
    --grad_accum 8 \
    --lr 2e-4 \
    --max_seq_len 1024 \
    --num_epochs 1 \
    --use_dora 1

EXIT=$?
echo "Exit code: $EXIT"
rm -rf $TMPDIR

if [ $EXIT -eq 0 ]; then
    echo "DONE! Adapter at $OUTPUT/"
    ls -lh $OUTPUT/adapter_model.safetensors 2>/dev/null || echo "(no safetensors yet)"
fi
