#!/bin/bash
#SBATCH --job-name=nem-m1
#SBATCH --partition=sixhour
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/m1_run1/train_%j.out
#SBATCH --error=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/m1_run1/train_%j.err

# M1 Training — Nemotron-3-Nano + DoRA on single MI210
# Uses pre-built Apptainer container + runtime mamba-ssm install

set -euo pipefail

CONTAINER=/home/a474r867/scratch/nemotron-eco-reasoner/nemotron-rocm.sif
MODEL_DIR=/home/a474r867/scratch/nemotron-model-cache
DATA=/home/a474r867/scratch/nemotron-eco-reasoner/data/kaggle_5k_train.jsonl
OUTPUT=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/m1_run1
SCRIPTS=/home/a474r867/scratch/nemotron-eco-reasoner/scripts
TMPDIR=/tmp/nem_m1_$$

mkdir -p $OUTPUT $TMPDIR

echo "=== M1 Training ==="
echo "Container: $CONTAINER"
echo "Model dir: $MODEL_DIR"
echo "Data: $DATA"
echo "Output: $OUTPUT"
echo "==================="

# Find the actual model snapshot path
MODEL_PATH=$(find $MODEL_DIR -name "config.json" -path "*/snapshots/*" 2>/dev/null | head -1 | xargs dirname)
if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: Model not found in $MODEL_DIR"
    exit 1
fi
echo "Model path: $MODEL_PATH"

# FIX: Container has CUDA torch (torch 2.12.0+cu130) — reinstall ROCm torch
# in writable tmpfs overlay so it takes priority over container's torch
echo "=== FIX: Reinstalling ROCm torch ==="
apptainer exec --rocm --writable-tmpfs $CONTAINER bash -c "
    set -euo pipefail
    pip install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/rocm6.1 -q 2>&1 | tail -3
    python3 -c 'import torch; v=torch.__version__; assert "+rocm" in v, f"Expected ROCm torch, got {v}"; print(f"torch {v} OK"); assert torch.cuda.is_available(), "GPU not available"; print(f"GPU: {torch.cuda.get_device_name(0)}")'
"
echo "ROCm torch ready."

# Install mamba-ssm at runtime (GPU visible with --rocm)
echo "Installing mamba-ssm..."
apptainer exec --rocm --writable-tmpfs --bind $TMPDIR:/tmp $CONTAINER bash -c "
    pip install --no-cache-dir mamba-ssm causal-conv1d 2>&1 | tail -5
    python3 -c 'import mamba_ssm; print(\"mamba-ssm OK\")'
"
echo "mamba-ssm ready."

# Train
echo "Starting training..."
apptainer exec --rocm --writable-tmpfs --bind $TMPDIR:/tmp $CONTAINER python3 -u $SCRIPTS/train_m1_container.py \
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
    ls -lh $OUTPUT/adapter_model.safetensors 2>/dev/null
fi
