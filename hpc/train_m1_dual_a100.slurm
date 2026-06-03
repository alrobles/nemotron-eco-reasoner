#!/bin/bash
#SBATCH --job-name=nem-a100
#SBATCH --partition=sixhour
#SBATCH --time=05:30:00
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/m1_dual_a100/train_%j.out
#SBATCH --error=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/m1_dual_a100/train_%j.err

# M1 Dual-GPU A100 — Nemotron-3-Nano-30B + DoRA on 2× A100 (FSDP full_shard)
# 2× 80GB A100 = 160GB total → ~30GB per GPU with FSDP full_shard
# Container: nemotron-rocm.sif (torch 2.12.0+cu130 CUDA)

set -euo pipefail

CONTAINER=/home/a474r867/scratch/nemotron-rocm.sif
MODEL_DIR=/home/a474r867/scratch/nemotron-model-cache
DATA=/home/a474r867/scratch/nemotron-eco-reasoner/data/kaggle_5k_train.jsonl
OUTPUT=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/m1_dual_a100
SCRIPTS=/home/a474r867/scratch/nemotron-eco-reasoner/scripts
TMPDIR=/tmp/nem_a100_$$

mkdir -p $OUTPUT $TMPDIR

echo "=== M1 Dual-GPU A100 Training ==="
echo "Nodes: $(hostname)"
echo "GPUs requested: 2× A100 (${SLURM_GPUS_ON_NODE:-2})"
echo "Container: $CONTAINER"
echo "Output: $OUTPUT"
echo "============================"

# Verify GPUs
echo "GPU visibility:"
apptainer exec --nv $CONTAINER python3 -c "
import torch
print(f'  torch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
print(f'  GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_memory/1024**3:.0f} GB)')
"

# Find model
MODEL_PATH=$(find $MODEL_DIR -name "config.json" -path "*/snapshots/*" 2>/dev/null | head -1 | xargs dirname)
if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: Model not found in $MODEL_DIR"
    exit 1
fi
echo "Model: $MODEL_PATH"

# Install mamba-ssm at runtime (CUDA toolkit present in container)
echo "Installing mamba-ssm (CUDA)..."
apptainer exec --nv --writable-tmpfs --bind $TMPDIR:/tmp $CONTAINER bash -c "
    pip install --no-cache-dir mamba-ssm causal-conv1d 2>&1 | tail -3
    python3 -c 'import mamba_ssm; print(\"mamba-ssm OK\")'
"

# Train with FSDP full_shard (2 GPUs)
echo "Starting FSDP training (2 GPUs)..."
apptainer exec --nv --writable-tmpfs --bind $TMPDIR:/tmp $CONTAINER \
    torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
        $SCRIPTS/train_m1_container.py \
            --model "$MODEL_PATH" \
            --data "$DATA" \
            --output "$OUTPUT" \
            --max_steps 500 \
            --batch_size 1 \
            --grad_accum 4 \
            --lr 2e-4 \
            --max_seq_len 1024 \
            --num_epochs 1 \
            --use_dora 1 \
            --fsdp 1

EXIT=$?
echo "Exit code: $EXIT"
rm -rf $TMPDIR

if [ $EXIT -eq 0 ]; then
    echo "DONE! Adapter at $OUTPUT/"
    ls -lh $OUTPUT/adapter_model.safetensors 2>/dev/null
fi
