#!/bin/bash
#SBATCH --job-name=nem-cuda
#SBATCH --partition=sixhour
#SBATCH --time=05:30:00
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/m1_cuda/train_%j.out
#SBATCH --error=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/m1_cuda/train_%j.err

# M1 Dual-GPU — Nemotron-3-Nano-30B + DoRA on 2x A100 (FSDP full_shard)
# 2x 80GB A100 = 160GB total -> ~30GB per GPU with FSDP full_shard
# Container: nemotron-cuda.sif (torch 2.5.1 CUDA 12.4, mamba-ssm pre-compiled)
#
# CRITICAL: on compute nodes, $HOME=/kuhpc/scratch/kbs/a474r867 NOT /home/a474r867

set -euo pipefail

CONTAINER=$HOME/scratch/nemotron-eco-reasoner/nemotron-cuda.sif
MODEL_DIR=$HOME/scratch/nemotron-model-cache
DATA=$HOME/scratch/nemotron-eco-reasoner/data/kaggle_5k_train.jsonl
OUTPUT=$HOME/scratch/nemotron-eco-reasoner/outputs/m1_cuda
SCRIPTS=$HOME/scratch/nemotron-eco-reasoner/scripts
FSDP_CONFIG=$HOME/scratch/nemotron-eco-reasoner/hpc/fsdp_config.yaml

mkdir -p $OUTPUT

echo "=== Nemotron-3-Nano M1 Training (CUDA/A100) ==="
echo "Host: $(hostname)"
echo "HOME: $HOME"
echo "Time: $(date)"
echo "GPUs: 2x A100 (80GB each)"
echo "Container: $CONTAINER"
echo "================================================"

# Verify container
if [ ! -f "$CONTAINER" ]; then
    echo "FATAL: Container not found: $CONTAINER"
    exit 1
fi

# Verify GPUs
echo ""
echo "--- GPU Verification ---"
apptainer exec --nv $CONTAINER python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {p.name} ({p.total_memory/1024**3:.0f} GB, SM{p.major}{p.minor})')
"

# Verify mamba-ssm
echo ""
echo "--- mamba-ssm Verification ---"
apptainer exec --nv $CONTAINER python3 -c "
import mamba_ssm; print('mamba-ssm: OK')
from mamba_ssm import Mamba; print('Mamba: OK')
try:
    import causal_conv1d; print('causal-conv1d: OK')
except ImportError:
    print('causal-conv1d: not available (PyTorch fallback)')
"

# Find model
MODEL_PATH=$(find $MODEL_DIR -name "config.json" -path "*/snapshots/*" 2>/dev/null | head -1 | xargs dirname)
if [ -z "$MODEL_PATH" ]; then
    echo "FATAL: Model not found in $MODEL_DIR"
    exit 1
fi
echo ""
echo "Model: $MODEL_PATH"
echo "Data: $DATA ($(wc -l < $DATA) examples)"

# Train
echo ""
echo "--- Starting FSDP Training (2x A100) ---"
apptainer exec --nv $CONTAINER \
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
            --fsdp 1 \
            --fsdp_config "$FSDP_CONFIG"

EXIT=$?
echo ""
echo "Exit code: $EXIT"

if [ $EXIT -eq 0 ]; then
    echo ""
    echo "=== TRAINING COMPLETE ==="
    echo "Adapter: $OUTPUT/"
    ls -lh $OUTPUT/adapter_model.safetensors 2>/dev/null || echo "(no safetensors)"
    ls -lh $OUTPUT/ 2>/dev/null
else
    echo "TRAINING FAILED (exit $EXIT)"
fi
