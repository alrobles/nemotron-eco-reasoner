#!/bin/bash
#SBATCH --job-name=nem-train
#SBATCH --partition=sixhour
#SBATCH --time=05:30:00
#SBATCH --gres=gpu:mi210:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/m1_dual/train_%j.out
#SBATCH --error=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/m1_dual/train_%j.err

# M1 Dual-GPU — Nemotron-3-Nano-30B + DoRA on 2x MI210 (FSDP full_shard)
# 2x 64GB MI210 = 128GB total -> ~30GB per GPU with FSDP full_shard
# Container: nemotron-train.sif (torch 2.6.0 ROCm 6.4, mamba-ssm pre-compiled)
#
# CRITICAL: on compute nodes, $HOME=/kuhpc/scratch/kbs/a474r867 NOT /home/a474r867
# All paths must use $HOME/scratch/... not /home/a474r867/scratch/...

set -euo pipefail

CONTAINER=$HOME/scratch/nemotron-eco-reasoner/nemotron-train.sif
MODEL_DIR=$HOME/scratch/nemotron-model-cache
DATA=$HOME/scratch/nemotron-eco-reasoner/data/kaggle_5k_train.jsonl
OUTPUT=$HOME/scratch/nemotron-eco-reasoner/outputs/m1_dual
SCRIPTS=$HOME/scratch/nemotron-eco-reasoner/scripts
FSDP_CONFIG=$HOME/scratch/nemotron-eco-reasoner/hpc/fsdp_config.yaml

mkdir -p $OUTPUT

echo "=== Nemotron-3-Nano M1 Training ==="
echo "Host: $(hostname)"
echo "HOME: $HOME"
echo "Time: $(date)"
echo "GPUs: 2x MI210"
echo "Container: $CONTAINER"
echo "===================================="

# Verify container exists
if [ ! -f "$CONTAINER" ]; then
    echo "FATAL: Container not found: $CONTAINER"
    exit 1
fi

# Verify GPUs
echo ""
echo "--- GPU Verification ---"
apptainer exec --rocm $CONTAINER python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'ROCm: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {p.name} ({p.total_memory/1024**3:.0f} GB)')
"

# Verify mamba-ssm (pre-compiled in container)
echo ""
echo "--- mamba-ssm Verification ---"
apptainer exec --rocm $CONTAINER python3 -c "
import mamba_ssm; print('mamba-ssm: OK')
from mamba_ssm import Mamba; print('Mamba import: OK')
"

# Find model path
MODEL_PATH=$(find $MODEL_DIR -name "config.json" -path "*/snapshots/*" 2>/dev/null | head -1 | xargs dirname)
if [ -z "$MODEL_PATH" ]; then
    echo "FATAL: Model not found in $MODEL_DIR"
    exit 1
fi
echo ""
echo "Model: $MODEL_PATH"
echo "Data: $DATA ($(wc -l < $DATA) examples)"

# Train with FSDP full_shard (2 GPUs)
echo ""
echo "--- Starting FSDP Training (2x MI210) ---"
apptainer exec --rocm $CONTAINER \
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
    ls -lh $OUTPUT/adapter_model.safetensors 2>/dev/null || echo "(no safetensors found)"
    ls -lh $OUTPUT/ 2>/dev/null
else
    echo "TRAINING FAILED (exit $EXIT)"
fi
