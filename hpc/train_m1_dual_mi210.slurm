#!/bin/bash
#SBATCH --job-name=nem-dual
#SBATCH --partition=sixhour
#SBATCH --time=05:30:00
#SBATCH --gres=gpu:mi210:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/m1_dual/train_%j.out
#SBATCH --error=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/m1_dual/train_%j.err

# M1 Dual-GPU — Nemotron-3-Nano-30B + DoRA on 2× MI210 (FSDP full_shard)
# 2× 64GB MI210 → 128GB total → ~30GB per GPU with FSDP full_shard
# Container: nemotron-rocm.sif (torch 2.6.0 ROCm 6.4)
#
# CRITICAL: on compute nodes, $HOME=/kuhpc/scratch/kbs/a474r867 NOT /home/a474r867
# All paths must use $HOME/scratch/... not /home/a474r867/scratch/...

set -euo pipefail

CONTAINER=$HOME/scratch/nemotron-eco-reasoner/nemotron-rocm.sif
MODEL_DIR=$HOME/scratch/nemotron-model-cache
DATA=$HOME/scratch/nemotron-eco-reasoner/data/kaggle_5k_train.jsonl
OUTPUT=$HOME/scratch/nemotron-eco-reasoner/outputs/m1_dual
SCRIPTS=$HOME/scratch/nemotron-eco-reasoner/scripts
FSDP_CONFIG=$HOME/scratch/nemotron-eco-reasoner/hpc/fsdp_config.yaml
TMPDIR=/tmp/nem_dual_$$

mkdir -p $OUTPUT $TMPDIR

echo "=== M1 Dual-GPU Training ==="
echo "Nodes: $(hostname)"
echo "HOME: $HOME"
echo "GPUs requested: 2× MI210 (${SLURM_GPUS_ON_NODE:-2})"
echo "Container: $CONTAINER"
echo "Output: $OUTPUT"
echo "============================"

# Verify GPUs
echo "GPU visibility:"
apptainer exec --rocm $CONTAINER python3 -c "
import torch
print(f'  torch: {torch.__version__}')
print(f'  ROCm: {torch.cuda.is_available()}')
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

# Install mamba-ssm at runtime
echo "Installing mamba-ssm..."
apptainer exec --rocm --writable-tmpfs --bind $TMPDIR:/tmp $CONTAINER bash -c "
    pip install --no-cache-dir mamba-ssm causal-conv1d 2>&1 | tail -3
    python3 -c 'import mamba_ssm; print(\"mamba-ssm OK\")'
"

# Train with FSDP full_shard (2 GPUs)
echo "Starting FSDP training (2 GPUs)..."
apptainer exec --rocm --writable-tmpfs --bind $TMPDIR:/tmp $CONTAINER \
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
echo "Exit code: $EXIT"
rm -rf $TMPDIR

if [ $EXIT -eq 0 ]; then
    echo "DONE! Adapter at $OUTPUT/"
    ls -lh $OUTPUT/adapter_model.safetensors 2>/dev/null
fi
