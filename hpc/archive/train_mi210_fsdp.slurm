#!/bin/bash
# === train_mi210_fsdp.slurm — MI210 DoRA + FSDP multi-GPU ===
#
# 3 nodes × 2 MI210s = 6 GPUs with FSDP full_shard.
# Effective batch: 2 × 4 × 6 = 48 per step.
#
# Strategy: train_mi210_dora.py with --fsdp 1
# Auto-wraps transformer layers in FSDP units.
# Each GPU holds a shard of the full model → fits seq=4096 if needed.
#
# Usage:
#   sbatch hpc/train_mi210_fsdp.slurm
#   SEQ_LEN=4096 sbatch hpc/train_mi210_fsdp.slurm

#SBATCH --job-name=mi210_fsdp
#SBATCH --partition=sixhour
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:mi210:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=05:50:00
#SBATCH --output=/home/a474r867/scratch/nemotron-eco-reasoner/logs/mi210_fsdp_%j.out
#SBATCH --error=/home/a474r867/scratch/nemotron-eco-reasoner/logs/mi210_fsdp_%j.err
#SBATCH --signal=B:SIGUSR1@120

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export TORCH_COMPILE_DISABLE=1
export ACCELERATE_USE_FSDP=true
export FSDP_SHARDING_STRATEGY="FULL_SHARD"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_DATASETS_OFFLINE=1

# ── Nodes info ────────────────────────────────────────────────────────
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address | awk '{print $1}')
echo "Head: $HEAD_NODE ($HEAD_NODE_IP)"
echo "Nodes: $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"

# ── Paths ─────────────────────────────────────────────────────────────
REPO_DIR="/home/a474r867/scratch/nemotron-eco-reasoner"
MODEL_DIR="/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848"
DATA="${DATA_PATH:-$REPO_DIR/data/train_cot_unified.jsonl}"
OUTPUT="$REPO_DIR/outputs/mi210_fsdp_v1"
VENV_DIR="/tmp/nemotron_mi210_fsdp_$$"

# ── Config ────────────────────────────────────────────────────────────
SEQ_LEN="${SEQ_LEN:-2048}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
PER_JOB_STEPS="${PER_JOB_STEPS:-300}"
LR="${LR:-1e-4}"
BATCH="${BATCH:-2}"
GRAD="${GRAD_ACCUM:-4}"

NUM_GPUS_PER_NODE=2
NUM_NODES=$SLURM_NNODES
TOTAL_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))
EFFECTIVE_BATCH=$((BATCH * GRAD * TOTAL_GPUS))

echo "=========================================="
echo "MI210 FSDP Multi-GPU — DoRA Training"
echo "=========================================="
echo "GPUs:   $TOTAL_GPUS (${NUM_NODES} nodes × $NUM_GPUS_PER_NODE MI210s)"
echo "Config: seq=$SEQ_LEN rank=$LORA_RANK steps=$PER_JOB_STEPS lr=$LR"
echo "Batch:  $BATCH × $GRAD × $TOTAL_GPUS = $EFFECTIVE_BATCH effective"
echo "Output: $OUTPUT"
echo "Job:    $SLURM_JOB_ID"
echo ""

mkdir -p "$OUTPUT"

# ── Venv setup (all nodes, parallel) ──────────────────────────────────
echo "=== Setting up venv on all nodes ==="
srun --ntasks=$TOTAL_GPUS --ntasks-per-node=$NUM_GPUS_PER_NODE bash -c "
    VDIR=/tmp/nemotron_mi210_fsdp_\${SLURM_JOB_ID}
    python3 -m venv --without-pip \$VDIR 2>/dev/null || python3 -m venv \$VDIR
    source \$VDIR/bin/activate
    pip install -q --upgrade pip 2>&1 | tail -1
    pip install --no-deps torch==2.6.0 torchvision==0.21.0 2>&1 | tail -1
    pip install --index-url https://download.pytorch.org/whl/rocm6.2 \
        torch==2.6.0 torchvision==0.21.0 2>&1 | tail -2
    pip install -q huggingface_hub 2>&1 | tail -1
    for pkg in \
        'transformers==4.57.6' \
        'peft==0.17.1' \
        'accelerate==1.10.1' \
        'datasets==4.5.0' \
        'trl==0.24.0' \
        sentencepiece numpy; do
        pip install -q --no-deps \$pkg 2>&1 | tail -1
    done
    echo \"\$HOSTNAME: venv ready\"
"

# ── Verify GPUs on all nodes ──────────────────────────────────────────
echo "=== GPU Check ==="
srun --ntasks=$TOTAL_GPUS --ntasks-per-node=$NUM_GPUS_PER_NODE bash -c "
    source /tmp/nemotron_mi210_fsdp_\${SLURM_JOB_ID}/bin/activate
    python -c \"
import torch
print(f'{\$HOSTNAME}: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB) BF16={torch.cuda.is_bf16_supported()}')
\"
"

# ── Train ─────────────────────────────────────────────────────────────
echo "=== Training ==="

cd "$REPO_DIR"

srun --ntasks=$TOTAL_GPUS --ntasks-per-node=$NUM_GPUS_PER_NODE bash -c "
    source /tmp/nemotron_mi210_fsdp_\${SLURM_JOB_ID}/bin/activate
    python scripts/train_mi210_dora.py \
        --model '$MODEL_DIR' \
        --data '$DATA' \
        --output '$OUTPUT' \
        --max_steps '$PER_JOB_STEPS' \
        --batch_size '$BATCH' \
        --grad_accum '$GRAD' \
        --lr '$LR' \
        --seq_len '$SEQ_LEN' \
        --use_dora 1 \
        --fsdp 1 \
        --packing 1
"

EXIT=$?

# ── Cleanup ───────────────────────────────────────────────────────────
echo "Exit: $EXIT"
if [ $EXIT -eq 0 ]; then
    echo "SUCCESS — multi-GPU FSDP training complete!"
    echo "Adapters in: $OUTPUT/final/"
    find "$OUTPUT" -name "*.safetensors" -type f 2>/dev/null | sort
elif [ $EXIT -eq 42 ]; then
    echo "SIGUSR1 — checkpoint saved for chain"
fi

exit $EXIT
