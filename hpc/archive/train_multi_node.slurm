#!/bin/bash
#SBATCH --job-name=nemotron-multi
#SBATCH --partition=gpu
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:mi210:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=logs/nemotron_multi_%j.out
#SBATCH --error=logs/nemotron_multi_%j.err
#SBATCH --signal=B:SIGUSR1@300

# Nemotron-3-Nano-30B QLoRA + FSDP multi-node training
# 3 nodes × 2 MI210s (64GB each) = 6 GPUs total
# With FSDP sharding: ~25 GB / 6 ≈ 4 GB per GPU for base weights
# Leaves plenty of room for activations and LoRA gradients

set -euo pipefail

PROJECT_DIR="$HOME/nemotron-eco-reasoner"
DATASET_DIR="$PROJECT_DIR/data"
OUTPUT_DIR="$HOME/scratch/nemotron-training/$(date +%Y%m%d_%H%M)"
KAGGLE_CSV="$DATASET_DIR/train.csv"

mkdir -p "$DATASET_DIR" "$OUTPUT_DIR" logs

# ── Environment ────────────────────────────────────────────────────
export NEMOTRON_BASE_MODEL="nvidia/Nemotron-3-Nano-30B-A3B-BF16"
export NEMOTRON_LORA_RANK=64        # Higher rank with more GPUs
export NEMOTRON_LORA_ALPHA=128
export NEMOTRON_EPOCHS=3
export NEMOTRON_BATCH_SIZE=2
export NEMOTRON_GRAD_ACCUM=4        # Lower accum with more GPUs
export NEMOTRON_LEARNING_RATE="1e-4" # Lower LR for larger batches
export NEMOTRON_MAX_LENGTH=2048
export NEMOTRON_SAVE_STEPS=500
export NEMOTRON_WANDB_PROJECT="nemotron-eco-reasoner"

# FSDP-specific
export FSDP_SHARDING_STRATEGY="HYBRID_SHARD"  # Shard within node, replicate across nodes
export FSDP_STATE_DICT_TYPE="SHARDED_STATE_DICT"

# ── Node metadata ──────────────────────────────────────────────────
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address | head -1)
echo "Head node: $HEAD_NODE ($HEAD_NODE_IP)"
echo "Nodes: $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"

# ── Prepare dataset (only on head node) ────────────────────────────
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -c "
    if [ ! -f '$DATASET_DIR/combined_dataset.jsonl' ]; then
        echo 'Generating combined dataset...'
        apptainer exec --rocm --bind $HOME:$HOME $HOME/images/rocm_pytorch.sif \
            python $PROJECT_DIR/scripts/prepare_dataset.py \
                --kaggle-csv $KAGGLE_CSV \
                --ecoagent-dir $HOME/ecoagent \
                --output $DATASET_DIR/combined_dataset.jsonl \
                --kaggle-ratio 0.5 \
                --total 50000
    fi
"

# ── Launch training with torchrun ──────────────────────────────────
NUM_GPUS_PER_NODE=2
NUM_NODES=$SLURM_NNODES
TOTAL_PROCS=$((NUM_NODES * NUM_GPUS_PER_NODE))

echo "Launching: $TOTAL_PROCS processes across $NUM_NODES nodes"
echo "Output: $OUTPUT_DIR"

srun --ntasks=$TOTAL_PROCS --ntasks-per-node=$NUM_GPUS_PER_NODE \
    apptainer exec --rocm --bind "$HOME:$HOME" "$HOME/images/rocm_pytorch.sif" \
    torchrun \
        --nnodes=$NUM_NODES \
        --nproc_per_node=$NUM_GPUS_PER_NODE \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint="$HEAD_NODE_IP:29500" \
        "$PROJECT_DIR/scripts/train_qlora.py" \
            --dataset "$DATASET_DIR/combined_dataset.jsonl" \
            --output "$OUTPUT_DIR"

echo "Training complete!"
echo "Adapter saved to: $OUTPUT_DIR/final/"

# ── Package submission ─────────────────────────────────────────────
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -c "
    apptainer exec --rocm --bind $HOME:$HOME $HOME/images/rocm_pytorch.sif \
        python $PROJECT_DIR/scripts/submit_kaggle.py \
            --adapter $OUTPUT_DIR/final/ \
            --output $OUTPUT_DIR/submission.zip
    echo 'Submission packaged: $OUTPUT_DIR/submission.zip'
"
