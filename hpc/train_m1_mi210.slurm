#!/bin/bash
#SBATCH --job-name=nem-m1
#SBATCH --partition=sixhour
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/m1_run1/train_%j.out
#SBATCH --error=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/m1_run1/train_%j.err

# M1 Training — Nemotron-3-Nano + DoRA on single MI210
# Uses pre-built Apptainer container (zero venv setup)

CONTAINER=/home/a474r867/scratch/nemotron-eco-reasoner/nemotron-rocm.sif
MODEL_DIR=/home/a474r867/scratch/nemotron-model-cache
DATA=/home/a474r867/scratch/nemotron-eco-reasoner/data/kaggle_5k_train.jsonl
OUTPUT=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/m1_run1
SCRIPTS=/home/a474r867/scratch/nemotron-eco-reasoner/scripts

mkdir -p $OUTPUT

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
    echo "Run download_model.sh first."
    exit 1
fi
echo "Model path: $MODEL_PATH"

apptainer exec --rocm $CONTAINER python3 -u $SCRIPTS/train_m1_container.py \
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

if [ $EXIT -eq 0 ]; then
    echo "DONE! Adapter at $OUTPUT/"
    ls -lh $OUTPUT/adapter_model.safetensors 2>/dev/null
fi
