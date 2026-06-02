#!/bin/bash
#SBATCH --job-name=nem-train
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=06:00:00
#SBATCH --output=/home/a474r867/scratch/nemotron-eco-reasoner/logs/train_%j.out
#SBATCH --error=/home/a474r867/scratch/nemotron-eco-reasoner/logs/train_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

echo "=== Job $(date) ==="
echo "Node: $(hostname)"

# Paths
SIF=/home/a474r867/scratch/nemotron-train.sif
TRAIN_SCRIPT=/home/a474r867/scratch/nemotron-eco-reasoner/train_dora_m1.py
DATA=/home/a474r867/scratch/nemotron-eco-reasoner/data/kaggle_5k_train.jsonl
MODEL=/home/a474r867/scratch/nemotron-model
OUTPUT=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/nem_dora_$(date +%Y%m%d_%H%M)

mkdir -p "$(dirname $DATA)" "$(dirname $TRAIN_SCRIPT)" "$OUTPUT"

echo "=== Apptainer exec --rocm ==="
apptainer exec --rocm \
    --bind /home/a474r867/scratch:/scratch \
    --bind /tmp:/tmp \
    $SIF \
    /usr/bin/python3 $TRAIN_SCRIPT \
    --data_path $DATA \
    --model_path $MODEL \
    --output_dir $OUTPUT \
    --max_steps 500 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --rank 32 \
    --use_dora

echo "=== Done $(date) ==="