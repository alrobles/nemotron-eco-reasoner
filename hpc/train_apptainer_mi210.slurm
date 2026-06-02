#!/bin/bash
#SBATCH --job-name=nem-dora-m1
#SBATCH --partition=sixhour
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:mi210:1
#SBATCH --mem=64G
#SBATCH --output=/home/a474r867/scratch/nemotron-eco-reasoner-full/logs/apptainer_%j.out
#SBATCH --error=/home/a474r867/scratch/nemotron-eco-reasoner-full/logs/apptainer_%j.err

set -euo pipefail

echo "=== Nemotron M1 Apptainer ==="
echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | $(date)"

SIF="/home/a474r867/scratch/rocm-pytorch-6.4.3.sif"
PROJECT="/home/a474r867/scratch/nemotron-eco-reasoner-full"

mkdir -p "$PROJECT/logs" "$PROJECT/data" "$PROJECT/outputs/nem-m1-apptainer"
mkdir -p /tmp/hf_${SLURM_JOB_ID}

export APPTAINER_BIND="$PROJECT/data:/data,$PROJECT/outputs/nem-m1-apptainer:/output,/home/a474r867/scratch/nemotron-kaggle:/kaggle"
export HF_HOME="/tmp/hf_${SLURM_JOB_ID}"

# ── SINGLE apptainer exec: install + verify + train, all in one tmpfs ──
echo "=== Launching container ==="

apptainer exec --rocm --writable-tmpfs \
    --bind "$PROJECT/data:/data" \
    --bind "$PROJECT/outputs/nem-m1-apptainer:/output" \
    --bind "/home/a474r867/scratch/nemotron-kaggle:/kaggle" \
    --bind "$PROJECT/scripts:/scripts" \
    "$SIF" bash -c '
set -euo pipefail

echo "=== Install ML stack ==="
pip install --cache-dir=/tmp/hf transformers peft datasets trl accelerate sentencepiece

echo "=== Verify ==="
python3 -c "
import torch, transformers, peft
print(f\"PyTorch {torch.__version__}\")
print(f\"ROCm {torch.version.hip}\")
print(f\"GPU: {torch.cuda.get_device_name(0)}\")
print(f\"VRAM: {torch.cuda.get_device_properties(0).total_mem // 1e9:.1f} GB\")
print(f\"transformers {transformers.__version__}\")
print(f\"peft {peft.__version__}\")
"

echo "=== Prepare dataset ==="
python3 -c "
import csv, json, os
os.makedirs(\"/data\", exist_ok=True)
data = []
with open(\"/kaggle/train.csv\") as f:
    for i, row in enumerate(csv.DictReader(f)):
        if i >= 5000: break
        data.append({\"messages\": [
            {\"role\": \"user\", \"content\": row[\"prompt\"]},
            {\"role\": \"assistant\", \"content\": row[\"answer\"]}
        ]})
with open(\"/data/kaggle_5k_train.jsonl\", \"w\") as f:
    for item in data:
        f.write(json.dumps(item) + \"\\n\")
print(f\"Dataset: {len(data)} examples\")
"

echo "=== Training ==="
export NEMOTRON_BASE_MODEL="nvidia/Nemotron-3-Nano-30B-A3B-BF16"
export NEMOTRON_LORA_RANK=32
export NEMOTRON_LORA_ALPHA=64
export NEMOTRON_USE_DORA=1
export NEMOTRON_EPOCHS=1
export NEMOTRON_BATCH_SIZE=1
export NEMOTRON_GRAD_ACCUM=4
export NEMOTRON_LEARNING_RATE=2e-4
export NEMOTRON_MAX_LENGTH=1024
export NEMOTRON_SAVE_STEPS=200

python3 /scripts/train_bf16_lora.py \
    --dataset /data/kaggle_5k_train.jsonl \
    --output /output
'

EXIT=$?
rm -rf /tmp/hf_${SLURM_JOB_ID}

echo "=== Done, exit=$EXIT ==="
date
exit $EXIT
