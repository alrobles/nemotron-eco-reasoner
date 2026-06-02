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
SITE_PKGS="/tmp/site_${SLURM_JOB_ID}"
HF_CACHE="/tmp/hf_${SLURM_JOB_ID}"

rm -rf "$SITE_PKGS" "$HF_CACHE" 2>/dev/null || true
mkdir -p "$SITE_PKGS" "$HF_CACHE" "$PROJECT/logs" "$PROJECT/data" "$PROJECT/outputs/nem-m1-apptainer"
export HF_HOME="$HF_CACHE"

# ── Step 1: Install packages to /tmp SSD (not tmpfs RAM) ──
echo "=== Step 1: Install ML stack to /tmp SSD ==="
apptainer exec --rocm \
    --bind "$SITE_PKGS:/pkgs" \
    "$SIF" \
    pip install --target=/pkgs --cache-dir="$HF_CACHE" \
    transformers peft datasets trl accelerate sentencepiece

# ── Step 2: Verify ──
echo "=== Step 2: Verify ==="
apptainer exec --rocm \
    --bind "$SITE_PKGS:/pkgs" \
    --env "PYTHONPATH=/pkgs" \
    "$SIF" python3 -c "
import torch, transformers, peft
print(f'PyTorch {torch.__version__}')
print(f'ROCm {torch.version.hip}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem // 1e9:.1f} GB')
print(f'transformers {transformers.__version__}')
print(f'peft {peft.__version__}')
"

# ── Step 3: Prepare dataset (ShareGPT) ──
echo "=== Step 3: Prepare dataset ==="
python3 -c "
import csv, json
data = []
with open('/home/a474r867/scratch/nemotron-kaggle/train.csv') as f:
    for i, row in enumerate(csv.DictReader(f)):
        if i >= 5000: break
        data.append({'messages': [
            {'role': 'user', 'content': row['prompt']},
            {'role': 'assistant', 'content': row['answer']}
        ]})
with open('$PROJECT/data/kaggle_5k_train.jsonl', 'w') as f:
    for item in data:
        f.write(json.dumps(item) + '\n')
print(f'Dataset: {len(data)} examples')
"

# ── Step 4: Train with DoRA BF16 ──
echo "=== Step 4: Training ==="
cd "$PROJECT"

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

apptainer exec --rocm \
    --bind "$SITE_PKGS:/pkgs" \
    --bind "$PROJECT/data:/data" \
    --bind "$PROJECT/outputs/nem-m1-apptainer:/output" \
    --bind "$PROJECT/scripts:/scripts" \
    --env "PYTHONPATH=/pkgs" \
    "$SIF" python3 /scripts/train_bf16_lora.py \
    --dataset /data/kaggle_5k_train.jsonl \
    --output /output

EXIT=$?

# ── Cleanup ──
rm -rf "$SITE_PKGS" "$HF_CACHE"

echo "=== Done, exit=$EXIT ==="
date
exit $EXIT
