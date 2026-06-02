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

# Paths
SIF="/home/a474r867/scratch/rocm-pytorch-6.4.3.sif"
OVERLAY="/tmp/overlay_${SLURM_JOB_ID}.img"
PROJECT="/home/a474r867/scratch/nemotron-eco-reasoner-full"
LOGDIR="$PROJECT/logs"
DATADIR="$PROJECT/data"
OUTDIR="$PROJECT/outputs/nem-m1-apptainer"
HF_CACHE="/tmp/hf_${SLURM_JOB_ID}"

mkdir -p "$LOGDIR" "$DATADIR" "$OUTDIR" "$HF_CACHE"
export HF_HOME="$HF_CACHE"

# ── Step 1: Overlay (8 GB in /tmp SSD) ──
echo "=== Step 1: Create overlay ==="
dd if=/dev/zero of="$OVERLAY" bs=1M count=8192 2>/dev/null
mkfs.ext3 -q -F "$OVERLAY" 2>/dev/null

# ── Step 2: Install ML packages ──
echo "=== Step 2: Install transformers, peft, datasets, trl, accelerate, sentencepiece ==="
apptainer exec --rocm --overlay "$OVERLAY" "$SIF" \
    pip install --cache-dir="$HF_CACHE" \
    transformers peft datasets trl accelerate sentencepiece

# ── Step 3: Verify ──
echo "=== Step 3: Verify ==="
apptainer exec --rocm --overlay "$OVERLAY" "$SIF" python3 -c "
import torch, transformers, peft
print(f'PyTorch {torch.__version__}')
print(f'ROCm {torch.version.hip}')
print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.device_count()} total)')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem // 1e9:.1f} GB')
print(f'transformers {transformers.__version__}')
print(f'peft {peft.__version__}')
print(f'DoRA: {hasattr(peft.tuners.lora.LoraConfig, \"use_dora\")}')
"

# ── Step 4: Prepare dataset (ShareGPT messages format) ──
echo "=== Step 4: Prepare dataset ==="
python3 -c "
import csv, json
data = []
with open('$DATADIR/../nemotron-kaggle/train.csv') as f:
    for i, row in enumerate(csv.DictReader(f)):
        if i >= 5000: break
        data.append({'messages': [
            {'role': 'user', 'content': row['prompt']},
            {'role': 'assistant', 'content': row['answer']}
        ]})
with open('$DATADIR/kaggle_5k_train.jsonl', 'w') as f:
    for item in data:
        f.write(json.dumps(item) + '\n')
print(f'Dataset: {len(data)} examples')
"

# ── Step 5: Train with DoRA BF16 ──
echo "=== Step 5: Training ==="
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

apptainer exec --rocm --overlay "$OVERLAY" \
    --bind "$DATADIR:/data" \
    --bind "$OUTDIR:/output" \
    "$SIF" \
    python3 "$PROJECT/scripts/train_bf16_lora.py" \
    --dataset /data/kaggle_5k_train.jsonl \
    --output /output

EXIT=$?

# ── Step 6: Cleanup ──
rm -f "$OVERLAY"
rm -rf "$HF_CACHE"

echo "=== Done, exit=$EXIT ==="
date
exit $EXIT
