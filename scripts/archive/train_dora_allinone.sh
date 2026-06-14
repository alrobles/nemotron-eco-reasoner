#!/bin/bash
# All-in-one Nemotron DoRA training on KU-HPC
# Solves: python3.9 venv, Lustre pip timeout, huggingface-hub dep hell
# Run: sbatch --partition=sixhour --gres=gpu:mi210:1 --mem=64G --time=6:00:00 scripts/train_dora_allinone.sh

# Hardcoded: Slurm copies script to /var/spool/slurmd/, BASH_SOURCE is unreliable
REPO_DIR="/home/a474r867/scratch/nemotron-eco-reasoner-full"
SCRIPT_DIR="$REPO_DIR/scripts"
MODEL_DIR="/home/a474r867/scratch/nemotron-model"
DATA_DIR="$REPO_DIR/data"
OUTPUT_DIR="$REPO_DIR/outputs"
LOG_DIR="$REPO_DIR/logs"
VENV_DIR="/tmp/nemotron_venv_$$"

N_SAMPLES="${NEMOTRON_N_SAMPLES:-5000}"
DORA="${NEMOTRON_USE_DORA:-1}"
EPOCHS="${NEMOTRON_EPOCHS:-3}"
BATCH_SIZE="${NEMOTRON_BATCH_SIZE:-4}"
GRAD_ACCUM="${NEMOTRON_GRAD_ACCUM:-4}"

echo "=========================================="
echo "Nemotron DoRA Training - All-in-One Runner"
echo "=========================================="
echo "Samples: $N_SAMPLES | DoRA: $DORA | Epochs: $EPOCHS"
echo "Batch: $BATCH_SIZE | GradAccum: $GRAD_ACCUM"
echo ""

set -euo pipefail

# ── Step 1: Create venv on fast local SSD ──
echo "=== Step 1: Creating venv on $VENV_DIR ==="
python3.11 -m venv --without-pip "$VENV_DIR"
source "$VENV_DIR/bin/activate"
curl -sS https://bootstrap.pypa.io/get-pip.py | python
echo "venv created OK"

# ── Step 2: PyTorch ROCm ──
echo "=== Step 2: PyTorch ROCm ==="
pip install --no-deps torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/rocm6.1 2>&1 | tail -3

# ── Step 3: huggingface-hub WITH deps (small, avoids dep hell) ──
echo "=== Step 3: huggingface-hub (with deps) ==="
pip install huggingface-hub 2>&1 | tail -3

# ── Step 4: Core ML deps (--no-deps, pure Python, fast) ──
echo "=== Step 4: Core ML deps ==="
for pkg in \
    "transformers==4.57.6" \
    "peft==0.17.1" \
    "accelerate==1.10.1" \
    "datasets==4.5.0" \
    "trl==0.24.0" \
    "sentencepiece" \
    "typing-extensions" \
    "sympy" \
    "mpmath" \
    "numpy" \
    "wandb" \
    "tensorboard"; do
    echo "  Installing $pkg..."
    pip install --no-deps "$pkg" 2>&1 | tail -1
done

# ── Step 5: Verify GPU ──
echo "=== Step 5: Verify GPU ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'ROCm available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'GPU name: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# ── Step 6: Prepare dataset ──
echo "=== Step 6: Prepare dataset ==="
mkdir -p "$DATA_DIR"
python "$SCRIPT_DIR/prepare_dataset.py" \
    --source kaggle \
    --kaggle-csv "$DATA_DIR/train.csv" \
    --output "$DATA_DIR/train_${N_SAMPLES}.jsonl" \
    --max-examples "$N_SAMPLES" \
    --kaggle-ratio 1.0

echo "Dataset: $(wc -l < "$DATA_DIR/train_${N_SAMPLES}.jsonl") examples"

# ── Step 7: Train! ──
echo "=== Step 7: Training ==="
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

export NEMOTRON_USE_DORA="$DORA"
export NEMOTRON_N_SAMPLES="$N_SAMPLES"

python "$SCRIPT_DIR/train_bf16_lora.py" \
    --model_name_or_path "$MODEL_DIR" \
    --dataset_path "$DATA_DIR/train_${N_SAMPLES}.jsonl" \
    --output_dir "$OUTPUT_DIR/nemotron-dora-m1-$(date +%Y%m%d_%H%M)" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate 2e-4 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --bf16 \
    --logging_steps 10 \
    --save_steps 200 \
    --eval_steps 200 \
    --save_total_limit 2 \
    --dataloader_num_workers 2 \
    --report_to tensorboard \
    --max_seq_length 2048 \
    --gradient_checkpointing

echo ""
echo "=========================================="
echo "Training complete!"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
