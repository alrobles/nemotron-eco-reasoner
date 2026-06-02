#!/bin/bash
#SBATCH --job-name=nem-dora
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:mi210:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/nem_dora_%j.out
#SBATCH --error=logs/nem_dora_%j.err
#SBATCH --signal=B:SIGUSR1@300

# Nemotron-3-Nano-30B BF16 DoRA — MI210 x1
# Venv + deps in /tmp SSD, no Lustre I/O during training
# Verified: pip install --dry-run shows 60+ packages, all covered

set -euo pipefail

PROJECT_DIR="$HOME/scratch/nemotron-eco-reasoner-full"
DATASET_JSONL="$HOME/scratch/nemotron-eco-reasoner/data/kaggle_5k_train.jsonl"
OUTPUT_DIR="$HOME/scratch/nemotron-training/$(date +%Y%m%d_%H%M)"
KAGGLE_CSV="$HOME/scratch/nemotron-kaggle/train.csv"
MODEL_DIR="$HOME/scratch/nemotron-model"

mkdir -p "$OUTPUT_DIR" logs

# ── Environment ────────────────────────────────────────────────────
export NEMOTRON_BASE_MODEL="$MODEL_DIR"
export NEMOTRON_LORA_RANK=32
export NEMOTRON_LORA_ALPHA=64
export NEMOTRON_USE_DORA=1
export NEMOTRON_EPOCHS=3
export NEMOTRON_BATCH_SIZE=2
export NEMOTRON_GRAD_ACCUM=8
export NEMOTRON_LEARNING_RATE="2e-4"
export NEMOTRON_MAX_LENGTH=2048
export NEMOTRON_SAVE_STEPS=200
export NEMOTRON_WANDB_PROJECT="nemotron-kaggle-reasoner"

# ── Venv in /tmp SSD (no Lustre) ──────────────────────────────────
VENV="/tmp/nem_venv_$$"
echo "=== Step 1: Create venv in /tmp ==="
python3.11 -m venv "$VENV" --without-pip
source "$VENV/bin/activate"
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

echo "=== Step 2: PyTorch ROCm ==="
pip install --no-deps torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

echo "=== Step 3: ML stack (CON dependencias, /tmp SSD) ==="
pip install transformers accelerate peft trl datasets \
    safetensors einops wandb pandas tqdm sentencepiece

echo "=== Step 4: Verify ==="
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "ROCm available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "VRAM: $(python -c 'import torch; t=torch.cuda.get_device_properties(0); print(f\"{t.total_memory/1e9:.1f} GB\")')"

# Verify ALL imports that failed in previous jobs
python -c "
import transformers, peft, accelerate, datasets, trl, torch
import packaging, safetensors, tokenizers, tqdm, yaml
import requests, huggingface_hub, httpcore
print('ALL imports OK')
"

echo "=== Step 5: Dataset ==="
echo "Dataset: $(wc -l < "$DATASET_JSONL") examples"

echo "=== Step 6: Training ==="
echo "Output: $OUTPUT_DIR"

python "$PROJECT_DIR/scripts/train_bf16_lora.py" \
    --dataset "$DATASET_JSONL" \
    --output "$OUTPUT_DIR"

echo ""
echo "========================================"
echo " Training complete!"
echo " Adapter: $OUTPUT_DIR/final/"
echo "========================================"
