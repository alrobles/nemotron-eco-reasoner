#!/bin/bash
#SBATCH --job-name=nem-dora
#SBATCH --partition=sixhour
#SBATCH --account=crcreuman
#SBATCH --gres=gpu:mi210:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/a474r867/scratch/nemotron-eco-reasoner/logs/nem_dora_%j.out
#SBATCH --error=/home/a474r867/scratch/nemotron-eco-reasoner/logs/nem_dora_%j.err
#SBATCH --signal=B:SIGUSR1@300

# Nemotron-3-Nano-30B BF16 DoRA — single MI210 64GB (ROCm)
# All pip installs in /tmp SSD, not Lustre.
# DoRA rank=32, BF16, Kaggle 5K (M1 proof of concept).

set -euo pipefail

PROJECT_DIR="/home/a474r867/scratch/nemotron-eco-reasoner-full"
MODEL_DIR="/home/a474r867/scratch/nemotron-model"
SCRIPTS_DIR="$PROJECT_DIR/scripts"
DATASET_JSONL="$PROJECT_DIR/data/kaggle_5k_train.jsonl"
OUTPUT_DIR="/home/a474r867/scratch/nemotron-training/nem_dora_$(date +%Y%m%d_%H%M)"
LOGS_DIR="/home/a474r867/scratch/nemotron-eco-reasoner/logs"

mkdir -p "$OUTPUT_DIR" "$LOGS_DIR"

# ===== Step 1: Venv in /tmp SSD (not Lustre) =====
echo "=== Step 1: Venv in /tmp SSD ==="
VENV_DIR="/tmp/nem_venv_$$"
python3.11 -m venv --without-pip "$VENV_DIR"
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip_$$.py
"$VENV_DIR/bin/python" /tmp/get-pip_$$.py --quiet
rm /tmp/get-pip_$$.py
source "$VENV_DIR/bin/activate"

echo "Python: $(python --version)"
echo "venv at: $VENV_DIR"

# ===== Step 2: PyTorch ROCm (--no-deps to avoid pulling CUDA variant) =====
echo "=== Step 2: PyTorch ROCm ==="
pip install --no-deps torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.1

# ===== Step 3: ML stack WITH dependencies (SSD handles this in seconds) =====
echo "=== Step 3: ML stack ==="
pip install transformers accelerate peft trl datasets \
    safetensors sentencepiece wandb

echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "ROCm available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "VRAM: $(python -c 'import torch; props=torch.cuda.get_device_properties(0); print(f\"{props.total_mem/1e9:.1f} GB\")')"

# ===== Step 4: Verify dataset exists =====
echo "=== Step 4: Dataset ==="
if [ -f "$DATASET_JSONL" ]; then
    echo "Dataset: $(wc -l < "$DATASET_JSONL") examples"
else
    echo "ERROR: Dataset not found at $DATASET_JSONL"
    exit 1
fi

# ===== Step 5: Verify model exists =====
echo "=== Step 5: Model ==="
if [ -d "$MODEL_DIR" ]; then
    echo "Model found at $MODEL_DIR"
    export NEMOTRON_MODEL_DIR="$MODEL_DIR"
else
    echo "Using HuggingFace Hub: nvidia/Nemotron-3-Nano-30B-A3B-BF16"
fi

# ===== Step 6: Train =====
echo "=== Step 6: Training ==="
echo "Output: $OUTPUT_DIR"

python "$SCRIPTS_DIR/train_bf16_lora.py" \
    --dataset "$DATASET_JSONL" \
    --output "$OUTPUT_DIR"

echo ""
echo "=== DONE ==="
echo "Adapter: $OUTPUT_DIR/final/"
