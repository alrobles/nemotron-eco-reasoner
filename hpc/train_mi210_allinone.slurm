#!/bin/bash
#SBATCH --job-name=nem-dora
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:mi210:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=05:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ── Nemotron DoRA M1: All-in-one (venv build + train) ─────────────
# Builds venv on local SSD (/tmp), installs deps, trains.
# No Lustre pip hell. No stale venvs. 100% self-contained.

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M)
MODEL_DIR="/home/a474r867/scratch/nemotron-model/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848"
KAGGLE_CSV="/home/a474r867/scratch/nemotron-kaggle/train.csv"
TRAIN_SCRIPT="/home/a474r867/scratch/nemotron-eco-reasoner-full/scripts/train_bf16_lora.py"
DATA_SCRIPT="/home/a474r867/scratch/nemotron-eco-reasoner-full/scripts/prepare_dataset_kaggle.py"
OUTPUT_DIR="/home/a474r867/scratch/nemotron-eco-reasoner/outputs/m1_${TIMESTAMP}"
DATASET_DIR="/home/a474r867/scratch/nemotron-eco-reasoner/data"

# ── Step 1: Build venv on local SSD ────────────────────────────
TTMP="/tmp/nemotron_venv_$$"
echo "=== Step 1: Creating venv on $TTMP ==="
python3 -m venv --without-pip "$TTMP"
curl -sS https://bootstrap.pypa.io/get-pip.py -o "$TTMP/get-pip.py"
"$TTMP/bin/python3" "$TTMP/get-pip.py" --quiet
PIP="$TTMP/bin/pip"

echo "=== Step 2: PyTorch ROCm ==="
$PIP install --no-deps torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/rocm6.1

echo "=== Step 3: Core ML deps ==="
$PIP install --no-deps \
    transformers==4.57.6 \
    peft==0.17.1 \
    accelerate==1.10.1 \
    datasets==4.5.0 \
    trl==0.24.0 \
    sentencepiece==0.2.1

echo "=== Step 4: Resolve torch sub-deps ==="
$PIP install --no-deps sympy typing-extensions numpy filelock fsspec jinja2 networkx pillow mpmath

echo "=== Step 5: Key extras ==="
$PIP install --no-deps wandb tensorboard scikit-learn

echo "=== Step 6: Verify ==="
source "$TTMP/bin/activate"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'ROCm available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
import transformers
print(f'Transformers: {transformers.__version__}')
import peft
print(f'PEFT: {peft.__version__} (DoRA: {\"use_dora\" in dir(peft.LoraConfig)})')
"

# ── Step 7: Prepare dataset ────────────────────────────────────
mkdir -p "$DATASET_DIR" "$OUTPUT_DIR"
DATASET_FILE="$DATASET_DIR/kaggle_5k_${TIMESTAMP}.jsonl"
echo "=== Step 7: Preparing dataset (5K Kaggle) ==="
python "$DATA_SCRIPT" \
    --kaggle-csv "$KAGGLE_CSV" \
    --output "$DATASET_FILE" \
    --total 5000
echo "Dataset: $(wc -l < "$DATASET_FILE") examples"

# ── Step 8: Environment settings ───────────────────────────────
export NEMOTRON_BASE_MODEL="$MODEL_DIR"
export NEMOTRON_LORA_RANK=32
export NEMOTRON_LORA_ALPHA=64
export NEMOTRON_USE_DORA=1
export NEMOTRON_EPOCHS=3
export NEMOTRON_BATCH_SIZE=2
export NEMOTRON_GRAD_ACCUM=8
export NEMOTRON_LEARNING_RATE="2e-4"
export NEMOTRON_MAX_LENGTH=2048
export NEMOTRON_SAVE_STEPS=100
export NEMOTRON_WANDB_PROJECT="nemotron-eco-reasoner"

echo "=== GPU Info ==="
rocm-smi --showproductname 2>/dev/null || true
python -c "
import torch
print(f'GPUs visible: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# ── Step 9: Train ──────────────────────────────────────────────
echo "=== STARTING TRAIN: $(date) ==="
echo "Model: $NEMOTRON_BASE_MODEL"
echo "DoRA rank=$NEMOTRON_LORA_RANK alpha=$NEMOTRON_LORA_ALPHA"
echo "Data: $DATASET_FILE"
echo "Output: $OUTPUT_DIR"

python "$TRAIN_SCRIPT" \
    --dataset "$DATASET_FILE" \
    --output "$OUTPUT_DIR"

EXIT_CODE=$?

echo "=== DONE: $(date) | Exit: $EXIT_CODE ==="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Adapter saved to: $OUTPUT_DIR/final/"
    ls -la "$OUTPUT_DIR/final/" 2>/dev/null || echo "(no final/ dir)"
fi

# ── Cleanup ────────────────────────────────────────────────────
rm -rf "$TTMP"

exit $EXIT_CODE
