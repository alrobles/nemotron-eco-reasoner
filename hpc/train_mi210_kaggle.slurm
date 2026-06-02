#!/bin/bash
#SBATCH --job-name=nem-dora
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:mi210:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --signal=B:SIGUSR1@300

# ============================================================
# Nemotron-3-Nano-30B BF16 DoRA — single MI210 64GB (ROCm)
# DoRA rank=32, BF16, Kaggle 5K (M1 proof of concept).
#
# Every run is recorded in training_logs/ and auto-committed
# to GitHub for permanent diagnostics. No more lost logs.
# ============================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────
PROJECT_DIR="/home/a474r867/scratch/nemotron-eco-reasoner-full"
MODEL_DIR="/home/a474r867/scratch/nemotron-model/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848"
SCRIPTS_DIR="$PROJECT_DIR/scripts"
DATASET_JSONL="$PROJECT_DIR/data/kaggle_5k_train.jsonl"
OUTPUT_DIR="/home/a474r867/scratch/nemotron-training/nem_dora_$(date +%Y%m%d_%H%M)"

TIMESTAMP="$(date +%Y%m%d_%H%M)"
LOG_DIR="$PROJECT_DIR/training_logs/${TIMESTAMP}_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Redirect ALL output to job.out inside the repo (git-tracked)
exec > >(tee "$LOG_DIR/job.out") 2>&1

echo "═══════════════════════════════════════════"
echo "  Nemotron DoRA Training — M1 Single GPU"
echo "═══════════════════════════════════════════"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs:      $SLURM_GPUS"
echo "Timestamp: $TIMESTAMP"
echo "Log dir:   $LOG_DIR"
echo "Output:    $OUTPUT_DIR"
echo ""

# ── Diagnostics snapshot (before training) ─────────────
echo "=== DIAGNOSTICS ==="
echo "Hostname: $(hostname)"
echo "Kernel:   $(uname -r)"
echo "CPU:      $(nproc) cores"
echo "Memory:   $(free -h | awk '/^Mem/{print $2}')"
echo ""
rocminfo 2>/dev/null | grep -E "Marketing Name|gfx" | head -4 || echo "rocminfo not available"
echo ""

# ── Signal handler for graceful shutdown ─────────────
trap 'echo "SIGUSR1 received — saving checkpoint..."; touch "$LOG_DIR/SIGUSR1_RECEIVED"; exit 0' SIGUSR1

# ── Step 1: Python3.11 venv in /tmp SSD (not Lustre) ──
echo "=== Step 1: Python 3.11 venv in /tmp ==="
VENV_DIR="/tmp/nem_venv_$$"
python3.11 -m venv --without-pip "$VENV_DIR" 2>&1
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip_$$.py
"$VENV_DIR/bin/python" /tmp/get-pip_$$.py --quiet 2>&1
rm /tmp/get-pip_$$.py
source "$VENV_DIR/bin/activate"
echo "Python: $(python --version)"
echo "venv:   $VENV_DIR"
echo ""

# ── Step 2: ML stack (WITH deps, in /tmp SSD — fast) ──
echo "=== Step 2: ML stack ==="
pip install transformers accelerate peft trl datasets \
    safetensors sentencepiece wandb 2>&1
echo "Install complete."
echo ""

# ── Step 3: PyTorch ROCm LAST (overwrite CUDA torch) ──
echo "=== Step 3: PyTorch ROCm ==="
pip install --force-reinstall --no-deps torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.1 2>&1
echo ""
python -c "
import torch
print(f'PyTorch:  {torch.__version__}')
print(f'ROCm:     {torch.cuda.is_available()}')
print(f'GPU:      {torch.cuda.get_device_name(0)}')
p = torch.cuda.get_device_properties(0)
print(f'VRAM:     {round(p.total_memory/1e9,1)} GB')
print(f'Compute:  gfx{p.gcn_arch_name}')
"
echo ""

# ── Step 4: Verify dataset and model ──────────────────
echo "=== Step 4: Dataset ==="
if [ ! -f "$DATASET_JSONL" ]; then
    echo "FATAL: Dataset not found: $DATASET_JSONL"
    exit 1
fi
echo "Path:     $DATASET_JSONL"
echo "Examples: $(wc -l < "$DATASET_JSONL")"
echo ""

echo "=== Step 5: Model ==="
if [ -d "$MODEL_DIR" ]; then
    echo "Model:    $MODEL_DIR"
    echo "Files:    $(ls "$MODEL_DIR"/*.json 2>/dev/null | wc -l) config files"
    export NEMOTRON_MODEL_DIR="$MODEL_DIR"
else
    echo "Model:    HuggingFace Hub (nvidia/Nemotron-3-Nano-30B-A3B-BF16)"
fi
echo ""

# ── Step 6: Train ─────────────────────────────────────
echo "=== Step 6: Training ==="
echo "Start: $(date -Iseconds)"
echo "Script: $SCRIPTS_DIR/train_bf16_lora.py"
echo ""

python "$SCRIPTS_DIR/train_bf16_lora.py" \
    --dataset "$DATASET_JSONL" \
    --output "$OUTPUT_DIR"

TRAIN_EXIT=$?
echo ""
echo "Train exit code: $TRAIN_EXIT"
echo "End:   $(date -Iseconds)"

# ── Step 7: Git commit logs (always, pass or fail) ─────
echo ""
echo "=== Step 7: Git commit log ==="
cd "$PROJECT_DIR"
git add training_logs/ 2>&1 || true
if git diff --cached --quiet; then
    echo "No log changes to commit."
else
    STATUS="SUCCESS"
    if [ $TRAIN_EXIT -ne 0 ]; then STATUS="FAILED"; fi
    git commit -m "training: nem-dora M1 ${STATUS} — job $SLURM_JOB_ID ($TIMESTAMP)" 2>&1
    git push origin master 2>&1 || echo "NOTE: git push failed (non-fatal)"
    echo "Log committed: training_logs/${TIMESTAMP}_${SLURM_JOB_ID}/"
fi

echo ""
echo "═══════════════════════════════════════════"
echo "  DONE (exit=$TRAIN_EXIT)"
echo "  Log:  $LOG_DIR/job.out"
echo "  Output: $OUTPUT_DIR"
echo "═══════════════════════════════════════════"
