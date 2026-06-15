#!/bin/bash
# NEM_EVAL_HIPREC — per-category eval at higher precision (plain transformers+peft,
# no Unsloth) to fix the 4-bit pred=None bug. QUANT picks the precision:
#   QUANT=8bit (default) LLM.int8, fits the 30B on ONE 48GB GPU (A40/L40).
#   QUANT=bf16           full precision, device_map=auto (needs ~60GB).
#   QUANT=4bit           nf4, single GPU (reproduces eval_checkpoint.py).
# Usage:
#   sbatch --gres=gpu:a40:1 --export=ALL,ADAPTER=/path/checkpoint-500,QUANT=8bit hpc/nem_eval_hiprec.slurm
#   # bf16 on 2 GPUs: sbatch --gres=gpu:a40:2 --export=ALL,ADAPTER=...,QUANT=bf16 hpc/nem_eval_hiprec.slurm
#SBATCH --job-name=nem-evhp
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/a474r867/scratch/nem_evhp_%j.out
#SBATCH --error=/home/a474r867/scratch/nem_evhp_%j.err

export APPTAINER_WRITABLE_TMPFS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_$SLURM_JOB_ID

REPO=/home/a474r867/scratch/nemotron-eco-reasoner
C=$REPO/nemotron-blackwell.sif
M=/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848
P=/tmp/pip_$SLURM_JOB_ID
ADAPTER=${ADAPTER:?set ADAPTER=/path/to/checkpoint-N}
N_PER_CAT=${N_PER_CAT:-8}
BATCH=${BATCH:-4}
CATS=${CATS:-}
QUANT=${QUANT:-8bit}
# NemotronH reasons in a long <think> block before \boxed{}; 4096 lets it finish.
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
rm -rf $P 2>/dev/null; mkdir -p $P

echo "=== EVAL_HIPREC job=$SLURM_JOB_ID $(date) adapter=$ADAPTER quant=$QUANT n_per_cat=$N_PER_CAT batch=$BATCH cats='$CATS' ==="

apptainer exec --nv --bind "/dev/shm,/tmp,$REPO" --env PYTHONUSERBASE=$P --env TRITON_CACHE_DIR=/tmp/triton_$SLURM_JOB_ID $C bash << 'INSTALL_EOF'
set -uo pipefail
pip install --user --no-cache-dir -q --no-deps "transformers==4.57.6" "tokenizers==0.22.2" 2>&1 | tail -1 || true
pip install --user --no-cache-dir -q --no-deps "datasets==4.3.0" bitsandbytes "peft==0.18.0" accelerate 2>&1 | tail -1 || true
pip install --user --no-cache-dir -q --no-deps einops 2>&1 | tail -1 || { echo "FATAL: einops"; exit 1; }
WHEEL_DIR=/home/a474r867/scratch/wheels
pip install --user --no-deps $WHEEL_DIR/causal_conv1d-1.4.0-cp312-cp312-linux_x86_64.whl 2>&1 | tail -1
pip install --user --no-deps $WHEEL_DIR/mamba_ssm-2.2.4-cp312-cp312-linux_x86_64.whl 2>&1 | tail -1
VENV=${PYTHONUSERBASE:-$HOME/.local}
rm -rf $VENV/lib/python3*/site-packages/torch* $VENV/lib/python3*/site-packages/triton* $VENV/lib/python3*/site-packages/nvidia* 2>/dev/null
python3 -c 'import transformers, peft, accelerate, bitsandbytes; print("deps OK peft", peft.__version__, "tf", transformers.__version__)'
INSTALL_EOF

[ $? -ne 0 ] && { echo "INSTALL FAILED"; exit 1; }

apptainer exec --nv --bind "/dev/shm,/tmp,$M,$REPO" \
    --env PYTHONUSERBASE=$P --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env TORCH_COMPILE_DISABLE=1 --env OMP_NUM_THREADS=1 \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env MODEL_PATH=$M --env QUANT=$QUANT --env TRITON_CACHE_DIR=/tmp/triton_$SLURM_JOB_ID \
    $C env PYTHONUNBUFFERED=1 python3 -u $REPO/scripts/eval_hiprec.py \
    --adapter "$ADAPTER" --data $REPO/data/kaggle_classified.jsonl \
    --n-per-cat "$N_PER_CAT" --max-new-tokens "$MAX_NEW_TOKENS" \
    --batch-size "$BATCH" --cats "$CATS"

echo "=== EVAL_HIPREC done $(date) ==="
