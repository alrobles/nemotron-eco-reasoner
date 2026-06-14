#!/bin/bash
# DIAG_GEN — one-shot raw-text generation diagnostic (NOT for PRs)
#   sbatch --export=ALL,ADAPTER=/path/to/checkpoint-N,CATS=numeral,cipher hpc/diag_gen.slurm
#SBATCH --job-name=nem-diag
#SBATCH --partition=sixhour
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/a474r867/scratch/nem_diag_%j.out
#SBATCH --error=/home/a474r867/scratch/nem_diag_%j.err

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
CATS=${CATS:-numeral,cipher}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-6000}
rm -rf $P 2>/dev/null; mkdir -p $P

echo "=== DIAG job=$SLURM_JOB_ID $(date) adapter=$ADAPTER cats=$CATS maxtok=$MAX_NEW_TOKENS ==="

apptainer exec --nv --bind "/dev/shm,/tmp" --env PYTHONUSERBASE=$P --env TRITON_CACHE_DIR=/tmp/triton_$SLURM_JOB_ID $C bash << 'INSTALL_EOF'
set -uo pipefail
pip install --user --no-cache-dir -q --no-deps "transformers==4.57.6" "tokenizers==0.22.2" 2>&1 | tail -1 || true
pip install --user --no-cache-dir -q --no-deps "datasets==4.3.0" bitsandbytes 2>&1 | tail -1 || true
pip install --user --no-cache-dir -q --no-deps unsloth_zoo unsloth rich "trl==0.12.0" 2>&1 | tail -1 || true
pip install --user --no-cache-dir -q --no-deps einops 2>&1 | tail -1 || { echo "FATAL: einops"; exit 1; }
WHEEL_DIR=/home/a474r867/scratch/wheels
pip install --user --no-deps $WHEEL_DIR/causal_conv1d-1.4.0-cp312-cp312-linux_x86_64.whl 2>&1 | tail -1
pip install --user --no-deps $WHEEL_DIR/mamba_ssm-2.2.4-cp312-cp312-linux_x86_64.whl 2>&1 | tail -1
VIRTUAL_ENV=${PYTHONUSERBASE:-$HOME/.local}
rm -rf $VIRTUAL_ENV/lib/python3*/site-packages/torch* $VIRTUAL_ENV/lib/python3*/site-packages/triton* $VIRTUAL_ENV/lib/python3*/site-packages/nvidia* 2>/dev/null
python3 -c 'import unsloth; print("unsloth OK")'
INSTALL_EOF

[ $? -ne 0 ] && { echo "INSTALL FAILED"; exit 1; }

apptainer exec --nv --bind "/dev/shm,/tmp" \
    --env PYTHONUSERBASE=$P --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env TORCH_COMPILE_DISABLE=1 --env OMP_NUM_THREADS=1 \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env MODEL_PATH=$M --env TRITON_CACHE_DIR=/tmp/triton_$SLURM_JOB_ID \
    $C env PYTHONUNBUFFERED=1 python3 -u $REPO/scripts/diag_gen.py \
    --adapter "$ADAPTER" --data $REPO/data/kaggle_classified.jsonl \
    --cats "$CATS" --max-new-tokens "$MAX_NEW_TOKENS"

echo "=== DIAG done $(date) ==="
