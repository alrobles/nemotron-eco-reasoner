#!/bin/bash
#SBATCH --job-name=setup-bw-venv
#SBATCH --partition=sixhour
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:pro6000:1
#SBATCH --ntasks=1
#SBATCH --output=/home/a474r867/scratch/setup_blackwell_venv_%j.out
#SBATCH --error=/home/a474r867/scratch/setup_blackwell_venv_%j.err

# Setup Blackwell-compatible venv for PRO6000 training
# PRO6000 Blackwell = sm_100/sm_120 → needs torch nightly with cu128
set -euo pipefail

VENV=/home/a474r867/scratch/nemotron-blackwell-venv
echo "=== BLACKWELL VENV SETUP ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
date

# Check for newer Python via modules
module load python/3.12 2>/dev/null || module load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null || true
PYTHON=$(command -v python3.12 2>/dev/null || command -v python3.11 2>/dev/null || command -v python3.10 2>/dev/null || command -v python3)
echo "Using Python: $PYTHON ($($PYTHON --version))"

# Clean slate
rm -rf "$VENV"
$PYTHON -m venv "$VENV"
source "$VENV/bin/activate"

# Upgrade pip (use pip3.9-compatible bootstrap if system python is old)
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    pip install --upgrade pip 2>&1 | tail -3
else
    # Python 3.9: get-pip.py dropped 3.9 support, use version-pinned bootstrap
    curl -sS https://bootstrap.pypa.io/pip/3.9/get-pip.py -o /tmp/get-pip-39.py
    python3 /tmp/get-pip-39.py 2>&1 | tail -3
fi
echo "pip: $(pip --version)"

# Install torch nightly with CUDA 12.8 (Blackwell support)
echo "Installing PyTorch nightly+cu128..."
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 2>&1 | tail -5
echo "torch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo FAILED)"

# Verify Blackwell GPU detection
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU name: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "CUDA arch: $(python3 -c 'import torch; print(torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "N/A")')"

# Install training dependencies
echo "Installing peft, trl, datasets, transformers, accelerate..."
pip install peft trl datasets transformers accelerate liger-kernel 2>&1 | tail -5

# Verify all imports
echo "=== VERIFICATION ==="
python3 -c "
import torch, peft, trl, datasets, transformers
print(f'torch={torch.__version__}')
print(f'peft={peft.__version__}')
print(f'trl={trl.__version__}')
print(f'datasets={datasets.__version__}')
print(f'transformers={transformers.__version__}')
print(f'cuda={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'gpu={torch.cuda.get_device_name(0)}')
    print(f'arch=sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}')
    # Quick kernel test
    x = torch.randn(100, 100, device='cuda', dtype=torch.bfloat16)
    y = x @ x.T
    print(f'bf16_matmul=OK (shape={y.shape})')
print('ALL_OK')
" 2>&1

echo "=== VENV READY at $VENV ==="
echo "Usage: sbatch --gres=gpu:pro6000:1 --export=ALL,VENV_PATH=$VENV hpc/eco_train_v2.slurm"
date
echo "SUCCESS"
