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

# Clean slate
rm -rf "$VENV"
python3 -m venv --without-pip "$VENV"
source "$VENV/bin/activate"

# Bootstrap pip (Python 3.9 bundled pip has broken resolvelib)
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python3 /tmp/get-pip.py 2>&1 | tail -3
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
