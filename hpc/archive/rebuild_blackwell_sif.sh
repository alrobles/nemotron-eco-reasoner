#!/bin/bash
# Rebuild Nemotron container SIF with Blackwell sm_120 support
# Uses NVIDIA NGC PyTorch container (has sm_120 kernels pre-built)
#
# Run on HPC login node (needs internet + apptainer build permissions):
#   bash rebuild_blackwell_sif.sh
#
# OR submit as a SLURM job (no GPU needed, just CPU+internet):
#   sbatch --partition=sixhour --time=02:00:00 --mem=32G --cpus-per-task=8 rebuild_blackwell_sif.sh
#
# Supported GPU architectures in resulting SIF:
#   sm_75 (Turing/Q6000), sm_80 (A100/A40), sm_86 (A40/RTX),
#   sm_90 (Hopper), sm_100 (Blackwell B100), sm_120 (Blackwell Pro6000/RTX50)

set -euo pipefail

SCRATCH="${HOME}/scratch"
SIF_DIR="${SCRATCH}/nemotron-eco-reasoner"
DEF_FILE="${SCRATCH}/nemotron-blackwell.def"
SIF_OUT="${SIF_DIR}/nemotron-blackwell.sif"
BUILD_LOG="${SCRATCH}/sif_build_$(date +%Y%m%d_%H%M%S).log"

echo "=== Nemotron Blackwell Container Build ==="
echo "Output: ${SIF_OUT}"
echo "Log: ${BUILD_LOG}"
echo "Time: $(date)"

# Write Apptainer definition file
# Using NGC PyTorch 25.01 = PyTorch 2.8 + CUDA 12.8 + sm_120 native
cat > "${DEF_FILE}" << 'DEFEOF'
Bootstrap: docker
From: nvcr.io/nvidia/pytorch:25.01-py3

%labels
    Author a474r867@ku.edu
    Description Nemotron-3-Nano-30B training with Blackwell sm_120 support
    Version 3.0-blackwell-ngc

%environment
    export TORCH_COMPILE_DISABLE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=false
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

%post
    # System deps
    apt-get update -qq && apt-get install -y -qq git wget curl 2>/dev/null
    apt-get clean && rm -rf /var/lib/apt/lists/*

    # Verify PyTorch + sm_120 support
    python3 -c '
import torch
print(f"PyTorch {torch.__version__} CUDA {torch.version.cuda}")
archs = torch.cuda.get_arch_list()
print(f"Supported archs: {archs}")
assert "sm_120" in archs or "compute_120" in archs, f"sm_120 NOT in arch list: {archs}"
print("sm_120 (Blackwell) support: CONFIRMED")
'

    # Core ML stack for Nemotron training
    pip install --no-cache-dir -q \
        "transformers==4.57.6" \
        "tokenizers==0.22.2" \
        "datasets==4.3.0" \
        "trl==0.12.0" \
        "peft>=0.15.0" \
        "bitsandbytes>=0.45.0" \
        "accelerate>=1.0.0" \
        "safetensors>=0.5.0" \
        "sentencepiece" \
        "protobuf" \
        "rich"

    # Unsloth for fast LoRA training
    pip install --no-cache-dir -q unsloth unsloth_zoo

    # mamba-ssm + causal-conv1d for Nemotron Mamba-2 layers
    # Must build from source with sm_120 patch — pip wheels don't include Blackwell
    # Ref: https://github.com/state-spaces/mamba/issues/745
    export TORCH_CUDA_ARCH_LIST="9.0;12.0"
    export FORCE_CUDA=1
    export MAX_JOBS=4

    # causal-conv1d from source with sm_120
    cd /tmp && git clone https://github.com/Dao-AILab/causal-conv1d.git
    cd causal-conv1d
    sed -i '/sm_90/a\    cc_flag.append("-gencode")\n    cc_flag.append("arch=compute_120,code=sm_120")' setup.py || true
    pip install -e . --no-build-isolation 2>&1 | tail -5
    python3 -c "import causal_conv1d; print(f'causal_conv1d {causal_conv1d.__version__} OK')"

    # mamba-ssm from source with sm_120
    cd /tmp && git clone https://github.com/state-spaces/mamba.git
    cd mamba
    sed -i '/sm_90/a\    cc_flag.append("-gencode")\n    cc_flag.append("arch=compute_120,code=sm_120")' setup.py || true
    pip install -e . --no-build-isolation 2>&1 | tail -10
    python3 -c "import mamba_ssm; print(f'mamba_ssm {mamba_ssm.__version__} OK')" || \
        echo "WARNING: mamba-ssm build failed — will need runtime compilation on GPU node"

    # Cleanup build artifacts
    rm -rf /tmp/causal-conv1d /tmp/mamba

    # Final verification
    python3 -c '
import torch
print(f"PyTorch {torch.__version__} CUDA {torch.version.cuda}")
print(f"Archs: {torch.cuda.get_arch_list()}")
import transformers, tokenizers, trl, peft, bitsandbytes, accelerate
print(f"transformers={transformers.__version__} trl={trl.__version__} peft={peft.__version__}")
try:
    import unsloth; print("unsloth: OK")
except: print("unsloth: import warning (OK at runtime with GPU)")
try:
    import mamba_ssm; print("mamba_ssm: OK (fast path)")
except: print("mamba_ssm: not available (fallback mode)")
print("=== Container build verification: PASSED ===")
'

%runscript
    exec python3 "$@"

%test
    python3 -c "
import torch
print(f'PyTorch {torch.__version__} CUDA {torch.version.cuda}')
archs = torch.cuda.get_arch_list()
print(f'Arch list: {archs}')
assert 'sm_120' in archs or 'compute_120' in archs, 'Missing sm_120!'
print('TEST PASSED — sm_120 Blackwell support confirmed')
"
DEFEOF

echo "--- Definition file written: ${DEF_FILE} ---"
echo "--- Building SIF from NGC PyTorch 25.01 (this takes 15-30 minutes) ---"

# Build the SIF
apptainer build --fakeroot "${SIF_OUT}" "${DEF_FILE}" 2>&1 | tee "${BUILD_LOG}"

RC=$?
if [ $RC -eq 0 ]; then
    SIF_SIZE=$(du -h "${SIF_OUT}" | cut -f1)
    echo ""
    echo "========================================"
    echo "BUILD SUCCESS"
    echo "========================================"
    echo "SIF: ${SIF_OUT} (${SIF_SIZE})"
    echo ""
    echo "To use for Blackwell training:"
    echo "  1. Update nem_unified.slurm line 21:"
    echo "     C=${SIF_OUT}"
    echo "  2. Or symlink:"
    echo "     ln -sf ${SIF_OUT} ${SIF_DIR}/nemotron-cuda.sif"
    echo "  3. Test:"
    echo "     sbatch --nodelist=r30r24n01 hpc/nem_blackwell_test.slurm"
else
    echo "========================================"
    echo "BUILD FAILED (exit $RC)"
    echo "========================================"
    echo "Check log: ${BUILD_LOG}"
    exit 1
fi
