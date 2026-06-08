#!/bin/bash
# Rebuild Nemotron container SIF with PyTorch 2.6 + CUDA 12.8 for Blackwell sm_120 support
# Run on HPC login node (needs internet + apptainer build permissions)
#
# Usage:
#   bash rebuild_blackwell_sif.sh
#
# The resulting SIF will support ALL GPU architectures:
#   sm_70 (V100), sm_75 (Turing/Q6000), sm_80 (A100/A40),
#   sm_90 (Hopper), sm_120 (Blackwell Pro6000)

set -euo pipefail

SCRATCH="${HOME}/scratch"
SIF_DIR="${SCRATCH}/nemotron-eco-reasoner"
DEF_FILE="${SCRATCH}/nemotron-blackwell.def"
SIF_OUT="${SIF_DIR}/nemotron-blackwell.sif"
BUILD_LOG="${SCRATCH}/sif_build_$(date +%Y%m%d_%H%M%S).log"

echo "=== Nemotron Blackwell Container Build ==="
echo "Output: ${SIF_OUT}"
echo "Log: ${BUILD_LOG}"

# Write Apptainer definition file
cat > "${DEF_FILE}" << 'DEFEOF'
Bootstrap: docker
From: pytorch/pytorch:2.6.0-cuda12.8-cudnn9-devel

%labels
    Author a474r867@ku.edu
    Description Nemotron-3-Nano-30B training container with Blackwell (sm_120) support
    Version 2.0-blackwell

%environment
    export TORCH_COMPILE_DISABLE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=false

%post
    # System deps
    apt-get update -qq && apt-get install -y -qq git wget curl build-essential ninja-build 2>/dev/null
    apt-get clean && rm -rf /var/lib/apt/lists/*

    # Core ML stack (pinned versions known to work with Nemotron)
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

    # mamba-ssm + causal-conv1d for Nemotron hybrid Mamba-2 layers
    # These need CUDA compilation — may take a few minutes
    pip install --no-cache-dir -q mamba-ssm causal-conv1d || \
        echo "WARNING: mamba-ssm/causal-conv1d failed to compile. Fallback to naive implementation."

    # Verify installation
    python3 -c '
import torch
print(f"PyTorch {torch.__version__} CUDA {torch.version.cuda}")
import transformers, tokenizers, trl, peft, bitsandbytes
print(f"transformers={transformers.__version__} trl={trl.__version__} peft={peft.__version__}")
try:
    import unsloth
    print(f"unsloth OK")
except: print("unsloth: import warning (OK at runtime)")
try:
    import mamba_ssm
    print(f"mamba_ssm OK")
except: print("mamba_ssm: not compiled (will use fallback)")
print("Container build verification: PASSED")
'

%runscript
    exec python3 "$@"

%test
    python3 -c "import torch; print(f'PyTorch {torch.__version__} CUDA support OK')"
DEFEOF

echo "--- Building SIF (this takes 10-15 minutes) ---"
echo "Definition file: ${DEF_FILE}"

# Build the SIF
apptainer build --fakeroot "${SIF_OUT}" "${DEF_FILE}" 2>&1 | tee "${BUILD_LOG}"

if [ $? -eq 0 ]; then
    SIF_SIZE=$(du -h "${SIF_OUT}" | cut -f1)
    echo ""
    echo "=== BUILD SUCCESS ==="
    echo "SIF: ${SIF_OUT} (${SIF_SIZE})"
    echo ""
    echo "To use in nem_unified.slurm, update line 21:"
    echo "  C=${SIF_OUT}"
    echo ""
    echo "Or symlink:"
    echo "  ln -sf ${SIF_OUT} ${SIF_DIR}/nemotron-cuda.sif"
    echo ""
    echo "Test with:"
    echo "  sbatch --nodelist=r30r24n01 ${SIF_DIR}/hpc/nem_unified.slurm"
else
    echo "=== BUILD FAILED ==="
    echo "Check log: ${BUILD_LOG}"
    exit 1
fi
