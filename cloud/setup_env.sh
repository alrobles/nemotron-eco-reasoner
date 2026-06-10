#!/usr/bin/env bash
# One-shot environment setup for cloud GPU training (Lightning.ai, RunPod, Colab terminal).
# Safe to re-run; idempotent. Run on CPU machine to save credits, then switch to GPU.
set -uo pipefail

PIP="python -m pip"

echo "== [1/4] unsloth (with deps: peft, trl, xformers, sentencepiece, ...) =="
$PIP install -q unsloth unsloth_zoo

echo "== [2/4] sync torchvision/triton with the torch that xformers selected =="
$PIP install -q --upgrade torchvision triton

echo "== [3/4] pinned data/model stack =="
$PIP install -q "transformers==4.57.6" "tokenizers==0.22.2" "datasets==4.3.0" bitsandbytes einops ninja packaging

echo "== [4/4] mamba kernels (compile against installed torch; ~10-15 min) =="
$PIP install -q causal-conv1d --no-build-isolation
$PIP install -q mamba-ssm --no-build-isolation

echo "== Verify =="
python - <<'EOF'
import torch, torchvision, triton, transformers, trl, peft, datasets, bitsandbytes
print("torch       ", torch.__version__)
print("torchvision ", torchvision.__version__)
print("triton      ", triton.__version__)
print("transformers", transformers.__version__)
print("trl         ", trl.__version__)
print("peft        ", peft.__version__)
import importlib
for m in ("mamba_ssm", "causal_conv1d"):
    importlib.import_module(m); print(m, "OK")
import unsloth
print("unsloth OK")
print("\nENV READY. Switch the Studio to a GPU (L40S / RTX Pro 6000) and run: python cloud/train.py")
EOF
