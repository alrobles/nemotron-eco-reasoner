#!/bin/bash
#SBATCH --job-name=smoke-bw
#SBATCH --partition=sixhour
#SBATCH --time=00:30:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/a474r867/scratch/smoke_test_bw_%j.out
#SBATCH --error=/home/a474r867/scratch/smoke_test_bw_%j.err

# Smoke test: validate full training pipeline on PRO6000 Blackwell
# Runs 5 training steps to confirm: model load, LoRA, forward, backward, save
set -euo pipefail

VENV=/home/a474r867/scratch/nemotron-blackwell-venv
R=/home/a474r867/scratch/nemotron-eco-reasoner
export M=/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/a474r867/scratch/nemotron-model-cache

echo "=== SMOKE TEST: PRO6000 BLACKWELL ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null
date

# Load Python 3.11+ module (needed for Blackwell venv)
module load python/3.12 2>/dev/null || module load python/3.11 2>/dev/null || true

source "$VENV/bin/activate"
echo "Python: $(python3 --version)"
echo "Torch: $(python3 -c 'import torch; print(torch.__version__, "cuda" if torch.cuda.is_available() else "cpu")')"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "Arch: $(python3 -c 'import torch; c=torch.cuda.get_device_capability(0); print(f"sm_{c[0]}{c[1]}")')"
echo "VRAM: $(python3 -c 'import torch; p=torch.cuda.get_device_properties(0); m=getattr(p,"total_memory",None) or getattr(p,"total_mem",0); print(f"{m/1e9:.1f} GB")')"

echo ""
echo "=== STEP 1: BF16 kernel test ==="
python3 -c "
import torch
x = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
y = x @ x.T
print(f'bf16 matmul OK: {y.shape}, sum={y.sum().item():.2f}')
"

echo ""
echo "=== STEP 2: Load tokenizer ==="
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('$M', trust_remote_code=True)
print(f'Tokenizer loaded: vocab={tok.vocab_size}, pad={tok.pad_token}')
test = tok('Hello ecological world', return_tensors='pt')
print(f'Encode OK: {test.input_ids.shape}')
"

echo ""
echo "=== STEP 3: Load model (BF16, this is the big one) ==="
python3 << PYEOF
import torch, time, os

t0 = time.time()
from transformers import AutoModelForCausalLM, AutoTokenizer

M = os.environ["M"]
print(f"Loading model from {M}...")

model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, local_files_only=True
)
load_time = time.time() - t0
print(f"Model loaded in {load_time:.1f}s")

# Memory check
mem_alloc = torch.cuda.memory_allocated() / 1e9
mem_reserved = torch.cuda.memory_reserved() / 1e9
print(f"GPU memory: {mem_alloc:.1f} GB allocated, {mem_reserved:.1f} GB reserved")

# Count parameters
total = sum(p.numel() for p in model.parameters())
print(f"Total params: {total/1e9:.2f}B")

# Forward pass test
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
inputs = tok("What is species distribution modeling?", return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model(**inputs)
    print(f"Forward pass OK: logits shape={out.logits.shape}, loss={out.logits.mean().item():.4f}")

print("STEP3_OK")
PYEOF

echo ""
echo "=== STEP 4: LoRA setup + fp32 cast ==="
python3 << PYEOF
import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

M = os.environ["M"]
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, local_files_only=True
)

# Architecture-aware LoRA targets (3rd place config)
candidate = {"q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj", "in_proj", "out_proj", "lm_head"}
target = sorted({n.split(".")[-1] for n, m in model.named_modules()
                 if isinstance(m, torch.nn.Linear)} & candidate)
print(f"LoRA targets: {target}")

lora_cfg = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM", target_modules=target)
model = get_peft_model(model, lora_cfg)

# fp32 cast (3rd place technique)
fp32_count = 0
for name, param in model.named_parameters():
    if ".lora_" in name and param.requires_grad:
        param.data = param.data.to(torch.float32)
        fp32_count += 1
print(f"LoRA fp32 cast: {fp32_count} params")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable/1e6:.1f}M / {total/1e9:.2f}B total ({100*trainable/total:.2f}%)")

mem = torch.cuda.memory_allocated() / 1e9
print(f"GPU after LoRA: {mem:.1f} GB")
print("STEP4_OK")
PYEOF

echo ""
echo "=== STEP 5: Mini training (5 steps) ==="
export M DATA="$R/data/ecoreasoner_train.jsonl" OUT="/tmp/smoke_test_out"
export TARGET_TOTAL=5 STEPS_PER_JOB=5 SEQ_LEN=2048 GRAD_ACCUM=1
export LR=2e-4 NUDGE_LR=5e-6 NUDGE_FRAC=0.3 NEFTUNE_ALPHA=5.0
export LORA_FP32=1 COMPLETION_ONLY=1 INCLUDE_LM_HEAD=1
export MASTER_ADDR=$(hostname) MASTER_PORT=29500
mkdir -p /tmp/smoke_test_out

python3 $R/hpc/eco_train_v2.py 2>&1 | tail -30

echo ""
echo "=== RESULTS ==="
ls -la /tmp/smoke_test_out/ 2>/dev/null | head -10
echo ""
echo "GPU final: $(python3 -c 'import torch; print(f"{torch.cuda.memory_allocated()/1e9:.1f} GB allocated")')"
date
echo "=== SMOKE TEST COMPLETE ==="
