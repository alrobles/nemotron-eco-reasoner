#!/bin/bash
#SBATCH --job-name=smoke-bw2
#SBATCH --partition=sixhour
#SBATCH --time=01:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/a474r867/scratch/smoke_test_bw2_%j.out
#SBATCH --error=/home/a474r867/scratch/smoke_test_bw2_%j.err

# Smoke test v2: copy model flat (no symlinks) then validate
set -euo pipefail

VENV=/home/a474r867/scratch/nemotron-blackwell-venv
R=/home/a474r867/scratch/nemotron-eco-reasoner
SRC=/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848
FLAT=/home/a474r867/scratch/nemotron-model-flat
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== SMOKE TEST v2: PRO6000 BLACKWELL ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null
date

# Step 0: Flatten model (dereference symlinks) — only if not done yet
if [ ! -f "$FLAT/config.json" ]; then
    echo "=== STEP 0: Copy model (dereference symlinks) ==="
    mkdir -p "$FLAT"
    cp -rL "$SRC/"* "$FLAT/" 2>&1
    echo "Copy done: $(ls "$FLAT" | wc -l) files, $(du -sh "$FLAT" | cut -f1)"
else
    echo "=== STEP 0: Model already flat at $FLAT ($(ls "$FLAT" | wc -l) files) ==="
fi

# Activate venv
module load python/3.12 2>/dev/null || module load python/3.11 2>/dev/null || true
source "$VENV/bin/activate"
echo "Python: $(python3 --version)"
echo "Torch: $(python3 -c 'import torch; print(torch.__version__)')"

echo ""
echo "=== STEP 1: BF16 kernel test ==="
python3 -c "
import torch
x = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
y = x @ x.T
print(f'bf16 matmul OK: {y.shape}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Arch: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}')
"

echo ""
echo "=== STEP 2: Load model from flat dir ==="
export MODEL_PATH="$FLAT"
python3 << 'ENDPY'
import torch, time, os

t0 = time.time()
from transformers import AutoModelForCausalLM, AutoTokenizer

M = os.environ["MODEL_PATH"]
print(f"Loading model from {M}...")
print(f"Files: {len(os.listdir(M))}")

model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
load_time = time.time() - t0
print(f"Model loaded in {load_time:.1f}s")

mem = torch.cuda.memory_allocated() / 1e9
total = sum(p.numel() for p in model.parameters())
print(f"GPU memory: {mem:.1f} GB")
print(f"Total params: {total/1e9:.2f}B")

# Forward pass
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
inputs = tok("What is species distribution modeling?", return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model(**inputs)
    print(f"Forward pass OK: logits={out.logits.shape}")

print("STEP2_OK")
ENDPY

echo ""
echo "=== STEP 3: LoRA + fp32 cast ==="
python3 << 'ENDPY'
import torch, os
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

M = os.environ["MODEL_PATH"]
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)

candidate = {"q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj", "in_proj", "out_proj", "lm_head"}
target = sorted({n.split(".")[-1] for n, m in model.named_modules()
                 if isinstance(m, torch.nn.Linear)} & candidate)
print(f"LoRA targets: {target}")

lora_cfg = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM", target_modules=target)
model = get_peft_model(model, lora_cfg)

fp32_count = 0
for name, param in model.named_parameters():
    if ".lora_" in name and param.requires_grad:
        param.data = param.data.to(torch.float32)
        fp32_count += 1

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"LoRA fp32 params: {fp32_count}")
print(f"Trainable: {trainable/1e6:.1f}M / {total/1e9:.2f}B ({100*trainable/total:.2f}%)")
print(f"GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")
print("STEP3_OK")
ENDPY

echo ""
echo "=== STEP 4: Mini training (3 steps) ==="
export MODEL_PATH="$FLAT"
python3 << 'ENDPY'
import torch, os, time
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

M = os.environ["MODEL_PATH"]
DATA = "/home/a474r867/scratch/nemotron-eco-reasoner/data/ecoreasoner_train.jsonl"

# Load model + LoRA
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

candidate = {"q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj", "in_proj", "out_proj", "lm_head"}
target = sorted({n.split(".")[-1] for n, m in model.named_modules()
                 if isinstance(m, torch.nn.Linear)} & candidate)
lora_cfg = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM", target_modules=target)
model = get_peft_model(model, lora_cfg)

# fp32 LoRA params
for name, param in model.named_parameters():
    if ".lora_" in name and param.requires_grad:
        param.data = param.data.to(torch.float32)

# Load tiny subset
ds = load_dataset("json", data_files=DATA, split="train[:20]")
print(f"Dataset: {len(ds)} examples")

def fmt(ex):
    return {"text": f"<extra_id_0>System\n{ex.get('system','')}\n<extra_id_1>User\n{ex['prompt']}\n<extra_id_1>Assistant\n{ex['response']}"}

ds = ds.map(fmt)

# Train 3 steps
OUT = "/tmp/smoke_train_out"
config = SFTConfig(
    output_dir=OUT,
    max_steps=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=1,
    save_steps=3,
    max_seq_length=512,
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(model=model, args=config, train_dataset=ds, processing_class=tok)

t0 = time.time()
result = trainer.train()
elapsed = time.time() - t0
print(f"Training: {result.training_loss:.4f} loss, {elapsed:.1f}s for 3 steps")
print(f"GPU peak: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

# Save adapter
trainer.save_model(OUT + "/adapter")
print(f"Adapter saved: {os.listdir(OUT + '/adapter')}")
print("STEP4_OK — ALL SMOKE TESTS PASSED")
ENDPY

date
echo "=== SMOKE TEST v2 COMPLETE ==="
