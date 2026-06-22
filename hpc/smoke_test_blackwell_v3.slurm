#!/bin/bash
#SBATCH --job-name=smoke-bw3
#SBATCH --partition=sixhour
#SBATCH --time=01:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/a474r867/scratch/smoke_test_bw3_%j.out
#SBATCH --error=/home/a474r867/scratch/smoke_test_bw3_%j.err

# Smoke test v3: load from snapshot path directly (has real files for .py configs)
set -euo pipefail

VENV=/home/a474r867/scratch/nemotron-blackwell-venv
R=/home/a474r867/scratch/nemotron-eco-reasoner
export MODEL_PATH=/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== SMOKE TEST v3: PRO6000 BLACKWELL ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null
date

module load python/3.12 2>/dev/null || module load python/3.11 2>/dev/null || true
source "$VENV/bin/activate"
echo "Python: $(python3 --version)"
echo "Torch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "Transformers: $(python3 -c 'import transformers; print(transformers.__version__)')"

echo ""
echo "=== STEP 1: BF16 kernel ==="
python3 -c "
import torch
x = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
y = x @ x.T
print(f'OK: {torch.cuda.get_device_name(0)}, sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}')
"
echo "STEP1_PASS"

echo ""
echo "=== STEP 2: Load model ==="
python3 << 'ENDPY'
import torch, time, os, sys

M = os.environ["MODEL_PATH"]
print(f"Model path: {M}")
print(f"Files: {sorted(os.listdir(M))}")

# Check that config files exist as real files
for f in ["config.json", "configuration_nemotron_h.py", "modeling_nemotron_h.py"]:
    path = os.path.join(M, f)
    exists = os.path.exists(path)
    is_link = os.path.islink(path)
    size = os.path.getsize(path) if exists else 0
    print(f"  {f}: exists={exists}, symlink={is_link}, size={size}")

t0 = time.time()
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# First test: can we load the config?
print("Loading config...")
try:
    cfg = AutoConfig.from_pretrained(M, trust_remote_code=True, local_files_only=True)
    print(f"Config loaded: {cfg.model_type}")
except Exception as e:
    print(f"Config FAILED: {e}")
    # Try without trust_remote_code
    try:
        cfg = AutoConfig.from_pretrained(M, local_files_only=True)
        print(f"Config loaded (no trust_remote_code): {cfg.model_type}")
    except Exception as e2:
        print(f"Config FAILED again: {e2}")
        sys.exit(1)

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True, local_files_only=True)
print(f"Tokenizer: vocab={tok.vocab_size}")

print("Loading model (BF16, ~60GB → 96GB VRAM)...")
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, local_files_only=True
)
load_time = time.time() - t0
print(f"Model loaded in {load_time:.1f}s")

mem = torch.cuda.memory_allocated() / 1e9
total = sum(p.numel() for p in model.parameters())
print(f"GPU: {mem:.1f} GB, Params: {total/1e9:.2f}B")

# Forward pass
inputs = tok("What is ecological niche modeling?", return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model(**inputs)
    print(f"Forward: logits={out.logits.shape}")
print("STEP2_PASS")
ENDPY

echo ""
echo "=== STEP 3: LoRA + fp32 ==="
python3 << 'ENDPY'
import torch, os
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

M = os.environ["MODEL_PATH"]
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, local_files_only=True
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
print(f"LoRA fp32: {fp32_count} params cast")
print(f"Trainable: {trainable/1e6:.1f}M / {total/1e9:.2f}B ({100*trainable/total:.2f}%)")
print(f"GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")
print("STEP3_PASS")
ENDPY

echo ""
echo "=== STEP 4: Training (3 steps) ==="
python3 << 'ENDPY'
import torch, os, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

M = os.environ["MODEL_PATH"]
DATA = "/home/a474r867/scratch/nemotron-eco-reasoner/data/ecoreasoner_train.jsonl"

model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, local_files_only=True
)
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True, local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

candidate = {"q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj", "in_proj", "out_proj", "lm_head"}
target = sorted({n.split(".")[-1] for n, m in model.named_modules()
                 if isinstance(m, torch.nn.Linear)} & candidate)
lora_cfg = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM", target_modules=target)
model = get_peft_model(model, lora_cfg)
for name, param in model.named_parameters():
    if ".lora_" in name and param.requires_grad:
        param.data = param.data.to(torch.float32)

ds = load_dataset("json", data_files=DATA, split="train[:20]")
print(f"Dataset: {len(ds)} examples, columns: {ds.column_names}")

# Dataset uses 'messages' format (chat completions style)
def fmt(ex):
    msgs = ex.get("messages", [])
    parts = []
    for m in msgs:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"<extra_id_0>System\n{content}")
        elif role == "user":
            parts.append(f"<extra_id_1>User\n{content}")
        elif role == "assistant":
            parts.append(f"<extra_id_1>Assistant\n{content}")
    return {"text": "\n".join(parts) if parts else "Hello world"}
ds = ds.map(fmt)

OUT = "/tmp/smoke_train_out"
try:
    config = SFTConfig(
        output_dir=OUT, max_steps=3, per_device_train_batch_size=1,
        gradient_accumulation_steps=1, learning_rate=2e-4, bf16=True,
        logging_steps=1, save_steps=3, max_seq_length=512,
        dataset_text_field="text", report_to="none",
    )
except TypeError:
    # trl >= 1.5: max_seq_length removed, use TrainingArguments style
    from transformers import TrainingArguments
    config = SFTConfig(
        output_dir=OUT, max_steps=3, per_device_train_batch_size=1,
        gradient_accumulation_steps=1, learning_rate=2e-4, bf16=True,
        logging_steps=1, save_steps=3, report_to="none",
    )
trainer = SFTTrainer(model=model, args=config, train_dataset=ds, processing_class=tok)

t0 = time.time()
result = trainer.train()
elapsed = time.time() - t0
print(f"Loss: {result.training_loss:.4f}, Time: {elapsed:.1f}s")
print(f"GPU peak: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
trainer.save_model(OUT + "/adapter")
print(f"Adapter: {os.listdir(OUT + '/adapter')}")
print("STEP4_PASS — ALL SMOKE TESTS PASSED")
ENDPY

date
echo "=== SMOKE TEST v3 COMPLETE ==="
