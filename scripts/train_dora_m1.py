#!/usr/bin/env python3
"""
M1 Training Script — Nemotron-3-Nano with DoRA on single MI210.
Self-contained: creates venv, installs deps, trains, saves adapter.

Usage:
  python3 train_dora_m1.py \
    --model /scratch/nemotron-model \
    --data /scratch/nemotron-eco-reasoner/data/kaggle_5k_train.jsonl \
    --output /scratch/nemotron-eco-reasoner/outputs/m1_run1 \
    --max_steps 500
"""

import argparse
import json
import os
import subprocess
import sys
import time

# ─── Step 0: Parse args ───────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Path to base model")
parser.add_argument("--data", required=True, help="Path to training JSONL")
parser.add_argument("--output", required=True, help="Output directory for adapter")
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--grad_accum", type=int, default=8)
parser.add_argument("--lr", type=float, default=2e-4)
args = parser.parse_args()

# ─── Constants ────────────────────────────────────────────────────
VENV_DIR = "/tmp/nemotron-venv"
VENV_TARBALL = os.path.expanduser("~/scratch/nemotron-rocm-venv.tar.gz")
LOG_FILE = os.path.join(args.output, "train.log")

def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    # Also write to log file if output dir exists
    try:
        os.makedirs(args.output, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass  # Lustre might not be ready yet

log("=" * 60)
log("M1 TRAINING — Nemotron-3-Nano + DoRA + MI210")
log("=" * 60)
log(f"Model: {args.model}")
log(f"Data: {args.data}")
log(f"Output: {args.output}")
log(f"Max steps: {args.max_steps}")
log(f"Batch size: {args.batch_size} x {args.grad_accum} grad accum")

# ─── Step 1: Setup venv ───────────────────────────────────────────
log("Step 1: Setting up venv...")
if not os.path.exists(VENV_DIR):
    if os.path.exists(VENV_TARBALL):
        log(f"  Extracting tarball from {VENV_TARBALL}...")
        os.makedirs(VENV_DIR, exist_ok=True)
        subprocess.run(["tar", "-xzf", VENV_TARBALL, "-C", VENV_DIR], check=True)
        log("  Tarball extracted.")
    else:
        log(f"  Creating fresh venv at {VENV_DIR}...")
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", VENV_DIR], check=True)
        # Bootstrap pip
        subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=True)
        pip = os.path.join(VENV_DIR, "bin", "pip")
        subprocess.run([sys.executable, "-m", "pip", "install", "--target",
                        os.path.join(VENV_DIR, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"),
                        "pip"], check=True)
        # Auto-detect GPU type and install correct PyTorch
        gpu_type = "cpu"
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, timeout=5, check=False)
            gpu_type = "cuda"
        except FileNotFoundError:
            pass
        if gpu_type == "cpu":
            try:
                subprocess.run(["rocm-smi"], capture_output=True, timeout=5, check=False)
                gpu_type = "rocm"
            except FileNotFoundError:
                pass
        if gpu_type == "rocm":
            log("  Installing PyTorch ROCm...")
            subprocess.run([pip, "install", "--no-deps", "torch==2.6.0",
                            "--index-url", "https://download.pytorch.org/whl/rocm6.1"], check=True)
        else:
            log("  Installing PyTorch CUDA...")
            subprocess.run([pip, "install", "torch==2.6.0",
                            "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)
        log("  Installing core packages (WITH deps on local SSD — fast)...")
        subprocess.run([pip, "install",
            "transformers", "peft", "accelerate", "datasets", "trl",
            "sentencepiece", "huggingface-hub", "httpcore",
            "wandb", "tensorboard",
        ], check=True)
        log("  Venv ready.")
python_exe = os.path.join(VENV_DIR, "bin", "python")
log("  Verifying imports...")
subprocess.run([python_exe, "-c",
    "import transformers, peft, datasets, trl, accelerate; "
    "print(f'transformers={transformers.__version__}, peft={peft.__version__}')"
], check=True)
log(f"  Python: {python_exe}")

# ─── Step 2: Check GPU ───────────────────────────────────────────
log("Step 2: Checking GPU...")
result = subprocess.run([python_exe, "-c", """
import torch
print(f"PyTorch: {torch.__version__}")
print(f"ROCm available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
"""], capture_output=True, text=True, check=True)
log(result.stdout.strip())

# ─── Step 3: Training ─────────────────────────────────────────────
log("Step 3: Starting training...")
os.makedirs(args.output, exist_ok=True)

training_args = [
    python_exe, "-u", "-c", f"""
import json, os, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import Dataset

# Load data
log = lambda m: print(f"[{{time.strftime('%H:%M:%S')}}] {{m}}", flush=True)

log("Loading dataset...")
with open("{args.data}") as f:
    data = [json.loads(line) for line in f]
log(f"  Loaded {{len(data)}} examples")

# Format for SFT
def format_example(ex):
    return {{
        "text": f"<|user|>\\n{{ex['prompt']}}\\n<|assistant|>\\n{{ex['answer']}}<|endoftext|>"
    }}

formatted = [format_example(ex) for ex in data]
dataset = Dataset.from_list(formatted)
log(f"  Dataset ready: {{len(dataset)}} examples")

# Load model
log("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "{args.model}",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("{args.model}", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

log("Configuring DoRA...")
# Target both Transformer and Mamba layers
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj", "out_proj",
]

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    use_dora=True,  # <-- Weight-Decomposed Low-Rank Adaptation
)
model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
log(f"  Trainable: {{trainable/1e6:.1f}}M params ({{100*trainable/total:.2f}}%)")

training_args = TrainingArguments(
    output_dir="{args.output}",
    num_train_epochs=3,
    per_device_train_batch_size={args.batch_size},
    gradient_accumulation_steps={args.grad_accum},
    learning_rate={args.lr},
    warmup_steps=50,
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    bf16=True,
    max_steps={args.max_steps},
    remove_unused_columns=False,
    report_to="none",
    dataloader_num_workers=0,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=1024,
)

log("Training...")
trainer.train()
log("Saving adapter...")
trainer.save_model("{args.output}")
log("DONE!")
"""]
subprocess.run(training_args, check=True)
log("Training complete!")
