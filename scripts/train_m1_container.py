#!/usr/bin/env python3
"""
M1 Training Script — Nemotron-3-Nano + DoRA + Apptainer container.
Zero venv setup — runs inside pre-built nemotron-rocm.sif.

Usage (inside container):
  python3 /scratch/nemotron-eco-reasoner/scripts/train_m1_container.py \
    --model /scratch/nemotron-model \
    --data /scratch/nemotron-eco-reasoner/data/kaggle_5k_train.jsonl \
    --output /scratch/nemotron-eco-reasoner/outputs/m1_run1 \
    --max_steps 500
"""

import argparse
import json
import os
import sys
import time
import torch
from pathlib import Path

# ─── Parse args ────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Path to base model")
parser.add_argument("--data", required=True, help="Path to training JSONL")
parser.add_argument("--output", required=True, help="Output directory for adapter")
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--grad_accum", type=int, default=8)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--max_seq_len", type=int, default=1024)
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--use_dora", type=int, default=1, help="1=DoRA, 0=standard LoRA")
args = parser.parse_args()

LOG_FILE = os.path.join(args.output, "train.log")


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        os.makedirs(args.output, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


log("=" * 60)
log("M1 TRAINING — Nemotron-3-Nano + DoRA + MI210 (Container)")
log("=" * 60)
log(f"PyTorch: {torch.__version__}")
log(f"CUDA/ROCm: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
log(f"Model: {args.model}")
log(f"Data: {args.data}")
log(f"Output: {args.output}")
log(f"Steps: {args.max_steps}, Batch: {args.batch_size}×{args.grad_accum}, LR: {args.lr}")
log(f"DoRA: {bool(args.use_dora)}")

# ─── Load data ─────────────────────────────────────────────────────
log("Loading dataset...")
with open(args.data) as f:
    raw = [json.loads(line) for line in f]
log(f"  {len(raw)} examples loaded")

# Transform to SFT format
from datasets import Dataset


def format_example(ex):
    user = ex.get("prompt", ex.get("user", ""))
    assistant = ex.get("answer", ex.get("assistant", ""))
    return {
        "text": f"<|user|>\n{user}\n<|assistant|>\n{assistant}<|endoftext|>"
    }


texts = [format_example(ex)["text"] for ex in raw]
dataset = Dataset.from_dict({"text": texts})
log(f"  Dataset ready: {len(dataset)} texts")

# ─── Load model ────────────────────────────────────────────────────
log("Loading model (BF16)...")
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

# trust_remote_code=False — let transformers 4.57.6 use native Nemotron support
# (the model's custom modeling_nemotron_h.py requires mamba-ssm which we don't have)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(args.model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ─── Configure DoRA ─────────────────────────────────────────────────
from peft import LoraConfig, get_peft_model, TaskType

use_dora = bool(args.use_dora)

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
    use_dora=use_dora,
)
model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
log(f"  Trainable: {trainable/1e6:.1f}M / {total/1e9:.2f}B ({100*trainable/total:.2f}%)")
log(f"  Mode: {'DoRA' if use_dora else 'LoRA'}")

# ─── Training config ────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=args.output,
    num_train_epochs=args.num_epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    warmup_steps=50,
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    bf16=True,
    max_steps=args.max_steps,
    remove_unused_columns=False,
    report_to="none",
    dataloader_num_workers=0,
)

from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=args.max_seq_len,
)

# ─── Train ──────────────────────────────────────────────────────────
log("Starting training...")
t0 = time.time()
trainer.train()
elapsed = time.time() - t0
log(f"Training complete in {elapsed/60:.1f} minutes")

# ─── Save ───────────────────────────────────────────────────────────
log(f"Saving adapter to {args.output}...")
trainer.save_model(args.output)
tokenizer.save_pretrained(args.output)

# Save training manifest
manifest = {
    "model": args.model,
    "dataset": args.data,
    "num_examples": len(raw),
    "max_steps": args.max_steps,
    "batch_size": args.batch_size,
    "grad_accum": args.grad_accum,
    "effective_batch": args.batch_size * args.grad_accum,
    "lr": args.lr,
    "max_seq_len": args.max_seq_len,
    "num_epochs": args.num_epochs,
    "use_dora": use_dora,
    "lora_rank": 32,
    "lora_alpha": 32,
    "trainable_params": trainable,
    "total_params": total,
    "elapsed_minutes": elapsed / 60,
    "torch_version": torch.__version__,
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
}
with open(os.path.join(args.output, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

log("DONE!")
log(f"Adapter saved to {args.output}")
log(f"Manifest: {os.path.join(args.output, 'manifest.json')}")
