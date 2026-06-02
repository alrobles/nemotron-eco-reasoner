#!/usr/bin/env python3
"""Train DoRA adapter on Nemotron-3-Nano using existing ROCm venv."""

import os, sys, json, time, argparse
from datetime import datetime

# ── Config ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", required=True)
parser.add_argument("--data-dir", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--max-steps", type=int, default=500)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--grad-accum", type=int, default=8)
args = parser.parse_args()

import torch
import transformers
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join(args.output_dir, f"nem_dora_{timestamp}")
os.makedirs(out_dir, exist_ok=True)

print(f"PyTorch {torch.__version__}")
print(f"ROCm available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Load dataset ────────────────────────────────────────
data_files = {"train": os.path.join(args.data_dir, "kaggle_5k_train.jsonl")}
dataset = load_dataset("json", data_files=data_files, split="train")
print(f"Loaded {len(dataset)} examples")

# ── Tokenizer & Model ───────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model (this takes ~2 min)...")
model = AutoModelForCausalLM.from_pretrained(
    args.model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# ── Tokenize ────────────────────────────────────────────
def tokenize(examples):
    texts = []
    for p, a in zip(examples["prompt"], examples["answer"]):
        texts.append(f"Question: {p}\nAnswer: {a}")
    return tokenizer(texts, truncation=True, max_length=1024, padding=False)

tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
print(f"Tokenized. Sample length: {len(tokenized[0]['input_ids'])} tokens")

# ── DoRA config ─────────────────────────────────────────
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    use_dora=True,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Train ───────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=out_dir,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    max_steps=args.max_steps,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    bf16=True,
    learning_rate=2e-4,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    report_to=[],
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print(f"Starting training at {timestamp}...")
start = time.time()
trainer.train()
elapsed = time.time() - start
print(f"Training complete in {elapsed/60:.1f} minutes")

# ── Save adapter ────────────────────────────────────────
adapter_path = os.path.join(out_dir, "adapter")
model.save_pretrained(adapter_path)
print(f"Adapter saved to {adapter_path}")
print("DONE")
