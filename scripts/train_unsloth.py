#!/usr/bin/env python3
"""Unsloth 4-bit QLoRA training for Nemotron-3-Nano-30B-A3B.
Unsloth's FastLanguageModel handles the hybrid Mamba2+MoE+Attention
architecture correctly, including num_logits_to_keep=1.

Env vars: MODEL_PATH, DATA_PATH, OUT_PATH
"""

import os, sys, json, time, torch
from datasets import load_dataset

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

torch.backends.cuda.matmul.allow_tf32 = True

MODEL_PATH = os.environ["MODEL_PATH"]
DATA_PATH = os.environ["DATA_PATH"]
OUT_PATH = os.environ["OUT_PATH"]

t0 = time.time()

# ── GPU info ──────────────────────────────────────────────────────
gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
ngpu = torch.cuda.device_count()
print(f"GPU: {gpu} ({vram:.1f} GB) x{ngpu}")
print(f"PyTorch: {torch.__version__}")

# ── Unsloth ───────────────────────────────────────────────────────
from unsloth import FastLanguageModel

print(f"Model: {MODEL_PATH}")
print("Loading model (Unsloth 4-bit)...")

model, tok = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    load_in_4bit=True,
    trust_remote_code=True,
)

# ── LoRA ──────────────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj", "out_proj",
        "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
)
model.print_trainable_parameters()

# ── Data ──────────────────────────────────────────────────────────
print(f"Data: {DATA_PATH}")
ds = load_dataset("json", data_files={"train": DATA_PATH}, split="train")
print(f"  {len(ds)} examples loaded")


def fmt(ex):
    """Convert prompt/answer to chat-template text."""
    msgs = ex.get("messages")
    if not msgs:
        msgs = [
            {"role": "user", "content": ex.get("prompt", "?")},
            {"role": "assistant", "content": ex.get("answer", "?")},
        ]
    return {
        "text": tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
    }


ds = ds.map(fmt)
print(f"  {len(ds)} texts formatted")

# ── Train ─────────────────────────────────────────────────────────
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir=OUT_PATH,
        max_steps=500,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        max_seq_length=2048,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
        packing=True,
    ),
    train_dataset=ds,
    processing_class=tok,
)

print("TRAINING with Unsloth 4-bit QLoRA...")
trainer.train()

elapsed = (time.time() - t0) / 60
print(f"DONE in {elapsed:.1f} minutes")

# ── Save ──────────────────────────────────────────────────────────
os.makedirs(OUT_PATH, exist_ok=True)
trainer.save_model(OUT_PATH)
tok.save_pretrained(OUT_PATH)
print(f"Adapter saved to {OUT_PATH}")

# Quick manifest
import json as _json
manifest = {
    "model": MODEL_PATH,
    "data": DATA_PATH,
    "num_examples": len(ds),
    "max_steps": 500,
    "batch_size": 1,
    "grad_accum": 8,
    "lr": 2e-4,
    "max_seq_len": 2048,
    "elapsed_min": elapsed,
    "gpu": gpu,
    "vram_gb": vram,
    "framework": "unsloth-4bit-qlora",
}
with open(os.path.join(OUT_PATH, "manifest.json"), "w") as f:
    _json.dump(manifest, f, indent=2)
