#!/usr/bin/env python3
"""
Nemotron-3-Nano DoRA training for AMD MI210 (ROCm).
Zero Unsloth — standard PEFT DoRA with BF16, works on ROCm.

Key features:
- DoRA (Weight-Decomposed LoRA) — better than LoRA for reasoning tasks
- Messages-format CoT data (chat template)
- Checkpoint chain via SIGUSR1 handler
- FSDP full_shard for multi-GPU (optional)
- BF16 native on MI210 (68.7GB VRAM → seq=2048, rank=64 easily)

Usage:
  # Single GPU
  python scripts/train_mi210_dora.py \
    --model /path/to/nemotron-model \
    --data data/train_cot_unified.jsonl \
    --output outputs/mi210_dora_run1

  # Multi-GPU (2+ GPUs, auto-detected via CUDA_VISIBLE_DEVICES)
  python scripts/train_mi210_dora.py ... --fsdp 1

  # Resume from checkpoint
  python scripts/train_mi210_dora.py ... --resume outputs/mi210_dora_run1/checkpoint-200

Env vars (override defaults):
  LORA_RANK=64 LORA_ALPHA=128 MAX_STEPS=1000 LR=1e-4 SEQ_LEN=2048
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

# ── Globals for SIGUSR1 checkpoint ─────────────────────────────────
_trainer = None
_save_path = None

def _sigusr1_handler(signum, frame):
    print("\n[SIGUSR1] Saving emergency checkpoint...")
    if _trainer is not None and _save_path is not None:
        ckpt_dir = os.path.join(_save_path, "sigusr1_checkpoint")
        _trainer.save_model(ckpt_dir)
        _trainer.save_state()
        print(f"[SIGUSR1] Checkpoint saved to {ckpt_dir}")
    print("[SIGUSR1] Exiting with code 42 (Slurm requeue signal)")
    sys.exit(42)

signal.signal(signal.SIGUSR1, _sigusr1_handler)


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    rank = int(os.environ.get("LOCAL_RANK", 0))
    prefix = f"[{ts}]" if rank == 0 else f"[{ts}][R{rank}]"
    print(f"{prefix} {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="MI210 DoRA training for Nemotron-3-Nano")
    p.add_argument("--model", required=True, help="Path to base model directory")
    p.add_argument("--data", required=True, help="Path to training JSONL (messages format)")
    p.add_argument("--output", required=True, help="Output directory for checkpoints")
    p.add_argument("--resume", default=None, help="Resume from checkpoint dir")
    p.add_argument("--max_steps", type=int, default=0, help="Max training steps (0=use epochs)")
    p.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    p.add_argument("--batch_size", type=int, default=2, help="Per-GPU micro batch size")
    p.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=0, help="Learning rate (0=use env/default)")
    p.add_argument("--seq_len", type=int, default=0, help="Max sequence length (0=use env/default)")
    p.add_argument("--use_dora", type=int, default=1, help="1=DoRA, 0=LoRA")
    p.add_argument("--fsdp", type=int, default=0, help="Enable FSDP full_shard for multi-GPU")
    p.add_argument("--packing", type=int, default=0, help="Enable sequence packing")
    return p.parse_args()


def get_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main():
    return get_rank() == 0


def main():
    global _trainer, _save_path

    args = parse_args()

    # ── Config from env vars with defaults ─────────────────────────
    lora_rank = int(os.environ.get("LORA_RANK", "64"))
    lora_alpha = int(os.environ.get("LORA_ALPHA", "128"))
    max_steps = args.max_steps or int(os.environ.get("MAX_STEPS", "0"))
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    grad_accum = args.grad_accum
    lr = args.lr or float(os.environ.get("LR", "1e-4"))
    seq_len = args.seq_len or int(os.environ.get("SEQ_LEN", "2048"))
    use_dora = bool(args.use_dora)
    use_fsdp = bool(args.fsdp) or os.environ.get("ACCELERATE_USE_FSDP") == "true"
    use_packing = bool(args.packing) or int(os.environ.get("PACKING", "0"))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", "50"))
    save_steps = int(os.environ.get("SAVE_STEPS", "100"))
    logging_steps = int(os.environ.get("LOGGING_STEPS", "10"))

    _save_path = args.output
    os.makedirs(args.output, exist_ok=True)

    # ── GPU check ──────────────────────────────────────────────────
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("No GPUs available!")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    bf16_ok = torch.cuda.is_bf16_supported()

    log("=" * 60)
    log(f"MI210 DoRA TRAINING — Nemotron-3-Nano-30B-A3B")
    log("=" * 60)
    log(f"GPU: {gpu_name} × {gpu_count} ({vram_gb:.1f} GB each)")
    log(f"BF16 supported: {bf16_ok}")
    log(f"DoRA: {use_dora} | FSDP: {use_fsdp} | Packing: {use_packing}")
    log(f"Seq len: {seq_len} | Rank: {lora_rank} | Alpha: {lora_alpha}")
    log(f"Batch: {batch_size} × {grad_accum} × {gpu_count} = {batch_size * grad_accum * gpu_count} effective")
    log(f"LR: {lr} | Warmup: {warmup_steps} | Epochs: {num_epochs}")
    if max_steps > 0:
        log(f"Max steps: {max_steps}")
    log(f"Data: {args.data}")
    log(f"Output: {args.output}")

    # ── Load dataset ───────────────────────────────────────────────
    log("Loading dataset...")
    ds = load_dataset("json", data_files={"train": args.data}, split="train")
    log(f"  {len(ds)} examples loaded")

    def format_chat(ex):
        """Convert messages to chat template text."""
        msgs = ex.get("messages")
        if msgs:
            # Ensure we have system+user+assistant
            return {"text": tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )}
        # Fallback: prompt/answer format
        prompt = ex.get("prompt", "")
        answer = ex.get("answer", "")
        return {"text": f"<|user|>\n{prompt}\n<|assistant|>\n{answer}<|endoftext|>"}

    log(f"Seq length: {seq_len}")

    # ── Load tokenizer ─────────────────────────────────────────────
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        # Add Nemotron-style chat template
        tokenizer.chat_template = "{{ bos_token }}{% for msg in messages %}{{ '<|' + msg['role'] + '|>\n' + msg['content'] + '<|endoftext|>\n' }}{% endfor %}"

    # Format dataset
    ds = ds.map(format_chat, desc="Formatting chat")

    # Split for eval
    if len(ds) > 5000:
        split = ds.train_test_split(test_size=0.05, seed=42)
        train_ds = split["train"]
        eval_ds = split["test"]
        log(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")
    else:
        train_ds = ds
        eval_ds = None
        log(f"Train: {len(train_ds)} (no eval split)")

    # ── Load model ─────────────────────────────────────────────────
    log("Loading Nemotron-3-Nano-30B-A3B in BF16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if bf16_ok else torch.float16,
        trust_remote_code=True,
        # No device_map with FSDP — it handles sharding itself
        device_map=None if use_fsdp else "auto",
    )

    # ── DoRA/LoRA config ───────────────────────────────────────────
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    log(f"Target modules: {target_modules}")

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_dora=use_dora,
    )
    model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"Trainable: {trainable/1e6:.1f}M / {total/1e9:.2f}B ({100*trainable/total:.2f}%)")

    # ── Training args ──────────────────────────────────────────────
    fsdp_kwargs = {}
    if use_fsdp:
        fsdp_kwargs = {
            "fsdp": "full_shard",
            "fsdp_config": {
                "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                "fsdp_transformer_layer_cls_to_wrap": None,
                "fsdp_state_dict_type": "SHARDED_STATE_DICT",
                "fsdp_forward_prefetch": True,
                "fsdp_use_orig_params": False,
                "fsdp_sync_module_states": True,
                "fsdp_cpu_ram_efficient_loading": True,
            },
        }
        log("FSDP enabled: full_shard auto-wrapping")

    # Determine precision
    if bf16_ok:
        bf16, fp16 = True, False
        log("Using BF16 precision")
    else:
        bf16, fp16 = False, True
        log("Using FP16 precision (BF16 not supported)")

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=num_epochs if max_steps == 0 else 100,
        max_steps=max_steps if max_steps > 0 else -1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=4,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=save_steps if eval_ds else None,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        load_best_model_at_end=False,
        **fsdp_kwargs,
    )

    # ── Trainer ────────────────────────────────────────────────────
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        max_seq_length=seq_len,
        dataset_text_field="text",
        packing=use_packing,
    )
    if eval_ds:
        trainer_kwargs["eval_dataset"] = eval_ds

    _trainer = SFTTrainer(**trainer_kwargs)

    # ── Resume if requested ────────────────────────────────────────
    if args.resume:
        log(f"Resuming from checkpoint: {args.resume}")
        _trainer.train(resume_from_checkpoint=args.resume)
    else:
        # Auto-detect latest checkpoint in output dir
        import glob
        ckpts = sorted(
            glob.glob(os.path.join(args.output, "checkpoint-*")),
            key=os.path.getmtime
        )
        if ckpts:
            latest = ckpts[-1]
            log(f"Resuming from latest checkpoint: {latest}")
            _trainer.train(resume_from_checkpoint=latest)
        else:
            _trainer.train()

    # ── Save final ─────────────────────────────────────────────────
    if is_main():
        final_path = os.path.join(args.output, "final")
        log(f"Saving final adapter to {final_path}")
        _trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)

        # Save manifest
        manifest = {
            "model": args.model,
            "data": args.data,
            "num_examples": len(ds),
            "seq_len": seq_len,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "use_dora": use_dora,
            "use_fsdp": use_fsdp,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "lr": lr,
            "num_epochs": num_epochs,
            "max_steps": max_steps,
            "gpu_count": gpu_count,
            "gpu_name": gpu_name,
            "vram_gb": vram_gb,
            "trainable_params": trainable,
            "total_params": total,
        }
        with open(os.path.join(final_path, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        log("DONE!")

    # Clean up distributed
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
