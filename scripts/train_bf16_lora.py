#!/usr/bin/env python3
"""
BF16 LoRA fine-tuning of Nemotron-3-Nano-30B-A3B for Kaggle reasoning puzzles.

No quantization — loads model in pure bfloat16. The MoE architecture has
~30B total params but only ~3B active, so BF16 + rank-32 LoRA fits within
64 GB MI210 VRAM without bitsandbytes.

Usage:
    python scripts/train_bf16_lora.py --dataset data/kaggle_dataset.jsonl --output checkpoints/

Environment variables (optional):
    NEMOTRON_BASE_MODEL: HuggingFace model ID (default: nvidia/Nemotron-3-Nano-30B-A3B-BF16)
    NEMOTRON_LORA_RANK: LoRA rank (default: 32)
    NEMOTRON_LORA_ALPHA: LoRA alpha (default: 64)
    NEMOTRON_EPOCHS: Number of epochs (default: 3)
    NEMOTRON_BATCH_SIZE: Micro-batch size per GPU (default: 2)
    NEMOTRON_GRAD_ACCUM: Gradient accumulation steps (default: 8)
    NEMOTRON_LEARNING_RATE: Learning rate (default: 2e-4)
    NEMOTRON_MAX_LENGTH: Max sequence length (default: 2048)
    NEMOTRON_SAVE_STEPS: Checkpoint interval (default: 200)
    NEMOTRON_WANDB_PROJECT: W&B project name (default: nemotron-kaggle-reasoner)
"""

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Globals for SIGUSR1 checkpoint ──────────────────────────────────────────
_trainer = None
_save_path = None


def _sigusr1_handler(signum, frame):
    """SIGUSR1: save checkpoint and exit with code 42 for Slurm re-submit."""
    global _trainer, _save_path
    logger.warning("SIGUSR1 received — saving checkpoint before walltime expiry")

    if _trainer is not None and _save_path is not None:
        ckpt_dir = os.path.join(_save_path, "sigusr1_checkpoint")
        logger.info(f"Saving checkpoint to {ckpt_dir}")
        _trainer.save_state()
        _trainer.save_model(ckpt_dir)
        logger.info("Checkpoint saved. Exiting with code 42 for Slurm requeue.")
        sys.exit(42)


signal.signal(signal.SIGUSR1, _sigusr1_handler)


def parse_args():
    p = argparse.ArgumentParser(description="BF16 LoRA train Nemotron-3-Nano for Kaggle")
    p.add_argument("--dataset", required=True, help="JSONL dataset path")
    p.add_argument("--output", required=True, help="Output directory for checkpoints")
    p.add_argument("--resume", default=None, help="Resume from checkpoint directory")
    return p.parse_args()


def get_target_modules(model):
    """Find all linear layers suitable for LoRA."""
    target_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            target_modules.add(name.split(".")[-1])
    candidates = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    found = list(candidates & target_modules) or list(target_modules)
    return found


def _format_messages(examples):
    """Format ShareGPT messages into text for SFTTrainer.

    Uses the Nemotron chat template if available, otherwise falls back
    to a simple format.
    """
    texts = []
    for messages in examples["messages"]:
        # Try tokenizer chat template first (set up in main)
        try:
            text = _tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        except Exception:
            # Fallback: simple format
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            texts.append("\n".join(parts))
    return {"text": texts}


# Module-level placeholder, set in main()
_tokenizer = None


def main():
    global _trainer, _save_path, _tokenizer

    args = parse_args()

    model_id = os.environ.get("NEMOTRON_BASE_MODEL", "nvidia/Nemotron-3-Nano-30B-A3B-BF16")
    lora_rank = int(os.environ.get("NEMOTRON_LORA_RANK", "32"))
    lora_alpha = int(os.environ.get("NEMOTRON_LORA_ALPHA", "64"))
    epochs = int(os.environ.get("NEMOTRON_EPOCHS", "3"))
    micro_batch = int(os.environ.get("NEMOTRON_BATCH_SIZE", "2"))
    grad_accum = int(os.environ.get("NEMOTRON_GRAD_ACCUM", "8"))
    lr = float(os.environ.get("NEMOTRON_LEARNING_RATE", "2e-4"))
    max_length = int(os.environ.get("NEMOTRON_MAX_LENGTH", "2048"))
    save_steps = int(os.environ.get("NEMOTRON_SAVE_STEPS", "200"))
    wandb_project = os.environ.get("NEMOTRON_WANDB_PROJECT", "nemotron-kaggle-reasoner")

    _save_path = args.output
    Path(args.output).mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset: {args.dataset}")
    dataset = Dataset.from_json(args.dataset)

    if len(dataset) > 5000:
        split = dataset.train_test_split(test_size=0.05, seed=42)
        train_ds = split["train"]
        eval_ds = split["test"]
    else:
        train_ds = dataset
        eval_ds = None

    logger.info(f"Train: {len(train_ds)} | Eval: {len(eval_ds) if eval_ds else 'none'}")

    # ── Load model in BF16 (no quantization) ────────────────────────────
    logger.info(f"Loading base model in BF16: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Make tokenizer available to the formatting function
    _tokenizer = tokenizer

    # ── Format dataset using chat template ──────────────────────────────
    logger.info("Formatting dataset with chat template...")
    train_ds = train_ds.map(
        _format_messages,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    if eval_ds is not None:
        eval_ds = eval_ds.map(
            _format_messages,
            batched=True,
            remove_columns=eval_ds.column_names,
        )

    # ── Find target modules ─────────────────────────────────────────────
    target_modules = get_target_modules(model)
    logger.info(f"LoRA target modules: {target_modules}")

    # ── LoRA config ─────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Training args ───────────────────────────────────────────────────
    effective_batch = micro_batch * grad_accum * torch.cuda.device_count()
    logger.info(
        f"Effective batch: {effective_batch} "
        f"(micro={micro_batch} x accum={grad_accum} x gpus={torch.cuda.device_count()})"
    )

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=epochs,
        per_device_train_batch_size=micro_batch,
        per_device_eval_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=save_steps if eval_ds else None,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name=wandb_project,
        remove_unused_columns=False,
    )

    # ── Trainer ─────────────────────────────────────────────────────────
    _trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_seq_length=max_length,
        dataset_text_field="text",
        packing=False,
    )

    # Resume if checkpoint exists
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        _trainer.train(resume_from_checkpoint=args.resume)
    else:
        _trainer.train()

    # ── Save final adapter ──────────────────────────────────────────────
    final_path = os.path.join(args.output, "final")
    logger.info(f"Saving final adapter to {final_path}")
    _trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
