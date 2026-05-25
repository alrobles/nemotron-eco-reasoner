#!/usr/bin/env python3
"""
QLoRA fine-tuning of Nemotron-3-Nano-30B-A3B for dual reasoning + ecology tasks.

Usage:
    python scripts/train_qlora.py --dataset data/combined_dataset.jsonl --output checkpoints/

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
    NEMOTRON_WANDB_PROJECT: W&B project name (default: nemotron-eco-reasoner)
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
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Globals for SIGUSR1 checkpoint ──────────────────────────────────────────
_trainer = None
_save_path = None
_sigusr1_received = False


def _sigusr1_handler(signum, frame):
    """SIGUSR1: save checkpoint and exit with code 42 for Slurm re-submit."""
    global _sigusr1_received
    _sigusr1_received = True
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
    p = argparse.ArgumentParser(description="QLoRA train Nemotron-3-Nano dual-purpose")
    p.add_argument("--dataset", required=True, help="Combined JSONL dataset path")
    p.add_argument("--output", required=True, help="Output directory for checkpoints")
    p.add_argument("--resume", default=None, help="Resume from checkpoint directory")
    return p.parse_args()


def get_target_modules(model):
    """Find all linear layers suitable for LoRA."""
    target_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Get the last part of the name (e.g., 'q_proj', 'out_proj')
            target_modules.add(name.split(".")[-1])
    # Ensure we have key projection layers
    candidates = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    return list(candidates & target_modules) or list(target_modules)


def main():
    global _trainer, _save_path

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
    wandb_project = os.environ.get("NEMOTRON_WANDB_PROJECT", "nemotron-eco-reasoner")

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

    logger.info(f"Train examples: {len(train_ds)}, Eval: {len(eval_ds) if eval_ds else 'none'}")

    # ── Quantization config ─────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Determine compute dtype — fall back to float16 if bfloat16 not supported
    try:
        _ = torch.tensor([1.0], dtype=torch.bfloat16, device="cuda")
        compute_dtype = torch.bfloat16
    except Exception:
        logger.warning("bfloat16 not supported on this GPU — falling back to float16")
        compute_dtype = torch.float16
        bnb_config.bnb_4bit_compute_dtype = torch.float16

    logger.info(f"Loading base model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Prepare for k-bit training ──────────────────────────────────────
    model = prepare_model_for_kbit_training(model)

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
    logger.info(f"Effective batch size: {effective_batch} "
                f"(micro={micro_batch} × accum={grad_accum} × gpus={torch.cuda.device_count()})")

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
        bf16=(compute_dtype == torch.bfloat16),
        fp16=(compute_dtype == torch.float16),
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
