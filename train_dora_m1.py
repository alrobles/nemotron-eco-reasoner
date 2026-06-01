#!/usr/bin/env python3
"""M1: DoRA fine-tuning of Nemotron-3-Nano on Kaggle puzzles.
Runs inside Apptainer container with pre-installed PyTorch ROCm.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--use-dora", action="store_true", default=True)
    parser.add_argument("--max-examples", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    # --- GPU check ---
    print(f"PyTorch {torch.__version__}")
    print(f"ROCm available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"VRAM: {gb:.1f} GB")

    # --- Load dataset ---
    data_file = os.path.join(args.data_dir, "kaggle_5k_train.jsonl")
    if not os.path.exists(data_file):
        data_file = os.path.join(args.data_dir, "kaggle_5k_train.jsonl")
    print(f"Loading: {data_file}")

    examples = []
    with open(data_file) as f:
        for line in f:
            examples.append(json.loads(line))

    if args.max_examples > 0:
        examples = examples[:args.max_examples]

    print(f"Examples: {len(examples)}")

    # Format as instruction-following
    texts = []
    for ex in examples:
        instruction = ex.get("instruction", ex.get("prompt", ""))
        response = ex.get("response", ex.get("answer", ""))
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        texts.append({"text": text})

    dataset = Dataset.from_list(texts)

    # --- Load model & tokenizer ---
    print(f"Loading model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # --- DoRA config ---
    # Target all linear layers in both Mamba and Transformer blocks
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",       # attention
        "gate_proj", "up_proj", "down_proj",            # MLP
        "in_proj", "out_proj",                          # Mamba
    ]

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_dora=args.use_dora,
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"DoRA rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # --- Tokenize ---
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Labels = input_ids for causal LM
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    tokenized = tokenized.map(add_labels)
    print(f"Tokenized: {len(tokenized)} examples, max_length={args.max_length}")

    # --- Training args ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    # --- Train ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )

    print(f"\n=== Starting training ===")
    print(f"Steps: ~{len(tokenized)//(args.batch_size*args.gradient_accumulation_steps)}")
    trainer.train()

    # --- Save ---
    adapter_path = os.path.join(args.output_dir, "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nAdapter saved to {adapter_path}")
    print("=== Training complete ===")


if __name__ == "__main__":
    main()
