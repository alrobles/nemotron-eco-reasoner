#!/usr/bin/env python3
"""Train Nemotron-3-Nano with DoRA using pre-built venv tarball.
No pip, no Lustre, no dependency resolution. Just extract and train.

Usage:
  python3 train_with_venv.py \
    --venv-tarball /path/to/nemotron-venv-rocm61.tar.gz \
    --model-path /path/to/nemotron-model \
    --dataset /path/to/kaggle_5k_train.jsonl \
    --output-dir /path/to/outputs
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time


def run(cmd, **kwargs):
    """Run a command and stream output."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, **kwargs)
    if result.returncode != 0:
        print(f"  ERROR: exit code {result.returncode}")
        sys.exit(result.returncode)


def verify_gpu():
    """Check GPU is available."""
    import torch
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        print("ERROR: No GPU detected")
        sys.exit(1)
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        mem_total = torch.cuda.get_device_properties(i).total_mem / 1e9
        print(f"  GPU {i}: {name} ({mem_total:.1f} GB)")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  ROCm available: {torch.cuda.is_available()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv-tarball", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    args = parser.parse_args()

    print("=" * 60)
    print("NEMOTRON DoRA TRAINING (venv tarball mode)")
    print("=" * 60)

    # Step 1: Verify inputs
    print("\n[1/5] Verifying inputs...")
    for path, name in [
        (args.venv_tarball, "venv tarball"),
        (args.model_path, "model"),
        (args.dataset, "dataset"),
    ]:
        if not os.path.exists(path):
            print(f"  ERROR: {name} not found at {path}")
            sys.exit(1)
    print(f"  venv tarball: {args.venv_tarball} ({os.path.getsize(args.venv_tarball)/1e9:.1f} GB)")
    print(f"  model: {args.model_path}")
    print(f"  dataset: {args.dataset}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"  output: {args.output_dir}")

    # Step 2: Extract venv to /tmp (SSD)
    print("\n[2/5] Extracting venv to /tmp...")
    t0 = time.time()
    venv_dir = os.path.join(tempfile.gettempdir(), "nemotron-venv")
    if not os.path.exists(venv_dir):
        os.makedirs(os.path.dirname(venv_dir), exist_ok=True)
        with tarfile.open(args.venv_tarball, "r:gz") as tar:
            tar.extractall(path=os.path.dirname(venv_dir))
    print(f"  Extracted to {venv_dir} in {time.time()-t0:.0f}s")

    # Step 3: Verify venv
    print("\n[3/5] Verifying venv...")
    python_bin = os.path.join(venv_dir, "nemotron-venv", "bin", "python3")
    verify_cmd = f"{python_bin} -c 'import torch; print(f\"PyTorch {{torch.__version__}}\"); import transformers; print(f\"Transformers {{transformers.__version__}}\"); import peft; print(f\"PEFT {{peft.__version__}}\")'"
    run(verify_cmd)

    # Step 4: Check GPU
    print("\n[4/5] Checking GPU...")
    gpu_cmd = f"{python_bin} -c 'import torch; [print(f\"GPU {{i}}: {{torch.cuda.get_device_name(i)}} ({{torch.cuda.get_device_properties(i).total_mem/1e9:.1f}} GB)\") for i in range(torch.cuda.device_count())]; print(f\"ROCm: {{torch.cuda.is_available()}}\")'"
    run(gpu_cmd)

    # Step 5: Train
    print("\n[5/5] Starting training...")
    print(f"  Config: DoRA rank={args.lora_rank}, LR={args.learning_rate}, BS={args.batch_size}x{args.gradient_accumulation}, max_steps={args.max_steps}")

    t0 = time.time()

    train_script = f'''
import json
import os
import time

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

MODEL_PATH = {repr(args.model_path)}
DATASET_PATH = {repr(args.dataset)}
OUTPUT_DIR = {repr(args.output_dir)}

LORA_RANK = {args.lora_rank}
LEARNING_RATE = {args.learning_rate}
BATCH_SIZE = {args.batch_size}
GRADIENT_ACCUMULATION = {args.gradient_accumulation}
MAX_STEPS = {args.max_steps}

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

print("Configuring DoRA...")
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "in_proj", "out_proj",
    ],
    lora_dropout=0.0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    use_dora=True,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Loading dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

num_examples = len(dataset)
tokens_est = num_examples * 512
print(f"  Dataset: {{num_examples}} examples, ~{{tokens_est}} tokens")
total_batch = BATCH_SIZE * GRADIENT_ACCUMULATION
steps_per_epoch = max(1, num_examples // total_batch)
print(f"  Steps per epoch (est): {{steps_per_epoch}}")
print(f"  epochs (est): {{MAX_STEPS / steps_per_epoch:.1f}}")

def formatting_func(example):
    # Return plain text, let SFTTrainer handle tokenization and EOS
    return example["text"]

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    max_steps=MAX_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    bf16=True,
    optim="adamw_torch",
    gradient_checkpointing=True,
    remove_unused_columns=False,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    formatting_func=formatting_func,
    max_seq_length=1024,
)

print("Training...")
trainer.train()

# Save final adapter
final_path = os.path.join(OUTPUT_DIR, "final_adapter")
trainer.model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
print(f"Final adapter saved to {{final_path}}")

elapsed = time.time() - {t0}
print(f"Training completed in {{elapsed:.0f}}s ({{elapsed/3600:.1f}}h)")
'''

    train_file = os.path.join(args.output_dir, "train_script.py")
    with open(train_file, "w") as f:
        f.write(train_script)

    run(f"{python_bin} {train_file}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE in {elapsed/3600:.1f}h")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
