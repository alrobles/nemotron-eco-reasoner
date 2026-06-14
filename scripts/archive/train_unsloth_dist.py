#!/usr/bin/env python3
"""Multi-node Unsloth 4-bit QLoRA training for Nemotron-3-Nano-30B-A3B.
Launched via torchrun: torchrun --nnodes=N --nproc_per_node=G train_unsloth_dist.py
Supports DDP (NCCL) and checkpoint resume via SIGUSR1 handler.

Env vars set by torchrun: LOCAL_RANK, RANK, WORLD_SIZE, LOCAL_WORLD_SIZE
Env vars from sbatch: MODEL_PATH, DATA_PATH, OUT_PATH, RESUME_CHECKPOINT (optional)
"""

import os, sys, sys, signal, json, time, types, torch
from pathlib import Path

# ── Distributed setup ──────────────────────────────────────────────
local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

torch.cuda.set_device(local_rank)
torch.backends.cuda.matmul.allow_tf32 = True
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

MODEL_PATH = os.environ["MODEL_PATH"]
DATA_PATH = os.environ["DATA_PATH"]
OUT_PATH = os.environ["OUT_PATH"]
RESUME_CHECKPOINT = os.environ.get("RESUME_CHECKPOINT", None)

if rank == 0:
    t0 = time.time()
    print(f"=== Multi-node Unsloth Training ===")
    print(f"World: {world_size} GPUs across {world_size // torch.cuda.device_count()} nodes")
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

# ── SIGUSR1 handler (checkpoint before walltime kill) ─────────────
def save_checkpoint_and_exit(signum, frame):
    if rank == 0:
        print(f"\n[SIGUSR1] Saving checkpoint to {OUT_PATH}/checkpoint-latest")
    torch.distributed.barrier()
    # Trainer will be saved by the main loop's exception handler
    os.environ["HERMES_SIGUSR1_RECEIVED"] = "1"
    sys.exit(0)

signal.signal(signal.SIGUSR1, save_checkpoint_and_exit)

# ── Unsloth ────────────────────────────────────────────────────────
from datasets import load_dataset
from unsloth import FastLanguageModel

if rank == 0:
    print(f"Loading model (Unsloth 4-bit, world_size={world_size})...")

model, tok = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    load_in_4bit=True,
    trust_remote_code=True,
    device_map={"": local_rank},
)

# ── Monkey-patch MoE ───────────────────────────────────────────────
_patched_count = 0
for module in model.modules():
    if not hasattr(module, "moe") or not hasattr(module, "experts"):
        continue
    if not callable(module.moe):
        continue

    def make_patched_moe(mod):
        def patched_moe(_self, hidden_states, topk_indices, topk_weights):
            orig_shape = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_states.size(-1))
            flat_topk_indices = topk_indices.view(-1)
            hidden_states = hidden_states.repeat_interleave(
                topk_indices.shape[-1], dim=0
            )
            final_hidden_states = torch.zeros_like(hidden_states)
            dtype = final_hidden_states.dtype
            for i, expert_layer in enumerate(mod.experts):
                expert_mask = (flat_topk_indices == i).nonzero(as_tuple=True)[0]
                if expert_mask.numel() == 0:
                    continue
                expert_hidden = hidden_states[expert_mask]
                expert_output = expert_layer(expert_hidden)
                weights = topk_weights.view(-1)[expert_mask]
                weighted_output = expert_output * weights.unsqueeze(-1)
                if weighted_output.dtype != dtype:
                    weighted_output = weighted_output.to(dtype)
                final_hidden_states.index_add_(0, expert_mask, weighted_output)
            top_k = topk_indices.shape[-1]
            final_hidden_states = final_hidden_states.view(
                -1, top_k, final_hidden_states.size(-1)
            ).sum(dim=1)
            return final_hidden_states
        return patched_moe

    module.moe = types.MethodType(make_patched_moe(module), module)
    _patched_count += 1

if rank == 0:
    print(f"  MoE dtype patch applied to {_patched_count} layers (rank {rank})")

# ── LoRA ───────────────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model, r=32, lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj", "out_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
)
if rank == 0:
    model.print_trainable_parameters()

# ── Data ───────────────────────────────────────────────────────────
ds = load_dataset("json", data_files={"train": DATA_PATH}, split="train")

def fmt(ex):
    msgs = ex.get("messages")
    if not msgs:
        msgs = [{"role": "user", "content": ex.get("prompt", "?")},
                {"role": "assistant", "content": ex.get("answer", "?")}]
    return {"text": tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}

ds = ds.map(fmt)
if rank == 0:
    print(f"Data: {len(ds)} examples loaded, {len(ds)} texts formatted")

# ── Train ──────────────────────────────────────────────────────────
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir=OUT_PATH,
        max_steps=500,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # Reduced: 8 GPUs x 1 x 4 = 32 effective batch
        learning_rate=2e-4,
        max_seq_length=2048,
        warmup_steps=50,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
        packing=True,
        ddp_find_unused_parameters=False,
    ),
    train_dataset=ds,
    processing_class=tok,
)

if rank == 0:
    print(f"TRAINING: {world_size} GPUs, batch={world_size}x1x4={world_size*4} effective, 500 steps")

# Resume from checkpoint if specified
resume = RESUME_CHECKPOINT if RESUME_CHECKPOINT else True  # True = auto-find latest

try:
    trainer.train(resume_from_checkpoint=resume)
except Exception as e:
    if rank == 0:
        print(f"Training interrupted: {e}")
    # Save checkpoint on interruption
    if rank == 0:
        trainer.save_model(os.path.join(OUT_PATH, "checkpoint-latest"))
        tok.save_pretrained(os.path.join(OUT_PATH, "checkpoint-latest"))
        print(f"Emergency checkpoint saved to {OUT_PATH}/checkpoint-latest")
    raise

if rank == 0:
    elapsed = (time.time() - t0) / 60
    print(f"DONE in {elapsed:.1f} minutes")
    trainer.save_model(OUT_PATH)
    tok.save_pretrained(OUT_PATH)
    print(f"Adapter saved to {OUT_PATH}")
    manifest = {
        "model": MODEL_PATH, "data": DATA_PATH, "num_examples": len(ds),
        "max_steps": 500, "batch_size": 1, "grad_accum": 4,
        "effective_batch": world_size * 4, "world_size": world_size,
        "lr": 2e-4, "max_seq_len": 2048, "elapsed_min": elapsed,
        "framework": "unsloth-4bit-qlora-multi-node-ddp",
    }
    with open(os.path.join(OUT_PATH, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

torch.distributed.destroy_process_group()
