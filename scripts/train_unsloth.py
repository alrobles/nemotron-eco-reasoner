#!/usr/bin/env python3
"""Unsloth 4-bit QLoRA training for Nemotron-3-Nano-30B-A3B.
BREAKTHROUGH (v15, Jun 5 2026): First successful training after 40+ failed attempts.
Key fixes: Unsloth FastLanguageModel + monkey-patched MoE forward with
dtype-safe index_add_ and top-k expert output aggregation.
Patches applied to 23 decoder layers. 883M trainable params (2.72%)."""

import os, sys, json, time, types, torch
from datasets import load_dataset

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
torch.backends.cuda.matmul.allow_tf32 = True

MODEL_PATH = os.environ["MODEL_PATH"]
DATA_PATH = os.environ["DATA_PATH"]
OUT_PATH = os.environ["OUT_PATH"]
t0 = time.time()

gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu} ({vram:.1f} GB) x{torch.cuda.device_count()}")
print(f"PyTorch: {torch.__version__}")

# ── Unsloth ───────────────────────────────────────────────────────
from unsloth import FastLanguageModel

print(f"Model: {MODEL_PATH}")
print("Loading model (Unsloth 4-bit)...")
model, tok = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH, max_seq_length=2048, load_in_4bit=True, trust_remote_code=True,
)

# ── Monkey-patch .moe() METHOD on decoder layers ──────────────────
# Nemotron-H MoE is NOT a submodule — it's a method self.moe() on
# NemotronHDecoderLayer (line 852 of modeling_nemotron_h.py).
# Two fixes needed for quantized training:
# 1. Cast weighted_output to match final_hidden_states.dtype (bf16 vs fp32)
# 2. Aggregate top-k expert outputs back to [B*S, D] from [B*S*top_k, D]
#    (Nemotron routes to 6 experts per token)

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

            # Aggregate top-k: [B*S*top_k, D] -> [B*S, D]
            top_k = topk_indices.shape[-1]
            final_hidden_states = final_hidden_states.view(
                -1, top_k, final_hidden_states.size(-1)
            ).sum(dim=1)
            return final_hidden_states

        return patched_moe

    module.moe = types.MethodType(make_patched_moe(module), module)
    _patched_count += 1

print(f"  MoE dtype patch applied to {_patched_count} layers")

# ── LoRA ──────────────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model, r=32, lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj", "out_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
)
model.print_trainable_parameters()

# ── Data ──────────────────────────────────────────────────────────
print(f"Data: {DATA_PATH}")
ds = load_dataset("json", data_files={"train": DATA_PATH}, split="train")
print(f"  {len(ds)} examples loaded")

def fmt(ex):
    msgs = ex.get("messages")
    if not msgs:
        msgs = [
            {"role": "user", "content": ex.get("prompt", "?")},
            {"role": "assistant", "content": ex.get("answer", "?")},
        ]
    return {"text": tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}

ds = ds.map(fmt)
print(f"  {len(ds)} texts formatted")

# ── Train ─────────────────────────────────────────────────────────
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir=OUT_PATH, max_steps=500, per_device_train_batch_size=1,
        gradient_accumulation_steps=8, learning_rate=2e-4, max_seq_length=2048,
        warmup_steps=50, logging_steps=10, save_steps=100, save_total_limit=3,
        bf16=True, remove_unused_columns=False, report_to="none",
        dataloader_num_workers=0, packing=True,
    ),
    train_dataset=ds, processing_class=tok,
)

print("TRAINING with Unsloth 4-bit QLoRA...")
trainer.train()

elapsed = (time.time() - t0) / 60
print(f"DONE in {elapsed:.1f} minutes")
os.makedirs(OUT_PATH, exist_ok=True)
trainer.save_model(OUT_PATH)
tok.save_pretrained(OUT_PATH)
print(f"Adapter saved to {OUT_PATH}")

import json as _json
manifest = {
    "model": MODEL_PATH, "data": DATA_PATH, "num_examples": len(ds),
    "max_steps": 500, "batch_size": 1, "grad_accum": 8, "lr": 2e-4,
    "max_seq_len": 2048, "elapsed_min": elapsed, "gpu": gpu, "vram_gb": vram,
    "framework": "unsloth-4bit-qlora-moe-patched-v15",
}
with open(os.path.join(OUT_PATH, "manifest.json"), "w") as f:
    _json.dump(manifest, f, indent=2)
