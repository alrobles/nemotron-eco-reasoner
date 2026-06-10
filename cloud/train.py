#!/usr/bin/env python
"""Single-command QLoRA training for Nemotron-3-Nano-30B on cloud GPUs.

Portable version of hpc/nem_chained.slurm. Requires >=40GB VRAM
(L40S 48GB / RTX Pro 6000 / A100). First run cloud/setup_env.sh.

Usage (from repo root):
    python cloud/train.py                      # 300 steps (default)
    TARGET_TOTAL=500 python cloud/train.py     # full run
    nohup python cloud/train.py > train.log 2>&1 &   # background, tail -f train.log

Checkpoints save every 50 steps to OUT_PATH and the script resumes from the
latest one automatically, so interruptions (credits, session limits) lose
at most 50 steps.
"""
import unsloth  # noqa: F401  must be first import for its patches
import glob
import json
import os

import torch
from unsloth import FastLanguageModel

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ID = os.environ.get("MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
DATA_PATH = os.environ.get("DATA_PATH", os.path.join(REPO_ROOT, "data", "train_deterministic_v7.jsonl"))
OUT_PATH = os.environ.get("OUT_PATH", os.path.join(REPO_ROOT, "outputs", "deterministic_v7"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", 2048))
RANK = int(os.environ.get("RANK", 32))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", 8))
TARGET_TOTAL = int(os.environ.get("TARGET_TOTAL", 300))

os.makedirs(OUT_PATH, exist_ok=True)

gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
bf16_ok = torch.cuda.is_bf16_supported()
print(f"GPU: {gpu} ({vram:.1f}GB) bf16={bf16_ok}")
assert vram >= 38, f"Need >=40GB VRAM for 30B QLoRA; got {vram:.1f}GB"

print(f"Loading {MODEL_ID} in 4-bit (downloads ~60GB on first run)...")
model, tok = FastLanguageModel.from_pretrained(
    MODEL_ID, max_seq_length=SEQ_LEN, load_in_4bit=True, trust_remote_code=True)

# Dense per-expert MoE dispatch (stock fused path breaks under 4-bit + grad ckpt)
import types as _types
patched = 0
for module in model.modules():
    if not hasattr(module, "moe") or not hasattr(module, "experts"):
        continue
    if not callable(module.moe):
        continue
    def mp(mod):
        def pm(_self, h, ti, tw):
            h = h.view(-1, h.size(-1)); ft = ti.view(-1)
            h = h.repeat_interleave(ti.shape[-1], dim=0)
            fh = torch.zeros_like(h); dt = fh.dtype
            for i, el in enumerate(mod.experts):
                m = (ft == i).nonzero(as_tuple=True)[0]
                if m.numel() == 0:
                    continue
                eh = h[m]; eo = el(eh); w = tw.view(-1)[m]; wo = eo * w.unsqueeze(-1)
                if wo.dtype != dt:
                    wo = wo.to(dt)
                fh.index_add_(0, m, wo)
            tk = ti.shape[-1]
            return fh.view(-1, tk, fh.size(-1)).sum(dim=1)
        return pm
    module.moe = _types.MethodType(mp(module), module); patched += 1
print(f"MoE patched: {patched} layers")

model = FastLanguageModel.get_peft_model(
    model, r=RANK, lora_alpha=RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth")
model.print_trainable_parameters()

from datasets import load_dataset

ds = load_dataset("json", data_files={"train": DATA_PATH}, split="train")
def fmt(ex):
    ms = ex.get("messages")
    if not ms:
        ms = [{"role": "user", "content": ex.get("prompt", "?")},
              {"role": "assistant", "content": ex.get("answer", "?")}]
    return {"text": tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=False)}
ds = ds.map(fmt)
print(f"{len(ds)} training records")

ckpts = sorted(glob.glob(os.path.join(OUT_PATH, "checkpoint-*", "trainer_state.json")), key=os.path.getmtime)
resume = os.path.dirname(ckpts[-1]) if ckpts else None
step = json.load(open(ckpts[-1])).get("global_step", 0) if ckpts else 0
print(f"Resume: {resume or 'FRESH START'} (step {step})")

from trl import SFTConfig, SFTTrainer

seq_kw = {"max_length": SEQ_LEN} if "max_length" in SFTConfig.__dataclass_fields__ else {"max_seq_length": SEQ_LEN}
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir=OUT_PATH, max_steps=TARGET_TOTAL,
        per_device_train_batch_size=1, gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=1e-4, **seq_kw,
        warmup_steps=min(30, TARGET_TOTAL // 3), lr_scheduler_type="constant_with_warmup",
        logging_steps=10, save_steps=50, save_total_limit=10,
        bf16=bf16_ok, fp16=not bf16_ok,
        remove_unused_columns=False, report_to="none",
        dataloader_num_workers=0, packing=True),
    train_dataset=ds, processing_class=tok)

trainer.train(resume_from_checkpoint=resume)

final = os.path.join(OUT_PATH, "final")
trainer.save_model(final)
tok.save_pretrained(final)

import shutil
zip_path = shutil.make_archive(os.path.join(OUT_PATH, "submission"), "zip", final)
print(f"DONE. Adapter: {final}\nSubmission zip: {zip_path}")
