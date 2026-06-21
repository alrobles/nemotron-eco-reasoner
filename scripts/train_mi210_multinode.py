#!/usr/bin/env python3
"""
MI210 Multi-Node DDP Training — Nemotron 30B BF16 + LoRA
Runs under torchrun: 1 GPU per process, DDP across nodes.

Key: uses SFTTrainer which handles DDP natively when launched with torchrun.
     Full model per GPU (~60GB) + LoRA adapters + gradient checkpointing.
"""
import os, sys, json, time, glob, shutil

# ── Patch modeling_nemotron_h.py for ROCm (no mamba_ssm) ─
MPATH = os.path.join(os.environ["M"], "modeling_nemotron_h.py")
OLD = '''except ImportError:
    raise ImportError("mamba-ssm is required by the Mamba model but cannot be imported")'''
NEW = '''except ImportError:
    def rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5, group_size=None, norm_before_gate=False):
        import torch
        dt = x.dtype
        x = x.float()
        if z is not None and not norm_before_gate:
            x = x * torch.nn.functional.silu(z.float())
        if group_size is not None and group_size != x.shape[-1]:
            s = x.shape
            xg = x.view(*s[:-1], s[-1] // group_size, group_size)
            xg = xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + eps)
            x = xg.view(s)
        else:
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        out = x * weight.float()
        if bias is not None:
            out = out + bias.float()
        if z is not None and norm_before_gate:
            out = out * torch.nn.functional.silu(z.float())
        return out.to(dt)'''
src = open(MPATH).read()
if OLD in src:
    open(MPATH, "w").write(src.replace(OLD, NEW))
    print("patched modeling_nemotron_h.py with torch rmsnorm_fn fallback")
for d in glob.glob(os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/cbd3fa9f*")):
    shutil.rmtree(d, ignore_errors=True)

# ── Imports ───────────────────────────────────────────────
import torch
import torch.distributed as dist

# ── DDP init ──────────────────────────────────────────────
RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
IS_MAIN = (RANK == 0)

torch.cuda.set_device(LOCAL_RANK)
dist.init_process_group(backend="nccl")  # ROCm maps nccl → rccl

if IS_MAIN:
    print(f"DDP INIT: world={WORLD_SIZE} rank={RANK} local={LOCAL_RANK}")
    print(f"  GPU: {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    sys.stdout.flush()

# ── Config ────────────────────────────────────────────────
M = os.environ["M"]
DATA_PATH = os.environ["DATA"]
OUT = os.environ["OUT"]
JOBID = os.environ["JOBID"]
TARGET_TOTAL = int(os.environ["TARGET_TOTAL"])
STEPS_PER_JOB = int(os.environ["STEPS_PER_JOB"])
SEQ_LEN = int(os.environ["SEQ_LEN"])
GRAD_ACCUM = int(os.environ["GRAD_ACCUM"])
LR = float(os.environ["LR"])

# ── Load tokenizer ────────────────────────────────────────
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# ── Load model (full model per GPU, BF16) ─────────────────
from transformers import AutoModelForCausalLM

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    M,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map={"": f"cuda:{LOCAL_RANK}"},
    max_memory={LOCAL_RANK: "62GiB"},  # 62GB to leave headroom on 68.7GB MI210
)
if IS_MAIN:
    print(f"MODEL LOADED in {time.time()-t0:.0f}s rank={RANK}")
    sys.stdout.flush()

# ── LoRA config ───────────────────────────────────────────
from peft import LoraConfig

# Find linear layers to target
target = sorted({
    n.split(".")[-1]
    for n, m in model.named_modules()
    if isinstance(m, torch.nn.Linear)
} & {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj", "in_proj", "out_proj",
})

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=target,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

if IS_MAIN:
    print(f"LoRA targets: {target}")
    sys.stdout.flush()

# ── Resume from checkpoint ────────────────────────────────
ckpts = sorted(
    glob.glob(os.path.join(OUT, "checkpoint-*", "trainer_state.json")),
    key=os.path.getmtime,
)
RESUME = os.path.dirname(ckpts[-1]) if ckpts else None
global_step = json.load(open(ckpts[-1])).get("global_step", 0) if ckpts else 0
if IS_MAIN:
    print(f"RESUME: {RESUME} (step {global_step})" if RESUME else "FRESH START")
    sys.stdout.flush()

# ── Dataset ───────────────────────────────────────────────
from datasets import Dataset
ds = Dataset.from_json(DATA_PATH)

def fmt(ex):
    ms = ex.get("messages") or [
        {"role": "user", "content": ex.get("prompt", "?")},
        {"role": "assistant", "content": ex.get("answer", "?")},
    ]
    return {
        "text": tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=False)
    }

ds = ds.map(fmt, remove_columns=ds.column_names)
if IS_MAIN:
    print(f"{len(ds)} training records")
    sys.stdout.flush()

# ── Training ──────────────────────────────────────────────
batch_steps = min(STEPS_PER_JOB, TARGET_TOTAL - global_step)
target_step = global_step + batch_steps

from trl import SFTTrainer, SFTConfig

cfg = SFTConfig(
    output_dir=OUT,
    max_steps=target_step,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    logging_steps=10,
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_torch",
    max_seq_length=SEQ_LEN,
    dataset_text_field="text",
    packing=False,
    warmup_steps=min(30, target_step // 3),
    lr_scheduler_type="constant_with_warmup",
    save_steps=50,
    save_total_limit=10,
    report_to="none",
    ddp_find_unused_parameters=True,   # Needed for LoRA + DDP
    ddp_backend="nccl",
    dataloader_num_workers=1,
)

trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=ds,
    processing_class=tok,
    peft_config=peft_config,
)

if IS_MAIN:
    print(f"TRAINING [MI210-MN]: step {global_step} -> {target_step} (target={TARGET_TOTAL})")
    print(f"  world_size={WORLD_SIZE} effective_batch={WORLD_SIZE * GRAD_ACCUM}")
    sys.stdout.flush()

t0 = time.time()
trainer.train(resume_from_checkpoint=RESUME)
elapsed = (time.time() - t0) / 60

final_loss = next(
    (e["loss"] for e in reversed(trainer.state.log_history) if "loss" in e),
    "?",
)

if IS_MAIN:
    print(f"BATCH DONE {elapsed:.1f}min steps={global_step}->{target_step} loss={final_loss}")
    if target_step < TARGET_TOTAL:
        print(f"NEED_RESUBMIT: target_step={target_step} target_total={TARGET_TOTAL}")
    else:
        print(f"TARGET_REACHED: {target_step}/{TARGET_TOTAL} steps")
    print("SUCCESS")
    sys.stdout.flush()

dist.destroy_process_group()
