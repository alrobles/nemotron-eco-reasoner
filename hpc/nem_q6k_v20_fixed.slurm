#!/bin/bash
# === nem_q6k_v20_fixed.slurm — Q6000 LoRA training with HF offline protection ===
# Fixes:
#   1. HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1 — block all HF network calls
#   2. local_files_only=True in from_pretrained() — prevents Unsloth telemetry HF calls
#   3. Removed UNSLOTH_USE_MODELSCOPE (Nemotron not on ModelScope, red herring)
#   4. Removed torch reinstall (container has 2.5.1+cu124, good enough)
#   5. local_files_only in tok.save_pretrained()
#
# Usage:
#   SEQ_LEN=48 LORA_RANK=8 sbatch hpc/nem_q6k_v20_fixed.slurm
#   # Or use defaults: seq=48, rank=4

#SBATCH --job-name=nqs
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --gres=gpu:q6000:1
#SBATCH --mem=62G
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/a474r867/scratch/nem_q6k_v20_%j.out
#SBATCH --error=/home/a474r867/scratch/nem_q6k_v20_%j.err
#SBATCH --signal=B:SIGUSR1@120

# --- Environment: BLOCK ALL HuggingFace network calls ---
export APPTAINER_WRITABLE_TMPFS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_DATASETS_OFFLINE=1

# --- Paths ---
C=/home/a474r867/scratch/nemotron-eco-reasoner/nemotron-cuda.sif
M=/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848
D=/home/a474r867/scratch/nemotron-eco-reasoner/data/train_final_v2.jsonl
O=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/v20_shared
P=/tmp/pip_${SLURM_JOB_ID:-$$}

rm -rf "$P" 2>/dev/null
mkdir -p "$O" "$P"

SEQ=${SEQ_LEN:-48}
RANK=${LORA_RANK:-4}
STEPS=${STEPS_PER_JOB:-100}
JOBID=${SLURM_JOB_ID:-$$}

echo "=== v20 FIXED seq=$SEQ rank=$RANK steps=$STEPS job=$JOBID $(date) ==="
echo "  HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"

# --- Install deps (no torch reinstall — container has 2.5.1+cu124) ---
apptainer exec --nv --bind "/dev/shm,/tmp:$P" \
    --env PYTHONUSERBASE="$P" \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env HF_HUB_DISABLE_TELEMETRY=1 --env HF_DATASETS_OFFLINE=1 \
    "$C" bash << 'INSTALL_EOF'
set -euo pipefail
# Q6000 deps only — NO mamba-ssm/causal-conv1d (those pull CUDA deps, break on r22/r15)
pip install --user --no-cache-dir -q --no-deps "datasets==4.3.0" bitsandbytes 2>&1 | tail -1
pip install --user --no-cache-dir -q --no-deps unsloth_zoo unsloth 2>&1 | tail -1
pip uninstall -y torchao 2>/dev/null || true
python3 -c 'import torch; print(f"torch:{torch.__version__} path:{torch.__file__}")'
python3 -c 'import datasets,transformers; print(f"ds:{datasets.__version__} tf:{transformers.__version__}")'
python3 -c 'import unsloth; from trl import SFTTrainer; print("unsloth+trl OK")'
INSTALL_EOF

if [ $? -ne 0 ]; then echo "INSTALL FAILED"; exit 1; fi

# --- Training ---
echo "=== Training ==="
apptainer exec --nv --bind "/dev/shm,/tmp:$P" \
    --env PYTHONUSERBASE="$P" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env TORCH_COMPILE_DISABLE=1 --env OMP_NUM_THREADS=1 \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env HF_HUB_DISABLE_TELEMETRY=1 --env HF_DATASETS_OFFLINE=1 \
    --env MODEL_PATH="$M" --env DATA_PATH="$D" --env OUT_PATH="$O" \
    --env SEQ_LEN="$SEQ" --env LORA_RANK="$RANK" --env JOBID="$JOBID" \
    --env STEPS_PER_JOB="$STEPS" \
    "$C" env PYTHONUNBUFFERED=1 python3 -u << 'PYEOF'
import os, sys, signal, types, torch, time, json, glob

# Block ALL HF network calls at Python level
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

t0 = time.time()
MODEL_PATH  = os.environ["MODEL_PATH"]
DATA_PATH   = os.environ["DATA_PATH"]
OUT_PATH    = os.environ["OUT_PATH"]
JOBID       = os.environ["JOBID"]
SEQ_LEN     = int(os.environ.get("SEQ_LEN", "48"))
RANK        = int(os.environ.get("LORA_RANK", "4"))
STEPS_PER_JOB = int(os.environ.get("STEPS_PER_JOB", "100"))

gpu  = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu} ({vram:.1f}GB) | seq={SEQ_LEN} rank={RANK} | HF_OFFLINE")

# SIGUSR1 handler for graceful checkpointing
_trainer = None
def _sigusr1(signum, frame):
    global _trainer
    if _trainer:
        _trainer.save_model(os.path.join(OUT_PATH, f"checkpoint-{JOBID}-sigusr1"))
    sys.exit(0)
signal.signal(signal.SIGUSR1, _sigusr1)

# --- Load model with local_files_only=True (prevents Unsloth telemetry HF call) ---
print("Loading model (local_files_only=True)...")
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(
    MODEL_PATH,
    max_seq_length=SEQ_LEN,
    load_in_4bit=True,
    trust_remote_code=True,
    local_files_only=True,          # <-- CRITICAL: blocks get_statistics() from calling HF
)

# MoE patch (custom forward for Nemotron-3-Nano MoE)
patched = 0
for module in model.modules():
    if not hasattr(module, 'moe') or not hasattr(module, 'experts'):
        continue
    if not callable(module.moe):
        continue
    def mp(mod):
        def pm(_self, h, ti, tw):
            h = h.view(-1, h.size(-1))
            ft = ti.view(-1)
            h = h.repeat_interleave(ti.shape[-1], dim=0)
            fh = torch.zeros_like(h)
            dt = fh.dtype
            for i, el in enumerate(mod.experts):
                m = (ft == i).nonzero(as_tuple=True)[0]
                if m.numel() == 0:
                    continue
                eh = h[m]
                eo = el(eh)
                w = tw.view(-1)[m]
                wo = eo * w.unsqueeze(-1)
                if wo.dtype != dt:
                    wo = wo.to(dt)
                fh.index_add_(0, m, wo)
            tk = ti.shape[-1]
            fh = fh.view(-1, tk, fh.size(-1)).sum(dim=1)
            return fh
        return pm
    module.moe = types.MethodType(mp(module), module)
    patched += 1
print(f"MoE patched: {patched} layers")

# LoRA config
model = FastLanguageModel.get_peft_model(
    model,
    r=RANK,
    lora_alpha=RANK,
    target_modules=[
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'in_proj', 'out_proj', 'up_proj', 'down_proj',
    ],
    use_gradient_checkpointing='unsloth',
)
model.print_trainable_parameters()

# --- Resume from latest checkpoint if available ---
from peft import PeftModel

ckpts = sorted(
    glob.glob(os.path.join(OUT_PATH, "checkpoint-*", "adapter_model.safetensors")),
    key=os.path.getmtime,
)
trainer_ckpts = [
    os.path.dirname(c)
    for c in ckpts
    if 'sigusr1' not in c
    and os.path.exists(os.path.join(os.path.dirname(c), "trainer_state.json"))
]
if trainer_ckpts:
    latest_ckpt = trainer_ckpts[-1]
    state_file = os.path.join(latest_ckpt, "trainer_state.json")
    try:
        state = json.load(open(state_file))
        global_step = state.get("global_step", 0)
        print(f"Loading adapter from: {latest_ckpt} (global step: {global_step})")
        model = PeftModel.from_pretrained(model, latest_ckpt, is_trainable=True)
        print(f"Adapter loaded! Continuing from step {global_step}")
    except Exception as e:
        global_step = 0
        print(f"Could not load state: {e}, starting fresh")
else:
    global_step = 0
    print("No prior checkpoint, starting fresh")

# --- Load dataset ---
from datasets import load_dataset

ds = load_dataset("json", data_files={"train": DATA_PATH}, split="train")

def fmt(ex):
    ms = ex.get("messages")
    if not ms:
        ms = [
            {"role": "user", "content": ex.get("prompt", "?")},
            {"role": "assistant", "content": ex.get("answer", "?")},
        ]
    return {"text": tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=False)}

ds = ds.map(fmt)

alloc = torch.cuda.memory_allocated() / 1e9
reserved = torch.cuda.memory_reserved() / 1e9
print(f"VRAM: {alloc:.1f}G allocated / {reserved:.1f}G reserved")

# --- Train ---
from trl import SFTTrainer, SFTConfig

target_total = global_step + STEPS_PER_JOB
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir=OUT_PATH,
        max_steps=target_total,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        max_seq_length=SEQ_LEN,
        warmup_steps=max(0, 50 - global_step),
        logging_steps=10,
        save_steps=50,
        save_total_limit=5,
        bf16=False,
        fp16=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
        packing=True,
    ),
    train_dataset=ds,
    processing_class=tok,
)
_trainer = trainer

print(f"TRAINING v20 fixed: step {global_step} -> {target_total}...")
trainer.train()

# --- Save final checkpoint ---
ckpt_dir = os.path.join(OUT_PATH, f"checkpoint-{JOBID}-step{target_total}")
trainer.save_model(ckpt_dir)
tok.save_pretrained(ckpt_dir, local_files_only=True)
print(f"Saved: {ckpt_dir}")

elapsed = (time.time() - t0) / 60
print(f"DONE {elapsed:.1f}min (steps {global_step}->{target_total})")

manifest = {
    "model": MODEL_PATH,
    "data": DATA_PATH,
    "seq_len": SEQ_LEN,
    "rank": RANK,
    "jobid": JOBID,
    "steps_done": target_total,
    "elapsed_min": elapsed,
    "gpu": gpu,
    "framework": "unsloth-4bit-qlora-q6000-v20-fixed-local-files-only",
}
with open(os.path.join(OUT_PATH, f"manifest-{JOBID}.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print("SUCCESS")
PYEOF

EXIT=$?
echo ""
echo "Exit: $EXIT"
if [ $EXIT -eq 0 ]; then
    echo "SUCCESS!"
    find "$O" -name "*.safetensors" -ls 2>/dev/null | tail -5 || true
else
    echo "FAILED with exit=$EXIT"
fi
rm -rf "$P"
