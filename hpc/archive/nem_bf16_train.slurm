#!/bin/bash
# NEM_BF16_TRAIN — full-precision BF16+LoRA training on 1x RTX PRO 6000 (96GB),
# no 4-bit quantization (no unsloth/bitsandbytes): pure transformers+peft+trl,
# same recipe validated on 2x MI210 (hpc/nem_mi210_train.slurm). The 60GB bf16
# weights + LoRA + gradient checkpointing fit in 96GB with packing=False.
# NOTE: precompiled mamba_ssm/causal_conv1d wheels lack Blackwell sm_100+ kernels,
# so we skip them entirely and use the same pure-torch rmsnorm_fn fallback as MI210.
# Chained: STEPS_PER_JOB per job + auto-resubmit to TARGET_TOTAL.
# Usage:
#   sbatch --export=ALL,DATA_PATH=.../train_deterministic_v9.jsonl,OUT_PATH=.../outputs/bf16_v9 hpc/nem_bf16_train.slurm
#SBATCH --job-name=nem-bf16
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=6
#SBATCH --output=/home/a474r867/scratch/nem_bf16_%j.out
#SBATCH --error=/home/a474r867/scratch/nem_bf16_%j.err

export PYTHONUNBUFFERED=1
export APPTAINER_WRITABLE_TMPFS=1

R=/home/a474r867/scratch/nemotron-eco-reasoner
C=$R/nemotron-blackwell.sif
M=/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848
DATA=${DATA_PATH:-$R/data/train_deterministic_v9.jsonl}
OUT=${OUT_PATH:-$R/outputs/bf16_v9}
TEMPLATE_PATH=$R/hpc/nem_bf16_train.slurm
TARGET_TOTAL=${TARGET_TOTAL:-500}
STEPS_PER_JOB=${STEPS_PER_JOB:-200}
SEQ_LEN=${SEQ_LEN:-2048}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-1e-4}
JOBID=${SLURM_JOB_ID:-$$}
P=/tmp/pip_$JOBID
rm -rf $P 2>/dev/null; mkdir -p $P

mkdir -p "$OUT"
echo "=== BF16 TRAIN job=$JOBID $(date) data=$DATA out=$OUT target=$TARGET_TOTAL batch=$STEPS_PER_JOB ==="

# trl>=0.21 for the max_length/processing_class SFT API (same as the ROCm
# container); the container's preinstalled trl 0.12-era is unsloth-only.
apptainer exec --nv --bind "/dev/shm,/tmp" --env PYTHONUSERBASE=$P $C bash << 'INSTALL_EOF'
set -uo pipefail
pip install --user --no-cache-dir -q --no-deps "transformers==4.57.6" "tokenizers==0.22.2" 2>&1 | tail -1 || true
pip install --user --no-cache-dir -q --no-deps "datasets==4.3.0" "trl==0.24.0" "peft==0.18.0" 2>&1 | tail -1 || true
pip install --user --no-cache-dir -q --no-deps einops 2>&1 | tail -1 || { echo "FATAL: einops"; exit 1; }
# Skip mamba_ssm & causal_conv1d wheels — their CUDA kernels lack Blackwell
# (sm_100+) support. The model code falls back to pure torch when these are
# absent (rmsnorm_fn patched in the snapshot, same as MI210 path).
VIRTUAL_ENV=${PYTHONUSERBASE:-$HOME/.local}
rm -rf $VIRTUAL_ENV/lib/python3*/site-packages/torch* $VIRTUAL_ENV/lib/python3*/site-packages/triton* $VIRTUAL_ENV/lib/python3*/site-packages/nvidia* 2>/dev/null
# Also purge any system-level mamba_ssm/causal_conv1d so they don't get imported
rm -rf $VIRTUAL_ENV/lib/python3*/site-packages/mamba_ssm* $VIRTUAL_ENV/lib/python3*/site-packages/causal_conv1d* 2>/dev/null
python3 -c 'import trl, peft, transformers; print("trl", trl.__version__, "| peft", peft.__version__, "| transformers", transformers.__version__)'
INSTALL_EOF
[ $? -ne 0 ] && { echo "INSTALL FAILED"; exit 1; }

apptainer exec --nv --bind "/dev/shm,/tmp,$M,$R" \
    --env PYTHONUSERBASE=$P \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env PYTHONUNBUFFERED=1 --env TORCH_COMPILE_DISABLE=1 \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env DATA="$DATA" --env OUT="$OUT" --env M="$M" --env JOBID=$JOBID \
    --env TARGET_TOTAL=$TARGET_TOTAL --env STEPS_PER_JOB=$STEPS_PER_JOB \
    --env SEQ_LEN=$SEQ_LEN --env GRAD_ACCUM=$GRAD_ACCUM --env LR=$LR \
    $C python3 - <<'PYEOF'
import os, time, glob, json, shutil, torch

# Torch rmsnorm_fn fallback patch (same as MI210 path) — the Blackwell PRO6000
# can't use the precompiled mamba_ssm wheels (no sm_100+ kernels).
M_PATH = os.path.join(os.environ["M"], "modeling_nemotron_h.py")
OLD = '''except ImportError:
    raise ImportError("mamba-ssm is required by the Mamba model but cannot be imported")'''
NEW = '''except ImportError:
    def rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5, group_size=None, norm_before_gate=False):
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
src = open(M_PATH).read()
if OLD in src:
    open(M_PATH, "w").write(src.replace(OLD, NEW))
    print("patched modeling_nemotron_h.py with torch rmsnorm_fn fallback")
for d in glob.glob(os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/cbd3fa9f*")):
    shutil.rmtree(d, ignore_errors=True)

M = os.environ["M"]; DATA = os.environ["DATA"]; OUT = os.environ["OUT"]
JOBID = os.environ["JOBID"]
TARGET_TOTAL = int(os.environ["TARGET_TOTAL"]); STEPS_PER_JOB = int(os.environ["STEPS_PER_JOB"])
SEQ_LEN = int(os.environ["SEQ_LEN"]); GRAD_ACCUM = int(os.environ["GRAD_ACCUM"])
LR = float(os.environ["LR"])

print(f"torch {torch.__version__} | GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  cuda:{i} {torch.cuda.get_device_name(i)} {torch.cuda.get_device_properties(i).total_memory/1e9:.1f}GB")

from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    M, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": 0})
print(f"MODEL LOADED in {time.time()-t0:.0f}s")

from peft import LoraConfig
target = sorted({n.split(".")[-1] for n, m in model.named_modules()
                 if isinstance(m, torch.nn.Linear)} &
                {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","in_proj","out_proj"})
peft_config = LoraConfig(r=32, lora_alpha=64, target_modules=target,
                         lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

ckpts = sorted(glob.glob(os.path.join(OUT, "checkpoint-*", "trainer_state.json")), key=os.path.getmtime)
RESUME = os.path.dirname(ckpts[-1]) if ckpts else None
global_step = json.load(open(ckpts[-1])).get("global_step", 0) if ckpts else 0
print(f"RESUME: {RESUME} (step {global_step})" if RESUME else "FRESH START")

from datasets import Dataset
ds = Dataset.from_json(DATA)
def fmt(ex):
    ms = ex.get("messages") or [{"role":"user","content":ex.get("prompt","?")},
                                {"role":"assistant","content":ex.get("answer","?")}]
    return {"text": tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=False)}
ds = ds.map(fmt, remove_columns=ds.column_names)
print(f"{len(ds)} training records")

batch_steps = min(STEPS_PER_JOB, TARGET_TOTAL - global_step)
target_step = global_step + batch_steps

from trl import SFTTrainer, SFTConfig
cfg = SFTConfig(output_dir=OUT, max_steps=target_step, per_device_train_batch_size=1,
                gradient_accumulation_steps=GRAD_ACCUM, learning_rate=LR, logging_steps=10,
                bf16=True, gradient_checkpointing=True, optim="adamw_torch",
                max_length=SEQ_LEN, dataset_text_field="text", packing=False,
                warmup_steps=min(30, target_step//3), lr_scheduler_type="constant_with_warmup",
                save_steps=50, save_total_limit=10, report_to="none")
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                     processing_class=tok, peft_config=peft_config)
print(f"TRAINING [BF16/PRO6000]: step {global_step} -> {target_step} (target={TARGET_TOTAL})")
t0 = time.time()
trainer.train(resume_from_checkpoint=RESUME)
elapsed = (time.time()-t0)/60
final_loss = next((e["loss"] for e in reversed(trainer.state.log_history) if "loss" in e), "?")
print(f"BATCH DONE {elapsed:.1f}min steps={global_step}->{target_step} loss={final_loss}")
if target_step < TARGET_TOTAL:
    print(f"NEED_RESUBMIT: target_step={target_step} target_total={TARGET_TOTAL}")
else:
    print(f"TARGET_REACHED: {target_step}/{TARGET_TOTAL} steps")
print("SUCCESS")
PYEOF
PY_EXIT=$?
echo "Exit: $PY_EXIT"

if [ $PY_EXIT -eq 0 ]; then
    NEED=$(grep 'NEED_RESUBMIT' /home/a474r867/scratch/nem_bf16_${JOBID}.out 2>/dev/null | tail -1)
    if [ -n "$NEED" ]; then
        TARGET_TOTAL_VAL=$(echo "$NEED" | grep -oP 'target_total=\K[0-9]+')
        echo "→ AUTO-RESUBMIT"
        NEW_JID=$(sbatch --export=ALL,TARGET_TOTAL=${TARGET_TOTAL_VAL},STEPS_PER_JOB=${STEPS_PER_JOB},DATA_PATH=${DATA},OUT_PATH=${OUT} ${TEMPLATE_PATH} 2>/dev/null | grep -oP '\d+')
        echo "→ RESUBMITTED: job ${NEW_JID}"
    fi
fi
exit $PY_EXIT
