#!/bin/bash
# NEM_SCRATCH — entrenamiento desde cero (sin warmstart) basado en v8_seq3072 (0.67)
# Mejoras sobre el original: LR linear decay, lora_dropout=0, seq_len=4096
#SBATCH --job-name=nem-scr
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=6
#SBATCH --output=/home/a474r867/scratch/nem_scr_%j.out
#SBATCH --error=/home/a474r867/scratch/nem_scr_%j.err

export PYTHONUNBUFFERED=1
export APPTAINER_WRITABLE_TMPFS=1

R=/home/a474r867/scratch/nemotron-eco-reasoner
C=$R/nemotron-blackwell.sif
M=/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848
DATA=${DATA_PATH:-$R/data/train_deterministic_v8.jsonl}
OUT=${OUT_PATH:-$R/outputs/scratch_v1}
TEMPLATE_PATH=$R/hpc/nem_scratch.slurm
TARGET_TOTAL=${TARGET_TOTAL:-500}
STEPS_PER_JOB=${STEPS_PER_JOB:-200}
SEQ_LEN=${SEQ_LEN:-4096}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-2e-4}
JOBID=${SLURM_JOB_ID:-$$}
P=/tmp/pip_$JOBID
rm -rf $P 2>/dev/null; mkdir -p $P

mkdir -p "$OUT"
echo "=== SCRATCH TRAIN job=$JOBID $(date) ==="
echo "  data=$DATA  seq=$SEQ_LEN  lr=$LR  grad_accum=$GRAD_ACCUM  target=$TARGET_TOTAL"

apptainer exec --nv --bind "/dev/shm,/tmp" --env PYTHONUSERBASE=$P $C bash << 'INSTALL_EOF'
set -uo pipefail
pip install --user --no-cache-dir -q --no-deps "transformers==4.57.6" "tokenizers==0.22.2" 2>&1 | tail -1 || true
pip install --user --no-cache-dir -q --no-deps "datasets==4.3.0" "trl==0.24.0" "peft==0.18.0" 2>&1 | tail -1 || true
pip install --user --no-cache-dir -q --no-deps einops liger_kernel 2>&1 | tail -1 || { echo "FATAL: einops"; exit 1; }
VIRTUAL_ENV=${PYTHONUSERBASE:-$HOME/.local}
rm -rf $VIRTUAL_ENV/lib/python3*/site-packages/torch* $VIRTUAL_ENV/lib/python3*/site-packages/triton* $VIRTUAL_ENV/lib/python3*/site-packages/nvidia* 2>/dev/null
rm -rf $VIRTUAL_ENV/lib/python3*/site-packages/mamba_ssm* $VIRTUAL_ENV/lib/python3*/site-packages/causal_conv1d* 2>/dev/null
python3 -c 'import trl, peft, transformers; print("OK trl",trl.__version__,"peft",peft.__version__)'
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

# Patch rmsnorm_fn
M_PATH = os.path.join(os.environ["M"], "modeling_nemotron_h.py")
OLD = 'except ImportError:\n    raise ImportError("mamba-ssm is required by the Mamba model but cannot be imported")'
NEW = 'except ImportError:\n    def rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5, group_size=None, norm_before_gate=False):\n        dt = x.dtype\n        x = x.float()\n        if z is not None and not norm_before_gate:\n            x = x * torch.nn.functional.silu(z.float())\n        if group_size is not None and group_size != x.shape[-1]:\n            s = x.shape\n            xg = x.view(*s[:-1], s[-1] // group_size, group_size)\n            xg = xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + eps)\n            x = x.view(s)\n        else:\n            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)\n        out = x * weight.float()\n        if bias is not None:\n            out = out + bias.float()\n        if z is not None and norm_before_gate:\n            out = out * torch.nn.functional.silu(z.float())\n        return out.to(dt)'
src = open(M_PATH).read()
if OLD in src:
    open(M_PATH, "w").write(src.replace(OLD, NEW))
    print("patched rmsnorm_fn")
for d in glob.glob(os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/cbd3fa9f*")):
    shutil.rmtree(d, ignore_errors=True)

M = os.environ["M"]; DATA = os.environ["DATA"]; OUT = os.environ["OUT"]
JOBID = os.environ["JOBID"]
TARGET_TOTAL = int(os.environ["TARGET_TOTAL"]); STEPS_PER_JOB = int(os.environ["STEPS_PER_JOB"])
SEQ_LEN = int(os.environ["SEQ_LEN"]); GRAD_ACCUM = int(os.environ["GRAD_ACCUM"])
LR = float(os.environ["LR"])

print(f"torch {torch.__version__} | GPU: {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    M, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": 0})
print(f"MODEL LOADED in {time.time()-t0:.0f}s")

# LoRA from scratch (NO warmstart) - same targets as v8_seq3072 (0.67 winner)
from peft import LoraConfig
target = sorted({n.split(".")[-1] for n, m in model.named_modules()
                 if isinstance(m, torch.nn.Linear)} &
                {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","in_proj","out_proj"})

RANK=32; ALPHA=32
peft_config = LoraConfig(
    r=RANK, lora_alpha=ALPHA, target_modules=target,
    lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")

# Resume check
ckpts = sorted(glob.glob(os.path.join(OUT, "checkpoint-*", "trainer_state.json")), key=os.path.getmtime)
RESUME = os.path.dirname(ckpts[-1]) if ckpts else None
global_step = json.load(open(ckpts[-1])).get("global_step", 0) if ckpts else 0
batch_steps = min(STEPS_PER_JOB, TARGET_TOTAL - global_step)
target_step = global_step + batch_steps
print(f"RESUME step {global_step}" if RESUME else "FRESH START")

# Data
from datasets import Dataset
ds = Dataset.from_json(DATA)
def fmt(ex):
    ms = ex.get("messages") or [{"role":"user","content":ex.get("prompt","?")},
                                {"role":"assistant","content":ex.get("answer","?")}]
    return {"text": tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=False)}
ds = ds.map(fmt, remove_columns=ds.column_names)
print(f"{len(ds)} training records")

from trl import SFTTrainer, SFTConfig

# MoETiedSFTTrainer: ties MoE LoRA gradients across experts after backward
class MoETiedSFTTrainer(SFTTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        self.accelerator.backward(loss)
        # Tie MoE grads across experts using trainer.model (PeftModel)
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if not p.requires_grad or p.grad is None or p.dim() < 2:
                    continue
                if ("gate_up_proj" in n or ".up_proj." in n or ".gate_proj." in n) and "lora_A" in n:
                    p.grad.copy_(p.grad.sum(dim=0, keepdim=True).expand_as(p.grad))
                elif "down_proj" in n and "lora_B" in n:
                    p.grad.copy_(p.grad.sum(dim=0, keepdim=True).expand_as(p.grad))
        return loss.detach() / self.args.gradient_accumulation_steps

warmup_steps = min(30, target_step // 4)
cfg = SFTConfig(
    output_dir=OUT, max_steps=target_step, per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM, learning_rate=LR, logging_steps=10,
    bf16=True, gradient_checkpointing=True, optim="adamw_torch",
    max_length=SEQ_LEN, dataset_text_field="text", packing=False,
    warmup_steps=warmup_steps, lr_scheduler_type="linear",
    save_steps=50, save_total_limit=15, report_to="none",
    adam_beta1=0.9, adam_beta2=0.95, adam_epsilon=1e-8,
    weight_decay=0.0, max_grad_norm=1.0,
    use_liger_kernel=True,
)

trainer = MoETiedSFTTrainer(model=model, args=cfg, train_dataset=ds,
                             processing_class=tok, peft_config=peft_config)

# === MoE weight tying init on PeftModel (must run AFTER PEFT wraps model) ===
print("Tying MoE expert LoRA weights...")
tied_count = 0
for n, p in trainer.model.named_parameters():
    if p.requires_grad and p.dim() >= 2:
        if ("gate_up_proj" in n or ".up_proj." in n or ".gate_proj." in n) and "lora_A" in n:
            p.data.copy_(p.data.mean(dim=0, keepdim=True).expand_as(p.data))
            tied_count += 1
        elif "down_proj" in n and "lora_B" in n:
            p.data.copy_(p.data.mean(dim=0, keepdim=True).expand_as(p.data))
            tied_count += 1
print(f"MoE tied: {tied_count} parameters")

print(f"TRAINING [SCRATCH]: step {global_step} -> {target_step} lr={LR} linear seq={SEQ_LEN} accum={GRAD_ACCUM}")
t0 = time.time()
trainer.train(resume_from_checkpoint=RESUME)
elapsed = (time.time()-t0)/60
final_loss = next((e["loss"] for e in reversed(trainer.state.log_history) if "loss" in e), "?")
print(f"DONE {elapsed:.1f}min loss={final_loss}")

# Save adapter explicitly
model.save_pretrained(os.path.join(OUT, f"checkpoint-{target_step}"))
print(f"Adapter saved to checkpoint-{target_step}")

if target_step < TARGET_TOTAL:
    print(f"NEED_RESUBMIT: target_step={target_step} target_total={TARGET_TOTAL}")
else:
    print(f"TARGET_REACHED: {target_step}/{TARGET_TOTAL}")
print("SUCCESS")
PYEOF
PY_EXIT=$?
echo "Exit: $PY_EXIT"

if [ $PY_EXIT -eq 0 ]; then
    NEED=$(grep 'NEED_RESUBMIT' /home/a474r867/scratch/nem_scr_${JOBID}.out 2>/dev/null | tail -1)
    if [ -n "$NEED" ]; then
        TARGET_TOTAL_VAL=$(echo "$NEED" | grep -oP 'target_total=\K[0-9]+')
        echo "AUTO-RESUBMIT target=$TARGET_TOTAL_VAL"
        NEW_JID=$(sbatch --export=ALL,TARGET_TOTAL=${TARGET_TOTAL_VAL},STEPS_PER_JOB=${STEPS_PER_JOB},DATA_PATH=${DATA},OUT_PATH=${OUT} ${TEMPLATE_PATH} 2>/dev/null | grep -oP '\d+')
        echo "RESUBMITTED: job ${NEW_JID}"
    fi
fi
exit $PY_EXIT
