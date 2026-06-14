#!/bin/bash
# MI210 BF16 LoRA smoke test — loads Nemotron-30B in bf16 sharded across 2 MI210
# (device_map=auto) and runs a few real training steps on the v8 dataset.
# Uses nemotron-rocm.sif (torch 2.6 ROCm, transformers 4.57.6, peft 0.17.1, trl 0.24.0).
# No bitsandbytes, no unsloth, no compiled mamba kernels (pure-Python Mamba fallback).
#SBATCH --job-name=nem-mi210-smoke
#SBATCH --partition=sixhour
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:mi210:2
#SBATCH --mem=96G
#SBATCH --cpus-per-task=6
#SBATCH --output=/home/a474r867/scratch/nem_mi210_smoke_%j.out
#SBATCH --error=/home/a474r867/scratch/nem_mi210_smoke_%j.err

export PYTHONUNBUFFERED=1

R=/home/a474r867/scratch/nemotron-eco-reasoner
C=$R/nemotron-rocm.sif
M=/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848
DATA=${DATA_PATH:-$R/data/train_deterministic_v8.jsonl}
OUT=${OUT_PATH:-$R/outputs/mi210_smoke}
MAX_STEPS=${MAX_STEPS:-10}
SEQ_LEN=${SEQ_LEN:-2048}

mkdir -p "$OUT"
echo "=== MI210 SMOKE job=$SLURM_JOB_ID $(date) data=$DATA out=$OUT steps=$MAX_STEPS ==="

apptainer exec --rocm --bind "/dev/shm,/tmp,$M,$R" \
    --env PYTHONUNBUFFERED=1 --env TORCH_COMPILE_DISABLE=1 \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env DATA="$DATA" --env OUT="$OUT" --env M="$M" \
    --env MAX_STEPS=$MAX_STEPS --env SEQ_LEN=$SEQ_LEN \
    $C python3 - <<'PYEOF'
import os, time, glob, shutil, torch

# The stock modeling_nemotron_h.py hard-raises if mamba_ssm is missing, even
# though every kernel it needs has a torch fallback except rmsnorm_fn. Patch
# the snapshot to provide a pure-torch rmsnorm_fn so the ROCm container
# (no mamba_ssm) can load the model. No-op when mamba_ssm is importable.
MPATH = os.path.join(os.environ["M"], "modeling_nemotron_h.py")
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
src = open(MPATH).read()
if OLD in src:
    open(MPATH, "w").write(src.replace(OLD, NEW))
    print("patched modeling_nemotron_h.py with torch rmsnorm_fn fallback")
for d in glob.glob(os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/cbd3fa9f*")):
    shutil.rmtree(d, ignore_errors=True)  # force re-copy of patched module

print(f"torch {torch.__version__} | GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} {torch.cuda.get_device_name(i)} {p.total_memory/1e9:.1f}GB")

M = os.environ["M"]; DATA = os.environ["DATA"]; OUT = os.environ["OUT"]
MAX_STEPS = int(os.environ["MAX_STEPS"]); SEQ_LEN = int(os.environ["SEQ_LEN"])

from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    M, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
print(f"MODEL LOADED in {time.time()-t0:.0f}s | hf_device_map entries: {len(model.hf_device_map)}")
for i in range(torch.cuda.device_count()):
    print(f"  cuda:{i} allocated {torch.cuda.memory_allocated(i)/1e9:.1f}GB")

from peft import LoraConfig
target = sorted({n.split(".")[-1] for n, m in model.named_modules()
                 if isinstance(m, torch.nn.Linear)} &
                {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","in_proj","out_proj"})
print("LoRA targets:", target)
peft_config = LoraConfig(r=32, lora_alpha=64, target_modules=target,
                         lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

from datasets import Dataset
ds = Dataset.from_json(DATA).select(range(200))
def fmt(ex):
    ms = ex.get("messages") or [{"role":"user","content":ex.get("prompt","?")},
                                {"role":"assistant","content":ex.get("answer","?")}]
    return {"text": tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=False)}
ds = ds.map(fmt, remove_columns=ds.column_names)
print(f"{len(ds)} smoke records")

from trl import SFTTrainer, SFTConfig
cfg = SFTConfig(output_dir=OUT, max_steps=MAX_STEPS, per_device_train_batch_size=1,
                gradient_accumulation_steps=4, learning_rate=1e-4, logging_steps=1,
                bf16=True, gradient_checkpointing=True, optim="adamw_torch",
                max_length=SEQ_LEN, dataset_text_field="text", packing=False,
                save_strategy="no", report_to="none")
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                     processing_class=tok, peft_config=peft_config)
t0 = time.time()
trainer.train()
dt = time.time() - t0
print(f"=== SMOKE OK: {MAX_STEPS} steps in {dt:.0f}s ({dt/MAX_STEPS:.1f}s/step) ===")
trainer.save_model(os.path.join(OUT, "smoke_adapter"))
print("adapter saved")
PYEOF
rc=$?
echo "=== DONE rc=$rc $(date) ==="
exit $rc
