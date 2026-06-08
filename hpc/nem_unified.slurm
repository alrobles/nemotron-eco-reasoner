#!/bin/bash
# UNIFIED Nemotron template — auto-detects GPU and configures optimally
#SBATCH --job-name=nem
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --gres=gpu:1
#SBATCH --mem=62G
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/a474r867/scratch/nem_unified_%j.out
#SBATCH --error=/home/a474r867/scratch/nem_unified_%j.err
#SBATCH --signal=B:SIGUSR1@120

export APPTAINER_WRITABLE_TMPFS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_$SLURM_JOB_ID

# SIF: use CONTAINER_SIF from orchestrator, or default to cuda
C=${CONTAINER_SIF:-/home/a474r867/scratch/nemotron-eco-reasoner/nemotron-cuda.sif}
M=/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848
D=/home/a474r867/scratch/nemotron-eco-reasoner/data/train_cot_unified.jsonl
O=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/unified
P=/tmp/pip_$SLURM_JOB_ID
rm -rf $P 2>/dev/null; mkdir -p $O $P

JOBID=${SLURM_JOB_ID:-$$}
START_TS=$(date +%s)

echo "=== UNIFIED job=$JOBID $(date) ==="
echo "SIF: $C"

# Auto-detect GPU runtime: ROCm for MI210, NVIDIA for everything else
GPU_FLAG="--nv"
if command -v rocminfo &>/dev/null 2>&1 || [[ "$C" == *mi210* ]]; then
    GPU_FLAG="--rocm"
fi

apptainer exec $GPU_FLAG --bind "/dev/shm,/tmp" --env PYTHONUSERBASE=$P --env TRITON_CACHE_DIR=/tmp/triton_$SLURM_JOB_ID $C bash << 'INSTALL_EOF'
set -uo pipefail
# Core deps (must succeed)
pip install --user --no-cache-dir -q --no-deps "transformers==4.57.6" "tokenizers==0.22.2" 2>&1 | tail -1 || true
pip install --user --no-cache-dir -q --no-deps "datasets==4.3.0" bitsandbytes 2>&1 | tail -1 || true
pip install --user --no-cache-dir -q --no-deps unsloth_zoo unsloth rich "trl==0.12.0" 2>&1 | tail -1 || true
pip install --user --no-cache-dir -q --no-deps einops 2>&1 | tail -1 || { echo "FATAL: einops install failed"; exit 1; }
# mamba-ssm + causal-conv1d: pre-built wheels for Blackwell, pip wheels for others
# Wheels at /home/a474r867/scratch/wheels/ (440MB, compiled on Blackwell GPU)
WHEEL_DIR=/home/a474r867/scratch/wheels
GPU_CHECK=$(python3 -c 'import torch; n=torch.cuda.get_device_name(0); print(n)' 2>/dev/null || echo "nogpu")
echo "GPU: $GPU_CHECK"
if echo "$GPU_CHECK" | grep -qi "PRO 6000\|blackwell"; then
    echo "=== Blackwell: installing pre-built mamba wheels (2s) ==="
    pip install --user --no-deps $WHEEL_DIR/causal_conv1d-1.4.0-cp312-cp312-linux_x86_64.whl 2>&1 | tail -1
    pip install --user --no-deps $WHEEL_DIR/mamba_ssm-2.2.4-cp312-cp312-linux_x86_64.whl 2>&1 | tail -1
elif echo "$GPU_CHECK" | grep -qi "MI210\|mi210\|instinct"; then
    echo "=== MI210 ROCm: installing pre-built ROCm wheels ==="
    ROCM_WHEEL_DIR=/home/a474r867/scratch/wheels_rocm
    pip install --user --no-deps $ROCM_WHEEL_DIR/causal_conv1d-*.whl 2>&1 | tail -1
    pip install --user --no-deps $ROCM_WHEEL_DIR/mamba_ssm-*.whl 2>&1 | tail -1
else
    echo "=== Non-Blackwell: installing mamba-ssm (required) ==="
    pip install --user --no-cache-dir -q --no-deps mamba-ssm causal-conv1d 2>&1 | tail -1
    python3 -c "import mamba_ssm; print("mamba-ssm OK")" || { echo "FATAL: mamba_ssm required but not installed"; exit 1; }
fi
# CRITICAL: purge any pip-installed torch that shadows the SIF's built-in torch
# (unsloth_zoo wheels can bundle torch; --no-deps doesn't prevent wheel-included deps)
SIF_TORCH_VER=$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)
echo "SIF torch: $SIF_TORCH_VER"
rm -rf $VIRTUAL_ENV/lib/python3*/site-packages/torch* \
       $VIRTUAL_ENV/lib/python3*/site-packages/triton* \
       $VIRTUAL_ENV/lib/python3*/site-packages/nvidia* 2>/dev/null
# If using --user installs
rm -rf $HOME/.local/lib/python3*/site-packages/torch* \
       $HOME/.local/lib/python3*/site-packages/triton* \
       $HOME/.local/lib/python3*/site-packages/nvidia* 2>/dev/null
# Verify SIF torch still intact (not shadowed)
ACTUAL_TORCH=$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)
if [ "$ACTUAL_TORCH" != "$SIF_TORCH_VER" ]; then
    echo "WARNING: torch changed from $SIF_TORCH_VER to $ACTUAL_TORCH after pip installs"
fi
echo "Active torch: $ACTUAL_TORCH"
# Verify critical imports
python3 -c 'import torch,transformers,tokenizers,trl; print(f"tf:{transformers.__version__} tok:{tokenizers.__version__} trl:{trl.__version__}")'
python3 -c 'import unsloth; from trl import SFTTrainer; print("OK")'
INSTALL_EOF

if [ $? -ne 0 ]; then echo "INSTALL FAILED"; exit 1; fi

echo "=== Training ==="
apptainer exec $GPU_FLAG --bind "/dev/shm,/tmp" \
    --env PYTHONUSERBASE=$P --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env TORCH_COMPILE_DISABLE=1 --env OMP_NUM_THREADS=1 \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env MODEL_PATH=$M --env DATA_PATH=$D --env OUT_PATH=$O \
    --env JOBID=$JOBID --env TRITON_CACHE_DIR=/tmp/triton_$SLURM_JOB_ID \
    $C env PYTHONUNBUFFERED=1 python3 -u << 'PYEOF'
import os, sys, signal, types, torch, time, json, glob
from datasets import load_dataset
os.environ['TORCH_COMPILE_DISABLE']='1'
t0=time.time()
MODEL_PATH=os.environ["MODEL_PATH"]; DATA_PATH=os.environ["DATA_PATH"]
OUT_PATH=os.environ["OUT_PATH"]; JOBID=os.environ["JOBID"]
STEPS_PER_JOB=int(os.environ.get("STEPS_PER_JOB","100"))

gpu_name=torch.cuda.get_device_name(0)
vram_gb=torch.cuda.get_device_properties(0).total_memory/1e9
bf16_ok=torch.cuda.is_bf16_supported()
cc=(torch.cuda.get_device_capability(0)[0]*10+torch.cuda.get_device_capability(0)[1])/10

# AUTO-CONFIG per GPU type
gpu_lower=gpu_name.lower()
if vram_gb>=90:
    SEQ_LEN,RANK,GRAD_ACCUM=2048,32,8;  TAG="BEAST"
elif vram_gb>=55:
    SEQ_LEN,RANK,GRAD_ACCUM=1024,16,8;  TAG="HEAVY"
elif vram_gb>=40:
    SEQ_LEN,RANK,GRAD_ACCUM=512,8,8;    TAG="MEDIUM"
else:
    SEQ_LEN,RANK,GRAD_ACCUM=128,4,4;    TAG="LIGHT"

print(f"GPU: {gpu_name} ({vram_gb:.1f}GB) CC={cc} bf16={bf16_ok} | seq={SEQ_LEN} rank={RANK} [{TAG}]")

_trainer=None
def _sigusr1(signum,frame):
    global _trainer
    if _trainer: _trainer.save_model(os.path.join(OUT_PATH,f"checkpoint-{JOBID}-sigusr1"))
    sys.exit(0)
signal.signal(signal.SIGUSR1,_sigusr1)

print("Loading model...")
from unsloth import FastLanguageModel
model,tok=FastLanguageModel.from_pretrained(MODEL_PATH,max_seq_length=SEQ_LEN,load_in_4bit=True,trust_remote_code=True)

patched=0
for module in model.modules():
    if not hasattr(module,'moe') or not hasattr(module,'experts'): continue
    if not callable(module.moe): continue
    def mp(mod):
        def pm(_self,h,ti,tw):
            h=h.view(-1,h.size(-1)); ft=ti.view(-1); h=h.repeat_interleave(ti.shape[-1],dim=0)
            fh=torch.zeros_like(h); dt=fh.dtype
            for i,el in enumerate(mod.experts):
                m=(ft==i).nonzero(as_tuple=True)[0]
                if m.numel()==0: continue
                eh=h[m]; eo=el(eh); w=tw.view(-1)[m]; wo=eo*w.unsqueeze(-1)
                if wo.dtype!=dt: wo=wo.to(dt)
                fh.index_add_(0,m,wo)
            tk=ti.shape[-1]; fh=fh.view(-1,tk,fh.size(-1)).sum(dim=1)
            return fh
        return pm
    module.moe=types.MethodType(mp(module),module); patched+=1
print(f"MoE patched: {patched} layers")

model=FastLanguageModel.get_peft_model(model,r=RANK,lora_alpha=RANK,
    target_modules=['q_proj','k_proj','v_proj','o_proj','in_proj','out_proj','up_proj','down_proj'],
    use_gradient_checkpointing='unsloth')
model.print_trainable_parameters()

# Load adapter from shared checkpoint pool
from peft import PeftModel
ckpts=sorted(glob.glob(os.path.join(OUT_PATH,"checkpoint-*","adapter_model.safetensors")),key=os.path.getmtime)
trainer_ckpts=[os.path.dirname(c) for c in ckpts if 'sigusr1' not in c and os.path.exists(os.path.join(os.path.dirname(c),"trainer_state.json"))]
if trainer_ckpts:
    latest_ckpt=trainer_ckpts[-1]
    state_file=os.path.join(latest_ckpt,"trainer_state.json")
    try:
        state=json.load(open(state_file))
        global_step=state.get("global_step",0)
        print(f"Loading adapter: {os.path.basename(latest_ckpt)} (step {global_step})")
        model=PeftModel.from_pretrained(model,latest_ckpt,is_trainable=True)
    except Exception as e:
        global_step=0
        print(f"No adapter loaded: {e}")
else:
    global_step=0
    print("Fresh start — no prior checkpoint")

ds=load_dataset("json",data_files={"train":DATA_PATH},split="train")
def fmt(ex):
    ms=ex.get("messages")
    if not ms: ms=[{"role":"user","content":ex.get("prompt","?")},{"role":"assistant","content":ex.get("answer","?")}]
    return {"text":tok.apply_chat_template(ms,tokenize=False,add_generation_prompt=False)}
ds=ds.map(fmt)

alloc=torch.cuda.memory_allocated()/1e9; reserved=torch.cuda.memory_reserved()/1e9
print(f"VRAM: {alloc:.1f}G/{reserved:.1f}G")

from trl import SFTTrainer,SFTConfig
# Constant LR after warmup — no scheduler reset between batches
target_total=global_step+STEPS_PER_JOB
USE_BF16=bf16_ok
trainer=SFTTrainer(model=model,args=SFTConfig(
    output_dir=OUT_PATH,max_steps=target_total,per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,learning_rate=1e-4,max_seq_length=SEQ_LEN,
    warmup_steps=30,lr_scheduler_type="constant_with_warmup",
    logging_steps=10,save_steps=50,save_total_limit=5,
    bf16=USE_BF16,fp16=not USE_BF16,
    remove_unused_columns=False,report_to="none",
    dataloader_num_workers=0,packing=True),
    train_dataset=ds,processing_class=tok)
_trainer=trainer

print(f"TRAINING [{TAG}]: step {global_step} -> {target_total} seq={SEQ_LEN} r={RANK} bf16={USE_BF16}")
trainer.train()

ckpt_dir=os.path.join(OUT_PATH,f"checkpoint-{JOBID}-step{target_total}")
trainer.save_model(ckpt_dir)
tok.save_pretrained(ckpt_dir)

elapsed=(time.time()-t0)/60
final_loss=trainer.state.log_history[-1].get('loss') or trainer.state.log_history[-1].get('train_loss','?') if trainer.state.log_history else '?'

print(f"DONE {elapsed:.1f}min steps={global_step}->{target_total} loss={final_loss}")

manifest={"gpu":gpu_name,"vram_gb":round(vram_gb,1),"cc":cc,"bf16":USE_BF16,
    "seq":SEQ_LEN,"rank":RANK,"tag":TAG,"jobid":JOBID,
    "steps_start":global_step,"steps_done":target_total,"elapsed_min":round(elapsed,1),
    "final_loss":final_loss}
with open(os.path.join(OUT_PATH,f"manifest-{JOBID}.json"),"w") as f: json.dump(manifest,f,indent=2)
print("SUCCESS")
PYEOF
echo "Exit: $?"
