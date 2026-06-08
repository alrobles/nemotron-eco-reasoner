#!/bin/bash
# Blackwell Pro6000 test — uses nemotron-blackwell.sif (NGC PyTorch 2.8 + sm_120)
#SBATCH --job-name=nem-bw-test
#SBATCH --partition=sixhour
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/a474r867/scratch/nem_blackwell_test_%j.out
#SBATCH --error=/home/a474r867/scratch/nem_blackwell_test_%j.err

export APPTAINER_WRITABLE_TMPFS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

# === USE NEW BLACKWELL SIF ===
C=/home/a474r867/scratch/nemotron-eco-reasoner/nemotron-blackwell.sif
M=/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848
D=/home/a474r867/scratch/nemotron-eco-reasoner/data/train_final_v2.jsonl
O=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/blackwell_test
P=/tmp/bw_test_$SLURM_JOB_ID

rm -rf $P 2>/dev/null; mkdir -p $O $P
JOBID=${SLURM_JOB_ID:-$$}

echo "=== BLACKWELL TEST job=$JOBID $(date) ==="
echo "SIF: $C"
ls -lh $C

echo "=== Phase 1: Verify PyTorch sm_120 support ==="
apptainer exec --nv --bind "/dev/shm,/tmp,$P" \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env PYTHONUNBUFFERED=1 --env TORCH_COMPILE_DISABLE=1 \
    $C python3 -c '
import torch
print(f"PyTorch {torch.__version__} CUDA {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
cc = torch.cuda.get_device_capability(0)
print(f"Compute Capability: {cc[0]}.{cc[1]}")
print(f"Supported archs: {torch.cuda.get_arch_list()}")
x = torch.randn(100, 100, device="cuda")
y = torch.matmul(x, x.T)
print(f"CUDA matmul test: OK ({y.shape})")
print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
print("=== Phase 1 PASSED ===")
'
if [ $? -ne 0 ]; then echo "PHASE 1 FAILED — PyTorch/CUDA check failed"; exit 1; fi

echo "=== Phase 1.5: Build mamba-ssm from source with sm_120 patch ==="
# Per https://github.com/state-spaces/mamba/issues/745
# mamba-ssm setup.py doesn't include sm_120 — must patch and compile from source
apptainer exec --nv --bind "/dev/shm,/tmp,$P" \
    --env PYTHONUNBUFFERED=1 --env TORCH_COMPILE_DISABLE=1 \
    --env TORCH_CUDA_ARCH_LIST="12.0" \
    --env FORCE_CUDA=1 \
    --env MAX_JOBS=4 \
    $C bash -c '
set -e

# Step 1: Build causal_conv1d from source with sm_120
echo "--- Building causal_conv1d from source ---"
cd /tmp
git clone https://github.com/Dao-AILab/causal-conv1d.git 2>/dev/null || (cd causal-conv1d && git pull)
cd causal-conv1d
# Patch setup.py to add sm_120
sed -i "/sm_90/a\\    cc_flag.append(\"-gencode\")\\n    cc_flag.append(\"arch=compute_120,code=sm_120\")" setup.py 2>/dev/null || true
TORCH_CUDA_ARCH_LIST="12.0" pip install -e . --no-build-isolation 2>&1 | tail -5
python3 -c "import causal_conv1d; print(f\"causal_conv1d {causal_conv1d.__version__} OK\")"

# Step 2: Build mamba-ssm from source with sm_120
echo "--- Building mamba-ssm from source ---"
cd /tmp
git clone https://github.com/state-spaces/mamba.git 2>/dev/null || (cd mamba && git pull)
cd mamba
# Patch setup.py to add sm_120 after sm_90 line
sed -i "/sm_90/a\\    cc_flag.append(\"-gencode\")\\n    cc_flag.append(\"arch=compute_120,code=sm_120\")" setup.py 2>/dev/null || true
TORCH_CUDA_ARCH_LIST="12.0" pip install -e . --no-build-isolation 2>&1 | tail -10
python3 -c "import mamba_ssm; print(f\"mamba_ssm {mamba_ssm.__version__} OK\")"
echo "=== Phase 1.5 PASSED ==="
'
MAMBA_RC=$?
if [ $MAMBA_RC -ne 0 ]; then echo "PHASE 1.5 FAILED (exit=$MAMBA_RC) — mamba-ssm compilation failed"; exit 1; fi

echo "=== Phase 2: Quick training test (10 steps) ==="

# Write training script
cat > $P/blackwell_train.py << 'PYEOF'
import os, sys, torch, time, json
from datasets import load_dataset
os.environ['TORCH_COMPILE_DISABLE']='1'
t0=time.time()
MODEL_PATH=os.environ["MODEL_PATH"]; DATA_PATH=os.environ["DATA_PATH"]
OUT_PATH=os.environ["OUT_PATH"]; JOBID=os.environ["JOBID"]
STEPS=int(os.environ.get("STEPS_PER_JOB","10"))

gpu_name=torch.cuda.get_device_name(0)
vram_gb=torch.cuda.get_device_properties(0).total_memory/1e9
bf16_ok=torch.cuda.is_bf16_supported()
cc=torch.cuda.get_device_capability(0)
print(f"GPU: {gpu_name} ({vram_gb:.1f}GB) CC={cc[0]}.{cc[1]} bf16={bf16_ok}")
print(f"PyTorch: {torch.__version__} CUDA: {torch.version.cuda}")

# BEAST config for Blackwell
SEQ_LEN,RANK=2048,32; TAG="BEAST"
print(f"Config: seq={SEQ_LEN} rank={RANK} [{TAG}] steps={STEPS}")

print("Loading model...")
from unsloth import FastLanguageModel
model,tok=FastLanguageModel.from_pretrained(MODEL_PATH,max_seq_length=SEQ_LEN,load_in_4bit=True,trust_remote_code=True)
load_time=time.time()-t0
print(f"Model loaded in {load_time:.1f}s")

# MoE patch for Nemotron hybrid architecture
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
            return fh
        return pm
    import types
    module.moe=types.MethodType(mp(module),module)
    patched+=1
print(f"MoE patched: {patched} layers")

# Apply LoRA
model=FastLanguageModel.get_peft_model(model,r=RANK,target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],lora_alpha=RANK*2,lora_dropout=0,bias="none",use_gradient_checkpointing="unsloth")
print(f"LoRA applied: rank={RANK}")

# Load data
from trl import SFTTrainer
from transformers import TrainingArguments
ds=load_dataset("json",data_files=DATA_PATH,split="train").select(range(min(100,5000)))
print(f"Dataset: {len(ds)} examples")

args=TrainingArguments(
    output_dir=OUT_PATH, max_steps=STEPS, per_device_train_batch_size=1,
    gradient_accumulation_steps=2, learning_rate=2e-4, bf16=bf16_ok, fp16=not bf16_ok,
    logging_steps=1, save_steps=STEPS, warmup_steps=0, lr_scheduler_type="constant",
    optim="adamw_8bit", report_to="none", seed=42
)

trainer=SFTTrainer(model=model,tokenizer=tok,train_dataset=ds,
    args=args, max_seq_length=SEQ_LEN,
    dataset_text_field=None, formatting_func=lambda ex: tok.apply_chat_template(ex["messages"],tokenize=False))

print(f"Starting training ({STEPS} steps)...")
step_times=[]
for step in range(STEPS):
    t1=time.time()
    trainer.train() if step==0 else None
    t2=time.time()
    if step==0: step_times.append(t2-t1)

train_time=time.time()-t0-load_time
vram_used=torch.cuda.max_memory_allocated()/1e9

# Output manifest
manifest={
    "status": "SUCCESS",
    "gpu": gpu_name,
    "vram_gb": round(vram_gb,1),
    "vram_used_gb": round(vram_used,1),
    "cc": f"{cc[0]}.{cc[1]}",
    "bf16": bf16_ok,
    "pytorch": torch.__version__,
    "cuda": torch.version.cuda,
    "config": {"seq_len": SEQ_LEN, "rank": RANK, "tag": TAG},
    "steps": STEPS,
    "load_time_s": round(load_time,1),
    "train_time_s": round(train_time,1),
    "moe_patched": patched,
    "job_id": JOBID
}
manifest_path=os.path.join(OUT_PATH, f"blackwell_manifest_{JOBID}.json")
with open(manifest_path,"w") as f: json.dump(manifest,f,indent=2)
print(f"\n{'='*50}")
print(json.dumps(manifest,indent=2))
print(f"{'='*50}")
print(f"\n*** SUCCESS — BLACKWELL IS READY ***")
print(f"Manifest: {manifest_path}")
PYEOF

apptainer exec --nv --bind "/dev/shm,/tmp,$P,$M,$O,$(dirname $D)" \
    --env MODEL_PATH=$M --env DATA_PATH=$D --env OUT_PATH=$O --env JOBID=$JOBID \
    --env STEPS_PER_JOB=10 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env PYTHONUNBUFFERED=1 --env TORCH_COMPILE_DISABLE=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $C python3 $P/blackwell_train.py

RC=$?
echo "=== DONE (exit=$RC) $(date) ==="
exit $RC
