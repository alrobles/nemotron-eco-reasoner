#!/bin/bash
# Blackwell Pro6000 test — 5 min quick test with PyTorch 2.6 upgrade
#SBATCH --job-name=nem-bw-test
#SBATCH --partition=sixhour
#SBATCH --time=00:15:00
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

C=/home/a474r867/scratch/nemotron-eco-reasoner/nemotron-cuda.sif
M=/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848
D=/home/a474r867/scratch/nemotron-eco-reasoner/data/train_final_v2.jsonl
O=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/blackwell_test
P=/tmp/pip_$SLURM_JOB_ID
rm -rf $P 2>/dev/null; mkdir -p $O $P

JOBID=${SLURM_JOB_ID:-$$}
echo "=== BLACKWELL TEST job=$JOBID $(date) ==="
echo "=== Phase 1: Upgrade PyTorch for sm_120 (Blackwell) support ==="

apptainer exec --nv --bind "/dev/shm,/tmp" --env PYTHONUSERBASE=$P \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    $C bash << 'INSTALL_EOF'
set -euo pipefail

echo "--- Step 1: Current PyTorch version ---"
python3 -c 'import torch; print(f"PyTorch {torch.__version__} CUDA {torch.version.cuda}")'
python3 -c 'import torch; print(f"GPU: {torch.cuda.get_device_name(0)}"); print(f"CC: {torch.cuda.get_device_capability(0)}")'

echo "--- Step 2: Upgrade PyTorch to 2.6+ for Blackwell sm_120 ---"
pip install --user --no-cache-dir -q \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3

echo "--- Step 3: Verify new PyTorch ---"
python3 -c '
import torch
print(f"PyTorch {torch.__version__} CUDA {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
cc = torch.cuda.get_device_capability(0)
print(f"Compute Capability: {cc[0]}.{cc[1]}")
# Quick CUDA test
x = torch.randn(100, 100, device="cuda")
y = torch.matmul(x, x.T)
print(f"CUDA matmul test: OK ({y.shape})")
print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
'

echo "--- Step 4: Install training deps ---"
pip install --user --no-cache-dir -q --no-deps "transformers==4.57.6" "tokenizers==0.22.2" 2>&1 | tail -1
pip install --user --no-cache-dir -q --no-deps "datasets==4.3.0" bitsandbytes 2>&1 | tail -1
pip install --user --no-cache-dir -q --no-deps unsloth_zoo unsloth rich "trl==0.12.0" 2>&1 | tail -1
python3 -c 'import transformers, trl; print(f"tf:{transformers.__version__} trl:{trl.__version__}")'
python3 -c 'import unsloth; from trl import SFTTrainer; print("Unsloth + SFTTrainer: OK")'
INSTALL_EOF

if [ $? -ne 0 ]; then echo "INSTALL FAILED"; exit 1; fi

echo "=== Phase 2: Quick training test (10 steps) ==="
apptainer exec --nv --bind "/dev/shm,/tmp" \
    --env PYTHONUSERBASE=$P --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env TORCH_COMPILE_DISABLE=1 --env OMP_NUM_THREADS=1 \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env MODEL_PATH=$M --env DATA_PATH=$D --env OUT_PATH=$O \
    --env JOBID=$JOBID --env STEPS_PER_JOB=10 \
    $C env PYTHONUNBUFFERED=1 python3 -u << 'PYEOF'
import os, sys, types, torch, time, json
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

# MoE patch
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

ds=load_dataset("json",data_files={"train":DATA_PATH},split="train")
def fmt(ex):
    ms=ex.get("messages")
    if not ms: ms=[{"role":"user","content":ex.get("prompt","?")},{"role":"assistant","content":ex.get("answer","?")}]
    return {"text":tok.apply_chat_template(ms,tokenize=False,add_generation_prompt=False)}
ds=ds.map(fmt)

alloc=torch.cuda.memory_allocated()/1e9; reserved=torch.cuda.memory_reserved()/1e9
print(f"VRAM: {alloc:.1f}G alloc / {reserved:.1f}G reserved")

from trl import SFTTrainer,SFTConfig
trainer=SFTTrainer(model=model,args=SFTConfig(
    output_dir=OUT_PATH,max_steps=STEPS,per_device_train_batch_size=1,
    gradient_accumulation_steps=8,learning_rate=1e-4,max_seq_length=SEQ_LEN,
    warmup_steps=2,lr_scheduler_type="constant_with_warmup",
    logging_steps=1,save_steps=STEPS,save_total_limit=2,
    bf16=True,fp16=False,
    remove_unused_columns=False,report_to="none",
    dataloader_num_workers=0,packing=True),
    train_dataset=ds,processing_class=tok)

print(f"TRAINING BLACKWELL TEST: {STEPS} steps, seq={SEQ_LEN}, rank={RANK}, bf16=True")
train_t0=time.time()
trainer.train()
train_time=time.time()-train_t0

ckpt_dir=os.path.join(OUT_PATH,f"checkpoint-blackwell-test-{JOBID}")
trainer.save_model(ckpt_dir)
tok.save_pretrained(ckpt_dir)

elapsed=(time.time()-t0)/60
final_loss=trainer.state.log_history[-1].get("loss","?") if trainer.state.log_history else "?"
step_time=train_time/STEPS

print(f"\n{'='*60}")
print(f"BLACKWELL TEST RESULTS")
print(f"{'='*60}")
print(f"GPU: {gpu_name} ({vram_gb:.1f}GB)")
print(f"PyTorch: {torch.__version__} CUDA: {torch.version.cuda}")
print(f"Config: seq={SEQ_LEN} rank={RANK} bf16=True [{TAG}]")
print(f"Model load: {load_time:.1f}s")
print(f"Training: {STEPS} steps in {train_time:.1f}s ({step_time:.1f}s/step)")
print(f"Final loss: {final_loss}")
print(f"Total elapsed: {elapsed:.1f} min")
print(f"Checkpoint: {ckpt_dir}")
print(f"VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.1f}GB / {vram_gb:.1f}GB")
print(f"{'='*60}")
print("SUCCESS — BLACKWELL IS READY FOR PRODUCTION TRAINING")

manifest={"gpu":gpu_name,"vram_gb":round(vram_gb,1),"cc":f"{cc[0]}.{cc[1]}",
    "pytorch":torch.__version__,"cuda":torch.version.cuda,
    "seq":SEQ_LEN,"rank":RANK,"tag":TAG,"bf16":True,
    "steps":STEPS,"step_time_s":round(step_time,1),
    "final_loss":str(final_loss),"total_min":round(elapsed,1),
    "vram_peak_gb":round(torch.cuda.max_memory_allocated()/1e9,1)}
with open(os.path.join(OUT_PATH,f"blackwell-test-{JOBID}.json"),"w") as f:
    json.dump(manifest,f,indent=2)
PYEOF
echo "Exit: $?"
