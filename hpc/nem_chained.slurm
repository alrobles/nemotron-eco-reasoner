#!/bin/bash
# NEM_CHAINED — 250 steps + auto-resubmit hasta 500
#SBATCH --job-name=nem-chain
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/a474r867/scratch/nem_chain_%j.out
#SBATCH --error=/home/a474r867/scratch/nem_chain_%j.err
#SBATCH --signal=B:SIGUSR1@120

export APPTAINER_WRITABLE_TMPFS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_$SLURM_JOB_ID

C=/home/a474r867/scratch/nemotron-eco-reasoner/nemotron-blackwell.sif
M=/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848
D=${DATA_PATH:-/home/a474r867/scratch/nemotron-eco-reasoner/data/train_deterministic_v6.jsonl}
O=${OUT_PATH:-/home/a474r867/scratch/nemotron-eco-reasoner/outputs/deterministic_v6}
P=/tmp/pip_$SLURM_JOB_ID
TEMPLATE_PATH=/home/a474r867/scratch/nemotron-eco-reasoner/hpc/nem_chained.slurm
TARGET_TOTAL=${TARGET_TOTAL:-500}
STEPS_PER_JOB=${STEPS_PER_JOB:-250}
rm -rf $P 2>/dev/null; mkdir -p $O $P

JOBID=${SLURM_JOB_ID:-$$}
START_TS=$(date +%s)

echo "=== CHAINED job=$JOBID $(date) === target=$TARGET_TOTAL batch=$STEPS_PER_JOB"

apptainer exec --nv --bind "/dev/shm,/tmp" --env PYTHONUSERBASE=$P --env TRITON_CACHE_DIR=/tmp/triton_$SLURM_JOB_ID $C bash << 'INSTALL_EOF'
set -uo pipefail
pip install --user --no-cache-dir -q --no-deps "transformers==4.57.6" "tokenizers==0.22.2" 2>&1 | tail -1 || true
pip install --user --no-cache-dir -q --no-deps "datasets==4.3.0" bitsandbytes 2>&1 | tail -1 || true
pip install --user --no-cache-dir -q --no-deps unsloth_zoo unsloth rich "trl==0.12.0" 2>&1 | tail -1 || true
pip install --user --no-cache-dir -q --no-deps einops 2>&1 | tail -1 || { echo "FATAL: einops"; exit 1; }
WHEEL_DIR=/home/a474r867/scratch/wheels
pip install --user --no-deps $WHEEL_DIR/causal_conv1d-1.4.0-cp312-cp312-linux_x86_64.whl 2>&1 | tail -1
pip install --user --no-deps $WHEEL_DIR/mamba_ssm-2.2.4-cp312-cp312-linux_x86_64.whl 2>&1 | tail -1
SIF_TORCH_VER=$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)
VIRTUAL_ENV=${PYTHONUSERBASE:-$HOME/.local}
rm -rf $VIRTUAL_ENV/lib/python3*/site-packages/torch* $VIRTUAL_ENV/lib/python3*/site-packages/triton* $VIRTUAL_ENV/lib/python3*/site-packages/nvidia* $HOME/.local/lib/python3*/site-packages/torch* $HOME/.local/lib/python3*/site-packages/triton* $HOME/.local/lib/python3*/site-packages/nvidia* 2>/dev/null
ACTUAL_TORCH=$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)
echo "torch: $ACTUAL_TORCH"
python3 -c 'import torch,transformers,tokenizers,trl; print(f"tf:{transformers.__version__} tok:{tokenizers.__version__} trl:{trl.__version__}")'
python3 -c 'import unsloth; print("OK")'
INSTALL_EOF

[ $? -ne 0 ] && { echo "INSTALL FAILED"; exit 1; }

echo "=== Training batch ==="
apptainer exec --nv --bind "/dev/shm,/tmp" \
    --env PYTHONUSERBASE=$P --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env TORCH_COMPILE_DISABLE=1 --env OMP_NUM_THREADS=1 \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env MODEL_PATH=$M --env DATA_PATH=$D --env OUT_PATH=$O \
    --env JOBID=$JOBID --env TRITON_CACHE_DIR=/tmp/triton_$SLURM_JOB_ID \
    --env TARGET_TOTAL=$TARGET_TOTAL --env STEPS_PER_JOB=$STEPS_PER_JOB \
    --env TEMPLATE_PATH=$TEMPLATE_PATH \
    $C env PYTHONUNBUFFERED=1 python3 -u << 'PYEOF'
import os, re, sys, signal, types, torch, time, json, glob, subprocess
from datasets import load_dataset
os.environ['TORCH_COMPILE_DISABLE']='1'
t0=time.time()
MODEL_PATH=os.environ["MODEL_PATH"]; DATA_PATH=os.environ["DATA_PATH"]
OUT_PATH=os.environ["OUT_PATH"]; JOBID=os.environ["JOBID"]
TARGET_TOTAL=int(os.environ.get("TARGET_TOTAL","500"))
STEPS_PER_JOB=int(os.environ.get("STEPS_PER_JOB","250"))
TEMPLATE_PATH=os.environ.get("TEMPLATE_PATH","")

gpu_name=torch.cuda.get_device_name(0)
vram_gb=torch.cuda.get_device_properties(0).total_memory/1e9
bf16_ok=torch.cuda.is_bf16_supported()
cc=(torch.cuda.get_device_capability(0)[0]*10+torch.cuda.get_device_capability(0)[1])/10

SEQ_LEN=2048; RANK=32; GRAD_ACCUM=8; TAG="BEAST"
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

import types as _types
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
    module.moe=_types.MethodType(mp(module),module); patched+=1
print(f"MoE patched: {patched} layers")

model=FastLanguageModel.get_peft_model(model,r=RANK,lora_alpha=RANK,
    target_modules=['q_proj','k_proj','v_proj','o_proj','in_proj','out_proj','up_proj','down_proj'],
    use_gradient_checkpointing='unsloth')
model.print_trainable_parameters()

# Resume from checkpoint (via trainer.train(resume_from_checkpoint=...) so the
# optimizer/scheduler state and global_step are restored — wrapping the already
# PEFT-wrapped model with PeftModel.from_pretrained double-wraps and loses state)
ckpts=sorted(glob.glob(os.path.join(OUT_PATH,"checkpoint-*","adapter_model.safetensors")),key=os.path.getmtime)
trainer_ckpts=[os.path.dirname(c) for c in ckpts if 'sigusr1' not in c and os.path.exists(os.path.join(os.path.dirname(c),"trainer_state.json"))]
RESUME_CKPT=None
if trainer_ckpts:
    RESUME_CKPT=trainer_ckpts[-1]
    state=json.load(open(os.path.join(RESUME_CKPT,"trainer_state.json")))
    global_step=state.get("global_step",0)
    print(f"RESUME: {os.path.basename(RESUME_CKPT)} (step {global_step})")
else:
    global_step=0
    print("FRESH START — no prior checkpoint")

ds=load_dataset("json",data_files={"train":DATA_PATH},split="train")
def fmt(ex):
    ms=ex.get("messages")
    if not ms: ms=[{"role":"user","content":ex.get("prompt","?")},{"role":"assistant","content":ex.get("answer","?")}]
    return {"text":tok.apply_chat_template(ms,tokenize=False,add_generation_prompt=False)}
ds=ds.map(fmt)

alloc=torch.cuda.memory_allocated()/1e9; reserved=torch.cuda.memory_reserved()/1e9
print(f"VRAM: {alloc:.1f}G/{reserved:.1f}G")

# How many steps this batch? Don't exceed TARGET_TOTAL
batch_steps = min(STEPS_PER_JOB, TARGET_TOTAL - global_step)
target_step = global_step + batch_steps

from trl import SFTTrainer,SFTConfig
USE_BF16=bf16_ok
trainer=SFTTrainer(model=model,args=SFTConfig(
    output_dir=OUT_PATH,max_steps=target_step,per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,learning_rate=1e-4,max_seq_length=SEQ_LEN,
    warmup_steps=min(30,target_step//3),lr_scheduler_type="constant_with_warmup",
    logging_steps=10,save_steps=50,save_total_limit=10,
    bf16=USE_BF16,fp16=not USE_BF16,
    remove_unused_columns=False,report_to="none",
    dataloader_num_workers=0,packing=True),
    train_dataset=ds,processing_class=tok)
_trainer=trainer

# === ACCURACY EVAL CALLBACK (every 25 steps) ===
EVAL_DATA = os.environ.get("EVAL_DATA", "/home/a474r867/scratch/nemotron-eco-reasoner/data/kaggle_5k_train.jsonl")
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "25"))
EVAL_N = int(os.environ.get("EVAL_N", "20"))
import random as _random
_eval_ds = None
_eval_samples = None

class EvalCallback:
    def __init__(self):
        global _eval_ds, _eval_samples
        if os.path.exists(EVAL_DATA):
            _eval_ds = load_dataset("json", data_files={"train": EVAL_DATA}, split="train")
            _eval_samples = _eval_ds if len(_eval_ds) <= EVAL_N else _eval_ds.select(_random.sample(range(len(_eval_ds)), EVAL_N))
            print(f"EVAL: loaded {len(_eval_ds)} puzzles, will sample {EVAL_N} every {EVAL_EVERY} steps")
        else:
            print(f"EVAL: data not found at {EVAL_DATA}, skipping accuracy checks")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % EVAL_EVERY != 0:
            return
        if _eval_ds is None:
            return
        try:
            self._run_eval(state, **kwargs)
        except Exception as e:
            print(f"EVAL step={state.global_step}: skipped ({type(e).__name__}: {e})")

    def _run_eval(self, state, **kwargs):

        model_to_eval = kwargs.get("model", model)
        _trainer = kwargs.get("trainer", trainer)
        tok_ev = _trainer.processing_class

        # Switch to eval mode
        training_mode = model_to_eval.training
        model_to_eval.eval()

        correct = 0
        total = 0
        t_start = time.time()

        samples = _eval_samples if _eval_samples is not None else _eval_ds.select(_random.sample(range(len(_eval_ds)), EVAL_N))

        for ex in samples:
            prompt = ex.get("prompt", "")
            expected = ex.get("answer", "").strip()

            exp_match = re.search(r"\\boxed\{([^}]+)\}", expected)
            expected_val = exp_match.group(1).strip() if exp_match else expected

            messages = [
                {"role": "system", "content": "You are an AI assistant specialized in logical reasoning puzzles. Think step by step and place your final answer inside \\boxed{...}."},
                {"role": "user", "content": prompt}
            ]
            text = tok_ev.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok_ev(text, return_tensors="pt").to(model_to_eval.device)

            with torch.no_grad():
                outputs = model_to_eval.generate(**inputs, max_new_tokens=512,
                    do_sample=False, pad_token_id=tok_ev.pad_token_id or tok_ev.eos_token_id)
            response = tok_ev.decode(outputs[0], skip_special_tokens=True)
            if response.startswith(text):
                response = response[len(text):]

            pred_match = re.search(r"\\boxed\{([^}]+)\}", response)
            predicted = pred_match.group(1).strip() if pred_match else None
            if predicted == expected_val:
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0
        elapsed = time.time() - t_start
        print(f"EVAL step={state.global_step}: acc={acc:.3f} ({correct}/{total}) in {elapsed:.1f}s")

        # Switch back to training mode
        if training_mode:
            model_to_eval.train()

# Instantiate callback
_eval_cb = EvalCallback()
trainer.add_callback(_eval_cb)

print(f"TRAINING [{TAG}]: step {global_step} -> {target_step} (batch={batch_steps}, target={TARGET_TOTAL})")
trainer.train(resume_from_checkpoint=RESUME_CKPT)

# Save checkpoint at end of batch
ckpt_dir=os.path.join(OUT_PATH,f"checkpoint-{JOBID}-step{target_step}")
trainer.save_model(ckpt_dir)
tok.save_pretrained(ckpt_dir)

elapsed=(time.time()-t0)/60
final_loss='?'
for entry in reversed(trainer.state.log_history):
    if 'loss' in entry:
        final_loss=entry['loss']; break

print(f"BATCH DONE {elapsed:.1f}min steps={global_step}->{target_step} loss={final_loss}")

# Will resubmit from bash level — just report status
if target_step < TARGET_TOTAL:
    print(f"NEED_RESUBMIT: target_step={target_step} target_total={TARGET_TOTAL}")
else:
    print(f"TARGET_REACHED: {target_step}/{TARGET_TOTAL} steps")

manifest={"gpu":gpu_name,"vram_gb":round(vram_gb,1),"cc":cc,"bf16":USE_BF16,
    "seq":SEQ_LEN,"rank":RANK,"tag":TAG,"jobid":JOBID,
    "steps_start":global_step,"steps_done":target_step,"target_total":TARGET_TOTAL,
    "elapsed_min":round(elapsed,1),"final_loss":final_loss}
with open(os.path.join(OUT_PATH,f"manifest-{JOBID}.json"),"w") as f: json.dump(manifest,f,indent=2)
# Save GPU GRES hint for auto-resubmit
gres_map = {"NVIDIA RTX PRO 6000":"pro6000", "NVIDIA A100":"a100", "AMD Instinct MI210":"mi210", "NVIDIA A40":"a40", "NVIDIA L40":"l40"}
gres_hint = gres_map.get(gpu_name.split("Blackwell")[0].strip(), "pro6000")
with open(f"/tmp/gpu_gres_{JOBID}", "w") as gf: gf.write(gres_hint)
print("SUCCESS")
PYEOF
PY_EXIT=$?
echo "Exit: $PY_EXIT"

# AUTO-RESUBMIT at bash level (sbatch available here)
if [ $PY_EXIT -eq 0 ]; then
    NEED=$(grep 'NEED_RESUBMIT' /home/a474r867/scratch/nem_chain_${JOBID}.out 2>/dev/null | tail -1)
    if [ -n "$NEED" ]; then
        TARGET_STEP=$(echo "$NEED" | grep -oP 'target_step=\K[0-9]+')
        TARGET_TOTAL_VAL=$(echo "$NEED" | grep -oP 'target_total=\K[0-9]+')
        REMAINING=$((TARGET_TOTAL_VAL - TARGET_STEP))
        NEXT_BATCH=$(( STEPS_PER_JOB < REMAINING ? STEPS_PER_JOB : REMAINING ))
        echo "→ AUTO-RESUBMIT: ${REMAINING} steps remaining, submitting ${NEXT_BATCH}"
        GRES_HINT=$(cat /tmp/gpu_gres_${JOBID} 2>/dev/null || echo pro6000)
        NEW_JID=$(sbatch --job-name=nem-chain --gres=gpu:${GRES_HINT}:1 --mem=48G             --export=ALL,TARGET_TOTAL=${TARGET_TOTAL_VAL},STEPS_PER_JOB=${NEXT_BATCH}             ${TEMPLATE_PATH} 2>/dev/null | grep -oP '\d+')
        echo "→ RESUBMITTED: job ${NEW_JID} (steps ${TARGET_STEP}→$((TARGET_STEP + NEXT_BATCH)))"
    fi
fi
