#!/bin/bash
# NEM_TIED_1GPU — v14 + MoE weight tying + STAGED EXPERT FREEZE on a single PRO6000.
# A/B partner to the 4-GPU tied run: same parameter-sharing tie, but here we turn ON
# the staged expert-LoRA freeze (EXPERT_FREEZE_FRAC) so attention adapts first and the
# experts join after warmup ("router/expert freeze" lever). World size = 1 (no DDP).
#SBATCH --job-name=nem-tfrz
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --output=/home/a474r867/scratch/nem_tfrz_%j.out
#SBATCH --error=/home/a474r867/scratch/nem_tfrz_%j.err

export PYTHONUNBUFFERED=1
export APPTAINER_WRITABLE_TMPFS=1

R=/home/a474r867/scratch/nemotron-eco-reasoner
C=$R/nemotron-blackwell.sif
M=/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848
DATA=${DATA_PATH:-$R/data/train_deterministic_v14.jsonl}
OUT=${OUT_PATH:-$R/outputs/tied_v14_freeze}
TEMPLATE_PATH=$R/hpc/nem_tied_1gpu.slurm
TRAIN_PY=$R/hpc/tied_train.py

TARGET_TOTAL=${TARGET_TOTAL:-500}
STEPS_PER_JOB=${STEPS_PER_JOB:-250}
SEQ_LEN=${SEQ_LEN:-3072}
GRAD_ACCUM=${GRAD_ACCUM:-8}
LR=${LR:-1e-4}
LR_SCHED=${LR_SCHED:-constant_with_warmup}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-32}
EXPERT_FREEZE_FRAC=${EXPERT_FREEZE_FRAC:-0.25}

JOBID=${SLURM_JOB_ID:-$$}
P=$R/.pip_tfrz_$JOBID
rm -rf $P 2>/dev/null; mkdir -p $P
mkdir -p "$OUT"

# single-process rendezvous (world=1) — harmless but keeps tied_train.py env-reads happy
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29518
export NCCL_DEBUG=WARN

echo "=== TIED-FREEZE TRAIN job=$JOBID $(date) ==="
echo "  nodes=$SLURM_JOB_NUM_NODES tasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT"
echo "  data=$DATA seq=$SEQ_LEN lr=$LR sched=$LR_SCHED grad_accum=$GRAD_ACCUM target=$TARGET_TOTAL freeze=$EXPERT_FREEZE_FRAC"
echo "  nodelist=$SLURM_JOB_NODELIST"

# ----------------------------------------------------------------- 1) deps (once)
apptainer exec --nv --bind "/dev/shm,/tmp,$R" --env PYTHONUSERBASE=$P $C bash << 'INSTALL_EOF'
set -uo pipefail
pip install --user --no-cache-dir -q --no-deps "transformers==4.57.6" "tokenizers==0.22.2" 2>&1 | tail -1 || true
pip install --user --no-cache-dir -q --no-deps "datasets==4.3.0" "trl==0.24.0" "peft==0.18.0" 2>&1 | tail -1 || true
pip install --user --no-cache-dir -q --no-deps einops liger_kernel accelerate 2>&1 | tail -1 || { echo "FATAL: deps"; exit 1; }
VENV=${PYTHONUSERBASE:-$HOME/.local}
rm -rf $VENV/lib/python3*/site-packages/torch* $VENV/lib/python3*/site-packages/triton* $VENV/lib/python3*/site-packages/nvidia* 2>/dev/null
rm -rf $VENV/lib/python3*/site-packages/mamba_ssm* $VENV/lib/python3*/site-packages/causal_conv1d* 2>/dev/null
python3 -c 'import trl, peft, transformers, accelerate; print("OK trl",trl.__version__,"peft",peft.__version__,"acc",accelerate.__version__)'
INSTALL_EOF
[ $? -ne 0 ] && { echo "INSTALL FAILED"; exit 1; }

# --------------------------------------------- 2) patch rmsnorm + prewarm module cache (once)
apptainer exec --nv --bind "/dev/shm,/tmp,$M,$R" \
    --env PYTHONUSERBASE=$P --env M="$M" \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 $C python3 - <<'PYEOF'
import os, glob, shutil, torch
M_PATH = os.path.join(os.environ["M"], "modeling_nemotron_h.py")
OLD = 'except ImportError:\n    raise ImportError("mamba-ssm is required by the Mamba model but cannot be imported")'
NEW = 'except ImportError:\n    def rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5, group_size=None, norm_before_gate=False):\n        dt = x.dtype\n        x = x.float()\n        if z is not None and not norm_before_gate:\n            x = x * torch.nn.functional.silu(z.float())\n        if group_size is not None and group_size != x.shape[-1]:\n            s = x.shape\n            xg = x.view(*s[:-1], s[-1] // group_size, group_size)\n            xg = xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + eps)\n            x = x.view(s)\n        else:\n            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)\n        out = x * weight.float()\n        if bias is not None:\n            out = out + bias.float()\n        if z is not None and norm_before_gate:\n            out = out * torch.nn.functional.silu(z.float())\n        return out.to(dt)'
src = open(M_PATH).read()
if OLD in src:
    open(M_PATH, "w").write(src.replace(OLD, NEW)); print("patched rmsnorm_fn")
else:
    print("rmsnorm_fn already patched (or pattern absent)")
for d in glob.glob(os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/cbd3fa9f*")):
    shutil.rmtree(d, ignore_errors=True)
from transformers import AutoConfig, AutoModelForCausalLM
cfg = AutoConfig.from_pretrained(os.environ["M"], trust_remote_code=True)
try:
    from accelerate import init_empty_weights
    with init_empty_weights():
        AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    print("prewarmed dynamic module cache")
except Exception as e:
    print("prewarm note:", e)
PYEOF
[ $? -ne 0 ] && { echo "PATCH/PREWARM FAILED"; exit 1; }

# ----------------------------------------------------------------- 3) launch (single GPU)
srun --kill-on-bad-exit=1 apptainer exec --nv --bind "/dev/shm,/tmp,$M,$R" \
    --env PYTHONUSERBASE=$P \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env PYTHONUNBUFFERED=1 --env TORCH_COMPILE_DISABLE=1 \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env MASTER_ADDR=$MASTER_ADDR --env MASTER_PORT=$MASTER_PORT --env NCCL_DEBUG=$NCCL_DEBUG \
    --env M="$M" --env DATA="$DATA" --env OUT="$OUT" \
    --env TARGET_TOTAL=$TARGET_TOTAL --env STEPS_PER_JOB=$STEPS_PER_JOB \
    --env SEQ_LEN=$SEQ_LEN --env GRAD_ACCUM=$GRAD_ACCUM --env LR=$LR --env LR_SCHED=$LR_SCHED \
    --env LORA_RANK=$LORA_RANK --env LORA_ALPHA=$LORA_ALPHA --env EXPERT_FREEZE_FRAC=$EXPERT_FREEZE_FRAC \
    $C python3 "$TRAIN_PY"
PY_EXIT=$?
echo "Exit: $PY_EXIT"

rm -rf $P 2>/dev/null

# ----------------------------------------------------------------- 4) auto-resubmit
if [ $PY_EXIT -eq 0 ]; then
    NEED=$(grep 'NEED_RESUBMIT' /home/a474r867/scratch/nem_tfrz_${JOBID}.out 2>/dev/null | tail -1)
    if [ -n "$NEED" ]; then
        TT=$(echo "$NEED" | grep -oP 'target_total=\K[0-9]+')
        echo "AUTO-RESUBMIT target=$TT"
        NEW_JID=$(sbatch --export=ALL,TARGET_TOTAL=${TT},STEPS_PER_JOB=${STEPS_PER_JOB},DATA_PATH=${DATA},OUT_PATH=${OUT},SEQ_LEN=${SEQ_LEN},GRAD_ACCUM=${GRAD_ACCUM},LR=${LR},LR_SCHED=${LR_SCHED},LORA_RANK=${LORA_RANK},LORA_ALPHA=${LORA_ALPHA},EXPERT_FREEZE_FRAC=${EXPERT_FREEZE_FRAC} ${TEMPLATE_PATH} 2>/dev/null | grep -oP '\d+')
        echo "RESUBMITTED: job ${NEW_JID}"
    fi
fi
exit $PY_EXIT
