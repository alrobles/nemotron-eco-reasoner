# EcoReasoner Training Plan — KU HPC Cluster

> Version: 1.1 | Date: 2026-06-22 | **Status: COMPLETED**
> Based on 3rd place Kaggle solution (0.900) + VCDAD silver techniques
> See [TRAINING_RESULTS.md](TRAINING_RESULTS.md) for final ablation results.

---

## 1. Objective

Fine-tune Nemotron-3-Nano-30B-A3B-BF16 on `ecoreasoner-cot-20k` (14,156 filtered
examples) for ecological reasoning. Target: a LoRA adapter that produces structured
`<think>...</think>` reasoning traces for ecological methods (MaxEnt, BRT, GLM,
occupancy modeling, phylogenetics, etc.).

---

## 2. Cluster Resource Strategy

### 2.1 GPU Selection Priority

```
Priority  GPU       VRAM   Nodes  Reason
-------   -------   -----  -----  ------
1st       A100      80 GB    6    BF16 native, torch 2.5.1+cu121 compatible, fastest
2nd       MI210     64 GB   27    BF16 via ROCm, most abundant (needs ROCm torch)
3rd       L40       48 GB    1    CUDA compatible, enough for QLoRA
4th       PRO6000   95 GB    3    Blackwell — needs torch >= 2.6 (cu128 nightly)
5th       V100      32 GB   14    fp16 only (no BF16), too small for full model
```

**Current default: A100** — best balance of compatibility, speed, and VRAM.

PRO6000 Blackwell (95GB) is ideal but requires PyTorch nightly with sm_100 kernels.
TODO: Create a Blackwell-ready venv with `torch nightly+cu128` for future runs.

### 2.2 Slurm Configurations

#### A100 (recommended — immediate)
```bash
sbatch hpc/eco_train_v2.slurm
# Default: 1x A100 (80GB), sixhour partition, 5h55m
```

#### A100 multi-GPU (faster, if available)
```bash
sbatch --gres=gpu:a100:2 --ntasks=2 --ntasks-per-node=2 hpc/eco_train_v2.slurm
# 2x A100 DDP — halves training time
```

#### MI210 (fallback — needs ROCm venv)
```bash
# First: create ROCm venv with torch+rocm
sbatch --gres=gpu:mi210:2 --ntasks=2 --ntasks-per-node=2 \
  --export=ALL,VENV_PATH=/home/a474r867/scratch/nemotron-rocm-venv \
  hpc/eco_train_v2.slurm
```

#### PRO6000 Blackwell (future — needs torch nightly)
```bash
# After creating Blackwell venv:
sbatch --gres=gpu:pro6000:1 --ntasks=1 --ntasks-per-node=1 \
  --export=ALL,VENV_PATH=/home/a474r867/scratch/nemotron-blackwell-venv \
  hpc/eco_train_v2.slurm
```

### 2.3 Storage Layout

```
/home/a474r867/scratch/                          # 501 TB NFS
  nemotron-eco-reasoner/                         # repo clone
    data/ecoreasoner_train.jsonl                 # 14,156 filtered examples
    outputs/eco_v2_twophase/                     # checkpoints + adapters
  nemotron-model-cache/                          # 173 GB cached base model
    models--nvidia--NVIDIA-Nemotron-3-Nano-.../
      snapshots/cbd3fa9f.../                     # BF16 weights
  ecocoder-training/venv/                        # torch 2.5.1+cu121 (A100/V100)
  eco_train_v2_*.out                             # Slurm job logs
```

---

## 3. Model Configuration

### 3.1 Base Model
- **nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16**
- Hybrid: 23 Mamba-2 + 23 MoE + 6 Attention layers (52 total)
- 30B total / ~3.5B active parameters
- Cached at `/home/a474r867/scratch/nemotron-model-cache/`

### 3.2 LoRA Configuration (from 3rd place + VCDAD)

```python
LoraConfig(
    r=32,
    lora_alpha=32,               # alpha/rank = 1.0
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "up_proj", "down_proj",                     # MLP/MoE
        "in_proj", "out_proj",                      # Mamba-2 SSM
        "gate_proj",                                # MoE gating
        "lm_head",                                  # output head (3rd place)
    ],
)
# Post-init: cast LoRA params to fp32 (LORA_FP32=1)
```

---

## 4. Training Strategy

### 4.1 Two-Phase Training (VCDAD silver approach)

```
Phase 1 — "Train" (aggressive exploration)
  Steps:     0 → 350 (70% of total)
  LR:        2e-4
  Schedule:  linear decay
  Clipping:  OFF (max_grad_norm=1e9)
  NEFTune:   alpha=5.0

Phase 2 — "Nudge" (precise refinement)
  Steps:     350 → 500 (30% of total)
  LR:        5e-6 (40x smaller)
  Schedule:  cosine
  Clipping:  ON (max_grad_norm=1.0)
  NEFTune:   alpha=5.0

Transition: TwoPhaseCallback switches at step 350
```

### 4.2 Key Techniques

| Technique | Source | Implementation |
|-----------|--------|----------------|
| Completion-only loss | 3rd place | `COMPLETION_ONLY=1` — mask prompt tokens |
| LoRA fp32 | 3rd place | `LORA_FP32=1` — higher precision for adapter weights |
| lm_head LoRA | 3rd place | `INCLUDE_LM_HEAD=1` — adapt output head |
| MoE gradient-sum tying | 3rd place | `MoEGradTieCallback` — sum gradients across experts |
| NEFTune noise | VCDAD silver | `neftune_noise_alpha=5.0` — embedding robustness |
| Stratified sampling | VCDAD/galaxy2025 | Round-robin by ecological method |
| No packing | VCDAD silver | `packing=False` — prevent cross-sample interference |
| Pure BF16 base | all top solutions | No quantization — precise trace reproduction |

### 4.3 Dataset

```
Source:     alrobles/ecoreasoner-cot-20k (HuggingFace)
Filtered:   14,156 / 20,000 examples (70.8%)
Filter:     require <think> tags (min 200 chars), min assistant 500 chars
Format:     messages: [{role: user, content: ...}, {role: assistant, content: ...}]

Method Distribution (top 15):
  other: 9,680 | maxent: 1,464 | occupancy: 368 | bayesian: 368
  network_analysis: 333 | glm: 278 | phylogenetics: 246
  hmm: 182 | brt: 182 | logistic_regression: 181
  linear_regression: 168 | gam: 161 | pca: 158 | edna: 154
  random_forest: 147
```

---

## 5. Training Runs

### Run 1: Baseline (current — job 23045706, FAILED on PRO6000)
- **Issue**: PRO6000 Blackwell needs PyTorch nightly (sm_100 kernels)
- **Fix**: Resubmit on A100

### Run 2: A100 baseline (next)
```bash
# Submit:
ku-hpc raw: cd /home/a474r867/scratch/nemotron-eco-reasoner && \
  sbatch --gres=gpu:a100:1 --ntasks=1 --ntasks-per-node=1 hpc/eco_train_v2.slurm

# Monitor:
ku-hpc raw: tail -30 /home/a474r867/scratch/eco_train_v2_<JOBID>.out

# Expected runtime: ~4-5 hours (500 steps, A100 80GB)
```

### Run 3: A100 multi-GPU (if Run 2 succeeds)
```bash
sbatch --gres=gpu:a100:2 --ntasks=2 --ntasks-per-node=2 \
  --export=ALL,TARGET_TOTAL=1000,SEQ_LEN=8192 hpc/eco_train_v2.slurm
# Double steps + full seq_len for best quality
```

### Run 4: Blackwell PRO6000 (after venv upgrade)
```bash
# Create Blackwell venv:
ku-hpc raw: python3 -m venv /home/a474r867/scratch/nemotron-blackwell-venv && \
  source /home/a474r867/scratch/nemotron-blackwell-venv/bin/activate && \
  pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 && \
  pip install peft trl datasets transformers accelerate

# Then submit with new venv
```

---

## 6. Monitoring & Recovery

### Check job status
```bash
ku-hpc raw: squeue -u a474r867 --format="%i %T %j %P %R %M" --noheader
```

### Check training progress
```bash
ku-hpc raw: tail -30 /home/a474r867/scratch/eco_train_v2_<JOBID>.out
```

### Check for errors
```bash
ku-hpc raw: tail -20 /home/a474r867/scratch/eco_train_v2_<JOBID>.err
```

### Auto-resubmit
The slurm script auto-resubmits if `NEED_RESUBMIT` appears in output
(i.e., `STEPS_PER_JOB < TARGET_TOTAL`). Default: 250 steps per job,
500 total = 2 jobs chained.

### Recovery from failures
```bash
# Resume from last checkpoint:
ku-hpc raw: ls /home/a474r867/scratch/nemotron-eco-reasoner/outputs/eco_v2_twophase/checkpoint-*/
# The trainer auto-detects and resumes from latest checkpoint
```

---

## 7. Post-Training Pipeline

### 7.1 Evaluate
```bash
# Load adapter and run inference on held-out examples
ku-hpc raw: python3 scripts/evaluate_adapter.py \
  --model /home/a474r867/scratch/nemotron-model-cache/.../cbd3fa9f... \
  --adapter outputs/eco_v2_twophase/checkpoint-500/ \
  --data data/ecoreasoner_eval.jsonl
```

### 7.2 Publish to HuggingFace
```bash
# Push adapter to HF
python3 -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16')
model = PeftModel.from_pretrained(model, 'outputs/eco_v2_twophase/checkpoint-500')
model.push_to_hub('alrobles/ecoreasoner-nemotron-v1')
"
```

### 7.3 Deploy for inference
```bash
# Via Ollama on HPC (existing infrastructure):
# Convert adapter to GGUF, load in Ollama
# Or: vLLM serve with --lora-modules
```

---

## 8. Environment Setup Checklist

### A100 (ready now)
- [x] Venv: `/home/a474r867/scratch/ecocoder-training/venv/` (torch 2.5.1+cu121)
- [x] Model: cached at `/home/a474r867/scratch/nemotron-model-cache/` (173GB)
- [x] Data: `data/ecoreasoner_train.jsonl` (14,156 records)
- [x] Scripts: `hpc/eco_train_v2.py` + `hpc/eco_train_v2.slurm`

### MI210 (needs ROCm venv)
- [ ] Create venv with `torch+rocm62` (ROCm 6.2)
- [ ] Install: peft, trl, datasets, transformers, accelerate
- [x] Model: same cache
- [x] Data: same

### PRO6000 Blackwell (needs nightly torch)
- [ ] Create venv with `torch nightly+cu128`
- [ ] Install: peft, trl, datasets, transformers, accelerate
- [x] Model: same cache
- [x] Data: same

---

## 9. Hermes Commands Quick Reference

```bash
# Submit A100 job:
ku-hpc raw: cd /home/a474r867/scratch/nemotron-eco-reasoner && \
  sbatch --gres=gpu:a100:1 --ntasks=1 --ntasks-per-node=1 hpc/eco_train_v2.slurm

# Check queue:
ku-hpc raw: squeue -u a474r867 --format="%i %T %j %R" --noheader

# Check training log:
ku-hpc raw: tail -30 /home/a474r867/scratch/eco_train_v2_<JOBID>.out

# Check GPU usage:
ku-hpc raw: scontrol show job <JOBID> | grep -E "NodeList|Gres"

# Cancel job:
ku-hpc raw: scancel <JOBID>

# Sync repo:
ku-hpc raw: cd /home/a474r867/scratch/nemotron-eco-reasoner && git pull

# List checkpoints:
ku-hpc raw: ls -la /home/a474r867/scratch/nemotron-eco-reasoner/outputs/eco_v2_twophase/
```
