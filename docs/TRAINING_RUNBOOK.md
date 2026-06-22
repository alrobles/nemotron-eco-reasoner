# EcoReasoner — Phase 2: Training Runbook

Status: ✅ Phase 1 (CoT generation) complete — 20K papers on HF
        🚀 Phase 2 (Training) — Devin orchestrates, Hermes supports

---

## 1. Dataset

**Source:** `alrobles/ecoreasoner-cot-20k` on HuggingFace
- File: `ecoreasoner_cot_20k.jsonl` (157 MB, 20,000 papers)
- Format: `{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`
- Compatible with `nem_q6000_train.slurm` and all Nemotron training scripts
- 6 sources: PubMed, GBIF, arXiv, bioRxiv, ecoevorxiv, Math Ecology

**Download command (on cluster):**
```bash
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('alrobles/ecoreasoner-cot-20k', 'ecoreasoner_cot_20k.jsonl',
                repo_type='dataset', local_dir='/home/a474r867/scratch/cot_generation/',
                token=open('/home/a474r867/.huggingface/token').read().strip())
"
```

---

## 2. Training Scripts (repo: alrobles/nemotron-eco-reasoner)

| Script | GPU | Method | VRAM | Max Seq | Notes |
|--------|-----|--------|------|---------|-------|
| `hpc/nem_q6000_train.slurm` | Q6000 | QLoRA 4-bit | 24 GB | 512 | Conservative, tested |
| `hpc/nem_mi210_train.slurm` | MI210 | LoRA BF16 | 64 GB | 1024 | 2× MI210, ROCm |
| `hpc/nem_mi210_multinode.slurm` | MI210 | LoRA BF16 | 64 GB | 2048 | Multi-node DDP |
| `hpc/nem_tied.slurm` | Multi | MoE Tied | Any | 2048 | 4-GPU, best quality |
| `hpc/nem_tied_1gpu.slurm` | Any | Tied 1GPU | 24+ GB | 1024 | Staged freeze |

**Base model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
**Container:** `nemotron-blackwell.sif` (CUDA) or `nemotron-mi210-rocm.def` (ROCm)
**Model cache:** `/home/a474r867/scratch/nemotron-model-cache/`

---

## 3. Recommended Training Grid (Phase 2A)

Launch 6-10 jobs in parallel with different hyperparameters:

| Job | GPU | Script | Seq | LR | Alpha | Steps | Notes |
|-----|-----|--------|-----|-----|-------|-------|-------|
| 1 | Q6000 ×1 | nem_q6000 | 512 | 1e-4 | 32 | 500 | QLoRA baseline |
| 2 | MI210 ×2 | nem_mi210 | 1024 | 1e-4 | 32 | 500 | BF16, fast |
| 3 | MI210 ×2 | nem_mi210 | 1024 | 5e-5 | 16 | 500 | Lower LR |
| 4 | MI210 ×2 | nem_mi210 | 2048 | 1e-4 | 32 | 500 | Long context |
| 5 | A100 ×3 | nem_mi210 | 2048 | 1e-4 | 32 | 500 | Best GPU |
| 6 | PRO6000 | nem_q6000* | 2048 | 1e-4 | 32 | 500 | Blackwell |

*PRO6000 uses Q6000 script but with BF16 instead of QLoRA (needs container tweak)

### Submission template:
```bash
# Q6000 baseline
sbatch --export=ALL,DATA_PATH=/home/a474r867/scratch/cot_generation/ecoreasoner_cot_20k.jsonl,OUT_PATH=/home/a474r867/scratch/nemotron_outputs/q6k_lr1e4,TARGET_TOTAL=500,LR=1e-4,SEQ_LEN=512 hpc/nem_q6000_train.slurm

# MI210 BF16
sbatch --export=ALL,DATA_PATH=...,OUT_PATH=...,TARGET_TOTAL=500,LR=1e-4,SEQ_LEN=1024 hpc/nem_mi210_train.slurm
```

**Key env vars:**
- `DATA_PATH` — path to JSONL dataset
- `OUT_PATH` — output directory for checkpoints
- `TARGET_TOTAL` — total training steps (500 default)
- `LR` — learning rate (1e-4 default)
- `SEQ_LEN` — max sequence length
- `LORA_ALPHA` — LoRA alpha (32 default)
- `GRAD_ACCUM` — gradient accumulation steps

---

## 4. Monitoring

### Check job status:
```bash
squeue -u a474r867 --format="%i|%j|%T|%M|%N" | grep nem
```

### Check training progress:
```bash
# Loss curve (from trainer logs)
grep "loss" /home/a474r867/scratch/nem_q6k_*.out | tail -5

# Checkpoint count
ls /home/a474r867/scratch/nemotron_outputs/*/checkpoint-*/trainer_state.json | wc -l
```

### Evaluate checkpoint:
```bash
python3 scripts/eval_checkpoint.py --adapter <path>/checkpoint-500 --data data/kaggle_classified.jsonl
```

---

## 5. Success Criteria

- ✅ Training completes without NaN/OOM
- ✅ Loss decreases monotonically
- ✅ Checkpoint evaluation > 0.5 accuracy on Kaggle categories
- ✅ LoRA adapter saved and portable (pure torch fallback)
- ✅ Adapter size < 100 MB (rank=32 constraint)

---

## 6. Fallback / Recovery

If training fails:
1. **OOM:** Reduce `SEQ_LEN` or `GRAD_ACCUM`
2. **NaN loss:** Reduce `LR` to 5e-5 or 1e-5
3. **Container crash:** Verify `nemotron-blackwell.sif` exists, rebuild if needed
4. **Data load error:** Verify JSONL format with `python3 -c "import json; [json.loads(l) for l in open('<file>')]"`

---

## 7. Next Steps (Phase 3)

After best checkpoint is selected:
1. Merge LoRA adapters (Devin's task)
2. Evaluate on held-out ecological QA
3. Generate ToT dataset (k=3, judge, <tree> format)
4. Train ToT model
5. Explore Mamba-3 + SSM for long-context reasoning

---

**Last updated:** 2026-06-21 — Hermes Agent (Phase 2 kickoff)
