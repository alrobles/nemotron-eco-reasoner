# Post-Hackathon Analysis: Lessons from Winning Solutions + Training Plan

> Generated: 2026-06-22. Based on review of silver-medal solutions, the
> `ecoreasoner-cot-20k` dataset, and existing nemotron-eco-reasoner infrastructure.

---

## 1. Competition Results Summary

| Team | Rank | Score | Key Technique |
|------|------|-------|---------------|
| **1st place** | 1/4163 | ~0.90 | Not yet published (likely GRPO/RL on top of SFT) |
| **VCDAD** (DaoyuanLi2816) | 65/4163 (Silver, Top 1.6%) | ~0.83 | Two-phase SFT: Train→Nudge |
| **galaxy2025** (galaxywk223) | 167/4163 (Silver) | 0.860 | Teacher-CoT distillation + stratified sampler |
| **msusol** | 2789/4163 | 0.67 | Per-expert MoE LoRA (PEFT-compatible) |
| **alrobles** (us) | — | 0.67 (v8 best) | Deterministic solvers + verified CoT |
| **InfoSage05** | — | — | GRPO/RL (single-GPU + 2-GPU split) |

**Gap to top: 0.67 → 0.90 = +0.23 points.** The top solutions used fundamentally
different approaches from ours.

---

## 2. Key Lessons from Winning Solutions

### 2.1 Two-Phase Training (VCDAD — Silver Top 1.6%)

The most detailed and actionable solution. Their recipe:

```
Phase 1 ("Train"): lr=2e-4, linear decay, NO gradient clipping (max_grad_norm=1e9)
Phase 2 ("Nudge"): lr=5e-6 (40× smaller), cosine decay, gradient clipping ON (1.0)
```

| Insight | Detail |
|---------|--------|
| **Architecture-aware LoRA** | Target `in_proj, out_proj` (Mamba-2 SSM) + `q/k/v/o_proj` (attention) + `up/down_proj` (MLP/MoE) — NOT just attention |
| **NEFTune noise** | `neftune_noise_alpha=5.0` in both phases — fights adversarial puzzle flavor text |
| **Stratified sampler** | Round-robin across puzzle types per effective batch — prevents gradient direction jitter |
| **Format contract** | Strip upstream `\boxed{}`, reattach official answer: decouples reasoning from answer |
| **Pure BF16** | No quantization — traces must be reproduced precisely |
| **No packing** | `packing=False` — prevents cross-sample token interference |
| **seq_len=8192** | Match eval context exactly |

### 2.2 Teacher-CoT Distillation (galaxy2025 — Silver #167)

- Use a strong teacher model (e.g. DeepSeek R1) to generate visible solution traces
- Filter: keep only records whose normalized teacher answer matches ground truth
- Adapter fusion (SVD, TIES, weight averaging) **all underperformed** vs single adapter
- Stratified sampler critical for balanced category representation

### 2.3 GRPO/Reinforcement Learning (InfoSage05)

- **GRPO** (Group Relative Policy Optimization): generate 4 drafts per puzzle, learn from the best
- Custom per-category scoring (numeric margins for math, letter similarity for ciphers)
- Format bonus (+0.1 for `<think>` + `\boxed{}`)
- RAFT++ rejection filter: skip too-easy and too-hard puzzles
- **This is likely what 1st place used on top of SFT** (gap from SFT-only ~0.83 → 0.90)

### 2.4 Per-Expert MoE LoRA (msusol)

- Standard LoRA misses MoE expert layers (Unsloth fused format incompatible with PEFT evaluator)
- Per-expert LoRA in PEFT-compatible format: `mixer.experts.{j}.up_proj.lora_A.weight`
- Our `tied_train.py` already handles this via weight tying — we were on the right track

---

## 3. What We Had Right vs What We Missed

### Right ✓
- Deterministic solvers for hard categories (73-78% coverage)
- MoE weight tying (`tied_train.py`) — similar idea to per-expert LoRA
- BF16 training on MI210 (no quantization noise)
- LoRA rank 32 (same as winners)
- Target modules included `up_proj, down_proj, in_proj, out_proj` (architecture-aware)

### Missed ✗
- **Two-phase training** (Train→Nudge) — we used single-phase
- **NEFTune noise** — not in our training scripts
- **Stratified sampling** — we used default random
- **GRPO/RL on top of SFT** — never attempted
- **Gradient clipping control** — we always used clipping (1.0)
- **Format alignment** — our eval format didn't precisely match training format
- **seq_len=8192** — we used 3072 (limited by GPU memory on some hardware)

---

## 4. Our Assets for Next Phase

### 4.1 Datasets on HuggingFace

| Dataset | Size | Domain | Format |
|---------|------|--------|--------|
| `ecoreasoner-cot-20k` | 20,000 | Ecological reasoning | `<think>` + code blocks |
| `ecoreasoner-cot-10k` | 10,000 | Ecological reasoning | Same format |
| `nemotron-eco-reasoner-v14` | 9,686 | Kaggle puzzles | Verified solver CoT |
| `nemotron-reasoning-v11` | ~15K | Kaggle puzzles | Augmented |
| `nemotron-reasoning-v3` | ~5K | Balanced puzzles | Balanced by category |
| `nemotron-reasoning-v2` | ~10K | Reasoning + ecology | Mixed |
| `ecocoder-scientific-reasoning` | ~5K | Scientific code | train/val/test split |
| `ecocoder-cot` | <1K | Ecological CoT | Prototype |
| `ecocoder-nemotron-kaggle` | ~10K | Unified v4/v5 | Competition format |

**Total unique training data: ~50K+ examples across ecological reasoning and puzzle-solving.**

### 4.2 Infrastructure

- KU HPC: PRO6000 (96GB), MI210 (64GB×2), V100, Q6000, A100, L40
- `tied_train.py` — production-ready multi-GPU DDP trainer with MoE weight tying
- Hermes gateway for token-efficient HPC orchestration
- 20 Ollama instances running (CoT generation active)

### 4.3 ecoreasoner-cot-20k Quality

- 20,000 examples from PubMed + bioRxiv ecological papers
- 70.8% have `<think>` reasoning traces (avg 3,081 chars)
- 55.1% have code blocks
- Top methods: MaxEnt, BRT, GLM, Occupancy, Random Forest, HMM
- **Needs filtering**: 29.2% missing `<think>` tags, some have short/empty responses

---

## 5. Training Plan: EcoReasoner v1.0

### Goal
Fine-tune an ecological reasoning model using the 20K CoT dataset + lessons from
winning solutions. Two tracks:

### Track A: Nemotron-3-Nano-30B (MoE, our existing infrastructure)
- **Advantage**: Existing `tied_train.py`, MoE weight tying, MI210/PRO6000 ready
- **Dataset**: `ecoreasoner-cot-20k` (filtered) + optionally `ecocoder-scientific-reasoning`
- **Approach**: Two-phase training (Train→Nudge) with winning techniques
- **Use case**: Powerful ecological reasoning model, potential for Ebbe Nielsen

### Track B: Smaller model (7B) for wider deployment
- Qwen2.5-7B-Instruct or DeepSeek-R1-distill-7B
- Single GPU training, faster iteration
- More accessible for deployment

### Training Recipe (incorporating winning lessons)

```python
# Phase 1 — Train (broad coverage, aggressive)
lr = 2e-4
lr_scheduler = "linear"
max_grad_norm = 1e9          # clipping OFF (from VCDAD)
neftune_noise_alpha = 5.0    # embedding noise (from VCDAD)
epochs = 1
packing = False
bf16 = True
stratified_sampler = True     # round-robin by method/category

# Phase 2 — Nudge (precision on hard categories)
lr = 5e-6                    # 40× smaller
lr_scheduler = "cosine"
max_grad_norm = 1.0          # clipping ON
warmup_steps = 10
epochs = 1
# Same NEFTune, BF16, no packing
```

### Data Preparation
1. Filter `ecoreasoner-cot-20k`: keep only examples with `<think>` tags (14,156 → ~14K)
2. Normalize format: ensure all have `<think>...</think>` + structured output
3. Optional: add `ecocoder-scientific-reasoning` for code quality boost
4. Stratify by `method` field for balanced sampling

### Infrastructure
- Use existing `tied_train.py` as base, add two-phase logic + NEFTune + stratified sampler
- Deploy on MI210 (BF16, no quantization) or PRO6000
- Monitor via Hermes (token-efficient `ku-hpc raw:` pattern)

---

## 6. Next Steps (Token-Efficient)

1. **Filter + prepare the 20K dataset** on HPC (0 tokens — direct SSH or Slurm)
2. **Create `eco_train_v2.py`** with two-phase training + winning techniques
3. **Submit training job** via Hermes (1 call, ~200 tokens)
4. **Monitor** via `ku-hpc raw: tail -30 ...` (200 tokens per check)
5. **Evaluate** the adapter on ecological reasoning benchmarks
6. **Publish** the fine-tuned adapter to HuggingFace

---

## 7. References

- DaoyuanLi2816/nvidia-nemotron-reasoning-challenge (Silver, Top 1.6%) — Two-phase SFT
- galaxywk223/kaggle-nvidia-nemotron-model-reasoning-challenge (Silver #167) — Teacher-CoT
- InfoSage05/NVIDIA-Nemotron-Model-Reasoning-Challenge-Solution-Kaggle — GRPO/RL
- msusol/kaggle-nemotron-model-reasoning-challenge — Per-expert MoE LoRA on DGX Spark
- huggingface.co/blog/nvidia/openreasoning-nemotron — NVIDIA's official reasoning pipeline
