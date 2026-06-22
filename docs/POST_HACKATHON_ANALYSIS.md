# Post-Hackathon Analysis: Lessons from Winning Solutions + Training Plan

> Generated: 2026-06-22. Updated with 2nd/3rd place writeups and YS-L repo.
> Based on review of top-3 solutions, silver-medal solutions, the
> `ecoreasoner-cot-20k` dataset, and existing nemotron-eco-reasoner infrastructure.

---

## 1. Competition Results Summary

| Team | Rank | Score | Key Technique |
|------|------|-------|---------------|
| **3rd place** (YS-L) | 3/4163 | **0.900** | Two-stage SFT: cryptarithm drill → mixed-task. MoE grad-tying. lm_head LoRA. |
| **VCDAD** (DaoyuanLi2816) | 65/4163 (Silver, Top 1.6%) | ~0.83 | Two-phase SFT: Train→Nudge. NEFTune. Stratified sampler. |
| **galaxy2025** (galaxywk223) | 167/4163 (Silver) | 0.860 | Teacher-CoT distillation + stratified sampler |
| **msusol** | 2789/4163 | 0.67 | Per-expert MoE LoRA (PEFT-compatible) |
| **alrobles** (us) | — | 0.67 (v8 best) | Deterministic solvers + verified CoT |
| **InfoSage05** | — | — | GRPO/RL (single-GPU + 2-GPU split) |

**Gap to top: 0.67 → 0.90 = +0.23 points.** SFT alone reached 0.90 (3rd place);
no RL was needed. The delta came from data engineering and training strategy.

Sources:
- 3rd place: https://github.com/YS-L/nvidia-nemotron-reasoning-3rd-place-solution
- 3rd place writeup: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/3rd-place-solution
- 2nd place writeup: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/2nd-place-solution

---

## 2. Key Lessons from Winning Solutions

### 2.0 3rd Place Solution (YS-L — Score 0.900, Full Code Available)

The most important reference — full open-source code, detailed config, 0.900 score.

**Two-stage training (not two-phase):**
```
Stage 1: Cryptarithm pattern RECALL DRILL
  lr=2e-4, 5 epochs, batch=8, seq=8192, completion_only_loss=1
  Dataset: cryptarithm-pat-drill-cap48 (specialized)
  Runtime: 25.2 hours on 1× RTX PRO 6000 (96GB)

Stage 2: Mixed-task from Stage 1 LoRA
  lr=2e-4, 1 epoch, batch=4, max_grad_norm=10, seq=8192
  Dataset: 7 combined dataset_configs + recall_replay (3000 examples)
  Runtime: 32.7 hours on 1× RTX PRO 6000 (96GB)
```

| Insight | Detail |
|---------|--------|
| **LoRA targets** | `q/k/v/o_proj` + `up/down_proj` + `in/out_proj` + **`lm_head`** (added manually!) |
| **LoRA dtype** | `fp32` for LoRA params (rest in bf16) — higher precision for adapter weights |
| **MoE gradient tying** | NOT parameter sharing — gradients are SUMMED across experts per step (`_build_moe_tie_hooks`) |
| **Completion-only loss** | Only train on assistant response tokens, not input prompt |
| **Recall replay** | Stage 2 replays 3000 cryptarithm drill examples to prevent forgetting |
| **Unsloth + Mamba fast path** | `unsloth_force_compile=1`, `force_mamba_fast_path=1` for speed |
| **Linear LR decay** | Custom `_linear_decay_lr(lr, step, total_steps)` |
| **Category weighting** | Per-category sample weights for balanced training |
| **Deterministic solvers** | Full solver suite in `src/solver/` generates verified traces |
| **Hardware** | Single RTX PRO 6000 Blackwell (96GB) — we have PRO6000 on HPC! |

**Critical difference from our approach:** They specialized Stage 1 entirely on
the hardest category (cryptarithm), then generalized in Stage 2. This is the
opposite of our "train everything at once" approach.

### 2.1 Two-Phase Training (VCDAD — Silver Top 1.6%)

Their recipe:

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
- 2-GPU split: vLLM on GPU 0 (generation), PEFT on GPU 1 (training)
- DAPO clipping to prevent catastrophic forgetting

### 2.4 Per-Expert MoE LoRA (msusol)

- Standard LoRA misses MoE expert layers (Unsloth fused format incompatible with PEFT evaluator)
- Per-expert LoRA in PEFT-compatible format: `mixer.experts.{j}.up_proj.lora_A.weight`
- Our `tied_train.py` already handles this via weight tying — we were on the right track

### 2.5 Common Patterns Across ALL Top Solutions

Every solution that scored > 0.80 shared these patterns:
1. **Deterministic solvers** for puzzle-type trace generation (not relying on LLM alone)
2. **Machine-verified traces** — discard any trace where final answer != ground truth
3. **Strict format alignment** — training target format byte-identical to eval protocol
4. **Architecture-aware LoRA** — must include Mamba `in/out_proj`, not just attention
5. **Pure BF16** (or fp32 for LoRA params) — no quantization during training
6. **seq_len >= 4096** (best results at 8192)
7. **SFT is sufficient** — no team needed RL/GRPO to reach 0.90

---

## 3. What We Had Right vs What We Missed

### Right
- Deterministic solvers for hard categories (73-78% coverage)
- MoE weight tying (`tied_train.py`) — 3rd place used gradient-tying variant
- BF16 training on MI210 (no quantization noise)
- LoRA rank 32 (same as winners)
- Target modules included `up_proj, down_proj, in_proj, out_proj` (architecture-aware)

### Missed
- **Two-stage specialized training** — 3rd place: drill hardest category first, then mixed
- **lm_head in LoRA targets** — 3rd place added it manually; we excluded it
- **LoRA params in fp32** — 3rd place cast LoRA weights to fp32 for precision
- **Completion-only loss** — train only on response tokens, not prompt
- **NEFTune noise** — not in our training scripts (VCDAD used alpha=5.0)
- **Recall replay** — prevent forgetting by replaying hard examples in Stage 2
- **Gradient clipping control** — VCDAD: off in Phase 1, on in Phase 2
- **seq_len=8192** — we used 3072 (limited by GPU memory on some hardware)
- **Category-weighted sampling** — 3rd place used per-category sample weights

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
3rd place (0.900) and silver-medal solutions. Two tracks:

### Track A: Nemotron-3-Nano-30B on PRO6000 (recommended — matches 3rd place hardware)
- **Hardware**: 1× PRO6000 (96GB) — same GPU family as 3rd place
- **Dataset**: `ecoreasoner-cot-20k` (filtered 14K) + `ecocoder-scientific-reasoning`
- **Approach**: Two-stage training following 3rd place pattern
- **Use case**: Ecological reasoning model for Ebbe Nielsen + general deployment

### Track B: Nemotron-3-Nano-30B on MI210 (multi-GPU alternative)
- **Hardware**: 2× MI210 (64GB each, BF16)
- **Same approach** but DDP across 2 GPUs

### Training Recipe (incorporating 3rd place + VCDAD lessons)

```python
# === STAGE 1: Specialized drill on hardest ecological methods ===
# Analogous to 3rd place's cryptarithm recall drill
# For ecology: MaxEnt + complex SDM methods are our "hard category"
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "up_proj", "down_proj", "in_proj", "out_proj",
                  "lm_head"]              # 3rd place included lm_head!
lora_rank = 32
lora_alpha = 32                           # alpha/rank = 1.0 (VCDAD)
lora_dropout = 0.0                        # VCDAD: no dropout
lora_dtype = "fp32"                       # 3rd place: LoRA params in fp32

lr = 2e-4                                 # both 3rd and VCDAD used 2e-4
lr_scheduler = "linear_decay"             # 3rd place: custom linear decay
num_epochs = 3                            # 3rd place used 5 for drill
batch_size = 4                            # depends on GPU memory
max_grad_norm = 10                        # 3rd place Stage 2 value
max_seq_len = 4096                        # 8192 ideal but OOM on MI210
completion_only_loss = True               # 3rd place: only train on response
packing = False                           # VCDAD: no packing
bf16 = True
neftune_noise_alpha = 5.0                 # VCDAD: embedding noise
moe_tie_weights = True                    # gradient-sum tying (3rd place)

# === STAGE 2: Mixed ecological reasoning from Stage 1 LoRA ===
input_lora = "stage1/adapter"
lr = 2e-4
num_epochs = 1                            # 3rd place: 1 epoch for mixed
recall_replay_num_examples = 2000         # replay hard examples (3rd place)

# Data: filtered ecoreasoner-cot-20k + ecocoder-scientific-reasoning
# Category-weighted sampling by ecological method
```

### Data Preparation
1. Filter `ecoreasoner-cot-20k`: keep only with `<think>` tags (14,156/20K = 70.8%)
2. Split by difficulty: isolate MaxEnt/complex SDM examples for Stage 1 drill
3. Normalize format: ensure `<think>...</think>` + structured output
4. Add `ecocoder-scientific-reasoning` for code quality boost
5. Category-weight by ecological method for balanced sampling

### Infrastructure
- PRO6000 preferred (single GPU, matches 3rd place exactly)
- MI210 fallback (2× GPU DDP, BF16)
- `eco_train_v2.py` has two-phase callback, adapt to two-stage
- Monitor via Hermes (`ku-hpc raw:` pattern, 200 tokens per check)

### New Techniques to Add to eco_train_v2.py
From 3rd place analysis:
- [ ] Add `lm_head` to LoRA target modules
- [ ] Cast LoRA params to fp32 (`lora_dtype="fp32"`)
- [ ] Completion-only loss (mask prompt tokens)
- [ ] Recall replay buffer for Stage 2
- [ ] MoE gradient-sum tying (vs our parameter sharing — different approach)
- [ ] Unsloth fast path support (if running on NVIDIA GPUs)

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

### Official Writeups
- 2nd place: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/2nd-place-solution
- 3rd place: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/3rd-place-solution

### Open-Source Repositories
- **YS-L/nvidia-nemotron-reasoning-3rd-place-solution** (3rd place, 0.900) — Two-stage SFT, full code
- DaoyuanLi2816/nvidia-nemotron-reasoning-challenge (Silver Top 1.6%) — Two-phase SFT: Train→Nudge
- galaxywk223/kaggle-nvidia-nemotron-model-reasoning-challenge (Silver #167) — Teacher-CoT
- InfoSage05/NVIDIA-Nemotron-Model-Reasoning-Challenge-Solution-Kaggle — GRPO/RL
- msusol/kaggle-nemotron-model-reasoning-challenge — Per-expert MoE LoRA on DGX Spark
- Anjana13Mohan/nemotron-reasoning-challenge — Data-centric LoRA fine-tuning
- huggingface.co/blog/nvidia/openreasoning-nemotron — NVIDIA's official reasoning pipeline
