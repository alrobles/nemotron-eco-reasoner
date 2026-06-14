# Nemotron Eco-Reasoner — NVIDIA Nemotron Model Reasoning Challenge

LoRA fine-tuning pipeline for **`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`** on the
Kaggle [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/)
("Alice's Wonderland" puzzles). The deliverable is a single **LoRA adapter of rank ≤ 32**
that outputs `<think>…</think>\boxed{answer}`.

Repository: `alrobles/nemotron-eco-reasoner`

---

## Current state

| Dataset | Kaggle score | Notes |
|---|---|---|
| **v8** | **0.67** | current best (`train_deterministic_v8.jsonl`, seq3072 ckpt-500) |
| v12 (ga8) | 0.66 | algorithmic long-CoT — did not beat v8 |
| v11 | 0.54 | over-augmented — regression |
| **v14** | TBD | v8 composition + **verified solver CoT** for the hard categories |
| **v15** | TBD | v14 + equation symbol-decoration + cryptarithm at a deeper search budget |

**Lesson (measured):** the lever is **CoT correctness on the hard categories**, not raw
volume (v11) or mega-long CoT that truncates `\boxed{}` (v12/v13). v14/v15 keep v8's exact
composition and short CoT, but replace heuristic CoT with traces **verified by a solver**.

---

## Evaluation categories

Seven categories, all framed as "In Alice's Wonderland, …":
`numeral`, `unit_conversion`, `cipher`, `bit_manipulation`,
`equation_numeric_deduce`, `cryptarithm_deduce`, `gravity`.

The three hard ones are attacked with deterministic solvers that fit the puzzle's
**operator family per-puzzle** and emit CoT only when the result matches the gold answer:

| Category | Solver | Verified coverage on real puzzles |
|---|---|---|
| equation | `scripts/solve_equation_ops.py` | 304/416 = **73.1%** |
| cryptarithm | `scripts/solve_cryptarithm_ops.py` | ~284/417 = **68%** (budget 15) |
| bit | `scripts/solve_bit_perbit.py` | 637/822 = **77.5%** |

---

## Repository layout

```
data/        canonical datasets (v8 record, v9 gravity source, v14, v15) + kaggle_classified (5k ground truth)
scripts/     dataset builders, per-category solvers, classifier, eval, Kaggle packaging
  archive/     superseded one-offs (kept for history)
hpc/         Slurm launchers + trainers for the KU HPC cluster
  archive/     superseded Slurm scripts
docs/        runbooks, milestones, design notes
  archive/     stale handoffs
containers/  Apptainer/Singularity definition files
cloud/       portable single-command cloud training
monitor/     cluster monitoring helper
notebooks/   portable QLoRA notebook
```

### Key scripts

| Script | Purpose |
|---|---|
| `scripts/classify_category.py` | classify a prompt into its eval category (100% on the 5k set) |
| `scripts/build_v14_dataset.py` | build the v14/v15 dataset (v8 base + v9 gravity + verified solver CoT) |
| `scripts/solve_equation_ops.py` / `solve_cryptarithm_ops.py` / `solve_bit_perbit.py` | per-category verified solvers |
| `scripts/eval_checkpoint.py` | offline per-category accuracy for a checkpoint |
| `scripts/submit_kaggle.py` | package a LoRA adapter into a submission zip |

### Datasets (`data/`)

| File | Role |
|---|---|
| `kaggle_classified.jsonl` | 5,000 real puzzles with category + answer (ground truth) |
| `train_deterministic_v8.jsonl` | the 0.67 record dataset (also a build input for num/unit/cipher) |
| `train_deterministic_v9.jsonl` | gravity (least-squares) CoT — build input for v14/v15 |
| `train_deterministic_v14.jsonl` | v14 (9,686 rows) |
| `train_deterministic_v15.jsonl` | v15 (9,686 rows) |

---

## Training (KU HPC)

All training produces a **portable LoRA adapter** (pure-torch `rmsnorm_fn` fallback, so the
same adapter trains on NVIDIA Blackwell PRO6000, MI210/ROCm, or A100 — see
`docs/MILESTONE_AMD_BF16.md`).

| Launcher | What |
|---|---|
| `hpc/nem_tied.slurm` + `hpc/tied_train.py` | 4-GPU multi-node DDP, **MoE weight tying** (expert LoRA factors shared → all experts learn together) |
| `hpc/nem_tied_1gpu.slurm` | single-GPU tied arm with staged expert-freeze (`EXPERT_FREEZE_FRAC`) |
| `hpc/nem_mi210_train.slurm` | 2× MI210 BF16 (diverse adapter for the final merge) |
| `hpc/nem_eval.slurm` | per-category eval pinned to an L40 |

Operational runbooks: **`docs/RUNBOOK_V14_TIED.md`** (tied training + endgame TIES-merge) and
**`docs/OPERATIONS.md`** (drive the whole pipeline via the Hermes gateway, no Devin needed).

---

## License

Apache 2.0 — see [LICENSE](LICENSE)
