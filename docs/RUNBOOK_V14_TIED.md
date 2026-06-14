# Runbook — v14 + MoE weight tying + router/expert freeze (multi-GPU)

This runbook documents the **MoE weight-tying** experiment on top of dataset **v14**
(verified solver CoT), how it is launched on the KU HPC cluster, and how to drive it
end-to-end **via Hermes** without a Devin session. It also records the idle-hardware
plan (the "4 fronts") so anyone can pick it up.

Everything here ships **only a standard PEFT LoRA adapter (rank ≤ 32)** → Kaggle-legal.
The weight tying and the staged freeze change *how* the adapter is trained, not what is
submitted.

---

## 0. TL;DR — what was done and why

- **Goal:** beat the current best Kaggle score **0.67** (v8, `v8_seq3072` ckpt-500).
  v11 = 0.54, `v12_ga8` ckpt-400 = 0.66 → more synthetic volume / mega-CoT did **not** help.
- **Lever 1 (data) = v14:** same composition as v8, but the CoT for the three hard
  categories (cryptarithm / equation / bit) is **verified by a solver** in the correct
  operator family, not heuristic. See `docs/OPERATIONS.md` and PR #33.
- **Lever 2 (optimization) = MoE weight tying:** in standard LoRA-MoE each token only
  updates the experts the router selected (top-k), so cold experts learn slowly from
  very few tokens — effectively **one expert at a time**. We tie the per-expert LoRA
  factors so a single shared `nn.Parameter` is used by every expert in a MoE layer
  → autograd **sums the gradients of all experts** → **all experts learn together each
  step**. This is the technique behind the strongest public Kaggle solution.
- **Lever 3 (router/expert freeze):** the router is **never LoRA-targeted**, so it is
  frozen by construction (this is the Kaggle-legal "router freeze"). Optionally we also
  **stage** the expert-LoRA: freeze it during a warmup fraction (`EXPERT_FREEZE_FRAC`)
  so attention adapts first, then the experts come online.

### The correct tie (and the bug it fixes)
- **Correct (`hpc/tied_train.py`):** parameter **sharing**. After the PEFT wrap, group
  expert LoRA modules per `(MoE layer, proj-type)` via regex `\.experts\.\d+\.` →
  `.experts.#.`; assign the **same** `nn.Parameter` to `up_proj.lora_A.default` and
  `down_proj.lora_B.default` across all experts (init = mean across experts). Leave
  `up_proj.lora_B` / `down_proj.lora_A` **free per expert**. `parameters()` dedupes the
  shared tensor → the optimizer steps it once; autograd sums grads across experts.
  Saved adapter is **standard PEFT** → Kaggle-legal.
- **Old buggy approach (do NOT port):** `p.grad.sum(dim=0)` in the old `nem_scratch.slurm`
  collapsed the **rank** dimension, not the **expert** dimension, and combined with
  `lr=2e-4 + linear` it blew up `grad_norm` (~1e11). The fix is the param-sharing above
  + **`lr=1e-4` + `constant_with_warmup` + `max_grad_norm=1.0`** (the v8 recipe).

**Runtime confirmation (job 22717744):** `World=4`, **46 tied groups**, 5842 duplicate
params collapsed, 381.37M trainable params; loss **5.85 → 3.16** with **grad_norm ~2–3.4**
(no explosion); checkpoint-50 saved. The instability is tamed.

---

## 1. Topology note — "4 GPUs" = multi-node DDP

No single node has 4× PRO6000 (they are spread 1 + 2 + 2 across nodes in `mix` state),
so **4 GPUs = 2 nodes × 2 PRO6000 = 4 DDP replicas**. The Slurm script requests
`--nodes=2 --ntasks-per-node=2 --gres=gpu:pro6000:2` and launches **one srun task per
GPU** (no `torchrun`); the distributed env is derived from `SLURM_*`
(`MASTER_ADDR = scontrol show hostnames | head -n1`, `MASTER_PORT=29517`,
`RANK=SLURM_PROCID`, `LOCAL_RANK=SLURM_LOCALID`, `WORLD_SIZE=SLURM_NTASKS`),
`device_map={"":local_rank}`, `ddp_find_unused_parameters=True`. The rmsnorm patch +
dynamic-module cache prewarm run **once on node0** before `srun` to avoid a 4-rank write
race.

---

## 2. Files

| File | Role |
|---|---|
| `hpc/tied_train.py` | BF16 multi-node DDP trainer + correct param-sharing tie + shape diagnostic + optional staged freeze. |
| `hpc/nem_tied.slurm` | **4-GPU** launcher (2 nodes × 2 PRO6000). Defaults: `OUT=outputs/tied_v14`, `SEQ_LEN=3072`, `GRAD_ACCUM=2`, `TARGET_TOTAL=500`, `STEPS_PER_JOB=200`, `EXPERT_FREEZE_FRAC=0.0`, `MASTER_PORT=29517`. |
| `hpc/nem_tied_1gpu.slurm` | **Freeze arm** (1× PRO6000). Defaults: `OUT=outputs/tied_v14_freeze`, `GRAD_ACCUM=8`, `STEPS_PER_JOB=250`, `EXPERT_FREEZE_FRAC=0.25`, `MASTER_PORT=29518`. |
| `hpc/nem_mi210_train.slurm` | **MI210 BF16 arm** (2× MI210, ROCm container, pure-torch rmsnorm). Diversity member for the TIES-merge. |
| `hpc/nem_eval.slurm` | Per-category eval (pin `--gres=gpu:l40:1`); writes `eval_results.json`. |

All paths assume `R=/home/a474r867/scratch/nemotron-eco-reasoner`.

---

## 3. Launch — paste into Hermes (in order, as cards free up)

> Direct-command pattern. System prompt: *"You are a terminal. Execute and return ONLY
> the raw output, verbatim. No commentary."* Prefix cluster commands with `ku-hpc raw:`.

### 3.1 Sync the cluster repo to the tied branch
```
ku-hpc raw: cd /home/a474r867/scratch/nemotron-eco-reasoner && git fetch origin && git checkout origin/devin/1781452595-v14-moe-tied -- hpc/tied_train.py hpc/nem_tied.slurm hpc/nem_tied_1gpu.slurm hpc/nem_mi210_train.slurm hpc/nem_eval.slurm && echo SYNCED
```

### 3.2 Front A — 4-GPU tied (the main bet)
```
ku-hpc raw: cd /home/a474r867/scratch/nemotron-eco-reasoner && sbatch hpc/nem_tied.slurm
```

### 3.3 Front B — freeze arm (1× PRO6000, attention adapts first)
```
ku-hpc raw: cd /home/a474r867/scratch/nemotron-eco-reasoner && sbatch hpc/nem_tied_1gpu.slurm
```

### 3.4 Front C — MI210 BF16 (diverse adapter for TIES-merge)
```
ku-hpc raw: cd /home/a474r867/scratch/nemotron-eco-reasoner && sbatch --export=ALL,DATA_PATH=$PWD/data/train_deterministic_v14.jsonl,OUT_PATH=$PWD/outputs/mi210_v14 hpc/nem_mi210_train.slurm
```

### 3.5 Front D — rolling eval on L40 (pin the L40 so it does not land on MI210)
```
ku-hpc raw: cd /home/a474r867/scratch/nemotron-eco-reasoner && for D in tied_v14 tied_v14_freeze mi210_v14 deterministic_v14 bf16_v14; do CK=$(ls -d outputs/$D/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1); [ -n "$CK" ] && { echo "EVAL $D -> $CK"; sbatch --export=ALL,ADAPTER=$PWD/$CK,N_PER_CAT=8 hpc/nem_eval.slurm; } || echo "$D: no checkpoint yet"; done
```

---

## 4. Monitor — one batched Hermes call
```
ku-hpc raw: cd /home/a474r867/scratch; squeue -u a474r867 -o '%.10i %.9T %.10M %.4D %R %j' | grep -E 'nem-|JOBID'; echo '== TIED =='; grep -E "'loss'|grad_norm" nem_tied_*.out | tail -3; echo '== FREEZE =='; grep -E "'loss'|reez" nem_tfrz_*.out | tail -3; echo '== CKPTS =='; for D in tied_v14 tied_v14_freeze mi210_v14 deterministic_v14 bf16_v14; do echo -n "$D: "; ls -d nemotron-eco-reasoner/outputs/$D/checkpoint-* 2>/dev/null | tr '\n' ' '; echo; done
```

Healthy signs: loss decreasing, **grad_norm O(1–10)** (NOT 1e11), `46 tied groups` in the
job header, checkpoints every 50 steps, auto-resubmit firing 200 → … → 500.

---

## 5. Collect eval results & pick the checkpoint
```
ku-hpc raw: cd /home/a474r867/scratch/nemotron-eco-reasoner; for D in tied_v14 tied_v14_freeze mi210_v14 deterministic_v14 bf16_v14; do echo "== $D =="; for F in outputs/$D/checkpoint-*/eval_results.json; do [ -f "$F" ] && python3 -c "import json;d=json.load(open('$F'));print('$F'.split('/')[-2], d['summary']['overall'])"; done; done
```

> Local eval = only 8 examples/category (56 total) → **noisy and underestimates Kaggle**
> (best local 0.429 ↔ 0.67 Kaggle). Use it to rank variants against each other; the **real
> arbiter is the Kaggle leaderboard** (5 submissions/day, 2 finals).

---

## 6. Package & submit to Kaggle
```
ku-hpc raw: cd /home/a474r867/scratch/nemotron-eco-reasoner/outputs/<VARIANT>/checkpoint-<N> && zip -j /home/a474r867/scratch/submission_v14_tied.zip adapter_model.safetensors adapter_config.json && unzip -l /home/a474r867/scratch/submission_v14_tied.zip
```
Then upload `submission_v14_tied.zip` to Kaggle. Replace `<VARIANT>` (`tied_v14`,
`tied_v14_freeze`, `mi210_v14`, …) and `<N>` (best checkpoint).

---

## 7. Endgame — TIES-merge (still rank ≤ 32, Kaggle-legal)
Merge the **diverse** adapters (tied-4gpu / freeze / MI210-BF16 / untied v14) into ONE
adapter, e.g. with `peft`'s TIES merge, keeping rank ≤ 32, then eval the merge vs the best
individual and submit whichever wins.

---

## 8. Comparison map (what each run isolates)

| Run | Data | Tie | Freeze | GPUs | Output dir | Compares against |
|---|---|---|---|---|---|---|
| **v8** (baseline) | v8 | no | no | 1 | — | **the 0.67 bar** |
| untied v14 | v14 | no | no | 1 | `outputs/deterministic_v14` (QLoRA), `outputs/bf16_v14` (BF16) | isolates the **data** lever (v14 vs v8) |
| **tied v14 (main)** | v14 | yes | no | 4 (2×2) | `outputs/tied_v14` | isolates the **tie** lever (vs untied v14) |
| tied v14 freeze | v14 | yes | 0.25 | 1 | `outputs/tied_v14_freeze` | isolates the **staged freeze** (vs tied v14) |
| MI210 BF16 | v14 | no | no | 2 (MI210) | `outputs/mi210_v14` | numerically diverse member for TIES-merge |

---

## 9. v15 dataset (the idle-CPU front)

v15 = **same v8 composition**, push verified-CoT coverage of the **real** puzzles higher
by deepening the solvers (NOT more volume — that is the v11 lesson; NOT mega-CoT — the
v12/v13 lesson). Measured on the 5000 ground-truth Kaggle puzzles:

| Solver | v14 | v15 | Note |
|---|---|---|---|
| `solve_equation_ops.py` | 294/416 = 70.7% | **304/416 = 73.1%** | **symbol decoration**: results that carry the operator symbol as a prefix/suffix (e.g. `92$58=$65`, `53%84=%31`, `11:92=81:`). Full query-op-group consistency + gold check; `decor='none'` tried first → **zero regression**. The 83 puzzles whose query operator never appears in the examples are genuinely unsolvable; realistic ceiling ≈ 73–77%. |
| `solve_bit_perbit.py` | 637/822 = 77.5% | 637/822 = 77.5% | A feature expansion (3-input gates / mux / and-not) raised the **gold-checked** number to 95.6%, **but honest no-gold accuracy was only 55.7%** and strict uniqueness 2.1% — the ~914-feature space over-fits the ~9 examples/puzzle (gold-leakage → spurious rules). **Reverted** to the original 242 features to avoid teaching spurious reasoning. Bit stays at the v14 level (no regression). |
| `solve_cryptarithm_ops.py` | 253/417 = 60.7% | (measuring) | Reuses `solve_equation_ops.CANDS`. The equation symbol-decoration edit keeps `CANDS` as base 4-tuples, so cryptarithm is unaffected by it. Deepening = larger `budget` / bigger `CANDS`; the joint shared-cipher constraint across examples makes spurious fits unlikely. |

Build v15 (same composition, re-verifying well-posedness):
```
python3 scripts/build_v14_dataset.py --out data/train_deterministic_v15.jsonl --workers 8
```

---

## 10. Known traps
- **Don't port the old `sum(dim=0)` tie** — it collapses rank, not experts.
- **Don't train the router as a full parameter** — only the LoRA adapter ships (not Kaggle-legal otherwise).
- **Don't over-augment** (v11 → 0.54) and **don't emit mega-CoT** that truncates `\boxed{}` at seq3072 (v12/v13).
- **Don't let a solver fit a spurious rule** — require full-group consistency + gold check
  (equation) and parsimony / uniqueness (bit). The bit gold-leakage finding above is the
  live example.
- **Pin the eval GPU** to L40 (`--gres=gpu:l40:1`); a generic `gpu:1` can land on an MI210.
- **`bitsandbytes`/`unsloth` do not exist on MI210** (CUDA-only) → BF16 + pure-torch rmsnorm.
