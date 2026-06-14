# Handoff — `scripts/eval_checkpoint.py` speed optimization

Goal: evaluate LoRA checkpoints **locally** (before Kaggle submission) as fast as
possible while keeping results correct. This doc is the handoff for the next Devin.

## TL;DR

- The original script generated **one prompt at a time** (batch=1) in a Python loop —
  the dominant bottleneck.
- The optimized script adds **batched generation** (left-padded, default `--batch-size 8`)
  so the GPU processes many prompts per `model.generate()` call. Greedy decoding
  (`do_sample=False`) is deterministic, so batched output is mathematically identical
  to single-sample output. `--batch-size 1` reproduces the legacy path exactly.
- It keeps (and hardens) the **KV-cache fix**: NemotronH is a hybrid Mamba/attention
  model and needs `HybridMambaAttentionDynamicCache` for O(n) decode instead of O(n^2).

## What changed vs the patched version

1. **Robust KV cache.** The previous patch hardcoded the snapshot-hash import path and
   called `HybridMambaAttentionDynamicCache()` with **no arguments**. On the current
   stack (transformers 4.57.6) that constructor **requires `(config, batch_size)`** and
   raises `TypeError` — i.e. the no-arg call crashes. We now:
   - locate the class dynamically via `sys.modules` (`get_cache_cls()`), no hardcoded hash;
   - build it per batch with the right size and the model's float dtype/device
     (`make_cache(cache_cls, model, batch_size)`);
   - fall back to `use_cache=True` (let `generate()` build its own cache) if construction
     fails, so the script never hard-crashes on a transformers version bump.
2. **Batched generation** (`generate_batch`): all prompts in a chunk are left-padded and
   generated together; prompt width is constant after left-padding, so completions are
   sliced with a single `plen`.
3. **Flattened sampling**: per-category samples are flattened into one list so batches
   stay full across category boundaries (max GPU utilization). Category is tracked per row.
4. **Micro-opts**: `eos_token_id` early-stop, correct `pad_token_id`, imports hoisted out
   of the loop, `torch.no_grad()`.

## CLI

```bash
MODEL_PATH=/path/to/nemotron/snapshot \
python scripts/eval_checkpoint.py \
  --adapter outputs/v8_seq4096_clean \
  --data data/kaggle_classified.jsonl \
  --n-per-cat 8 \
  --max-new-tokens 1024 \
  --seq-len 2048 \
  --batch-size 8         # 1 = legacy one-at-a-time
```

Output: per-category accuracy + a JSON results file (`--out`).

## Correctness / speed verification (HPC)

`scripts/_verify_batch.py` loads the model **once** and runs the same prompts both
single (bs=1) and batched (bs=8), then prints:
- `single_time=… batched_time=… speedup=…x`
- `BOXED MATCH k/N` (greedy => expect a full match)

Run via `hpc/_verify_batch.slurm` (a100 GPU):

```bash
sbatch --export=ALL,ADAPTER=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/v8_seq4096_clean,NPC=2,BS=8,MNT=1024 _verify_batch.slurm
```

Notes:
- Model load is ~12 min (13 shards) on the a100, so each verify run is ~18-20 min total.
- `_verify_batch.py` / `_verify_batch.slurm` are **test scaffolding**, not meant to merge.
  They live on branch `devin/1781440738-eval-batch-test`.

## Status (2026-06-14)

- Optimized script: written, ruff-clean, py_compile OK.
- First HPC verify run surfaced the `HybridMambaAttentionDynamicCache()` `TypeError`
  (no-arg constructor). Fixed via `make_cache(...)`. Re-verification in progress.
- Once verified (match + speedup), open/merge the PR and switch local evals to
  `--batch-size 8` (or higher if VRAM allows).

## Gotchas

- Keep `do_sample=False` (greedy) — that is what makes batched == single.
- `tok.padding_side = "left"` is **required** for batched generation alignment.
- If you bump transformers and the cache constructor signature changes again, extend the
  `make_cache` attempt list; the `use_cache=True` fallback keeps it correct (just slower).
- Kaggle itself runs vLLM (`max_num_seqs=64`, `max_model_len=8192`, `temperature=0.0`).
  This script is the **local** Unsloth/PyTorch eval, not the Kaggle inference path.
