#!/usr/bin/env python3
"""Offline per-category accuracy eval of a LoRA checkpoint.

Loads the base model in 4-bit + an adapter checkpoint, samples N puzzles per
category from data/kaggle_classified.jsonl, generates and compares the
\\boxed{...} answer. Runs on its own GPU so it never touches the training job.

Speed:
  * KV cache — generation reuses a HybridMambaAttentionDynamicCache so decode is
    O(n) instead of O(n^2). (NemotronH is a hybrid Mamba/attention model and
    needs its own cache class; stock generate() in this stack did not build it.)
  * Batched generation — prompts are left-padded and generated --batch-size at a
    time so the GPU stays busy instead of running one sequence at a time. Greedy
    decoding is deterministic, so batched and unbatched results match; use
    --batch-size 1 to fall back to exact one-at-a-time behaviour.

Usage:
    python scripts/eval_checkpoint.py --adapter outputs/deterministic_v7/checkpoint-100 \
        --data data/kaggle_classified.jsonl --n-per-cat 8 --batch-size 8 --out eval_results.json
Env overrides: MODEL_PATH (local snapshot dir or HF id).
"""

import argparse
import json
import os
import random
import re
import sys
import time
import types
from collections import defaultdict

import torch


def extract_boxed(text):
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    return m[-1].strip() if m else None


def get_cache_cls():
    """Return NemotronH's HybridMambaAttentionDynamicCache without hardcoding the
    model snapshot hash, or None if it cannot be located (then generate() falls
    back to building its own cache from use_cache=True)."""
    for name, mod in list(sys.modules.items()):
        if name.endswith("modeling_nemotron_h") and hasattr(
            mod, "HybridMambaAttentionDynamicCache"
        ):
            return mod.HybridMambaAttentionDynamicCache
    return None


def make_cache(cache_cls, model, batch_size):
    """Build a hybrid Mamba/attention KV cache sized for this batch. On this stack
    the class requires (config, batch_size); we also pass the model's float dtype
    and device so the Mamba conv/ssm states line up with the hidden states. Returns
    None if it cannot be built, in which case generate() builds its own cache from
    use_cache=True."""
    if cache_cls is None:
        return None
    fp = next((p for p in model.parameters() if p.is_floating_point()), None)
    dtype = fp.dtype if fp is not None else torch.bfloat16
    device = fp.device if fp is not None else getattr(model, "device", "cuda")
    for args, kwargs in (
        ((model.config, batch_size), {"dtype": dtype, "device": device}),
        ((model.config, batch_size), {}),
    ):
        try:
            return cache_cls(*args, **kwargs)
        except TypeError:
            continue
    return None


def patch_moe(model):
    """Same dense per-expert MoE dispatch as training (hpc/nem_chained.slurm):
    the stock NemotronH dispatch hits a bf16/fp32 index_add_ dtype mismatch when
    LoRA adapters are active. Batch-agnostic (operates on flattened tokens)."""
    patched = 0
    for module in model.modules():
        if not hasattr(module, "moe") or not hasattr(module, "experts"):
            continue
        if not callable(module.moe):
            continue

        def mp(mod):
            def pm(_self, h, ti, tw):
                h = h.view(-1, h.size(-1))
                ft = ti.view(-1)
                h = h.repeat_interleave(ti.shape[-1], dim=0)
                fh = torch.zeros_like(h)
                dt = fh.dtype
                for i, el in enumerate(mod.experts):
                    m = (ft == i).nonzero(as_tuple=True)[0]
                    if m.numel() == 0:
                        continue
                    eh = h[m]
                    eo = el(eh)
                    w = tw.view(-1)[m]
                    wo = eo * w.unsqueeze(-1)
                    if wo.dtype != dt:
                        wo = wo.to(dt)
                    fh.index_add_(0, m, wo)
                tk = ti.shape[-1]
                return fh.view(-1, tk, fh.size(-1)).sum(dim=1)

            return pm

        module.moe = types.MethodType(mp(module), module)
        patched += 1
    print(f"MoE patched: {patched} layers", flush=True)


SYS_PROMPT = (
    "You are an expert puzzle solver. Think step by step and place "
    "your final answer inside \\boxed{}."
)


def build_msgs(prompt):
    return [
        {"role": "system", "content": SYS_PROMPT},
        {
            "role": "user",
            "content": prompt + "\n\nPlease put your final answer inside \\boxed{}.",
        },
    ]


def generate_batch(model, tok, cache_cls, prompts, max_new_tokens):
    """Generate completions for a list of prompts in a single batched call.
    Returns the decoded completion text (prompt stripped) for each prompt."""
    enc = tok.apply_chat_template(
        [build_msgs(p) for p in prompts],
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True,
    )
    cache = make_cache(cache_cls, model, enc["input_ids"].shape[0])
    if cache is not None:
        gen_kwargs["past_key_values"] = cache
    with torch.no_grad():
        out = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            **gen_kwargs,
        )
    plen = enc["input_ids"].shape[1]  # left-padded => same prompt width for all rows
    return [tok.decode(row[plen:], skip_special_tokens=True) for row in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--adapter", required=True, help="checkpoint dir with adapter weights"
    )
    ap.add_argument("--data", default="data/kaggle_classified.jsonl")
    ap.add_argument("--n-per-cat", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="prompts generated per batch (1 = legacy one-at-a-time)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_path = os.environ.get(
        "MODEL_PATH", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    )

    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        model_path,
        max_seq_length=args.seq_len,
        load_in_4bit=True,
        trust_remote_code=True,
        device_map={"": 0},
    )

    patch_moe(model)
    model.load_adapter(args.adapter)
    FastLanguageModel.for_inference(model)

    # Left padding is required so every prompt in a batch ends at the same column
    # (the next generated token aligns across the batch).
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cache_cls = get_cache_cls()
    print(
        f"KV cache class: {cache_cls.__name__ if cache_cls else 'auto'} | batch_size={args.batch_size}",
        flush=True,
    )

    by_cat = defaultdict(list)
    with open(args.data) as f:
        for line in f:
            r = json.loads(line)
            by_cat[r["category"]].append(r)

    rng = random.Random(args.seed)
    # Flatten the per-category samples into one list so batches stay full even
    # across category boundaries (max GPU utilisation); category is tracked per row.
    todo = []
    for cat in sorted(by_cat):
        for r in rng.sample(by_cat[cat], min(args.n_per_cat, len(by_cat[cat]))):
            todo.append((cat, r))

    cat_stats = defaultdict(lambda: {"n": 0, "correct": 0})
    details = []
    bs = max(1, args.batch_size)
    for start in range(0, len(todo), bs):
        chunk = todo[start : start + bs]
        prompts = [r["prompt"] for _, r in chunk]
        t0 = time.time()
        texts = generate_batch(model, tok, cache_cls, prompts, args.max_new_tokens)
        per = round((time.time() - t0) / len(chunk), 1)
        for (cat, r), text in zip(chunk, texts):
            pred = extract_boxed(text)
            gold = extract_boxed(r["answer"]) or r["answer"].strip()
            ok = pred is not None and pred == gold
            cat_stats[cat]["n"] += 1
            cat_stats[cat]["correct"] += int(ok)
            details.append(
                {
                    "category": cat,
                    "index": r.get("index"),
                    "pred": pred,
                    "gold": gold,
                    "ok": ok,
                    "gen_s": per,
                }
            )
            print(
                f"[{cat}] idx={r.get('index')} ok={ok} pred={pred!r} gold={gold!r} (~{per}s)",
                flush=True,
            )

    results = {}
    for cat in sorted(cat_stats):
        n = cat_stats[cat]["n"]
        c = cat_stats[cat]["correct"]
        results[cat] = {"n": n, "correct": c, "acc": round(c / n, 3) if n else 0.0}
        print(f"== {cat}: {c}/{n} = {results[cat]['acc']}", flush=True)

    total_n = sum(v["n"] for v in results.values())
    total_c = sum(v["correct"] for v in results.values())
    summary = {
        "adapter": args.adapter,
        "n_per_cat": args.n_per_cat,
        "batch_size": bs,
        "overall": {
            "n": total_n,
            "correct": total_c,
            "acc": round(total_c / total_n, 3) if total_n else 0.0,
        },
        "by_category": results,
    }
    print(json.dumps(summary, indent=2))
    out_path = args.out or os.path.join(args.adapter, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "details": details}, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
