#!/usr/bin/env python3
"""High-precision per-category accuracy eval of a LoRA checkpoint.

Same harness as scripts/eval_checkpoint.py but loads the base model with
*plain transformers + peft* (no Unsloth) so the numerical precision can be
chosen with the QUANT env var:

    QUANT=8bit  (default)  LLM.int8 — fits the 30B on ONE 48GB GPU (A40/L40),
                           far less lossy than nf4-4bit for a reasoning model.
    QUANT=bf16             full precision, device_map="auto" across GPUs
                           (needs ~60GB: 1x PRO6000 96GB, or 2x A40 / 2x MI210).
    QUANT=4bit            nf4 4-bit, single GPU — reproduces eval_checkpoint.py.

Why this exists: eval_checkpoint.py loads the base in 4-bit. For NemotronH (a
reasoning model) the 4-bit base degrades the chain-of-thought enough that the
adapter never closes </think> / emits \\boxed{} within the token budget, so every
category scores 0.0 LOCALLY even though the SAME adapter scores ~0.67 on Kaggle
(which runs the adapter on the BF16 base). Raising precision restores boxing.

Usage:
    python scripts/eval_hiprec.py --adapter outputs/tied_v14/checkpoint-500 \
        --data data/kaggle_classified.jsonl --n-per-cat 8 --batch-size 4
Env: MODEL_PATH (local snapshot dir or HF id), QUANT (8bit|bf16|4bit).
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
    """NemotronH's HybridMambaAttentionDynamicCache without hardcoding the model
    snapshot hash, or None (then generate() builds its own from use_cache=True)."""
    for name, mod in list(sys.modules.items()):
        if name.endswith("modeling_nemotron_h") and hasattr(
            mod, "HybridMambaAttentionDynamicCache"
        ):
            return mod.HybridMambaAttentionDynamicCache
    return None


def make_cache(cache_cls, model, batch_size):
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


def force_torch_mamba_path():
    """Force the Mamba-2 mixer onto its pure-torch path *only* where the fused
    kernel would crash.

    NemotronH's fast path calls a precompiled ``causal_conv1d`` CUDA kernel whose
    wheel ships no Blackwell (sm_100) image, so on an RTX PRO 6000 the forward
    dies with ``cudaErrorNoKernelImageForDevice``. On sm_80/sm_86 (A100/A40) the
    fused kernels exist and are *far* faster than the unfused torch scan (whose
    cached generation is ~O(n^2)), so we keep them there. Auto-detects the GPU
    arch; override with FORCE_TORCH_MAMBA=1 (force torch) or =0 (keep fused)."""
    env = os.environ.get("FORCE_TORCH_MAMBA", "auto").lower()
    if env in ("0", "false", "no"):
        print("FORCE_TORCH_MAMBA=0: keeping fused mamba kernels", flush=True)
        return
    if env not in ("1", "true", "yes"):  # auto
        try:
            major = torch.cuda.get_device_capability()[0]
        except Exception:
            major = 0
        if major < 10:  # < Blackwell: fused causal_conv1d kernel image exists
            print(f"sm_{major}x: fused mamba kernels OK, keeping fast path", flush=True)
            return
    n = 0
    for name, mod in list(sys.modules.items()):
        if name.endswith("modeling_nemotron_h") and hasattr(
            mod, "is_fast_path_available"
        ):
            mod.is_fast_path_available = False
            n += 1
    print(f"forced torch mamba path on {n} module(s)", flush=True)


def patch_moe(model):
    """Dense per-expert MoE dispatch (same as training): the stock NemotronH
    dispatch hits a bf16/fp32 index_add_ dtype mismatch when LoRA is active."""
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


# Must match the training records EXACTLY (data/train_deterministic_v*.jsonl):
# system = this string, user = the bare prompt. Any deviation pushes the model
# off-distribution and it never closes / emits \boxed within the budget.
SYS_PROMPT = "You are a helpful assistant that solves puzzles step by step."


def build_msgs(prompt):
    return [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": prompt},
    ]


def generate_batch(model, tok, cache_cls, prompts, max_new_tokens, use_explicit_cache):
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
    if use_explicit_cache:
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


def load_model(model_path, adapter_path, quant):
    """Plain transformers + peft load. Returns (model, tok, use_explicit_cache)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=True)
    quant = quant.lower()
    if quant == "bf16":
        load_kwargs.update(dtype=torch.bfloat16, device_map="auto")
    elif quant == "8bit":
        from transformers import BitsAndBytesConfig

        load_kwargs.update(
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map={"": 0},
        )
    elif quant == "4bit":
        from transformers import BitsAndBytesConfig

        load_kwargs.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            ),
            device_map={"": 0},
        )
    else:
        raise SystemExit(f"unknown QUANT={quant!r} (use 8bit|bf16|4bit)")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    print(f"BASE LOADED quant={quant} in {time.time()-t0:.0f}s", flush=True)

    from peft import PeftModel

    model = PeftModel.from_pretrained(model, adapter_path)
    patch_moe(model)
    force_torch_mamba_path()
    model.eval()

    # An explicit single-device cache only makes sense when the whole model lives
    # on one device. With device_map="auto" (bf16, sharded) let generate() build
    # its own cache so tensors land on the right shard.
    devices = {str(p.device) for p in model.parameters()}
    use_explicit_cache = len(devices) == 1
    print(
        f"param devices={sorted(devices)} explicit_cache={use_explicit_cache}",
        flush=True,
    )
    return model, tok, use_explicit_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="checkpoint dir with adapter")
    ap.add_argument("--data", default="data/kaggle_classified.jsonl")
    ap.add_argument("--n-per-cat", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--cats", default="", help="comma list to restrict categories")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_path = os.environ.get(
        "MODEL_PATH", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    )
    quant = os.environ.get("QUANT", "8bit")

    model, tok, use_explicit_cache = load_model(model_path, args.adapter, quant)

    # Left padding so every prompt in a batch ends at the same column.
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cache_cls = get_cache_cls()
    print(
        f"KV cache class: {cache_cls.__name__ if cache_cls else 'auto'} | "
        f"quant={quant} | batch_size={args.batch_size}",
        flush=True,
    )

    only = {c.strip() for c in args.cats.split(",") if c.strip()}
    by_cat = defaultdict(list)
    with open(args.data) as f:
        for line in f:
            r = json.loads(line)
            if only and r["category"] not in only:
                continue
            by_cat[r["category"]].append(r)

    rng = random.Random(args.seed)
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
        texts = generate_batch(
            model, tok, cache_cls, prompts, args.max_new_tokens, use_explicit_cache
        )
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
                    "gen_chars": len(text),
                }
            )
            print(
                f"[{cat}] idx={r.get('index')} ok={ok} pred={pred!r} "
                f"gold={gold!r} chars={len(text)} (~{per}s)",
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
        "quant": quant,
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
    out_path = args.out or os.path.join(args.adapter, f"eval_results_{quant}.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "details": details}, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
