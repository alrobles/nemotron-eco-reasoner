#!/usr/bin/env python3
"""Rejection sampling (RFT): generate K candidate solutions per puzzle with the
best checkpoint, keep only traces whose \\boxed{} answer matches the gold.

Output records use the same messages schema as train_deterministic_v*.jsonl so
accepted traces can be concatenated into the next dataset directly.

Shardable: run multiple jobs with --shard i --num-shards N (one GPU each).

Usage:
    python scripts/rejection_sample.py --adapter outputs/deterministic_v7/checkpoint-250 \
        --data data/kaggle_classified.jsonl --k 4 --out data/rft/shard0.jsonl \
        --shard 0 --num-shards 4
Env overrides: MODEL_PATH (local snapshot dir or HF id).
"""
import argparse
import json
import os
import random
import re
import time

import torch

SYSTEM = "You are an expert puzzle solver. Think step by step and place your final answer inside \\boxed{}."


def extract_boxed(text):
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    return m[-1].strip() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", default="data/kaggle_classified.jsonl")
    ap.add_argument("--k", type=int, default=4, help="candidates per puzzle")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--categories", default="cryptarithm_deduce,equation_numeric_deduce,bit_manipulation,gravity,cipher",
                    help="comma-separated; weakest categories first")
    ap.add_argument("--max-puzzles", type=int, default=200, help="per shard cap")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model_path = os.environ.get("MODEL_PATH", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        model_path, max_seq_length=args.seq_len, load_in_4bit=True, trust_remote_code=True,
        device_map={"": 0})

    # Same dense per-expert MoE dispatch as eval_checkpoint.py (bf16/fp32
    # index_add_ mismatch in the stock dispatch when LoRA is active).
    import types as _types
    patched = 0
    for module in model.modules():
        if not hasattr(module, "moe") or not hasattr(module, "experts"):
            continue
        if not callable(module.moe):
            continue

        def mp(mod):
            def pm(_self, h, ti, tw):
                h = h.view(-1, h.size(-1)); ft = ti.view(-1)
                h = h.repeat_interleave(ti.shape[-1], dim=0)
                fh = torch.zeros_like(h); dt = fh.dtype
                for i, el in enumerate(mod.experts):
                    m = (ft == i).nonzero(as_tuple=True)[0]
                    if m.numel() == 0:
                        continue
                    eh = h[m]; eo = el(eh); w = tw.view(-1)[m]; wo = eo * w.unsqueeze(-1)
                    if wo.dtype != dt:
                        wo = wo.to(dt)
                    fh.index_add_(0, m, wo)
                tk = ti.shape[-1]
                return fh.view(-1, tk, fh.size(-1)).sum(dim=1)
            return pm

        module.moe = _types.MethodType(mp(module), module); patched += 1
    print(f"MoE patched: {patched} layers", flush=True)

    model.load_adapter(args.adapter)
    FastLanguageModel.for_inference(model)

    cats = set(args.categories.split(","))
    puzzles = []
    with open(args.data) as f:
        for line in f:
            r = json.loads(line)
            if r.get("category") in cats:
                puzzles.append(r)
    rng = random.Random(args.seed)
    rng.shuffle(puzzles)
    puzzles = [p for i, p in enumerate(puzzles) if i % args.num_shards == args.shard]
    puzzles = puzzles[:args.max_puzzles]
    print(f"shard {args.shard}/{args.num_shards}: {len(puzzles)} puzzles x {args.k} candidates", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    accepted = 0
    with open(args.out, "w") as fout:
        for n, r in enumerate(puzzles):
            gold = extract_boxed(r["answer"]) or r["answer"].strip()
            user = r["prompt"] + "\n\nPlease put your final answer inside \\boxed{}."
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
            ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                          return_tensors="pt").to(model.device)
            got = False
            for k in range(args.k):
                t0 = time.time()
                with torch.no_grad():
                    out = model.generate(ids, max_new_tokens=args.max_new_tokens,
                                         do_sample=k > 0, temperature=args.temperature if k > 0 else None,
                                         top_p=0.95 if k > 0 else None,
                                         pad_token_id=tok.eos_token_id)
                text = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
                pred = extract_boxed(text)
                ok = pred is not None and pred == gold
                print(f"[{n+1}/{len(puzzles)}] {r['category']} idx={r.get('index')} k={k} ok={ok} "
                      f"pred={pred!r} gold={gold!r} ({time.time()-t0:.0f}s)", flush=True)
                if ok:
                    rec = {"messages": [{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": user},
                                        {"role": "assistant", "content": text.strip()}],
                           "category": r["category"], "index": r.get("index"), "source": "rft"}
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                    accepted += 1
                    got = True
                    break  # one accepted trace per puzzle
            if not got:
                pass
    print(f"DONE: accepted {accepted}/{len(puzzles)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
