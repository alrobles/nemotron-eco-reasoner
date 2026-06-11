#!/usr/bin/env python3
"""Offline per-category accuracy eval of a LoRA checkpoint.

Loads the base model in 4-bit + an adapter checkpoint, samples N puzzles per
category from data/kaggle_classified.jsonl, generates and compares the
\\boxed{...} answer. Runs on its own GPU so it never touches the training job.

NOTE: Nemotron generate() in this stack has no KV cache (quadratic in output
length) — keep N_PER_CAT and MAX_NEW_TOKENS modest.

Usage:
    python scripts/eval_checkpoint.py --adapter outputs/deterministic_v7/checkpoint-100 \
        --data data/kaggle_classified.jsonl --n-per-cat 8 --out eval_results.json
Env overrides: MODEL_PATH (local snapshot dir or HF id).
"""
import argparse
import json
import os
import random
import re
import time
from collections import defaultdict

import torch


def extract_boxed(text):
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    return m[-1].strip() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="checkpoint dir with adapter weights")
    ap.add_argument("--data", default="data/kaggle_classified.jsonl")
    ap.add_argument("--n-per-cat", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_path = os.environ.get("MODEL_PATH", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")

    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        model_path, max_seq_length=args.seq_len, load_in_4bit=True, trust_remote_code=True)
    model.load_adapter(args.adapter)
    FastLanguageModel.for_inference(model)

    by_cat = defaultdict(list)
    with open(args.data) as f:
        for line in f:
            r = json.loads(line)
            by_cat[r["category"]].append(r)

    rng = random.Random(args.seed)
    results = {}
    details = []
    for cat in sorted(by_cat):
        sample = rng.sample(by_cat[cat], min(args.n_per_cat, len(by_cat[cat])))
        correct = 0
        for r in sample:
            msgs = [
                {"role": "system", "content": "You are an expert puzzle solver. Think step by step and place your final answer inside \\boxed{}."},
                {"role": "user", "content": r["prompt"] + "\n\nPlease put your final answer inside \\boxed{}."},
            ]
            ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
            pred = extract_boxed(text)
            gold = extract_boxed(r["answer"]) or r["answer"].strip()
            ok = pred is not None and pred == gold
            correct += ok
            details.append({"category": cat, "index": r.get("index"), "pred": pred,
                            "gold": gold, "ok": ok, "gen_s": round(time.time() - t0, 1)})
            print(f"[{cat}] idx={r.get('index')} ok={ok} pred={pred!r} gold={gold!r} ({details[-1]['gen_s']}s)", flush=True)
        results[cat] = {"n": len(sample), "correct": correct, "acc": round(correct / len(sample), 3)}
        print(f"== {cat}: {correct}/{len(sample)} = {results[cat]['acc']}", flush=True)

    total_n = sum(v["n"] for v in results.values())
    total_c = sum(v["correct"] for v in results.values())
    summary = {"adapter": args.adapter, "n_per_cat": args.n_per_cat,
               "overall": {"n": total_n, "correct": total_c, "acc": round(total_c / total_n, 3)},
               "by_category": results}
    print(json.dumps(summary, indent=2))
    out_path = args.out or os.path.join(args.adapter, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "details": details}, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
