#!/usr/bin/env python3
"""Build training dataset v7: v6 base + synthetic cryptarithm + augmented cryptarithm + new CSP v2 solutions.

Merges:
1. data/train_deterministic_v6.jsonl (4230 verified traces)
2. data/synthetic_cryptarithm.jsonl (2000 synthetic puzzles teaching the CSP reasoning pattern)
3. data/augmented_cryptarithm.jsonl (symbol-permuted variants of solved puzzles)
4. data/cryptarithm_solutions_v2_*.jsonl (new solutions from expanded CSP solver)

For (4), generates CoT traces using the v2 solver's output and the original prompts.

Output: data/train_deterministic_v7.jsonl
"""

import glob
import hashlib
import json
import re
import sys


def load_jsonl(path):
    records = []
    for line in open(path):
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def load_classified():
    by_md5 = {}
    for line in open("data/kaggle_classified.jsonl"):
        r = json.loads(line)
        md5 = hashlib.md5(r["prompt"].encode()).hexdigest()
        by_md5[md5] = r
    return by_md5


def make_training_record(prompt, answer, cot):
    system_msg = "You are a helpful assistant that solves puzzles step by step."
    assistant_msg = f"<think>\n{cot}\n</think>\n\\boxed{{{answer}}}"
    return {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def main():
    out_path = "data/train_deterministic_v7.jsonl"

    # 1. Load v6 base
    v6 = load_jsonl("data/train_deterministic_v6.jsonl")
    print(f"v6 base: {len(v6)} records")

    # Track v6 prompts to avoid duplicates
    v6_prompts = set()
    for r in v6:
        if len(r["messages"]) > 1:
            v6_prompts.add(r["messages"][1]["content"])

    # 2. Load synthetic cryptarithm
    synth = load_jsonl("data/synthetic_cryptarithm.jsonl")
    print(f"Synthetic cryptarithm: {len(synth)} records")

    # 3. Load augmented cryptarithm
    aug = load_jsonl("data/augmented_cryptarithm.jsonl")
    print(f"Augmented cryptarithm: {len(aug)} records")

    # 4. Load new CSP v2 solutions
    classified = load_classified()
    v2_files = sorted(glob.glob("data/cryptarithm_solutions_v2_*.jsonl"))
    # Also check for merged v2 file
    if not v2_files:
        v2_files = glob.glob("data/cryptarithm_solutions_v2.jsonl")

    v2_new = []
    v1_md5s = set()
    for line in open("data/cryptarithm_solutions.jsonl"):
        r = json.loads(line)
        v1_md5s.add(r["md5"])

    for vf in v2_files:
        for line in open(vf):
            r = json.loads(line)
            if r["md5"] not in v1_md5s:
                # New solution not in v1
                puzzle = classified.get(r["md5"])
                if puzzle:
                    rec = make_training_record(puzzle["prompt"], r["answer"], r["cot"])
                    if rec["messages"][1]["content"] not in v6_prompts:
                        v2_new.append(rec)
    print(f"New CSP v2 solutions (not in v1): {len(v2_new)} records")

    # Combine
    all_records = v6 + synth + aug + v2_new

    with open(out_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    print(f"\n=== v7 dataset: {len(all_records)} total records ===")
    print(f"  v6 base:       {len(v6)}")
    print(f"  synthetic:     {len(synth)}")
    print(f"  augmented:     {len(aug)}")
    print(f"  new v2 sols:   {len(v2_new)}")
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
