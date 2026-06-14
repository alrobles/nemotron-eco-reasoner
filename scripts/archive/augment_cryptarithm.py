#!/usr/bin/env python3
"""Augment existing cryptarithm solutions by permuting symbol assignments.

For each solved puzzle, creates N variants by shuffling which symbols map to which
digits.  The puzzle structure (operands, operators, results) stays the same, but
the visual encoding changes — teaching the model that symbol identity is arbitrary.

Also oversamples existing solutions to balance the dataset.

Usage:
    python3 scripts/augment_cryptarithm.py [--variants N] [--oversample M]
    Default: 8 variants per solution, 5x oversample of originals.
Output: data/augmented_cryptarithm.jsonl
"""

import argparse
import hashlib
import json
import random
import re
import sys


SYMBOL_POOL = list("!@#$%^&*()[]{}<>|/\\~`':;,.-_+=?\"")


def load_solutions(path="data/cryptarithm_solutions.jsonl"):
    sols = []
    for line in open(path):
        r = json.loads(line)
        sols.append(r)
    return sols


def load_classified(path="data/kaggle_classified.jsonl"):
    by_md5 = {}
    for line in open(path):
        r = json.loads(line)
        if r["category"] == "cryptarithm_deduce":
            md5 = hashlib.md5(r["prompt"].encode()).hexdigest()
            by_md5[md5] = r
    return by_md5


def parse_mapping_from_cot(cot):
    """Extract symbol→digit mapping and operator info from CoT text."""
    mapping = {}
    op_info = {}
    in_mapping = False
    in_ops = False
    for line in cot.split("\n"):
        line = line.strip()
        if "symbol-to-digit" in line.lower():
            in_mapping = True
            in_ops = False
            continue
        if "operators:" in line.lower():
            in_ops = True
            in_mapping = False
            continue
        if in_mapping and line.startswith("'"):
            m = re.match(r"'(.)'\s*=\s*(\d+)", line)
            if m:
                mapping[m.group(1)] = int(m.group(2))
        if in_ops and line.startswith("'"):
            m = re.match(r"'(.)'\s*=\s*(\w+)", line)
            if m:
                op_info[m.group(1)] = m.group(2)
        if line.startswith("Verify") or line.startswith("Apply"):
            in_mapping = False
            in_ops = False
    return mapping, op_info


def make_variant(prompt, answer, cot, mapping, rng):
    """Create a variant by reassigning symbols to different characters."""
    # Get all symbols used in this puzzle
    used_syms = set(mapping.keys())

    # Pick new symbols (same count, different chars)
    available = [s for s in SYMBOL_POOL if s not in used_syms]
    if len(available) < len(used_syms):
        return None

    old_syms = sorted(used_syms)
    new_syms = rng.sample(available, len(old_syms))
    sym_remap = dict(zip(old_syms, new_syms))

    # Apply remap to prompt
    new_prompt = ""
    for c in prompt:
        new_prompt += sym_remap.get(c, c)

    # Apply remap to answer
    new_answer = ""
    for c in answer:
        new_answer += sym_remap.get(c, c)

    # Apply remap to CoT
    new_cot = ""
    for c in cot:
        new_cot += sym_remap.get(c, c)

    return new_prompt, new_answer, new_cot


def make_training_record(prompt, answer, cot):
    """Format as training message (same schema as v6)."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=int, default=8,
                        help="Number of symbol-permuted variants per solution")
    parser.add_argument("--oversample", type=int, default=5,
                        help="Times to repeat original solutions")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", default="data/augmented_cryptarithm.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    solutions = load_solutions()
    classified = load_classified()
    print(f"Loaded {len(solutions)} solutions, {len(classified)} classified puzzles")

    records = []

    for sol in solutions:
        md5 = sol["md5"]
        puzzle = classified.get(md5)
        if not puzzle:
            continue

        prompt = puzzle["prompt"]
        answer = sol["answer"]
        cot = sol["cot"]
        mapping, op_info = parse_mapping_from_cot(cot)

        if not mapping:
            continue

        # Original (oversampled)
        for _ in range(args.oversample):
            records.append(make_training_record(prompt, answer, cot))

        # Variants with permuted symbols
        for _ in range(args.variants):
            result = make_variant(prompt, answer, cot, mapping, rng)
            if result:
                vp, va, vc = result
                records.append(make_training_record(vp, va, vc))

    rng.shuffle(records)

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Generated {len(records)} augmented cryptarithm records → {args.out}")
    print(f"  {len(solutions)} originals × {args.oversample} oversample = {len(solutions)*args.oversample}")
    print(f"  {len(solutions)} originals × {args.variants} variants = {len(solutions)*args.variants}")


if __name__ == "__main__":
    main()
