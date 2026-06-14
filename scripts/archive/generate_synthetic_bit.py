#!/usr/bin/env python3
"""Generate synthetic bit_manipulation puzzles with verified CoT traces.

Same format as the Kaggle bit_manipulation category: 8-10 input -> output pairs
of 8-bit binary strings produced by a hidden pipeline of bitwise operations,
then a query input. The CoT names the deduced rule, verifies it against every
example, and applies it to the query — all correct by construction.

Usage:
    python3 scripts/generate_synthetic_bit.py [--count N] [--seed S]
Output: data/synthetic_bit.jsonl
"""

import argparse
import json
import random


def rotl8(x, n):
    n %= 8
    return ((x << n) | (x >> (8 - n))) & 0xFF


def rotr8(x, n):
    n %= 8
    return ((x >> n) | (x << (8 - n))) & 0xFF


def rev8(x):
    r = 0
    for i in range(8):
        r = (r << 1) | ((x >> i) & 1)
    return r


def b(x):
    return format(x, "08b")


# name -> (param_sampler, apply(x, p), describe(p))
PRIMITIVES = {
    "not":   (lambda rng: None, lambda x, p: (~x) & 0xFF, lambda p: "NOT (invert every bit)"),
    "shl":   (lambda rng: rng.randint(1, 7), lambda x, p: (x << p) & 0xFF, lambda p: f"left shift by {p} (zero-fill)"),
    "shr":   (lambda rng: rng.randint(1, 7), lambda x, p: x >> p, lambda p: f"right shift by {p} (zero-fill)"),
    "rotl":  (lambda rng: rng.randint(1, 7), rotl8 if False else (lambda x, p: rotl8(x, p)), lambda p: f"rotate left by {p}"),
    "rotr":  (lambda rng: rng.randint(1, 7), lambda x, p: rotr8(x, p), lambda p: f"rotate right by {p}"),
    "rev":   (lambda rng: None, lambda x, p: rev8(x), lambda p: "reverse the bit order"),
    "xor":   (lambda rng: rng.randint(1, 254), lambda x, p: x ^ p, lambda p: f"XOR with mask {b(p)}"),
    "and":   (lambda rng: rng.randint(1, 254), lambda x, p: x & p, lambda p: f"AND with mask {b(p)}"),
    "or":    (lambda rng: rng.randint(1, 254), lambda x, p: x | p, lambda p: f"OR with mask {b(p)}"),
}

PIPELINE_CHOICES = [
    ["not"], ["rev"], ["shl"], ["shr"], ["rotl"], ["rotr"], ["xor"], ["and"], ["or"],
    ["rotl", "xor"], ["rotr", "xor"], ["shl", "xor"], ["shr", "xor"],
    ["and", "xor"], ["rev", "xor"], ["not", "rotl"], ["rev", "rotr"],
    ["xor", "rotl"], ["or", "rotr"],
]

PROMPT_HEADER = (
    "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. "
    "The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, "
    "and possibly majority or choice functions.\n\nHere are some examples of input -> output:"
)


def generate_one(rng):
    names = rng.choice(PIPELINE_CHOICES)
    steps = []
    for name in names:
        sampler, fn, desc = PRIMITIVES[name]
        p = sampler(rng)
        steps.append((name, p, fn, desc(p)))

    def apply(x):
        for _, p, fn, _ in steps:
            x = fn(x, p)
        return x

    n_examples = rng.choice([8, 9, 10])
    inputs = rng.sample(range(256), n_examples + 1)
    query = inputs.pop()
    pairs = [(x, apply(x)) for x in inputs]
    answer = apply(query)

    # Degenerate guard: rule must not be constant or identity over the examples
    outs = {o for _, o in pairs}
    if len(outs) == 1 or all(x == o for x, o in pairs):
        return None

    prompt = PROMPT_HEADER + "\n" + "\n".join(f"{b(x)} -> {b(o)}" for x, o in pairs)
    prompt += f"\n\nNow, determine the output for: {b(query)}"

    rule_desc = ", then ".join(d for _, _, _, d in steps)
    cot = ["We look for a bitwise rule consistent with every example.",
           f"Hypothesis: {rule_desc}.", "", "Verify against the examples:"]
    for x, o in pairs:
        trace = [b(x)]
        cur = x
        for _, p, fn, _ in steps:
            cur = fn(cur, p)
            trace.append(b(cur))
        cot.append(f"  {' -> '.join(trace)}  matches {b(o)}")
    cot += ["", f"Apply the rule to {b(query)}:"]
    cur = query
    for _, p, fn, d in steps:
        prev = cur
        cur = fn(cur, p)
        cot.append(f"  {d}: {b(prev)} -> {b(cur)}")
    cot.append(f"The output is {b(answer)}.")
    return prompt, b(answer), "\n".join(cot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/synthetic_bit.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    records, attempts = [], 0
    while len(records) < args.count and attempts < args.count * 5:
        attempts += 1
        res = generate_one(rng)
        if res is None:
            continue
        prompt, answer, cot = res
        records.append({"messages": [
            {"role": "system", "content": "You are a helpful assistant that solves puzzles step by step."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"<think>\n{cot}\n</think>\n\\boxed{{{answer}}}"},
        ]})
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Generated {len(records)} synthetic bit puzzles -> {args.out}")


if __name__ == "__main__":
    main()
