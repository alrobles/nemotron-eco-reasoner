#!/usr/bin/env python3
"""Generate synthetic equation_numeric_deduce puzzles with verified CoT traces.

Same surface format as the Kaggle equation_numeric_deduce category: examples
like `61"88 = 27` where operands/results are plain digits and the operator is a
hidden symbol with secret semantics. The model learns to hypothesise an
operator meaning, verify it against ALL examples, then apply it to the query.

Usage:
    python3 scripts/generate_synthetic_equation.py [--count N] [--seed S]
Output: data/synthetic_equation.jsonl
"""

import argparse
import json
import random

OP_REGISTRY = {}


def _reg(name, func, fmt):
    OP_REGISTRY[name] = (func, fmt)


_reg("add", lambda a, b: a + b, lambda a, b, r: f"{a} + {b} = {r}")
_reg("abs_diff", lambda a, b: abs(a - b), lambda a, b, r: f"|{a} - {b}| = {r}")
_reg("mul", lambda a, b: a * b, lambda a, b, r: f"{a} * {b} = {r}")
_reg("concat", lambda a, b: int(f"{a:02d}{b:02d}"), lambda a, b, r: f"concat({a:02d}, {b:02d}) = {r}")
_reg("rev_concat", lambda a, b: int(f"{b:02d}{a:02d}"), lambda a, b, r: f"concat reversed ({b:02d}, {a:02d}) = {r}")
_reg("concat_rev_digits", lambda a, b: int(f"{a:02d}{b:02d}"[::-1]), lambda a, b, r: f"reverse digits of concat({a:02d}, {b:02d}) = {r}")
_reg("add_rev", lambda a, b: int(str(a + b)[::-1]), lambda a, b, r: f"reverse({a} + {b} = {a + b}) = {r}")
_reg("mul_rev", lambda a, b: int(str(a * b)[::-1]), lambda a, b, r: f"reverse({a} * {b} = {a * b}) = {r}")
_reg("sub_mod100", lambda a, b: (a - b) % 100, lambda a, b, r: f"({a} - {b}) mod 100 = {r}")
_reg("add_mod100", lambda a, b: (a + b) % 100, lambda a, b, r: f"({a} + {b}) mod 100 = {r}")
_reg("mul_mod100", lambda a, b: (a * b) % 100, lambda a, b, r: f"({a} * {b}) mod 100 = {r}")
_reg("digit_add", lambda a, b: ((a // 10 + b // 10) % 10) * 10 + (a % 10 + b % 10) % 10,
     lambda a, b, r: f"digit-wise add mod 10: ({a}, {b}) = {r}")
_reg("digit_diff", lambda a, b: abs(a // 10 - b // 10) * 10 + abs(a % 10 - b % 10),
     lambda a, b, r: f"digit-wise abs diff: ({a}, {b}) = {r}")
_reg("digit_mul", lambda a, b: ((a // 10 * (b // 10)) % 10) * 10 + (a % 10 * (b % 10)) % 10,
     lambda a, b, r: f"digit-wise mul mod 10: ({a}, {b}) = {r}")
_reg("sum_sq_digits", lambda a, b: sum(int(c) ** 2 for c in f"{a}{b}"),
     lambda a, b, r: f"sum of squared digits of {a}{b} = {r}")
_reg("digit_sum", lambda a, b: sum(int(c) for c in f"{a}{b}"),
     lambda a, b, r: f"sum of all digits of {a} and {b} = {r}")

OP_SYMBOLS = list("\"')(}{][<>|/\\~`:;,.@#$%^&*-_+=?!")


def generate_one(rng):
    num_ops = rng.choice([1, 1, 2])
    op_names = rng.sample(list(OP_REGISTRY), num_ops)
    op_syms = rng.sample(OP_SYMBOLS, num_ops)
    op_map = dict(zip(op_syms, op_names))

    n_examples = rng.choice([4, 5, 6])
    examples = []
    for _ in range(n_examples * 2):
        if len(examples) >= n_examples:
            break
        a, bb = rng.randint(10, 99), rng.randint(10, 99)
        sym = rng.choice(op_syms)
        func, fmt = OP_REGISTRY[op_map[sym]]
        r = func(a, bb)
        if r < 0 or r > 9999:
            continue
        examples.append((a, sym, bb, r, fmt(a, bb, r)))
    if len(examples) < 4:
        return None

    qa, qb = rng.randint(10, 99), rng.randint(10, 99)
    q_sym = rng.choice(op_syms)
    q_func, q_fmt = OP_REGISTRY[op_map[q_sym]]
    q_r = q_func(qa, qb)
    if q_r < 0 or q_r > 9999:
        return None

    lines = ["In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:"]
    for a, sym, bb, r, _ in examples:
        lines.append(f"{a:02d}{sym}{bb:02d} = {r}")
    lines.append(f"Now, determine the result for: {qa:02d}{q_sym}{qb:02d}")
    prompt = "\n".join(lines)

    cot = ["The operands and results are plain numbers; only the operator symbol has hidden semantics.",
           "Deduced operators:"]
    for s in op_syms:
        cot.append(f"  '{s}' = {op_map[s]}")
    cot += ["", "Verify against the examples:"]
    for a, sym, bb, r, fmt_str in examples:
        cot.append(f"  {a:02d}{sym}{bb:02d} = {r}  =>  {fmt_str}")
    cot += ["", f"Apply to the question {qa:02d}{q_sym}{qb:02d}:", f"  {q_fmt(qa, qb, q_r)}"]
    return prompt, str(q_r), "\n".join(cot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/synthetic_equation.jsonl")
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
    print(f"Generated {len(records)} synthetic equation puzzles -> {args.out}")


if __name__ == "__main__":
    main()
