#!/usr/bin/env python3
"""Generate synthetic cryptarithm puzzles with known solutions for training.

Creates puzzles in the same format as the Kaggle competition's cryptarithm_deduce
category, with verified CoT traces.  The model learns the *reasoning pattern*
(symbol→digit mapping + operator deduction) rather than memorising specific puzzles.

Usage:
    python3 scripts/generate_synthetic_cryptarithm.py [--count N] [--seed S]
    Default: 2000 puzzles, seed 42.
Output: data/synthetic_cryptarithm.jsonl  (one JSON per line, same schema as v6)
"""

import argparse
import hashlib
import json
import random
import string
import sys
from itertools import combinations

# ── Operator library (same 13 ops as solve_cryptarithm_csp + extras) ──────────

def _num_str(n):
    """Non-negative int → its digit string (no leading zeros except for 0)."""
    return str(n)

def _zpad(n, width):
    return str(n).zfill(width)


OP_REGISTRY = {}  # name → (func, fmt_func)

def _reg(name, func, fmt):
    OP_REGISTRY[name] = (func, fmt)

_reg("add",        lambda a, b: a + b,
     lambda a, b, r: f"{a} + {b} = {r}")
_reg("abs_diff",   lambda a, b: abs(a - b),
     lambda a, b, r: f"|{a} - {b}| = {r}")
_reg("mul",        lambda a, b: a * b,
     lambda a, b, r: f"{a} * {b} = {r}")
_reg("concat",     lambda a, b: int(f"{a:02d}{b:02d}"),
     lambda a, b, r: f"concat({a:02d}, {b:02d}) = {r:04d}")
_reg("rev_concat", lambda a, b: int(f"{b:02d}{a:02d}"),
     lambda a, b, r: f"rev_concat({a:02d}, {b:02d}) = {r:04d}")
_reg("sub_mod100", lambda a, b: (a - b) % 100,
     lambda a, b, r: f"({a} - {b}) mod 100 = {r}")
_reg("add_mod100", lambda a, b: (a + b) % 100,
     lambda a, b, r: f"({a} + {b}) mod 100 = {r}")
_reg("mul_mod100", lambda a, b: (a * b) % 100,
     lambda a, b, r: f"({a} * {b}) mod 100 = {r}")
_reg("digit_add",  lambda a, b: ((a // 10 + b // 10) % 10) * 10 + (a % 10 + b % 10) % 10,
     lambda a, b, r: f"digit_add({a}, {b}) = {r}")
_reg("digit_diff", lambda a, b: abs(a // 10 - b // 10) * 10 + abs(a % 10 - b % 10),
     lambda a, b, r: f"digit_diff({a}, {b}) = {r}")
_reg("digit_mul",  lambda a, b: ((a // 10 * b // 10) % 10) * 10 + (a % 10 * b % 10) % 10,
     lambda a, b, r: f"digit_mul({a}, {b}) = {r}")
_reg("add_rev",    lambda a, b: int(str(a + b)[::-1]),
     lambda a, b, r: f"reverse({a} + {b}) = {r}")
_reg("mul_rev",    lambda a, b: int(str(a * b)[::-1]),
     lambda a, b, r: f"reverse({a} * {b}) = {r}")

# Operators that the original solver doesn't cover — expose the model to more
_reg("xor",        lambda a, b: a ^ b,
     lambda a, b, r: f"{a} XOR {b} = {r}")
_reg("digit_xor",  lambda a, b: ((a // 10 ^ b // 10) % 10) * 10 + (a % 10 ^ b % 10) % 10,
     lambda a, b, r: f"digit_xor({a}, {b}) = {r}")

# Subset of ops suitable for training (avoids degenerate cases)
TRAINABLE_OPS = [
    "add", "abs_diff", "mul", "sub_mod100", "add_mod100", "mul_mod100",
    "digit_add", "digit_diff", "digit_mul", "add_rev", "mul_rev",
    "concat", "rev_concat", "xor", "digit_xor",
]


# ── Symbol pool ───────────────────────────────────────────────────────────────

# Same symbol pool as the Kaggle puzzles: printable non-alphanumeric chars
SYMBOL_POOL = list("!@#$%^&*()[]{}<>|/\\~`':;,.-_+=?\"")


def _pick_symbols(rng, n):
    """Pick n distinct symbols from the pool."""
    return rng.sample(SYMBOL_POOL, n)


# ── Puzzle generator ──────────────────────────────────────────────────────────

def _digits_to_syms(n, d2s):
    """Convert a non-negative integer to its symbol representation."""
    return "".join(d2s[int(c)] for c in str(n))


def generate_one_puzzle(rng, num_examples=None, num_ops=None):
    """Generate a single cryptarithm puzzle with known solution.

    Returns (prompt, answer, cot) or None if generation fails.
    """
    if num_examples is None:
        num_examples = rng.choice([3, 4, 5])
    if num_ops is None:
        num_ops = rng.choice([1, 2, 3])
    num_ops = min(num_ops, num_examples)

    # Pick distinct digit→symbol mapping (injective: 10 digits → 10 symbols)
    syms = _pick_symbols(rng, 10)
    digit_to_sym = {d: s for d, s in enumerate(syms)}
    sym_to_digit = {s: d for d, s in enumerate(syms)}

    # Pick operator symbols (from our pool, not overlapping digit symbols)
    remaining = [s for s in SYMBOL_POOL if s not in syms]
    if len(remaining) < num_ops:
        return None
    op_syms = rng.sample(remaining, num_ops)

    # Pick operator semantics
    op_names = rng.sample(TRAINABLE_OPS, num_ops)
    op_map = dict(zip(op_syms, op_names))

    # Generate examples
    examples_text = []
    examples_detail = []
    op_sym_list = list(op_map.keys())

    for _ in range(num_examples):
        # Pick operands (2-digit numbers, 10-99 to avoid leading-zero ambiguity)
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        op_sym = rng.choice(op_sym_list)
        op_name = op_map[op_sym]
        func, fmt_func = OP_REGISTRY[op_name]
        result = func(a, b)
        if result < 0 or result > 9999:
            continue  # skip degenerate results

        # Encode
        a_sym = digit_to_sym[a // 10] + digit_to_sym[a % 10]
        b_sym = digit_to_sym[b // 10] + digit_to_sym[b % 10]
        r_sym = _digits_to_syms(result, digit_to_sym)
        lhs = f"{a_sym}{op_sym}{b_sym}"
        examples_text.append(f"{lhs} = {r_sym}")
        examples_detail.append((a, b, op_name, result, fmt_func(a, b, result)))

    if len(examples_text) < 3:
        return None  # need at least 3 examples

    # Generate query
    qa = rng.randint(10, 99)
    qb = rng.randint(10, 99)
    q_op_sym = rng.choice(op_sym_list)
    q_op_name = op_map[q_op_sym]
    q_func, q_fmt = OP_REGISTRY[q_op_name]
    q_result = q_func(qa, qb)
    if q_result < 0 or q_result > 9999:
        return None

    qa_sym = digit_to_sym[qa // 10] + digit_to_sym[qa % 10]
    qb_sym = digit_to_sym[qb // 10] + digit_to_sym[qb % 10]
    qr_sym = _digits_to_syms(q_result, digit_to_sym)
    query_text = f"{qa_sym}{q_op_sym}{qb_sym}"

    # Build prompt (same style as Kaggle)
    prompt_lines = [
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:"
    ]
    for ex in examples_text:
        prompt_lines.append(ex)
    prompt_lines.append(f"Now, determine the result for: {query_text}")
    prompt = "\n".join(prompt_lines)

    answer = qr_sym

    # Build CoT
    cot_lines = [
        "We deduce a symbol-to-digit mapping and operator meanings from the examples.",
        "Each lhs is two 2-digit numbers joined by an operator symbol; each symbol maps to a unique digit.",
        "",
        "Deduced symbol-to-digit mapping:",
    ]
    # Only include symbols that actually appear in examples + query
    all_chars = set()
    for ex in examples_text:
        for c in ex:
            if c in sym_to_digit:
                all_chars.add(c)
    for c in query_text + answer:
        if c in sym_to_digit:
            all_chars.add(c)
    for s in sorted(all_chars):
        cot_lines.append(f"  '{s}' = {sym_to_digit[s]}")

    cot_lines.append("Deduced operators:")
    for s in sorted(op_map.keys()):
        cot_lines.append(f"  '{s}' = {op_map[s]}")

    cot_lines.append("")
    cot_lines.append("Verify against the examples:")
    for ex_text, (a, b, op_name, result, fmt_str) in zip(examples_text, examples_detail):
        cot_lines.append(f"  {ex_text}  =>  {fmt_str}")

    cot_lines.append("")
    cot_lines.append(f"Apply to the question {query_text}:")
    cot_lines.append(f"  {q_fmt(qa, qb, q_result)}")
    cot_lines.append(f"  Mapping the digits back to symbols gives {answer}.")

    cot = "\n".join(cot_lines)
    return prompt, answer, cot


def generate_dataset(count, seed):
    rng = random.Random(seed)
    records = []
    attempts = 0
    max_attempts = count * 5

    while len(records) < count and attempts < max_attempts:
        attempts += 1
        result = generate_one_puzzle(rng)
        if result is None:
            continue
        prompt, answer, cot = result

        # Format as training message (same schema as v6)
        system_msg = "You are a helpful assistant that solves puzzles step by step."
        user_msg = prompt
        assistant_msg = f"<think>\n{cot}\n</think>\n\\boxed{{{answer}}}"

        record = {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        }
        records.append(record)

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/synthetic_cryptarithm.jsonl")
    args = parser.parse_args()

    records = generate_dataset(args.count, args.seed)
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Generated {len(records)} synthetic cryptarithm puzzles → {args.out}")

    # Stats
    from collections import Counter
    ops = Counter()
    for r in records:
        cot = r["messages"][2]["content"]
        for line in cot.split("\n"):
            if line.strip().startswith("'") and "' = " in line and "Deduced operators" not in line:
                if "' = " in line and len(line.strip()) < 40:
                    parts = line.strip().split("' = ")
                    if len(parts) == 2:
                        ops[parts[1]] += 1
    print(f"Operator distribution: {ops.most_common()}")


if __name__ == "__main__":
    main()
