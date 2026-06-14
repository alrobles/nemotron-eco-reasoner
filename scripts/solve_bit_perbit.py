#!/usr/bin/env python3
"""Per-output-bit boolean-feature solver for bit_manipulation.

Each of the 8 output bits is modelled independently as a simple boolean feature
of the input byte (constant 0/1, a single input bit +/- negation, a pairwise
AND/OR/XOR +/- negation, or majority-of-3). We pick, per output bit, any feature
consistent across all examples (plus the gold query row when verifying), then
apply to the query. Verified coverage on the real Kaggle puzzles: 77.5% (637/822).

Also provides verified CoT (`cot_for`) and search-free augmentation (`augment`)
used to build dataset v14.
"""
import json
import os
import random
import re
import sys
import time
from itertools import combinations


def bits(s):
    return [int(c) for c in s]


def feature_fns():
    fns, names = [], []
    fns.append(lambda b: 0); names.append('0')
    fns.append(lambda b: 1); names.append('1')
    for i in range(8):
        fns.append(lambda b, i=i: b[i]); names.append(f'b{i}')
        fns.append(lambda b, i=i: 1 - b[i]); names.append(f'~b{i}')
    for i, k in combinations(range(8), 2):
        fns.append(lambda b, i=i, k=k: b[i] & b[k]); names.append(f'b{i}&b{k}')
        fns.append(lambda b, i=i, k=k: 1 - (b[i] & b[k])); names.append(f'~(b{i}&b{k})')
        fns.append(lambda b, i=i, k=k: b[i] | b[k]); names.append(f'b{i}|b{k}')
        fns.append(lambda b, i=i, k=k: 1 - (b[i] | b[k])); names.append(f'~(b{i}|b{k})')
        fns.append(lambda b, i=i, k=k: b[i] ^ b[k]); names.append(f'b{i}^b{k}')
        fns.append(lambda b, i=i, k=k: 1 - (b[i] ^ b[k])); names.append(f'~(b{i}^b{k})')
    for i, k, m in combinations(range(8), 3):
        fns.append(lambda b, i=i, k=k, m=m: 1 if (b[i] + b[k] + b[m]) >= 2 else 0)
        names.append(f'maj(b{i},b{k},b{m})')
    return fns, names


FNS, NAMES = feature_fns()


def parse(prompt):
    pairs = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    qm = re.search(r'determine the output for:\s*([01]{8})', prompt)
    if not qm or len(pairs) < 1:
        return None
    return pairs, qm.group(1)


def fit(pairs, q, gold=None):
    """Return the list of 8 chosen feature indices, or None."""
    rows = list(pairs)
    if gold is not None:
        rows = pairs + [(q, gold)]
    ins = [bits(a) for a, b in rows]
    outs = [bits(b) for a, b in rows]
    cols = [[fn(x) for x in ins] for fn in FNS]
    chosen = []
    for j in range(8):
        target = [o[j] for o in outs]
        found = None
        for fi, col in enumerate(cols):
            if col == target:
                found = fi
                break
        if found is None:
            return None
        chosen.append(found)
    return chosen


def apply_rule(chosen, q):
    qb = bits(q)
    return ''.join(str(FNS[chosen[j]](qb)) for j in range(8))


def solve(prompt, gold=None):
    pr = parse(prompt)
    if not pr:
        return None
    pairs, q = pr
    chosen = fit(pairs, q, gold)
    if chosen is None:
        return None
    pred = apply_rule(chosen, q)
    if gold is not None and pred != gold:
        return None
    return pred, chosen


def gold_of(ans):
    m = re.search(r'\\boxed\{([^}]*)\}', ans)
    return m.group(1).strip() if m else ans.strip()


def format_cot(pairs, q, pred, chosen):
    lines = ["<think>"]
    lines.append("We fit each output bit independently as a boolean function of "
                 "the input bits (positions 0=MSB .. 7=LSB).")
    lines.append("Deduced per-output-bit rules:")
    for j in range(8):
        lines.append(f"  out[{j}] = {NAMES[chosen[j]]}")
    lines.append("Verify against the examples:")
    for a, b in pairs[:6]:
        lines.append(f"  {a} -> {apply_rule(chosen, a)}  (expected {b})")
    lines.append(f"Apply the rule to {q}:")
    lines.append(f"  -> {pred}")
    lines.append("</think>")
    return "\n".join(lines) + f"\n\\boxed{{{pred}}}"


def cot_for(prompt, gold=None):
    res = solve(prompt, gold)
    if not res:
        return None
    pred, chosen = res
    pairs, q = parse(prompt)
    return pred, format_cot(pairs, q, pred, chosen)


# ----------------------------------------------------------------------------
# Search-free augmentation: pick a random per-output-bit rule, then mint
# verified puzzles directly by computing outputs from the rule.
# ----------------------------------------------------------------------------
PROMPT_HEAD = ("In Alice's Wonderland, a secret bit manipulation rule transforms "
               "8-bit binary numbers. The transformation involves operations like "
               "bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority "
               "or choice functions.\n\nHere are some examples of input -> output:")


def random_rule(rng):
    return [rng.randrange(len(FNS)) for _ in range(8)]


def make_puzzle(chosen, rng, n_examples=9):
    seen = set()
    lines = []
    while len(lines) < n_examples and len(seen) < 256:
        x = rng.randint(0, 255)
        if x in seen:
            continue
        seen.add(x)
        a = f"{x:08b}"
        lines.append(f"{a} -> {apply_rule(chosen, a)}")
    for _ in range(256):
        x = rng.randint(0, 255)
        if x not in seen:
            q = f"{x:08b}"
            break
    else:
        return None
    gold = apply_rule(chosen, q)
    body = "\n".join(lines)
    prompt = (PROMPT_HEAD + "\n" + body
              + f"\n\nNow, determine the output for: {q}")
    pairs = [tuple(ln.split(' -> ')) for ln in lines]
    return prompt, gold, pairs, q


def augment(chosen, n, seed=0):
    rng = random.Random(seed)
    out = 0
    while out < n:
        mk = make_puzzle(chosen, rng)
        if not mk:
            break
        prompt, gold, pairs, q = mk
        yield prompt, gold, format_cot(pairs, q, gold, chosen)
        out += 1


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(here, '..', 'data', 'kaggle_classified.jsonl')
    rows = [json.loads(l) for l in open(data)]
    bit = [r for r in rows if r['category'] == 'bit_manipulation']
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(bit)
    t0 = time.time()
    solved = sum(1 for r in bit[:n] if cot_for(r['prompt'], gold_of(r['answer'])))
    print(f"bit per-bit: {solved}/{n} = {100 * solved / n:.1f}% in "
          f"{time.time() - t0:.0f}s")
