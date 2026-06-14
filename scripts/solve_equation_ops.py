#!/usr/bin/env python3
"""Per-puzzle integer/string-level operator search for equation_numeric_deduce.

f(A,B) = OUTER( BASE( TA(A), TB(B) ) ), TA/TB/OUTER in {id,rev}, string-level so
leading zeros from reversal are preserved. BASE is an expanded integer/string
family (arithmetic with small const offsets, div/mod/min/max, concat).

The operands are literal numbers (identity digit mapping); only the operator
symbol has hidden semantics, fit PER-PUZZLE by grouping the examples that share
the query's operator symbol. Verified coverage on the real Kaggle puzzles: 70.7%.

Also provides verified CoT generation (`cot_for`) and on-distribution
augmentation (`augment`) used to build dataset v14.
"""
import json
import os
import random
import re
import sys


def rev(s):
    neg = s.startswith('-')
    if neg:
        s = s[1:]
    r = s[::-1]
    return ('-' + r) if neg else r


def T_id(s):
    return s


def T_rev(s):
    return rev(s)


TRANSFORMS = {'id': T_id, 'rev': T_rev}


def to_int(s):
    try:
        return int(s)
    except Exception:
        return None


def _int_base(fn):
    """wrap an int->int (or None) op as a string-returning base on two strings."""
    def g(x, y):
        a, b = to_int(x), to_int(y)
        if a is None or b is None:
            return None
        v = fn(a, b)
        if v is None:
            return None
        return str(v)
    return g


BASES = {}
# arithmetic with small constant offsets
for k in (0, 1, 2, -1, -2):
    suf = '' if k == 0 else f'{k:+d}'
    BASES[f'add{suf}'] = _int_base(lambda a, b, k=k: a + b + k)
    BASES[f'sub{suf}'] = _int_base(lambda a, b, k=k: a - b + k)
    BASES[f'rsub{suf}'] = _int_base(lambda a, b, k=k: b - a + k)
    BASES[f'mul{suf}'] = _int_base(lambda a, b, k=k: a * b + k)
    BASES[f'absdiff{suf}'] = _int_base(lambda a, b, k=k: abs(a - b) + k)
# mod-100 (keep 2 digits) variants
BASES['addmod'] = lambda x, y: (None if to_int(x) is None or to_int(y) is None
                                else f"{(int(x) + int(y)) % 100:02d}")
BASES['submod'] = lambda x, y: (None if to_int(x) is None or to_int(y) is None
                                else f"{(int(x) - int(y)) % 100:02d}")
BASES['mulmod'] = lambda x, y: (None if to_int(x) is None or to_int(y) is None
                                else f"{(int(x) * int(y)) % 100:02d}")
# div / mod / min / max
BASES['fdiv'] = _int_base(lambda a, b: (a // b) if b else None)
BASES['mod'] = _int_base(lambda a, b: (a % b) if b else None)
BASES['rfdiv'] = _int_base(lambda a, b: (b // a) if a else None)
BASES['rmod'] = _int_base(lambda a, b: (b % a) if a else None)
BASES['min'] = _int_base(lambda a, b: min(a, b))
BASES['max'] = _int_base(lambda a, b: max(a, b))
# string concat
BASES['concat'] = lambda x, y: x + y
BASES['rconcat'] = lambda x, y: y + x

CANDS = []
for ta in TRANSFORMS:
    for tb in TRANSFORMS:
        for bn in BASES:
            for outer in ('id', 'rev'):
                CANDS.append((ta, tb, bn, outer))


def apply_cand(cand, A, B):
    ta, tb, bn, outer = cand
    x = TRANSFORMS[ta](A)
    y = TRANSFORMS[tb](B)
    r = BASES[bn](x, y)
    if r is None:
        return None
    if outer == 'rev':
        r = rev(r)
    return r


def cand_str(cand, sym=''):
    ta, tb, bn, outer = cand
    wa = "rev(A)" if ta == 'rev' else "A"
    wb = "rev(B)" if tb == 'rev' else "B"
    expr = f"{bn}({wa},{wb})"
    if outer == 'rev':
        expr = f"rev({expr})"
    return expr


def parse(prompt):
    exs = []
    for m in re.finditer(r'(\d+)\s*(\D)\s*(\d+)\s*=\s*(-?\d+)', prompt):
        exs.append((m.group(1), m.group(2), m.group(3), m.group(4)))
    qm = re.search(r'determine the result for:\s*(\d+)\s*(\D)\s*(\d+)', prompt)
    if not qm:
        return None, None
    return exs, (qm.group(1), qm.group(2), qm.group(3))


def gold_of(ans):
    m = re.search(r'\\boxed\{([^}]*)\}', ans)
    return m.group(1).strip() if m else None


def solve(prompt, gold=None):
    """Return (ans, cand) if a single operator family reproduces the query
    group (and matches gold when provided), else None."""
    exs, q = parse(prompt)
    if not q:
        return None
    qa, qsym, qb = q
    group = [(a, b, r) for (a, sym, b, r) in exs if sym == qsym]
    if not group:
        return None
    for cand in CANDS:
        ok = True
        for (a, b, r) in group:
            if apply_cand(cand, a, b) != r:
                ok = False
                break
        if not ok:
            continue
        ans = apply_cand(cand, qa, qb)
        if ans is None:
            continue
        if gold is not None and ans != gold:
            continue
        return ans, cand
    return None


def format_cot(exs, q, ans, cand):
    """Build a compact, verified chain-of-thought (v8 style)."""
    qa, qsym, qb = q
    group = [(a, b, r) for (a, sym, b, r) in exs if sym == qsym]
    expr = cand_str(cand)
    lines = ["<think>"]
    lines.append("The operands and results are plain numbers; only the operator "
                 "symbol has hidden semantics.")
    lines.append(f"Deduced rule for operator '{qsym}':  f(A,B) = {expr}")
    lines.append("Verify against the examples:")
    for (a, b, r) in group[:6]:
        lines.append(f"  {a} {qsym} {b} = {r}  =>  f({a},{b}) = {apply_cand(cand, a, b)}")
    lines.append(f"Apply to the question {qa} {qsym} {qb}:")
    lines.append(f"  f({qa},{qb}) = {ans}")
    lines.append("</think>")
    return "\n".join(lines) + f"\n\\boxed{{{ans}}}"


def cot_for(prompt, gold=None):
    res = solve(prompt, gold)
    if not res:
        return None
    ans, cand = res
    exs, q = parse(prompt)
    return ans, format_cot(exs, q, ans, cand)


# ----------------------------------------------------------------------------
# On-distribution augmentation: given a discovered operator family, mint fresh
# verified puzzles by resampling operands (no search needed -> fast & parallel).
# ----------------------------------------------------------------------------
PROMPT_HEAD = ("In Alice's Wonderland, a secret set of transformation rules is "
               "applied to equations. Below are a few examples:")
SYMBOLS = [c for c in "!\"#$%&'()*+,-./:;<=>?@[]^_`{|}~" ]


def _operand(rng):
    return f"{rng.randint(0, 99):02d}"


def make_puzzle(cand, rng, n_examples=3, distractor=True):
    """Generate a verified equation puzzle for a given operator family `cand`.

    Returns (prompt, gold, exs, q, cand) or None if degenerate."""
    qsym = rng.choice(SYMBOLS)
    lines = []
    seen = set()
    tries = 0
    while len(lines) < n_examples and tries < 200:
        tries += 1
        a, b = _operand(rng), _operand(rng)
        r = apply_cand(cand, a, b)
        if r is None or (a, b) in seen:
            continue
        seen.add((a, b))
        lines.append(f"{a}{qsym}{b} = {r}")
    if len(lines) < n_examples:
        return None
    # query (distinct from examples)
    for _ in range(200):
        qa, qb = _operand(rng), _operand(rng)
        if (qa, qb) in seen:
            continue
        gold = apply_cand(cand, qa, qb)
        if gold is not None:
            break
    else:
        return None
    # optional distractor line with a different symbol + different family
    if distractor:
        dsym = rng.choice([s for s in SYMBOLS if s != qsym])
        dcand = rng.choice(CANDS)
        for _ in range(50):
            da, db = _operand(rng), _operand(rng)
            dr = apply_cand(dcand, da, db)
            if dr is not None:
                lines.append(f"{da}{dsym}{db} = {dr}")
                break
    rng.shuffle(lines)
    prompt = (PROMPT_HEAD + "\n" + "\n".join(lines)
              + f"\nNow, determine the result for: {qa}{qsym}{qb}")
    exs, q = parse(prompt)
    return prompt, gold, exs, q, cand


def augment(cand, n, seed=0):
    """Yield up to n verified (prompt, gold, cot) for the family `cand`."""
    rng = random.Random(seed)
    out = 0
    attempts = 0
    while out < n and attempts < n * 20:
        attempts += 1
        mk = make_puzzle(cand, rng)
        if not mk:
            continue
        prompt, gold, exs, q, _ = mk
        # confirm the puzzle is uniquely solvable to the same answer
        res = solve(prompt, gold)
        if not res:
            continue
        ans, fcand = res
        if ans != gold:
            continue
        yield prompt, gold, format_cot(exs, q, ans, fcand)
        out += 1


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(here, '..', 'data', 'kaggle_classified.jsonl')
    rows = [json.loads(l) for l in open(data)]
    eq = [r for r in rows if r['category'] == 'equation_numeric_deduce']
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(eq)
    solved = 0
    for r in eq[:n]:
        gold = gold_of(r['answer'])
        if cot_for(r['prompt'], gold):
            solved += 1
    print(f"equation integer-level: {solved}/{n} = {100 * solved / n:.1f}%  (cands={len(CANDS)})")
