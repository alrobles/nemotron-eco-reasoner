#!/usr/bin/env python3
"""Cryptarithm solver (cryptarithm_deduce).

Structure: each LHS is 5 symbols "o1[0] o1[1] OP o2[0] o2[1]"; the symbol at
position 2 is the operator (its meaning is PER-SYMBOL), and a single shared
injective symbol->digit cipher applies across all examples. RHS is the
symbol-encoding of the integer result (variable length, leading-zero
preserving).

We filter the examples to those whose operator matches the query's operator,
then jointly solve (cipher, operator family) by forward-checking, reusing the
integer-op family from `solve_equation_ops`. Verified coverage on the real
Kaggle puzzles: 60.7% (~76% of the solvable subset).

Also provides verified CoT (`cot_for`) and search-free augmentation (`augment`)
used to build dataset v14.
"""
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solve_equation_ops as E


def parse(prompt):
    lhs = []
    for ln in prompt.splitlines():
        ln = ln.strip()
        if '=' not in ln or 'determine' in ln.lower():
            continue
        a, b = ln.split('=', 1)
        a, b = a.strip(), b.strip()
        if len(a) == 5 and b:
            lhs.append((a, b))
    qm = re.search(r'determine the result for:\s*(\S+)', prompt)
    if not qm or len(lhs) < 1:
        return None
    q = qm.group(1)
    if len(q) != 5:
        return None
    op = q[2]
    same = [(a, b) for (a, b) in lhs if a[2] == op]
    if not same:
        return None          # query operator unseen in examples
    return same, q


def operand_syms(constraints):
    s = set()
    for a, b in constraints:
        s |= {a[0], a[1], a[3], a[4]}
    return sorted(s)


def solve(prompt, gold=None, budget=5.0):
    pr = parse(prompt)
    if not pr:
        return None
    lhs, q = pr
    constraints = list(lhs)
    if gold is not None:
        constraints = lhs + [(q, gold)]
    osyms = operand_syms(constraints)
    if len(osyms) > 10:
        return None
    order = []
    for a, b in constraints:
        for c in (a[0], a[1], a[3], a[4]):
            if c not in order:
                order.append(c)
    deadline = time.time() + budget
    for cand in E.CANDS:
        res = _search(cand, constraints, order, 0, {}, [False] * 10, deadline)
        if res is None:
            continue
        m = res
        o1 = str(m[q[0]]) + str(m[q[1]])
        o2 = str(m[q[3]]) + str(m[q[4]])
        rstr = E.apply_cand(cand, o1, o2)
        if rstr is None or '-' in rstr:
            continue
        ans = _map_back(rstr, m)
        if ans is None:
            continue
        if gold is not None and ans != gold:
            continue
        return ans, cand, dict(m)
    return None


def _derive(cand, a, b, m):
    if any(s not in m for s in (a[0], a[1], a[3], a[4])):
        return True, {}
    o1 = str(m[a[0]]) + str(m[a[1]])
    o2 = str(m[a[3]]) + str(m[a[4]])
    rstr = E.apply_cand(cand, o1, o2)
    if rstr is None or '-' in rstr:
        return False, None
    if len(rstr) != len(b):
        return False, None
    used = set(m.values())
    add = {}
    for ch, sym in zip(rstr, b):
        d = int(ch)
        cur = m.get(sym, add.get(sym))
        if cur is not None:
            if cur != d:
                return False, None
        else:
            if d in used or d in add.values():
                return False, None
            add[sym] = d
    return True, add


def _search(cand, constraints, order, i, m, used, deadline):
    if time.time() > deadline:
        return None
    if i == len(order):
        full = dict(m)
        for a, b in constraints:
            ok, add = _derive(cand, a, b, full)
            if not ok:
                return None
            for s, d in add.items():
                if s in full and full[s] != d:
                    return None
                full[s] = d
        return full
    s = order[i]
    for d in range(10):
        if used[d]:
            continue
        m[s] = d
        used[d] = True
        ok = True
        for a, b in constraints:
            if all(x in m for x in (a[0], a[1], a[3], a[4])):
                good, add = _derive(cand, a, b, m)
                if not good:
                    ok = False
                    break
        if ok:
            r = _search(cand, constraints, order, i + 1, m, used, deadline)
            if r is not None:
                return r
        del m[s]
        used[d] = False
    return None


def _map_back(rstr, m):
    inv = {v: k for k, v in m.items()}
    out = []
    for ch in rstr:
        d = int(ch)
        if d not in inv:
            return None
        out.append(inv[d])
    return ''.join(out)


def format_cot(lhs, q, ans, cand, m):
    """Compact verified CoT. `lhs` are examples sharing the query operator."""
    op = q[2]
    # only display symbols that actually appear in the puzzle
    used_syms = set()
    for a, b in list(lhs) + [(q, ans)]:
        used_syms |= set(a) | set(b)
    used_syms.discard(op)
    expr = E.cand_str(cand)
    lines = ["<think>"]
    lines.append(f"We deduce a symbol->digit cipher and the meaning of operator "
                 f"'{op}' from the examples.")
    lines.append("Each lhs is two 2-digit numbers joined by an operator symbol; "
                 "each symbol maps to a unique digit.")
    lines.append("Deduced symbol-to-digit mapping:")
    for s in sorted(used_syms):
        if s in m:
            lines.append(f"  '{s}' = {m[s]}")
    lines.append(f"Operator '{op}':  f(A,B) = {expr}")
    lines.append("Verify against the examples:")
    for a, b in list(lhs)[:5]:
        o1 = str(m[a[0]]) + str(m[a[1]])
        o2 = str(m[a[3]]) + str(m[a[4]])
        r = E.apply_cand(cand, o1, o2)
        lines.append(f"  {a} = {b}  =>  f({o1},{o2}) = {r}")
    o1 = str(m[q[0]]) + str(m[q[1]])
    o2 = str(m[q[3]]) + str(m[q[4]])
    rstr = E.apply_cand(cand, o1, o2)
    lines.append(f"Apply to the question {q}:")
    lines.append(f"  f({o1},{o2}) = {rstr}; re-encoding with the mapping gives {ans}.")
    lines.append("</think>")
    return "\n".join(lines) + f"\n\\boxed{{{ans}}}"


def cot_for(prompt, gold=None, budget=5.0):
    res = solve(prompt, gold, budget)
    if not res:
        return None
    ans, cand, m = res
    lhs, q = parse(prompt)
    return ans, format_cot(lhs, q, ans, cand, m)


# ----------------------------------------------------------------------------
# Search-free augmentation: pick a fresh full bijection cipher + an operator
# family, then mint verified puzzles directly (no CSP search needed).
# ----------------------------------------------------------------------------
PROMPT_HEAD = ("In Alice's Wonderland, a secret set of transformation rules is "
               "applied to equations. Below are a few examples:")
ALPHABET = list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _encode(rstr, inv):
    out = []
    for ch in rstr:
        d = int(ch)
        if d not in inv:
            return None
        out.append(inv[d])
    return ''.join(out)


def make_puzzle(cand, rng, n_examples=4):
    """Generate a verified cryptarithm puzzle for operator family `cand`.

    Uses a fresh full bijection (10 symbols -> 0..9) so any result digit
    encodes. Returns (prompt, gold, lhs, q, m) or None."""
    syms = rng.sample(ALPHABET, 10)
    m = {s: d for d, s in enumerate(syms)}        # symbol -> digit
    inv = {d: s for s, d in m.items()}            # digit -> symbol
    opsym = rng.choice([c for c in ALPHABET if c not in syms])
    osyms = syms                                  # operand symbols

    def gen_line():
        for _ in range(40):
            a = rng.choice(osyms) + rng.choice(osyms)
            b = rng.choice(osyms) + rng.choice(osyms)
            o1 = str(m[a[0]]) + str(m[a[1]])
            o2 = str(m[b[0]]) + str(m[b[1]])
            r = E.apply_cand(cand, o1, o2)
            if r is None or '-' in r:
                continue
            enc = _encode(r, inv)
            if enc is None:
                continue
            lhs_sym = a[0] + a[1] + opsym + b[0] + b[1]
            return lhs_sym, enc
        return None

    lines = []
    seen = set()
    while len(lines) < n_examples:
        g = gen_line()
        if not g:
            return None
        if g[0] in seen:
            continue
        seen.add(g[0])
        lines.append(g)
    # query
    for _ in range(60):
        g = gen_line()
        if not g:
            return None
        if g[0] not in seen:
            qln, gold = g
            break
    else:
        return None
    body = "\n".join(f"{a} = {b}" for a, b in lines)
    prompt = (PROMPT_HEAD + "\n" + body
              + f"\nNow, determine the result for: {qln}")
    lhs = [(a, b) for a, b in lines]
    return prompt, gold, lhs, qln, dict(m)


def augment(cand, n, seed=0):
    rng = random.Random(seed)
    out = 0
    attempts = 0
    while out < n and attempts < n * 30:
        attempts += 1
        mk = make_puzzle(cand, rng)
        if not mk:
            continue
        prompt, gold, lhs, q, m = mk
        yield prompt, gold, format_cot(lhs, q, gold, cand, m)
        out += 1


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(here, '..', 'data', 'kaggle_classified.jsonl')
    rows = [json.loads(l) for l in open(data)]
    cr = [r for r in rows if r['category'] == 'cryptarithm_deduce']
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(cr)
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    t0 = time.time()
    solved = sum(1 for r in cr[:n]
                 if cot_for(r['prompt'], E.gold_of(r['answer']), budget))
    print(f"cryptarithm: {solved}/{n} = {100 * solved / n:.1f}% in "
          f"{time.time() - t0:.0f}s (budget={budget})")
