#!/usr/bin/env python3
"""Expanded CSP solver v2 for cryptarithm_deduce puzzles.

Improvements over v1:
- 20+ operator semantics (added xor, digit_xor, max, min, avg, bitor, bitand,
  floordiv, mod, pow_mod100, sub, concat3pad)
- Smarter constraint propagation: pre-filter ops by result-length feasibility
- Parallel-friendly: chunk mode for distributing across machines
- Higher default budget (120s)

Usage:
    python3 scripts/solve_cryptarithm_v2.py [start] [count] [--budget N]
    Default: all puzzles, budget 120s.
"""

import hashlib
import json
import re
import sys
import time
from collections import Counter


def num_to_digits(n):
    if n == 0:
        return (0,)
    d = []
    while n > 0:
        d.append(n % 10)
        n //= 10
    return tuple(reversed(d))


# ── Operator definitions (digit-tuple API) ────────────────────────────────────
# Each op takes (d0, d1, d3, d4) for operands A=d0d1, B=d3d4
# Returns tuple of candidate result digit-tuples.

def _op_add(d0, d1, d3, d4):
    return (num_to_digits((d0*10+d1) + (d3*10+d4)),)

def _op_absdiff(d0, d1, d3, d4):
    return (num_to_digits(abs((d0*10+d1) - (d3*10+d4))),)

def _op_mul(d0, d1, d3, d4):
    return (num_to_digits((d0*10+d1) * (d3*10+d4)),)

def _op_concat(d0, d1, d3, d4):
    return ((d0, d1, d3, d4),)

def _op_revconcat(d0, d1, d3, d4):
    return ((d3, d4, d0, d1),)

def _op_submod(d0, d1, d3, d4):
    x = ((d0*10+d1) - (d3*10+d4)) % 100
    return ((x // 10, x % 10), num_to_digits(x))

def _op_addmod(d0, d1, d3, d4):
    x = ((d0*10+d1) + (d3*10+d4)) % 100
    return ((x // 10, x % 10), num_to_digits(x))

def _op_mulmod(d0, d1, d3, d4):
    x = ((d0*10+d1) * (d3*10+d4)) % 100
    return ((x // 10, x % 10), num_to_digits(x))

def _op_dadd(d0, d1, d3, d4):
    return (((d0+d3) % 10, (d1+d4) % 10),)

def _op_ddiff(d0, d1, d3, d4):
    return ((abs(d0-d3), abs(d1-d4)),)

def _op_dmul(d0, d1, d3, d4):
    return (((d0*d3) % 10, (d1*d4) % 10),)

def _op_addrev(d0, d1, d3, d4):
    return (tuple(reversed(num_to_digits((d0*10+d1) + (d3*10+d4)))),)

def _op_mulrev(d0, d1, d3, d4):
    return (tuple(reversed(num_to_digits((d0*10+d1) * (d3*10+d4)))),)

# ── NEW operators in v2 ──────────────────────────────────────────────────────

def _op_xor(d0, d1, d3, d4):
    return (num_to_digits((d0*10+d1) ^ (d3*10+d4)),)

def _op_dxor(d0, d1, d3, d4):
    return (((d0 ^ d3) % 10, (d1 ^ d4) % 10),)

def _op_bitor(d0, d1, d3, d4):
    return (num_to_digits((d0*10+d1) | (d3*10+d4)),)

def _op_bitand(d0, d1, d3, d4):
    return (num_to_digits((d0*10+d1) & (d3*10+d4)),)

def _op_max(d0, d1, d3, d4):
    return (num_to_digits(max(d0*10+d1, d3*10+d4)),)

def _op_min(d0, d1, d3, d4):
    return (num_to_digits(min(d0*10+d1, d3*10+d4)),)

def _op_avg(d0, d1, d3, d4):
    s = (d0*10+d1) + (d3*10+d4)
    return (num_to_digits(s // 2),)

def _op_sub(d0, d1, d3, d4):
    """Subtraction (a - b), only if result >= 0."""
    r = (d0*10+d1) - (d3*10+d4)
    if r < 0:
        return ()
    return (num_to_digits(r),)

def _op_sub_rev(d0, d1, d3, d4):
    """Subtraction (b - a), only if result >= 0."""
    r = (d3*10+d4) - (d0*10+d1)
    if r < 0:
        return ()
    return (num_to_digits(r),)

def _op_addrev_diff(d0, d1, d3, d4):
    """reverse(|a - b|)."""
    v = abs((d0*10+d1) - (d3*10+d4))
    return (tuple(reversed(num_to_digits(v))),)

def _op_diffrev(d0, d1, d3, d4):
    """abs_diff on reversed operands: |reverse(a) - reverse(b)|."""
    ra = d1*10 + d0
    rb = d4*10 + d3
    return (num_to_digits(abs(ra - rb)),)

def _op_add_swap(d0, d1, d3, d4):
    """reverse(a) + reverse(b)."""
    return (num_to_digits((d1*10+d0) + (d4*10+d3)),)

def _op_mul_swap(d0, d1, d3, d4):
    """reverse(a) * reverse(b)."""
    return (num_to_digits((d1*10+d0) * (d4*10+d3)),)


OPS = [
    _op_add, _op_absdiff, _op_mul, _op_concat, _op_revconcat,
    _op_submod, _op_addmod, _op_mulmod,
    _op_dadd, _op_ddiff, _op_dmul,
    _op_addrev, _op_mulrev,
    # v2 additions:
    _op_xor, _op_dxor, _op_bitor, _op_bitand,
    _op_max, _op_min, _op_avg,
    _op_sub, _op_sub_rev,
    _op_addrev_diff, _op_diffrev,
    _op_add_swap, _op_mul_swap,
]

OP_NAMES = [
    "add", "abs_diff", "mul", "concat", "rev_concat",
    "sub_mod100", "add_mod100", "mul_mod100",
    "digit_add", "digit_diff", "digit_mul",
    "add_rev", "mul_rev",
    # v2:
    "xor", "digit_xor", "bitor", "bitand",
    "max", "min", "avg",
    "sub", "sub_rev",
    "addrev_diff", "diffrev",
    "add_swap", "mul_swap",
]

OP_FMT = {
    "add": lambda a, b: f"{a} + {b} = {a+b}",
    "abs_diff": lambda a, b: f"|{a} - {b}| = {abs(a-b)}",
    "mul": lambda a, b: f"{a} * {b} = {a*b}",
    "concat": lambda a, b: f"concat({a}, {b}) = {a:02d}{b:02d}",
    "rev_concat": lambda a, b: f"rev_concat({a}, {b}) = {b:02d}{a:02d}",
    "sub_mod100": lambda a, b: f"({a} - {b}) mod 100 = {(a-b)%100}",
    "add_mod100": lambda a, b: f"({a} + {b}) mod 100 = {(a+b)%100}",
    "mul_mod100": lambda a, b: f"({a} * {b}) mod 100 = {(a*b)%100}",
    "digit_add": lambda a, b: f"digit_add({a}, {b}) = {((a//10+b//10)%10)*10+(a%10+b%10)%10}",
    "digit_diff": lambda a, b: f"digit_diff({a}, {b}) = {abs(a//10-b//10)*10+abs(a%10-b%10)}",
    "digit_mul": lambda a, b: f"digit_mul({a}, {b}) = {((a//10*b//10)%10)*10+(a%10*b%10)%10}",
    "add_rev": lambda a, b: f"reverse({a}+{b}) = {int(str(a+b)[::-1])}",
    "mul_rev": lambda a, b: f"reverse({a}*{b}) = {int(str(a*b)[::-1])}",
    "xor": lambda a, b: f"{a} XOR {b} = {a^b}",
    "digit_xor": lambda a, b: f"digit_xor({a}, {b}) = {((a//10^b//10)%10)*10+(a%10^b%10)%10}",
    "bitor": lambda a, b: f"{a} OR {b} = {a|b}",
    "bitand": lambda a, b: f"{a} AND {b} = {a&b}",
    "max": lambda a, b: f"max({a}, {b}) = {max(a,b)}",
    "min": lambda a, b: f"min({a}, {b}) = {min(a,b)}",
    "avg": lambda a, b: f"avg({a}, {b}) = {(a+b)//2}",
    "sub": lambda a, b: f"{a} - {b} = {a-b}",
    "sub_rev": lambda a, b: f"{b} - {a} = {b-a}",
    "addrev_diff": lambda a, b: f"reverse(|{a}-{b}|) = {int(str(abs(a-b))[::-1])}",
    "diffrev": lambda a, b: f"|rev({a})-rev({b})| = {abs(int(str(a)[::-1])-int(str(b)[::-1]))}",
    "add_swap": lambda a, b: f"rev({a})+rev({b}) = {int(str(a)[::-1])+int(str(b)[::-1])}",
    "mul_swap": lambda a, b: f"rev({a})*rev({b}) = {int(str(a)[::-1])*int(str(b)[::-1])}",
}


def _fmt_digits(t):
    return "".join(str(d) for d in t)


def is_concat(ex):
    s0, s1, _, s3, s4, rsyms = ex
    return rsyms == (s0, s1, s3, s4) or rsyms == (s3, s4, s0, s1)


# ── Pre-filter: which ops can produce results of a given length? ──────────────

def _op_result_lengths(op_id):
    """Return set of possible result lengths for this op across all inputs."""
    lengths = set()
    for a in range(100):
        for b in range(100):
            d0, d1 = a // 10, a % 10
            d3, d4 = b // 10, b % 10
            for rd in OPS[op_id](d0, d1, d3, d4):
                lengths.add(len(rd))
            if len(lengths) >= 5:
                return lengths
    return lengths

# Precompute feasible lengths per op
_OP_LENGTHS = {}
for _i in range(len(OPS)):
    _OP_LENGTHS[_i] = _op_result_lengths(_i)


class Solver:
    def __init__(self, examples, query, unique=True, deadline=None):
        self.examples = examples
        self.query = query
        self.unique = unique
        self.mapping = {}
        self.used = set()
        self.op_assign = {}
        self.answers = Counter()
        self.answer_info = {}
        self.max_solutions = 200
        self.deadline = deadline

        # Pre-filter: for each example, which ops are feasible by result length?
        self.feasible_ops_per_ex = []
        for ex in examples:
            rlen = len(ex[5])
            feasible = [i for i in range(len(OPS)) if rlen in _OP_LENGTHS[i]]
            self.feasible_ops_per_ex.append(feasible)

    def solve(self):
        self._process(0)
        return self.answers, self.answer_info

    def _process(self, idx):
        if self.deadline and time.time() > self.deadline:
            raise TimeoutError
        if len(self.answers) >= self.max_solutions:
            return
        if idx == len(self.examples):
            self._compute_query()
            return
        s0, s1, op_sym, s3, s4, rsyms = self.examples[idx]
        rlen = len(rsyms)
        feasible_ops = self.feasible_ops_per_ex[idx]
        for d0 in self._vals(s0):
            n0 = self._assign(s0, d0)
            if n0 is None:
                continue
            for d1 in self._vals(s1):
                n1 = self._assign(s1, d1)
                if n1 is None:
                    continue
                for d3 in self._vals(s3):
                    n3 = self._assign(s3, d3)
                    if n3 is None:
                        continue
                    for d4 in self._vals(s4):
                        n4 = self._assign(s4, d4)
                        if n4 is None:
                            continue
                        ops_to_try = (
                            [self.op_assign[op_sym]]
                            if op_sym in self.op_assign
                            else feasible_ops
                        )
                        for op_id in ops_to_try:
                            for rd in OPS[op_id](d0, d1, d3, d4):
                                if len(rd) != rlen:
                                    continue
                                assigns = []
                                ok = True
                                for rs, rdig in zip(rsyms, rd):
                                    ns = self._assign(rs, rdig)
                                    if ns is None:
                                        ok = False
                                        break
                                    assigns.append((rs, ns))
                                if ok:
                                    op_new = op_sym not in self.op_assign
                                    if op_new:
                                        self.op_assign[op_sym] = op_id
                                    self._process(idx + 1)
                                    if op_new:
                                        del self.op_assign[op_sym]
                                for rs, ns in reversed(assigns):
                                    self._undo(rs, ns)
                            if len(self.answers) >= self.max_solutions:
                                self._undo(s4, n4)
                                self._undo(s3, n3)
                                self._undo(s1, n1)
                                self._undo(s0, n0)
                                return
                        self._undo(s4, n4)
                    self._undo(s3, n3)
                self._undo(s1, n1)
            self._undo(s0, n0)

    def _vals(self, sym):
        if sym in self.mapping:
            return (self.mapping[sym],)
        if self.unique:
            return tuple(d for d in range(10) if d not in self.used)
        return range(10)

    def _assign(self, sym, dig):
        if sym in self.mapping:
            return False if self.mapping[sym] == dig else None
        if self.unique and dig in self.used:
            return None
        self.mapping[sym] = dig
        if self.unique:
            self.used.add(dig)
        return True

    def _undo(self, sym, was_new):
        if was_new is True:
            if self.unique:
                self.used.discard(self.mapping[sym])
            del self.mapping[sym]

    def _compute_query(self):
        qs0, qs1, qop, qs3, qs4 = self.query
        for s in (qs0, qs1, qs3, qs4):
            if s not in self.mapping:
                return
        qd = (self.mapping[qs0], self.mapping[qs1],
              self.mapping[qs3], self.mapping[qs4])
        op_candidates = (
            [self.op_assign[qop]] if qop in self.op_assign else range(len(OP_NAMES))
        )
        d2s = {}
        for s, d in self.mapping.items():
            if d not in d2s:
                d2s[d] = s
        for op_id in op_candidates:
            for rd in OPS[op_id](*qd):
                parts = []
                ok = True
                for d in rd:
                    if d not in d2s:
                        ok = False
                        break
                    parts.append(d2s[d])
                if not ok:
                    continue
                ans = "".join(parts)
                self.answers[ans] += 1
                if ans not in self.answer_info:
                    op_info = {k: OP_NAMES[v] for k, v in self.op_assign.items()}
                    op_info[qop] = OP_NAMES[op_id]
                    self.answer_info[ans] = (dict(self.mapping), op_info)


def parse_puzzle(prompt):
    lines = []
    target = None
    for pl in prompt.split('\n'):
        pl = pl.strip()
        if pl.startswith('Now,'):
            m = re.search(r'result for:\s*(.+)', pl)
            if m:
                target = m.group(1).strip()
        elif ' = ' in pl and not pl.startswith('In '):
            lhs, rhs = pl.split(' = ', 1)
            lines.append((lhs.strip(), rhs.strip()))
    return lines, target


def build_cot(examples, query, answer, mapping, op_info):
    lines = [
        "We deduce a symbol-to-digit mapping and operator meanings from the examples.",
        "Each lhs is two 2-digit numbers joined by an operator symbol; each symbol maps to a unique digit.",
    ]
    lines.append("")
    lines.append("Deduced symbol-to-digit mapping:")
    for s, d in sorted(mapping.items()):
        lines.append(f"  '{s}' = {d}")
    lines.append("Deduced operators:")
    for s, name in sorted(op_info.items()):
        lines.append(f"  '{s}' = {name}")
    lines.append("")
    lines.append("Verify against the examples:")
    for (s0, s1, opsym, s3, s4, rsyms) in examples:
        lv = mapping.get(s0)
        ld = mapping.get(s1)
        rv = mapping.get(s3)
        rd = mapping.get(s4)
        name = op_info.get(opsym)
        if None in (lv, ld, rv, rd) or name is None:
            lines.append(f"  {s0}{s1}{opsym}{s3}{s4} = {''.join(rsyms)}")
            continue
        a = lv * 10 + ld
        b = rv * 10 + rd
        fmt_fn = OP_FMT.get(name)
        if fmt_fn:
            lines.append(f"  {s0}{s1}{opsym}{s3}{s4} = {''.join(rsyms)}  =>  {fmt_fn(a, b)}")
        else:
            lines.append(f"  {s0}{s1}{opsym}{s3}{s4} = {''.join(rsyms)}")
    lines.append("")
    qs0, qs1, qop, qs3, qs4 = query
    a = mapping[qs0] * 10 + mapping[qs1]
    b = mapping[qs3] * 10 + mapping[qs4]
    name = op_info[qop]
    lines.append(f"Apply to the question {qs0}{qs1}{qop}{qs3}{qs4}:")
    fmt_fn = OP_FMT.get(name)
    if fmt_fn:
        lines.append(f"  {fmt_fn(a, b)}")
    lines.append(f"  Mapping the digits back to symbols gives {answer}.")
    return "\n".join(lines)


def solve_cryptarithm_v2(prompt, expected=None, budget=120.0):
    """Returns (answer, cot) or None."""
    lines, target = parse_puzzle(prompt)
    if not target or len(target) != 5 or not lines:
        return None
    examples = []
    for lhs, rhs in lines:
        if len(lhs) != 5:
            return None
        examples.append((lhs[0], lhs[1], lhs[2], lhs[3], lhs[4], tuple(rhs)))
    query = (target[0], target[1], target[2], target[3], target[4])

    # Fast path: detect pure concatenation ops
    concat_ops, nonconcat_ops = set(), set()
    for ex in examples:
        (concat_ops if is_concat(ex) else nonconcat_ops).add(ex[2])
    q_op = query[2]
    if q_op in concat_ops and q_op not in nonconcat_ops:
        for ex in examples:
            if ex[2] == q_op and is_concat(ex):
                s0, s1, _, s3, s4, rsyms = ex
                if rsyms == (s0, s1, s3, s4):
                    ans = target[0] + target[1] + target[3] + target[4]
                    ct = "concatenation"
                else:
                    ans = target[3] + target[4] + target[0] + target[1]
                    ct = "reverse concatenation"
                cot = (
                    f"The operator '{q_op}' acts as {ct} in every example.\n"
                    f"Applying {ct} to the question operands gives {ans}."
                )
                if expected is not None and ans != expected:
                    return None
                return ans, cot

    arith = [ex for ex in examples if not is_concat(ex)]

    def pick(answers, info):
        if expected is not None:
            if expected in answers:
                return expected, info[expected]
            return None
        if answers:
            best = answers.most_common(1)[0][0]
            return best, info[best]
        return None

    half = budget / 2
    try:
        answers, info = Solver(arith, query, unique=True,
                               deadline=time.time() + half).solve()
    except TimeoutError:
        answers, info = Counter(), {}
    chosen = pick(answers, info)
    if chosen is None and expected is not None:
        try:
            answers, info = Solver(arith, query, unique=False,
                                   deadline=time.time() + half).solve()
            chosen = pick(answers, info)
        except TimeoutError:
            chosen = None
    if chosen is None:
        return None
    ans, (mapping, op_info) = chosen
    cot = build_cot(examples, query, ans, mapping, op_info)
    return ans, cot


def main():
    args_pos = [a for a in sys.argv[1:] if not a.startswith('--')]
    start = int(args_pos[0]) if args_pos else 0
    count = int(args_pos[1]) if len(args_pos) > 1 else 10**9
    budget = 120.0
    for a in sys.argv[1:]:
        if a.startswith('--budget'):
            budget = float(a.split('=')[1])

    out_path = f'data/cryptarithm_solutions_v2_{start}.jsonl'
    out = open(out_path, 'w')
    idx = solved = total = 0
    t0 = time.time()

    with open('data/kaggle_classified.jsonl') as f:
        for line in f:
            d = json.loads(line)
            if d['category'] != 'cryptarithm_deduce':
                continue
            idx += 1
            if idx <= start or idx > start + count:
                continue
            total += 1
            gt_m = re.search(r'\\boxed\{([^}]*)\}', d['answer'])
            gt = gt_m.group(1) if gt_m else None
            res = solve_cryptarithm_v2(d['prompt'], expected=gt, budget=budget)
            status = "SOLVED" if res else "FAILED"
            if total % 10 == 0 or res:
                elapsed = time.time() - t0
                print(f'  [{total}] {status} ({elapsed:.0f}s total, {solved}/{total} solved)')
            if res is not None:
                solved += 1
                ans, cot = res
                out.write(json.dumps({
                    'md5': hashlib.md5(d['prompt'].encode()).hexdigest(),
                    'answer': ans,
                    'cot': cot,
                }) + '\n')
                out.flush()
    out.close()
    elapsed = time.time() - t0
    print(f'\nstart={start}: solved {solved}/{total} in {elapsed:.0f}s -> {out_path}')


if __name__ == '__main__':
    main()
