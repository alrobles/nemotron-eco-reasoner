#!/usr/bin/env python3
"""CSP solver for cryptarithm_deduce puzzles.

Model (validated against ground truth, same family as the Progress Prize winner):
- Every lhs is 5 symbols: A(2) op(1) B(2). Each non-op symbol maps to a unique
  digit 0-9 (injective). Operator symbols map to one of:
  add, abs_diff, mul, concat (A*100+B, zero-padded to 4), rev_concat (B*100+A).
- Backtracking over examples assigns digits; the query is decoded through the
  inverse mapping.
- With the ground-truth answer available (training data), we accept any
  consistent solution whose query decoding matches it.

Writes data/cryptarithm_solutions.jsonl: {"md5": ..., "answer": ..., "cot": ...}
Usage: python3 scripts/solve_cryptarithm_csp.py [start] [count] [--budget N]
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


# Each op takes the four operand digits (a = d0d1, b = d3d4) and returns the
# candidate result digit-tuples it can produce.
def _op_add(d0, d1, d3, d4):
    return (num_to_digits((d0 * 10 + d1) + (d3 * 10 + d4)),)


def _op_absdiff(d0, d1, d3, d4):
    return (num_to_digits(abs((d0 * 10 + d1) - (d3 * 10 + d4))),)


def _op_mul(d0, d1, d3, d4):
    return (num_to_digits((d0 * 10 + d1) * (d3 * 10 + d4)),)


def _op_concat(d0, d1, d3, d4):
    return ((d0, d1, d3, d4),)


def _op_revconcat(d0, d1, d3, d4):
    return ((d3, d4, d0, d1),)


def _op_submod(d0, d1, d3, d4):
    x = ((d0 * 10 + d1) - (d3 * 10 + d4)) % 100
    return ((x // 10, x % 10), num_to_digits(x))


def _op_addmod(d0, d1, d3, d4):
    x = ((d0 * 10 + d1) + (d3 * 10 + d4)) % 100
    return ((x // 10, x % 10), num_to_digits(x))


def _op_mulmod(d0, d1, d3, d4):
    x = ((d0 * 10 + d1) * (d3 * 10 + d4)) % 100
    return ((x // 10, x % 10), num_to_digits(x))


def _op_dadd(d0, d1, d3, d4):
    return (((d0 + d3) % 10, (d1 + d4) % 10),)


def _op_ddiff(d0, d1, d3, d4):
    return ((abs(d0 - d3), abs(d1 - d4)),)


def _op_dmul(d0, d1, d3, d4):
    return (((d0 * d3) % 10, (d1 * d4) % 10),)


def _op_addrev(d0, d1, d3, d4):
    return (tuple(reversed(num_to_digits((d0 * 10 + d1) + (d3 * 10 + d4)))),)


def _op_mulrev(d0, d1, d3, d4):
    return (tuple(reversed(num_to_digits((d0 * 10 + d1) * (d3 * 10 + d4)))),)


OPS = [_op_add, _op_absdiff, _op_mul, _op_concat, _op_revconcat, _op_submod,
       _op_addmod, _op_mulmod, _op_dadd, _op_ddiff, _op_dmul, _op_addrev,
       _op_mulrev]
OP_NAMES = ["add", "abs_diff", "mul", "concat", "rev_concat", "sub_mod100",
            "add_mod100", "mul_mod100", "digit_add", "digit_diff", "digit_mul",
            "add_rev", "mul_rev"]


def _fmt_digits(t):
    return "".join(str(d) for d in t)


OP_FMT = {
    "add": lambda a, b: f"{a} + {b} = {a + b}",
    "abs_diff": lambda a, b: f"|{a} - {b}| = {abs(a - b)}",
    "mul": lambda a, b: f"{a} * {b} = {a * b}",
    "concat": lambda a, b: f"concat({a}, {b}) = {a:02d}{b:02d}",
    "rev_concat": lambda a, b: f"rev_concat({a}, {b}) = {b:02d}{a:02d}",
    "sub_mod100": lambda a, b: f"({a} - {b}) mod 100 = {(a - b) % 100}",
    "add_mod100": lambda a, b: f"({a} + {b}) mod 100 = {(a + b) % 100}",
    "mul_mod100": lambda a, b: f"({a} * {b}) mod 100 = {(a * b) % 100}",
    "digit_add": lambda a, b: f"digit-wise ({a} + {b}) mod 10 = {_fmt_digits(_op_dadd(a // 10, a % 10, b // 10, b % 10)[0])}",
    "digit_diff": lambda a, b: f"digit-wise |{a} - {b}| = {_fmt_digits(_op_ddiff(a // 10, a % 10, b // 10, b % 10)[0])}",
    "digit_mul": lambda a, b: f"digit-wise ({a} * {b}) mod 10 = {_fmt_digits(_op_dmul(a // 10, a % 10, b // 10, b % 10)[0])}",
    "add_rev": lambda a, b: f"reverse({a} + {b}) = {_fmt_digits(_op_addrev(a // 10, a % 10, b // 10, b % 10)[0])}",
    "mul_rev": lambda a, b: f"reverse({a} * {b}) = {_fmt_digits(_op_mulrev(a // 10, a % 10, b // 10, b % 10)[0])}",
}


def is_concat(ex):
    s0, s1, _, s3, s4, rsyms = ex
    return rsyms == (s0, s1, s3, s4) or rsyms == (s3, s4, s0, s1)


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
        feasible_ops = list(range(len(OPS)))
        for d0 in self._vals(s0):
            n0 = self._assign(s0, d0)
            if n0 is None:
                continue
            for d1 in self._vals(s1):
                n1 = self._assign(s1, d1)
                if n1 is None:
                    continue
                lv = d0 * 10 + d1
                for d3 in self._vals(s3):
                    n3 = self._assign(s3, d3)
                    if n3 is None:
                        continue
                    for d4 in self._vals(s4):
                        n4 = self._assign(s4, d4)
                        if n4 is None:
                            continue
                        rv = d3 * 10 + d4
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
    lines = ["We deduce a symbol-to-digit mapping and operator meanings from the examples.",
             "Each lhs is two 2-digit numbers joined by an operator symbol; each symbol maps to a unique digit."]
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
        lines.append(
            f"  {s0}{s1}{opsym}{s3}{s4} = {''.join(rsyms)}  =>  {OP_FMT[name](a, b)}"
        )
    lines.append("")
    qs0, qs1, qop, qs3, qs4 = query
    a = mapping[qs0] * 10 + mapping[qs1]
    b = mapping[qs3] * 10 + mapping[qs4]
    name = op_info[qop]
    lines.append(f"Apply to the question {qs0}{qs1}{qop}{qs3}{qs4}:")
    lines.append(f"  {OP_FMT[name](a, b)}")
    lines.append(f"  Mapping the digits back to symbols gives {answer}.")
    return "\n".join(lines)


def solve_cryptarithm_csp(prompt, expected=None, budget=30.0):
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

    try:
        answers, info = Solver(arith, query, unique=True,
                               deadline=time.time() + budget).solve()
    except TimeoutError:
        answers, info = Counter(), {}
    chosen = pick(answers, info)
    if chosen is None and expected is not None:
        try:
            answers, info = Solver(arith, query, unique=False,
                                   deadline=time.time() + budget).solve()
            chosen = pick(answers, info)
        except TimeoutError:
            chosen = None
    if chosen is None:
        return None
    ans, (mapping, op_info) = chosen
    cot = build_cot(examples, query, ans, mapping, op_info)
    return ans, cot


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    start = int(args[0]) if args else 0
    count = int(args[1]) if len(args) > 1 else 10**9
    budget = 30.0
    for a in sys.argv[1:]:
        if a.startswith('--budget'):
            budget = float(a.split('=')[1])
    out_path = f'data/cryptarithm_solutions_{start}.jsonl' if count < 10**9 else 'data/cryptarithm_solutions.jsonl'
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
            res = solve_cryptarithm_csp(d['prompt'], expected=gt, budget=budget)
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
    print(f'start={start}: solved {solved}/{total} in {time.time()-t0:.0f}s -> {out_path}')


if __name__ == '__main__':
    main()
