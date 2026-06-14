#!/usr/bin/env python3
"""Build dataset v14.

Recipe (approved by the user, "arranca la granja con la receta v14"):

  * Keep v8's winning DNA: full category coverage, short COMPLETE chain-of-thought
    that always ends in the correct \\boxed{}, calibrated synthetic volume,
    seq_len 3072.
  * Replicate v8's exact per-category counts, BUT replace the chain-of-thought of
    the three hard categories (equation / cryptarithm / bit) with VERIFIED traces
    produced by the operator-discovery solvers:
        - solve the REAL Kaggle puzzles first (correct-by-construction CoT,
          gold-checked), then
        - top up to v8's count with on-distribution augmentation that resamples
          operands/inputs from the *discovered* operator families (so the
          synthetic operator distribution matches the real test distribution).
  * num / unit / cipher  -> taken verbatim from v8 (proven base).
  * gravity              -> taken verbatim from v9 (least-squares estimator).

Diagnosis lever: v8 is stuck at 0.67 because its hard-category CoT was heuristic
and in the WRONG operator family (cryptarithm scored 0/8 locally).  v14 keeps the
same distribution but upgrades those CoTs to verified, correct-family deductions.

Run:
    python3 scripts/build_v14_dataset.py --workers 8

Output: data/train_deterministic_v14.jsonl (messages schema, ready to train).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import classify_category as C  # noqa: E402
import solve_equation_ops as EQ  # noqa: E402
import solve_cryptarithm_ops as CR  # noqa: E402
import solve_bit_perbit as BT  # noqa: E402

SYSTEM = "You are a helpful assistant that solves puzzles step by step."


def rec(prompt: str, cot: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": cot},
        ]
    }


# ---------------------------------------------------------------------------
# per-puzzle solve workers (top-level so multiprocessing can pickle them)
# ---------------------------------------------------------------------------
def _solve_eqn(args):
    prompt, gold = args
    res = EQ.solve(prompt, gold)
    if not res:
        return None
    ans, cand = res
    exs, q = EQ.parse(prompt)
    return prompt, EQ.format_cot(exs, q, ans, cand), cand


def _solve_bit(args):
    prompt, gold = args
    res = BT.solve(prompt, gold)
    if not res:
        return None
    pred, chosen = res
    pairs, q = BT.parse(prompt)
    return prompt, BT.format_cot(pairs, q, pred, chosen), tuple(chosen)


def _solve_cryp(args):
    prompt, gold, budget = args
    res = CR.solve(prompt, gold, budget)
    if not res:
        return None
    ans, cand, m = res
    lhs, q = CR.parse(prompt)
    return prompt, CR.format_cot(lhs, q, ans, cand, m), cand


def gold_of(answer: str) -> str:
    return EQ.gold_of(answer)


def load_real(kaggle_path, canon):
    rows = [json.loads(l) for l in open(kaggle_path)]
    out = [(r["prompt"], gold_of(r["answer"])) for r in rows if r["category"] == canon]
    return out


def _verify_one(args):
    """Return prompt if well-posed (solvable to its stated gold), else None."""
    solve_fn, prompt, gold, vbudget = args
    task = (prompt, gold, vbudget) if vbudget is not None else (prompt, gold)
    res = solve_fn(task)
    return prompt if res else None


# ---------------------------------------------------------------------------
def build_hard(name, canon, kaggle_path, solve_fn, augment_fn, target,
               workers, seed, crypt_budget=None, surplus=1.6,
               verify_budget=None):
    """Return (records, stats) for one hard category.

    Real puzzles are solved (gold-checked verified CoT).  Augmentation cycles
    the *discovered* operator rules so the synthetic distribution matches the
    real test distribution, and every augmented puzzle is RE-VERIFIED to be
    well-posed (uniquely solvable to its stated answer) so the CoT teaches a
    genuinely deducible deduction."""
    real = load_real(kaggle_path, canon)
    if crypt_budget is not None:
        tasks = [(p, g, crypt_budget) for (p, g) in real]
    else:
        tasks = real
    t0 = time.time()
    if workers > 1:
        with Pool(workers) as pool:
            solved = pool.map(solve_fn, tasks, chunksize=8)
    else:
        solved = [solve_fn(t) for t in tasks]
    solved = [s for s in solved if s]
    real_records = [rec(p, cot) for (p, cot, _rule) in solved]
    rules = [rule for (_p, _cot, rule) in solved]
    n_real = len(real_records)
    solve_t = time.time() - t0

    # augment to fill target, cycling the discovered rules (matches real distro)
    need = max(0, target - n_real)
    aug_records = []
    n_generated = 0
    if rules and need:
        rng = random.Random(seed)
        # generate a surplus so we can drop under-constrained puzzles
        want = int(need * surplus)
        cand = {}  # prompt -> (gold, cot)
        i = 0
        per = max(1, want // len(rules) + 1)
        attempts = 0
        while len(cand) < want and attempts < want * 50:
            rule = rules[i % len(rules)]
            i += 1
            attempts += 1
            s = rng.randint(0, 2**31 - 1)
            for (prompt, g, cot) in augment_fn(rule, per, seed=s):
                if len(cand) >= want:
                    break
                if prompt not in cand:
                    cand[prompt] = (g, cot)
        n_generated = len(cand)
        # well-posedness verification (parallel)
        vb = verify_budget if verify_budget is not None else crypt_budget
        vtasks = [(solve_fn, p, g, vb) for p, (g, _c) in cand.items()]
        if workers > 1:
            with Pool(workers) as pool:
                verified = pool.map(_verify_one, vtasks, chunksize=8)
        else:
            verified = [_verify_one(t) for t in vtasks]
        wellposed = {p for p in verified if p}
        for p, (g, c) in cand.items():
            if p in wellposed:
                aug_records.append(rec(p, c))
            if len(aug_records) >= need:
                break

    records = real_records + aug_records[:need]
    stats = dict(category=name, target=target, real=n_real,
                 real_total=len(real), aug=len(records) - n_real,
                 aug_generated=n_generated, total=len(records),
                 solve_s=round(time.time() - t0, 1))
    return records, stats


def pull_from(path, cats):
    """Return {cat: [records]} for the requested short cats, taken verbatim."""
    buckets = {c: [] for c in cats}
    for l in open(path):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        c = C.classify(C.user_prompt(r))
        if c in buckets:
            buckets[c].append(r)
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v8", default=os.path.join(HERE, "..", "data", "train_deterministic_v8.jsonl"))
    ap.add_argument("--v9", default=os.path.join(HERE, "..", "data", "train_deterministic_v9.jsonl"))
    ap.add_argument("--kaggle", default=os.path.join(HERE, "..", "data", "kaggle_classified.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "train_deterministic_v14.jsonl"))
    ap.add_argument("--target-eqn", type=int, default=1702)
    ap.add_argument("--target-cryp", type=int, default=2610)
    ap.add_argument("--target-bit", type=int, default=2128)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--crypt-budget", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    print(f"[v14] workers={args.workers} crypt_budget={args.crypt_budget}")

    all_stats = []
    out_records = []

    # --- easy categories straight from v8 -----------------------------------
    v8 = pull_from(args.v8, ["num", "unit", "ciph"])
    for c in ["num", "unit", "ciph"]:
        out_records += v8[c]
        all_stats.append(dict(category=c, source="v8", total=len(v8[c])))

    # --- gravity from v9 -----------------------------------------------------
    v9 = pull_from(args.v9, ["grav"])
    out_records += v9["grav"]
    all_stats.append(dict(category="grav", source="v9", total=len(v9["grav"])))

    # --- hard categories: verified solver CoT + matched augmentation --------
    eqn_recs, st = build_hard("eqn", "equation_numeric_deduce", args.kaggle,
                              _solve_eqn, EQ.augment, args.target_eqn,
                              args.workers, args.seed + 1)
    out_records += eqn_recs
    all_stats.append(st)
    print("[v14]", st)

    bit_recs, st = build_hard("bit", "bit_manipulation", args.kaggle,
                              _solve_bit, BT.augment, args.target_bit,
                              args.workers, args.seed + 2)
    out_records += bit_recs
    all_stats.append(st)
    print("[v14]", st)

    cryp_recs, st = build_hard("cryp", "cryptarithm_deduce", args.kaggle,
                               _solve_cryp, CR.augment, args.target_cryp,
                               args.workers, args.seed + 3,
                               crypt_budget=args.crypt_budget,
                               surplus=1.5, verify_budget=3.0)
    out_records += cryp_recs
    all_stats.append(st)
    print("[v14]", st)

    # --- dedup by user prompt, shuffle, write -------------------------------
    seen = set()
    deduped = []
    for r in out_records:
        u = C.user_prompt(r)
        if u in seen:
            continue
        seen.add(u)
        deduped.append(r)
    rng = random.Random(args.seed)
    rng.shuffle(deduped)

    with open(args.out, "w") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[v14] wrote {len(deduped)} records -> {args.out}")
    print("[v14] composition:")
    os.system(f"python3 {os.path.join(HERE, 'classify_category.py')} {args.out}")


if __name__ == "__main__":
    main()
