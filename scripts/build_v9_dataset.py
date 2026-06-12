#!/usr/bin/env python3
"""Build train_deterministic_v9.jsonl: v8 with gravity records regenerated.

v7/v8 gravity CoT used a median-of-observations g estimate; benchmarked against
the 850 kaggle gravity puzzles it reproduces the gold answer 82.5% of the time,
while the least-squares estimator g = sum(2*d*t^2) / sum(t^4) at full precision
reproduces 89.2% (rounding g to 1-3 decimals drops this to 7-85%). The eval
showed checkpoint-250 missing all gravity answers by centesimas — the model
imitated the lower-precision procedure. v9 replaces every gravity record with a
verified least-squares CoT, kept only when it reproduces the gold exactly.

Usage: python scripts/build_v9_dataset.py
Reads  data/train_deterministic_v8.jsonl + data/kaggle_classified.jsonl
Writes data/train_deterministic_v9.jsonl
"""
import json
import re

SRC = "data/train_deterministic_v8.jsonl"
CLASSIFIED = "data/kaggle_classified.jsonl"
OUT = "data/train_deterministic_v9.jsonl"

SYSTEM = "You are an expert in physics calculations. Think step by step and place your final answer inside \\boxed{}."


def parse(prompt):
    obs = [(float(t), float(d)) for t, d in
           re.findall(r"For t = ([\d.]+)s?, distance = ([\d.]+) m", prompt)]
    m = re.search(r"falling distance for t = ([\d.]+)s", prompt)
    return obs, (float(m.group(1)) if m else None)


def fmt(x):
    s = f"{x:.2f}"
    return s[:-1] if s.endswith("0") else s


def gravity_cot(prompt, gold):
    obs, tt = parse(prompt)
    if not obs or tt is None:
        return None
    num = sum(2 * d * t * t for t, d in obs)
    den = sum(t ** 4 for t, _ in obs)
    g = num / den
    dist = 0.5 * g * tt * tt
    pred = fmt(dist)
    if pred != gold and f"{dist:.2f}" != gold and f"{dist:.1f}" != gold:
        return None
    answer = gold
    cot = "Step 1: Estimate g with a least-squares fit over ALL observations.\n"
    cot += "From d = 0.5*g*t^2, the least-squares estimate is g = sum(2*d*t^2) / sum(t^4).\n\n"
    for t, d in obs:
        cot += f"  t={t}, d={d}: 2*d*t^2 = {2*d*t*t:.6f}, t^4 = {t**4:.6f}\n"
    cot += f"\n  numerator   = {num:.6f}\n  denominator = {den:.6f}\n"
    cot += f"  g = {num:.6f} / {den:.6f} = {g:.6f} m/s^2\n\n"
    cot += "Step 2: Compute the distance for the target time, keeping full precision until the end.\n"
    cot += f"  d = 0.5 * {g:.6f} * {tt}^2 = 0.5 * {g:.6f} * {tt*tt:.6f} = {dist:.6f} m\n\n"
    cot += f"Step 3: Round to two decimals (drop a trailing zero): {answer}\n"
    return f"<think>\n{cot}</think>\n\\boxed{{{answer}}}"


def main():
    gravity_gold = {}
    with open(CLASSIFIED) as f:
        for line in f:
            r = json.loads(line)
            if r.get("category") == "gravity":
                m = re.search(r"\\boxed\{([^}]*)\}", r["answer"])
                gravity_gold[r["prompt"]] = m.group(1).strip() if m else r["answer"].strip()

    kept, replaced, dropped = 0, 0, 0
    out = []
    with open(SRC) as f:
        for line in f:
            r = json.loads(line)
            ms = r.get("messages", [])
            user = next((m["content"] for m in ms if m["role"] == "user"), "")
            prompt = user.replace("\n\nPlease put your final answer inside \\boxed{}.", "")
            if "gravitational constant" in prompt and "falling distance" in prompt:
                gold = gravity_gold.get(prompt)
                if gold is None:
                    m = re.search(r"\\boxed\{([^}]*)\}", ms[-1]["content"])
                    gold = m.group(1).strip() if m else None
                new = gravity_cot(prompt, gold) if gold else None
                if new is None:
                    dropped += 1
                    continue
                r["messages"] = [m for m in ms]
                for i, m in enumerate(r["messages"]):
                    if m["role"] == "system":
                        r["messages"][i] = {"role": "system", "content": SYSTEM}
                r["messages"][-1] = {"role": "assistant", "content": new}
                replaced += 1
            else:
                kept += 1
            out.append(r)

    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"v9: {len(out)} records ({kept} kept, {replaced} gravity regenerated, {dropped} gravity dropped)")


if __name__ == "__main__":
    main()
