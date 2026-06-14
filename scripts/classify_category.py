#!/usr/bin/env python3
"""Category classifier for the Nemotron reasoning puzzles.

Maps a puzzle *user prompt* to one of the 7 evaluation categories.  Verified
100% accurate against the 5000 ground-truth-labelled Kaggle puzzles in
data/kaggle_classified.jsonl (the canonical `category` field).

Short codes used internally:
    num  -> numeral
    unit -> unit_conversion
    ciph -> cipher
    bit  -> bit_manipulation
    eqn  -> equation_numeric_deduce
    cryp -> cryptarithm_deduce
    grav -> gravity

The only non-trivial case is equation vs cryptarithm: both share the header
"... transformation rules ... applied to equations ...".  They are separated by
the fraction of digit characters in the example operands/results (equation uses
plain numbers, cryptarithm encodes everything with symbols).
"""
from __future__ import annotations

import collections
import json
import re
import statistics
import sys

# short-code -> canonical Kaggle category name
CANON = {
    "num": "numeral",
    "unit": "unit_conversion",
    "ciph": "cipher",
    "bit": "bit_manipulation",
    "eqn": "equation_numeric_deduce",
    "cryp": "cryptarithm_deduce",
    "grav": "gravity",
}
SHORT = {v: k for k, v in CANON.items()}
CATS = ["num", "unit", "ciph", "bit", "eqn", "cryp", "grav", "?"]


def classify(prompt: str) -> str:
    """Return the short category code for a puzzle user-prompt."""
    low = prompt.lower()
    if "bit manipulation" in low or "8-bit binary" in low:
        return "bit"
    if "numeral system" in low:
        return "num"
    if "unit conversion is applied" in low or " becomes " in low:
        return "unit"
    if "gravitational constant" in low or "distance =" in low:
        return "grav"
    if "encryption rules" in low:
        return "ciph"
    if (
        "transformation rules is applied to equations" in low
        or "transformation rules are applied to equations" in low
    ):
        m = re.findall(r"([^\s=]+)\s*=\s*([^\s=]+)", prompt)
        sm = "".join(a + b for a, b in m[:4])
        if not sm:
            return "eqn"
        d = sum(c.isdigit() for c in sm)
        t = sum(not c.isspace() for c in sm)
        return "eqn" if t and d / t > 0.5 else "cryp"
    return "?"


def classify_canon(prompt: str) -> str:
    """Return the canonical Kaggle category name (or '?')."""
    return CANON.get(classify(prompt), "?")


def user_prompt(record: dict) -> str:
    """Extract the user prompt from a training record (messages schema)."""
    msgs = record.get("messages") or record.get("conversations") or []
    return next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")


def assistant_text(record: dict) -> str:
    msgs = record.get("messages") or record.get("conversations") or []
    return next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")


def _selftest() -> None:
    """Verify 100% accuracy against the labelled Kaggle puzzles."""
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "data", "kaggle_classified.jsonl")
    ok = n = 0
    confusion = collections.Counter()
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        gold = r["category"]
        pred = classify_canon(r["prompt"])
        n += 1
        if pred == gold:
            ok += 1
        else:
            confusion[(gold, pred)] += 1
    print(f"classifier self-test: {ok}/{n} = {ok / n:.4%}")
    for (g, p), c in confusion.most_common(20):
        print(f"  MISS gold={g} pred={p} x{c}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args == ["--selftest"]:
        _selftest()
        sys.exit(0)
    # otherwise: tally category composition of each given jsonl file
    for path in args:
        cnt = collections.Counter()
        lens = collections.defaultdict(list)
        n = 0
        try:
            for line in open(path):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                u = user_prompt(r)
                a = assistant_text(r)
                c = classify(u)
                cnt[c] += 1
                lens[c].append(len(a))
                n += 1
        except Exception as e:  # noqa: BLE001
            print(path, "ERR", e)
            continue
        name = path.split("/")[-1]
        print(f"{name:<32} N={n}")
        print("  cnt :", " ".join(f"{c}={cnt.get(c, 0)}" for c in CATS))
        print(
            "  cot :",
            " ".join(
                f"{c}={int(statistics.mean(lens[c])) if lens.get(c) else 0}"
                for c in CATS[:-1]
            ),
        )
