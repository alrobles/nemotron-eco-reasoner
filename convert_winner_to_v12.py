#!/usr/bin/env python3
"""Convert winner's reasoning traces to our JSONL training format.

Reads:
  - ~/scratch/huikang-nemotron/train.csv (prompts + answers)
  - ~/scratch/huikang-nemotron/problems.jsonl (categories)
  - ~/scratch/huikang-nemotron/reasoning/<id>.txt (verified CoT traces)

Outputs:
  - data/train_v12.jsonl (our messages format)
"""

import csv
import json
import re
from pathlib import Path

WINNER_REPO = Path("/home/a474r867/scratch/huikang-nemotron")
TRAIN_CSV = WINNER_REPO / "train.csv"
PROBLEMS_FILE = WINNER_REPO / "problems.jsonl"
REASONING_DIR = WINNER_REPO / "reasoning"
OUTPUT_FILE = Path("/home/a474r867/scratch/nemotron-eco-reasoner/data/train_v12.jsonl")

SYSTEM_PROMPT = "You are a helpful assistant that solves puzzles step by step."
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def extract_answer(trace: str) -> str:
    """Extract final answer from \\boxed{...} in the trace."""
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", trace)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
    return ""


def main():
    print(f"Loading prompts from {TRAIN_CSV}...")
    prompts = {}
    with open(TRAIN_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts[row["id"]] = row["prompt"]

    print(f"Loading categories from {PROBLEMS_FILE}...")
    categories = {}
    with open(PROBLEMS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                p = json.loads(line)
                categories[p["id"]] = p.get("category", "unknown")

    trace_files = sorted(REASONING_DIR.glob("*.txt"))
    print(f"Found {len(trace_files)} reasoning traces")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    cat_counts = {}

    with open(OUTPUT_FILE, "w") as out:
        for tf in trace_files:
            pid = tf.stem
            if pid not in prompts:
                skipped += 1
                continue

            trace = tf.read_text().strip()
            if not trace:
                skipped += 1
                continue

            answer = extract_answer(trace)
            if not answer:
                skipped += 1
                continue

            cat = categories.get(pid, "unknown")
            user_content = prompts[pid] + PROMPT_SUFFIX
            assistant_content = f"<think>\n{trace}\n</think>\n\\boxed{{{answer}}}"

            entry = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
            }

            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

    print(f"\n=== RESULTS ===")
    print(f"  Written: {written}")
    print(f"  Skipped: {skipped}")
    print(f"\n=== CATEGORY DISTRIBUTION ===")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / written if written else 0
        print(f"  {cat}: {count} ({pct:.1f}%)")
    print(f"\nOutput: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
