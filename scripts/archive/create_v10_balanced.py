#!/usr/bin/env python3
"""
Create balanced v10 dataset matching Kaggle's test distribution.

Problem: v9 dataset is 44.5% cryptarithm + 22% bit_manipulation, but Kaggle
test is ~16.7% each category. Models trained on v9 overfit to cryptarithm and
lose performance on gravity, cipher, numeral, unit_conversion.

Solution: Sample from all available data to match Kaggle's uniform ~16.7%
distribution across 6 categories. Prefer v6 "expert" traces as base,
supplement with best v9 traces for cryptarithm (which v6 lacked).

Kaggle categories (each ~16.7%):
  gravity, numeral, cryptarithm_deduce, cipher, bit_manipulation, unit_conversion
"""

import json
import hashlib
import random
import sys
from pathlib import Path
from collections import defaultdict

random.seed(42)

DATA_DIR = Path(__file__).parent.parent / "data"


def categorize(prompt: str) -> str:
    lc = prompt.lower()
    if "bit manipulation" in lc:
        return "bit_manipulation"
    if "numeral" in lc or ("base" in lc and "convert" in lc):
        return "numeral"
    if "unit" in lc and "conversion" in lc:
        return "unit_conversion"
    if "gravitational" in lc or "free fall" in lc:
        return "gravity"
    if "encryption" in lc or "cipher" in lc or "decrypt" in lc:
        return "cipher"
    if "transformation rules" in lc:
        return "cryptarithm_deduce"
    return "UNKNOWN"


def get_user_content(example: dict) -> str:
    for m in example.get("messages", []):
        if m["role"] == "user":
            return m["content"]
    return ""


def get_assistant_content(example: dict) -> str:
    for m in example.get("messages", []):
        if m["role"] == "assistant":
            return m["content"]
    return ""


def prompt_hash(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()


def has_boxed(text: str) -> bool:
    return "\\boxed" in text


def load_dataset(path: str) -> list:
    examples = []
    for line in open(path):
        d = json.loads(line.strip())
        examples.append(d)
    return examples


def main():
    # Load all datasets
    v6 = load_dataset(DATA_DIR / "train_deterministic_v6.jsonl")
    v9 = load_dataset(DATA_DIR / "train_deterministic_v9.jsonl")

    # Categorize and deduplicate
    # Priority: v6 traces first (proven quality), then v9 for gaps
    by_category = defaultdict(list)  # cat -> [(example, source, priority)]

    seen_hashes = set()

    # v6 first (higher priority = lower number)
    for ex in v6:
        user = get_user_content(ex)
        asst = get_assistant_content(ex)
        cat = categorize(user)
        h = prompt_hash(user)

        if cat == "UNKNOWN":
            continue
        if not has_boxed(asst):
            continue

        if h not in seen_hashes:
            seen_hashes.add(h)
            by_category[cat].append((ex, "v6", 0))

    # v9 for additional examples (lower priority)
    for ex in v9:
        user = get_user_content(ex)
        asst = get_assistant_content(ex)
        cat = categorize(user)
        h = prompt_hash(user)

        if cat == "UNKNOWN":
            continue
        if not has_boxed(asst):
            continue

        if h not in seen_hashes:
            seen_hashes.add(h)
            by_category[cat].append((ex, "v9", 1))

    # Report available examples per category
    print("Available examples per category:")
    for cat in sorted(by_category.keys()):
        examples = by_category[cat]
        v6_count = sum(1 for _, src, _ in examples if src == "v6")
        v9_count = sum(1 for _, src, _ in examples if src == "v9")
        print(f"  {cat:25s}: {len(examples):5d} (v6={v6_count}, v9={v9_count})")

    # Target: balanced distribution matching Kaggle test (~16.7% each)
    # Use the minimum available category as the cap to keep balance
    TARGET_PER_CAT = 833  # Match the smallest Kaggle category
    MIN_AVAILABLE = min(len(by_category[c]) for c in by_category)
    ACTUAL_PER_CAT = min(TARGET_PER_CAT, MIN_AVAILABLE)

    print(f"\nTarget per category: {ACTUAL_PER_CAT}")
    print(f"Total target: {ACTUAL_PER_CAT * len(by_category)}")

    # Sample from each category, preferring v6 traces
    final_dataset = []
    for cat in sorted(by_category.keys()):
        examples = by_category[cat]
        # Sort by priority (v6 first)
        examples.sort(key=lambda x: (x[2], random.random()))
        selected = examples[:ACTUAL_PER_CAT]
        final_dataset.extend([(ex, cat, src) for ex, src, _ in selected])
        v6_sel = sum(1 for _, src, _ in selected if src == "v6")
        v9_sel = sum(1 for _, src, _ in selected if src == "v9")
        print(f"  {cat:25s}: selected {len(selected)} (v6={v6_sel}, v9={v9_sel})")

    # Shuffle the final dataset
    random.shuffle(final_dataset)

    # Write output
    output_path = DATA_DIR / "train_balanced_v10.jsonl"
    with open(output_path, "w") as f:
        for ex, cat, src in final_dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nWritten {len(final_dataset)} examples to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Verify distribution
    print("\nVerification:")
    cats = defaultdict(int)
    for _, cat, _ in final_dataset:
        cats[cat] += 1
    for c, n in sorted(cats.items()):
        print(f"  {c:25s}: {n:5d} ({100*n/len(final_dataset):.1f}%)")


if __name__ == "__main__":
    main()
