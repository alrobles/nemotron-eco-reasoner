#!/usr/bin/env python3
"""
Merge and prepare training datasets for Nemotron-3-Nano fine-tuning.

Prioritizes chain-of-thought (CoT) data for reasoning improvement.
Merges:
  1. train_cot_unified.jsonl (7,076 Kaggle puzzles with full CoT)
  2. cryptarithm_cot.jsonl (500 synthetic cryptarithms with CoT)
  3. ecology_chat.jsonl (1,586 ecology CoT traces)

Output: Single JSONL in messages format ready for SFTTrainer.

Usage:
  python scripts/prepare_cot_dataset.py \
    --cot-data data/train_cot_unified.jsonl \
    --cryptarithm data/cryptarithm_cot.jsonl \
    --ecology data/ecology_chat.jsonl \
    --output data/train_sprint_v2.jsonl
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    """Load JSONL, skip empty/broken lines."""
    examples = []
    skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                examples.append(ex)
            except json.JSONDecodeError:
                skipped += 1
    if skipped:
        print(f"  Skipped {skipped} malformed lines in {path}")
    return examples


def validate_messages(ex):
    """Check that an example has proper messages format."""
    msgs = ex.get("messages", [])
    if not msgs or len(msgs) < 2:
        return False

    roles = [m.get("role", "") for m in msgs]
    has_assistant = "assistant" in roles
    has_user = "user" in roles
    if not has_assistant or not has_user:
        return False

    # Assistant must have content (not just \boxed{})
    for m in msgs:
        if m.get("role") == "assistant":
            content = m.get("content", "")
            if len(content.strip()) < 10:
                return False

    return True


def has_cot(ex):
    """Check if the assistant response has chain-of-thought (>200 chars)."""
    for m in ex.get("messages", []):
        if m.get("role") == "assistant":
            content = m.get("content", "")
            # CoT has step-by-step reasoning, not just answer
            if "STEP" in content or "step" in content.lower():
                return True
            # Long response is likely CoT
            if len(content) > 300:
                return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Prepare CoT training dataset")
    parser.add_argument("--cot-data", required=True, help="Path to train_cot_unified.jsonl")
    parser.add_argument("--cryptarithm", default=None, help="Path to cryptarithm_cot.jsonl")
    parser.add_argument("--ecology", default=None, help="Path to ecology_chat.jsonl")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max-examples", type=int, default=0,
                        help="Cap total examples (0=use all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    all_examples = []

    # ── 1. Load CoT Kaggle data ───────────────────────────────────
    print(f"Loading CoT Kaggle data: {args.cot_data}")
    cot = load_jsonl(args.cot_data)
    valid_cot = [e for e in cot if validate_messages(e)]
    print(f"  {len(cot)} total, {len(valid_cot)} valid messages")

    cot_with_cot = [e for e in valid_cot if has_cot(e)]
    cot_no_cot = [e for e in valid_cot if not has_cot(e)]
    print(f"  With CoT: {len(cot_with_cot)} | Without CoT: {len(cot_no_cot)}")

    all_examples.extend(cot_with_cot)
    # Only include prompt/answer ones if we need more data
    all_examples.extend(cot_no_cot)

    # ── 2. Load cryptarithm CoT ───────────────────────────────────
    if args.cryptarithm and Path(args.cryptarithm).exists():
        print(f"Loading cryptarithm CoT: {args.cryptarithm}")
        crypt = load_jsonl(args.cryptarithm)
        valid_crypt = [e for e in crypt if validate_messages(e)]
        print(f"  {len(crypt)} total, {len(valid_crypt)} valid")
        all_examples.extend(valid_crypt)

    # ── 3. Load ecology CoT ───────────────────────────────────────
    if args.ecology and Path(args.ecology).exists():
        print(f"Loading ecology CoT: {args.ecology}")
        eco = load_jsonl(args.ecology)
        valid_eco = [e for e in eco if validate_messages(e)]
        print(f"  {len(eco)} total, {len(valid_eco)} valid")
        all_examples.extend(valid_eco)

    # ── Shuffle ───────────────────────────────────────────────────
    random.shuffle(all_examples)

    # Cap if needed
    if args.max_examples > 0 and len(all_examples) > args.max_examples:
        all_examples = all_examples[:args.max_examples]
        print(f"Capped to {args.max_examples} examples")

    # ── Stats ─────────────────────────────────────────────────────
    total_cot = sum(1 for e in all_examples if has_cot(e))
    avg_assistant_len = 0
    for e in all_examples:
        for m in e.get("messages", []):
            if m.get("role") == "assistant":
                avg_assistant_len += len(m.get("content", ""))
    avg_assistant_len = avg_assistant_len / len(all_examples) if all_examples else 0

    print(f"\n=== Final Dataset ===")
    print(f"Total: {len(all_examples)} examples")
    print(f"With CoT: {total_cot} ({100*total_cot/len(all_examples):.1f}%)")
    print(f"Avg assistant length: {avg_assistant_len:.0f} chars")

    # ── Write ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for ex in all_examples:
            f.write(json.dumps({"messages": ex["messages"]}) + "\n")

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Saved: {args.output} ({size_mb:.1f} MB)")
    print("DONE!")


if __name__ == "__main__":
    main()
