#!/usr/bin/env python3
"""
Consolidate all data sources into a category-aware, balanced training dataset.

Data sources:
  1. Kaggle CoT traces (with category classification)
  2. Cryptarithm CoT traces (synthetic)
  3. Ecology reasoning traces

Features:
  - Classifies all puzzles by the 9 evaluation categories
  - Computes per-category statistics (count, match rate)
  - Balance strategy: oversample weak categories, cap strong ones
  - Integrates all sources into unified ShareGPT JSONL
  - Category metadata for downstream analysis

Usage:
    python3 consolidate_dataset_v2.py \
        --classified data/kaggle_classified.jsonl \
        --cot-file data/train_cot_unified.jsonl \
        --cryptarithm-file data/cryptarithm_cot.jsonl \
        --output data/train_cot_unified_v3.jsonl \
        --balance \
        --target-per-category 800
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)

# ── Weightage from evaluation table (user provided) ──
# These determine how much each category matters for overall score
EVAL_WEIGHTAGE = {
    'bit_manipulation': 17.8,
    'cipher': 17.1,
    'cryptarithm_deduce': 7.5,
    'cryptarithm_guess': 1.5,
    'equation_numeric_deduce': 5.1,
    'equation_numeric_guess': 0.7,
    'gravity': 16.7,
    'numeral': 15.7,
    'unit_conversion': 18.0,
}

# ── Category-specific system prompts ──

SYSTEM_PROMPTS = {
    'bit_manipulation': (
        "You are an expert in binary arithmetic and bit manipulation. "
        "For bit manipulation puzzles, analyze the input-output pairs to identify "
        "the bitwise operation pattern (AND, OR, XOR, shift, rotate, NOT). "
        "Work through bit by bit. Place your final answer inside \\boxed{}."
    ),
    'cipher': (
        "You are an expert cryptanalyst. For cipher/encryption puzzles, "
        "analyze the example mappings to identify the substitution pattern "
        "(letter mapping, shift cipher, or more complex transformation). "
        "Build a mapping table from examples and apply it. "
        "Place your final answer inside \\boxed{}."
    ),
    'cryptarithm_deduce': (
        "You are an expert in symbolic reasoning and constraint propagation. "
        "For transformation puzzles, systematically analyze each example to "
        "identify the mapping rule. Work position by position, build a "
        "substitution table, and verify against all examples before applying. "
        "Show your complete reasoning. Place your final answer inside \\boxed{}."
    ),
    'cryptarithm_guess': (
        "You are an expert in symbolic reasoning. For puzzles with limited "
        "examples, identify the most likely transformation pattern. "
        "Consider the structure and symmetry of the mappings. "
        "Place your final answer inside \\boxed{}."
    ),
    'equation_numeric_deduce': (
        "You are an expert in mathematical pattern recognition. "
        "For equation transformation puzzles, analyze the numeric examples "
        "to identify the hidden operation rule. Test your hypothesis against "
        "all examples. Place your final answer inside \\boxed{}."
    ),
    'equation_numeric_guess': (
        "You are an expert in mathematical reasoning. For puzzles with "
        "limited numeric examples, identify the most probable operation rule. "
        "Consider common patterns (digit reversal, arithmetic combinations). "
        "Place your final answer inside \\boxed{}."
    ),
    'gravity': (
        "You are an expert in physics computation. For gravity puzzles, "
        "use d = 0.5*g*t^2. First compute g from the examples (linear regression "
        "or averaging), then apply to the target time. Round to 2 decimal places. "
        "Place your final answer inside \\boxed{}."
    ),
    'numeral': (
        "You are an expert in numeral systems. For Roman numeral conversion, "
        "decompose the number into thousands, hundreds, tens, ones and apply "
        "Roman numeral rules (subtractive notation). "
        "Place your final answer inside \\boxed{}."
    ),
    'unit_conversion': (
        "You are an expert in dimensional analysis. For unit conversion puzzles, "
        "first find the conversion factor from the examples (divide output by input), "
        "verify consistency, then apply to the target. Round appropriately. "
        "Place your final answer inside \\boxed{}."
    ),
}

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert reasoning model. Solve puzzles step by step, "
    "showing all your work. Place your final answer inside \\boxed{}."
)


def load_jsonl(path: str) -> list:
    """Load a JSONL file."""
    if not os.path.exists(path):
        return []
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def classify_puzzle(prompt: str) -> str:
    """Classify a puzzle into its category (same logic as classify_puzzles.py)."""
    import re
    p = prompt.lower()

    if 'bit manipulation' in p:
        return 'bit_manipulation'
    if 'secret encryption' in p or 'decrypt the following' in p:
        return 'cipher'
    if 'gravitational constant' in p:
        return 'gravity'
    if 'unit conversion' in p or 'convert the following measurement' in p:
        return 'unit_conversion'
    if 'numeral system' in p:
        return 'numeral'
    if 'transformation rules is applied to equations' in p:
        has_numbers = bool(re.search(r'\d', p))
        if has_numbers:
            return 'equation_numeric_deduce'
        else:
            return 'cryptarithm_deduce'
    return 'unknown'


def compute_impact_score(category: str, count: int, match_rate: float) -> float:
    """
    Compute an impact score for a category.
    High impact = high weightage × low accuracy.
    This tells us where improving training data matters most.
    """
    weight = EVAL_WEIGHTAGE.get(category, 1.0)
    error_rate = 1.0 - match_rate
    return weight * error_rate


def main():
    parser = argparse.ArgumentParser(
        description='Consolidate training data with category awareness')
    parser.add_argument('--classified', help='Classified puzzles JSONL')
    parser.add_argument('--cot-file', required=True, help='Existing CoT traces (train_cot_unified.jsonl)')
    parser.add_argument('--cryptarithm-file', help='Cryptarithm CoT traces')
    parser.add_argument('--output', required=True, help='Output unified JSONL')
    parser.add_argument('--balance', action='store_true', help='Balance categories')
    parser.add_argument('--target-per-category', type=int, default=800,
                        help='Target examples per category when balancing')
    parser.add_argument('--max-per-category', type=int, default=1500,
                        help='Max examples per category (cap oversampling)')
    parser.add_argument('--stats', action='store_true', help='Print detailed statistics')
    parser.add_argument('--min-assistant-length', type=int, default=50,
                        help='Min assistant response length')
    args = parser.parse_args()

    # ── Load classified puzzles (for category labels) ──
    classified_map = {}
    if args.classified and os.path.exists(args.classified):
        with open(args.classified) as f:
            for i, line in enumerate(f):
                p = json.loads(line)
                classified_map[p.get('index', i)] = p.get('category', 'unknown')
        print(f"Loaded {len(classified_map)} classified puzzles")

    # ── Load existing CoT traces ──
    all_traces = load_jsonl(args.cot_file)
    print(f"Loaded {len(all_traces)} existing CoT traces")

    # ── Separate kaggle CoT from ecology ──
    kaggle_traces = []
    ecology_traces = []
    for t in all_traces:
        meta = t.get('metadata', {})
        if meta.get('source') == 'ecology':
            ecology_traces.append(t)
        else:
            kaggle_traces.append(t)

    print(f"  Kaggle CoT: {len(kaggle_traces)}")
    print(f"  Ecology: {len(ecology_traces)}")

    # ── Load cryptarithm traces ──
    crypt_traces = []
    if args.cryptarithm_file:
        crypt_traces = load_jsonl(args.cryptarithm_file)
        print(f"  Cryptarithm: {len(crypt_traces)}")

    # ── Classify and enrich Kaggle traces ──
    per_category = defaultdict(list)
    skipped = 0

    for trace in kaggle_traces:
        msgs = trace.get('messages', [])
        if len(msgs) < 2:
            skipped += 1
            continue

        # Get category
        idx = trace.get('index', -1)
        user_msg = msgs[1].get('content', '') if len(msgs) > 1 else ''

        category = classified_map.get(idx)
        if not category:
            category = classify_puzzle(user_msg)

        # Skip unknown
        if category == 'unknown':
            skipped += 1
            continue

        # Check assistant length
        assistant = msgs[-1].get('content', '')
        if len(assistant) < args.min_assistant_length:
            skipped += 1
            continue

        # Enrich with category and improved system prompt
        # Replace system prompt with category-specific one
        system_prompt = SYSTEM_PROMPTS.get(category, DEFAULT_SYSTEM_PROMPT)
        if msgs[0].get('role') == 'system':
            msgs[0]['content'] = system_prompt

        meta = trace.get('metadata', {})
        enriched = {
            'messages': msgs,
            'metadata': {
                **meta,
                'category': category,
                'weightage': EVAL_WEIGHTAGE.get(category, 0),
            }
        }
        per_category[category].append(enriched)

    # ── Process ecology traces (mark as 'ecology' category) ──
    for trace in ecology_traces:
        msgs = trace.get('messages', [])
        if len(msgs) < 2:
            continue
        assistant = msgs[-1].get('content', '')
        if len(assistant) < args.min_assistant_length:
            continue
        enriched = {
            'messages': msgs,
            'metadata': {
                'source': 'ecology',
                'category': 'ecology',
                'weightage': 0,
            }
        }
        per_category['ecology'].append(enriched)

    # ── Process cryptarithm traces ──
    for trace in crypt_traces:
        msgs = trace.get('messages', [])
        if len(msgs) < 2:
            continue
        assistant = msgs[-1].get('content', '')
        if len(assistant) < args.min_assistant_length:
            continue

        # Determine if it's deduce or guess
        user_msg = msgs[1].get('content', '')
        n_examples = user_msg.count('\n') - 2  # rough estimate
        subtype = 'cryptarithm_deduce'  # default

        system_prompt = SYSTEM_PROMPTS.get(subtype, DEFAULT_SYSTEM_PROMPT)
        if msgs[0].get('role') == 'system':
            msgs[0]['content'] = system_prompt

        enriched = {
            'messages': msgs,
            'metadata': {
                'source': 'cryptarithm_synthetic',
                'category': subtype,
                'weightage': EVAL_WEIGHTAGE.get(subtype, 0),
            }
        }
        per_category[subtype].append(enriched)

    # ── Print statistics ──
    print(f"\n=== Per-Category Statistics (after classification) ===")
    print(f"{'Category':<30} {'Count':>6} {'Weight%':>8} {'Impact*':>8}")
    print('-' * 56)

    impact_scores = {}
    total_impact = 0
    for cat in sorted(per_category.keys()):
        count = len(per_category[cat])
        weight = EVAL_WEIGHTAGE.get(cat, 0)
        # Use eval accuracy to estimate match rate
        match_rate = 0.88  # default
        impact = compute_impact_score(cat, count, match_rate)
        impact_scores[cat] = impact
        total_impact += impact
        print(f"{cat:<30} {count:>6} {weight:>7.1f}% {impact:>7.1f}")

    print(f"\nTotal traces (after filter): {sum(len(v) for v in per_category.values())}")
    print(f"Skipped: {skipped}")

    # ── Apply balancing ──
    if args.balance:
        print(f"\n=== Applying Balance (target={args.target_per_category}) ===")
        balanced = []

        for cat, traces in sorted(per_category.items()):
            count = len(traces)
            weight = EVAL_WEIGHTAGE.get(cat, 0)

            # Determine target: weight-adjusted or flat target
            target = args.target_per_category

            if count < target:
                # Oversample with repetition until we reach target
                oversampled = list(traces)
                while len(oversampled) < target:
                    needed = target - len(oversampled)
                    extra = random.sample(traces, min(needed, count))
                    oversampled.extend(extra)
                selected = oversampled[:target]
                strategy = f"oversampled {count}→{len(selected)}"
            elif count > args.max_per_category:
                # Undersample large categories
                selected = random.sample(traces, args.max_per_category)
                strategy = f"capped {count}→{len(selected)}"
            else:
                selected = traces
                strategy = f"kept {count}"

            balanced.extend(selected)
            print(f"  {cat:<30} {strategy}")

        # Shuffle final
        random.shuffle(balanced)
        final_traces = balanced
    else:
        # No balancing: just combine all
        final_traces = []
        for traces in per_category.values():
            final_traces.extend(traces)
        random.shuffle(final_traces)

    # ── Write output ──
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        for trace in final_traces:
            f.write(json.dumps(trace) + '\n')

    print(f"\n=== Final Dataset ===")
    print(f"Total examples: {len(final_traces)}")

    # Final category breakdown
    final_cats = Counter()
    for t in final_traces:
        final_cats[t['metadata']['category']] += 1

    print(f"\nFinal category distribution:")
    for cat in sorted(final_cats.keys(), key=lambda c: -final_cats[c]):
        print(f"  {cat}: {final_cats[cat]}")

    print(f"\nOutput: {args.output}")


if __name__ == '__main__':
    main()
