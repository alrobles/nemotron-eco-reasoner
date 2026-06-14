#!/usr/bin/env python3
"""
Classify Kaggle training puzzles into 9 categories matching the evaluation table.

Categories (from Nemotron event evaluation):
  bit_manipulation    - 8-bit binary transformation puzzles
  cipher              - Text encryption/decryption puzzles
  cryptarithm_deduce  - Symbol/character equation deduction
  cryptarithm_guess   - Symbol mapping with fewer examples (guess)
  equation_numeric_deduce - Numeric equation rule deduction
  equation_numeric_guess   - Numeric equation with fewer examples (guess)
  gravity             - Gravitational constant physics puzzles
  numeral             - Roman numeral conversion
  unit_conversion     - Unit conversion puzzles

Distinction between deduce vs guess:
  - deduce: 3-4+ examples provided (pattern can be deduced)
  - guess:   1-2 examples provided (requires more guessing)

Usage:
    python3 classify_puzzles.py \
        --input data/kaggle_puzzles_raw.jsonl \
        --output data/kaggle_classified.jsonl
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict


def count_examples(prompt: str) -> int:
    """Count number of input-output examples in the prompt."""
    # Count lines with patterns like "X -> Y" or "X = Y"
    lines = prompt.split('\n')
    count = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith('In Alice') or line.startswith('Now,'):
            continue
        if '->' in line or ' = ' in line or ' becomes ' in line:
            count += 1
    return count


def classify_puzzle(prompt: str) -> dict:
    """
    Classify a puzzle into its category and sub-type.
    Returns {'category': str, 'examples': int}
    """
    p = prompt.lower()
    n_examples = count_examples(prompt)

    if 'bit manipulation' in p:
        return {'category': 'bit_manipulation', 'examples': n_examples}

    if 'secret encryption' in p or 'decrypt the following' in p:
        return {'category': 'cipher', 'examples': n_examples}

    if 'gravitational constant' in p:
        return {'category': 'gravity', 'examples': n_examples}

    if 'unit conversion' in p or 'convert the following measurement' in p:
        return {'category': 'unit_conversion', 'examples': n_examples}

    if 'numeral system' in p:
        return {'category': 'numeral', 'examples': n_examples}

    if 'transformation rules is applied to equations' in p:
        # Distinguish cryptarithm (symbols/letters) from numeric equations
        # Extract example lines
        example_lines = []
        in_examples = False
        for line in prompt.split('\n'):
            line = line.strip()
            if 'examples:' in line.lower():
                in_examples = True
                continue
            if in_examples and line.startswith('Now,'):
                break
            if in_examples and line:
                example_lines.append(line)

        # Check if examples use symbols/letters vs numbers
        has_letters = any(re.search(r'[A-Za-z]', l) for l in example_lines)
        has_numbers = any(re.search(r'\d', l) for l in example_lines)

        # If predominantly letters/symbols, it's cryptarithm-like
        if has_letters and not has_numbers:
            subtype = 'cryptarithm_deduce' if n_examples >= 3 else 'cryptarithm_guess'
            return {'category': subtype, 'examples': n_examples}
        elif has_numbers:
            subtype = 'equation_numeric_deduce' if n_examples >= 3 else 'equation_numeric_guess'
            return {'category': subtype, 'examples': n_examples}
        else:
            # Pure symbols only — treat as cryptarithm_deduce
            subtype = 'cryptarithm_deduce' if n_examples >= 3 else 'cryptarithm_guess'
            return {'category': subtype, 'examples': n_examples}

    return {'category': 'unknown', 'examples': n_examples}


def main():
    parser = argparse.ArgumentParser(description='Classify Kaggle puzzles by category')
    parser.add_argument('--input', required=True, help='Input JSONL with puzzles')
    parser.add_argument('--output', required=True, help='Output classified JSONL')
    parser.add_argument('--stats', action='store_true', help='Print classification statistics')
    args = parser.parse_args()

    puzzles = []
    with open(args.input) as f:
        for line in f:
            puzzles.append(json.loads(line))

    print(f"Loaded {len(puzzles)} puzzles")

    # Classify
    categories = Counter()
    example_dist = defaultdict(Counter)

    classified = []
    for i, puzzle in enumerate(puzzles):
        result = classify_puzzle(puzzle['prompt'])
        puzzle['category'] = result['category']
        puzzle['examples_count'] = result['examples']
        puzzle['index'] = i
        classified.append(puzzle)

        categories[result['category']] += 1
        example_dist[result['category']][result['examples']] += 1

    # Write output
    with open(args.output, 'w') as f:
        for p in classified:
            f.write(json.dumps(p) + '\n')

    # Stats
    print(f"\n=== Category Distribution ===")
    print(f"{'Category':<30} {'Count':>6} {'%':>7}")
    print('-' * 45)
    for cat in sorted(categories.keys(), key=lambda c: -categories[c]):
        pct = categories[cat] / len(puzzles) * 100
        print(f"{cat:<30} {categories[cat]:>6} {pct:>6.1f}%")
    print(f"{'TOTAL':<30} {sum(categories.values()):>6}")

    # Example distribution per category
    if args.stats:
        print(f"\n=== Example Count Distribution per Category ===")
        for cat in sorted(example_dist.keys()):
            dist = example_dist[cat]
            print(f"\n{cat}:")
            for n_ex in sorted(dist.keys()):
                print(f"  {n_ex} examples: {dist[n_ex]} puzzles")

    print(f"\nOutput: {args.output}")


if __name__ == '__main__':
    main()
