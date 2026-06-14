#!/usr/bin/env python3
"""
Generate synthetic "guess" puzzles from "deduce" puzzles by reducing examples.

For categories cryptarithm_guess and equation_numeric_guess:
  - Take puzzles from the "deduce" variant (3-5 examples)
  - Randomly keep only 1-2 examples
  - This simulates the evaluation where fewer examples are given

Usage:
    python3 scripts/generate_guess_puzzles.py \
        --classified data/kaggle_classified.jsonl \
        --cryptarithm-out data/cryptarithm_guess_puzzles.jsonl \
        --equation-out data/equation_numeric_guess_puzzles.jsonl
"""

import argparse
import json
import random

random.seed(42)


def reduce_examples(prompt: str, n_keep: int) -> str:
    """Reduce a puzzle prompt to only keep n_keep examples."""
    parts = prompt.split('examples:\n')
    if len(parts) < 2:
        return prompt
    
    rest = parts[1]
    lines = rest.split('\n')
    examples = []
    question = ""
    
    for line in lines:
        if line.startswith('Now,'):
            question = line
            break
        elif '=' in line and line.strip():
            examples.append(line.strip())
    
    if len(examples) <= n_keep:
        return prompt
    
    selected = random.sample(examples, n_keep)
    
    new_prompt = parts[0] + 'examples:\n'
    new_prompt += '\n'.join(selected) + '\n'
    new_prompt += question
    
    return new_prompt


def main():
    parser = argparse.ArgumentParser(description='Generate guess puzzles from deduce')
    parser.add_argument('--classified', required=True, help='Classified puzzles JSONL')
    parser.add_argument('--cryptarithm-out', required=True, help='Output for cryptarithm_guess')
    parser.add_argument('--equation-out', required=True, help='Output for equation_numeric_guess')
    args = parser.parse_args()

    with open(args.classified) as f:
        puzzles = [json.loads(line) for line in f]

    crypt_guess = []
    eq_guess = []

    for p in puzzles:
        cat = p['category']
        if cat not in ('cryptarithm_deduce', 'equation_numeric_deduce'):
            continue
        if p.get('examples_count', 0) < 3:
            continue

        n_keep = random.choice([1, 2])
        new_prompt = reduce_examples(p['prompt'], n_keep)
        
        target_cat = cat.replace('deduce', 'guess')
        new_puzzle = {
            'prompt': new_prompt,
            'answer': p['answer'],
            'category': target_cat,
            'examples_count': n_keep,
            'original_index': p['index'],
        }

        if cat == 'cryptarithm_deduce':
            crypt_guess.append(new_puzzle)
        else:
            eq_guess.append(new_puzzle)

    with open(args.cryptarithm_out, 'w') as f:
        for p in crypt_guess:
            f.write(json.dumps(p) + '\n')

    with open(args.equation_out, 'w') as f:
        for p in eq_guess:
            f.write(json.dumps(p) + '\n')

    print(f"Cryptarithm guess: {len(crypt_guess)} puzzles")
    print(f"Equation numeric guess: {len(eq_guess)} puzzles")


if __name__ == '__main__':
    main()
