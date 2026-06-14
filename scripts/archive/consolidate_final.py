#!/usr/bin/env python3
"""
Final consolidation: create optimized training dataset with category-aware strategy.

Strategy per category:
  - cryptarithm_deduce: answer-only format (LLM can't solve symbol-transformation, 2.4% CoT match)
  - bit_manipulation: keep existing CoT + flag as noisy (11.2% match)
  - equation_numeric_*: answer-only for unsolved, CoT where available
  - cipher, gravity, numeral, unit_conversion: keep existing high-quality CoT
  - ecology: keep existing traces
  - cryptarithm (traditional): include from cryptarithm_cot.jsonl (good CoT)

Key distinction:
  - Symbol-transformation puzzles (cryptarithm_deduce from Kaggle): answer-only
  - Traditional cryptarithms (XNI + QSQ = QRRS): keep full CoT

Usage:
    python3 scripts/consolidate_final.py \
        --classified data/kaggle_classified.jsonl \
        --cot-file data/train_cot_unified.jsonl \
        --cryptarithm-file data/cryptarithm_cot.jsonl \
        --output data/train_cot_unified_v4.jsonl \
        --balance
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict

random.seed(42)

# ── Category configuration ──
# answer_only: categories where CoT generation is not feasible
# good_cot: categories where existing CoT is high quality
# synthesized: categories where we use synthetic traces

CATEGORY_CONFIG = {
    'bit_manipulation':    {'mode': 'cot_keep',   'match_rate': 11.2, 'weight': 17.8},
    'cipher':              {'mode': 'cot_keep',   'match_rate': 72.4, 'weight': 17.1},
    'cryptarithm_deduce':  {'mode': 'answer_only', 'match_rate': 2.4,  'weight': 7.5},
    'cryptarithm_guess':   {'mode': 'answer_only',  'match_rate': 0,    'weight': 1.5},
    'equation_numeric_deduce': {'mode': 'cot_keep','match_rate': 32.7, 'weight': 5.1},
    'equation_numeric_guess':  {'mode': 'answer_only','match_rate': 0,   'weight': 0.7},
    'gravity':             {'mode': 'cot_keep',   'match_rate': 98.6, 'weight': 16.7},
    'numeral':             {'mode': 'cot_keep',   'match_rate': 100.0,'weight': 15.7},
    'unit_conversion':     {'mode': 'cot_keep',   'match_rate': 90.7, 'weight': 18.0},
    'ecology':             {'mode': 'cot_keep',   'match_rate': 100,  'weight': 0},
}

SYSTEM_PROMPTS = {
    'bit_manipulation': (
        "You are an expert in binary arithmetic and bit manipulation. "
        "Place your final answer inside \\boxed{}."
    ),
    'cipher': (
        "You are an expert cryptanalyst. Analyze substitution patterns. "
        "Place your final answer inside \\boxed{}."
    ),
    'cryptarithm_deduce': (
        "You are an expert puzzle solver. For transformation puzzles, "
        "identify the pattern and compute the result. "
        "Place your final answer inside \\boxed{}."
    ),
    'equation_numeric_deduce': (
        "You are an expert in mathematical pattern recognition. "
        "Place your final answer inside \\boxed{}."
    ),
    'gravity': (
        "You are an expert in physics. Use d = 0.5*g*t^2. "
        "Place your final answer inside \\boxed{}."
    ),
    'numeral': (
        "You are an expert in numeral systems. "
        "Place your final answer inside \\boxed{}."
    ),
    'unit_conversion': (
        "You are an expert in dimensional analysis. "
        "Place your final answer inside \\boxed{}."
    ),
    'cryptarithm_traditional': (
        "You are an expert mathematical reasoner. Solve cryptarithms using "
        "constraint propagation and logical deduction. Show your work. "
        "Place your final answer inside \\boxed{}."
    ),
}

DEFAULT_SYSTEM = (
    "You are an expert reasoning model. Solve problems step by step. "
    "Place your final answer inside \\boxed{}."
)


def load_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def classify_puzzle(prompt: str) -> str:
    """Classify a puzzle prompt into a category."""
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
        return 'equation_numeric_deduce' if has_numbers else 'cryptarithm_deduce'
    return 'unknown'


def create_answer_only(puzzle: dict) -> dict:
    """Create an answer-only training example (no CoT)."""
    category = puzzle.get('category', 'unknown')
    system_prompt = SYSTEM_PROMPTS.get(category, DEFAULT_SYSTEM)
    answer = puzzle['answer']
    if answer.startswith('\\boxed{') and answer.endswith('}'):
        answer_text = answer
    else:
        answer_text = f"\\boxed{{{answer}}}"

    return {
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': puzzle['prompt']},
            {'role': 'assistant', 'content': answer_text},
        ],
        'metadata': {
            'source': 'kaggle_answer_only',
            'category': category,
            'weight': CATEGORY_CONFIG.get(category, {}).get('weight', 0),
            'format': 'answer_only',
        }
    }


def convert_cot_trace(trace: dict, category: str, system_prompt: str) -> dict:
    """Convert and enrich a CoT trace."""
    msgs = trace.get('messages', [])
    if msgs and msgs[0].get('role') == 'system':
        msgs[0]['content'] = system_prompt

    meta = trace.get('metadata', {})
    return {
        'messages': msgs,
        'metadata': {
            **meta,
            'category': category,
            'weight': CATEGORY_CONFIG.get(category, {}).get('weight', 0),
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Final consolidated dataset')
    parser.add_argument('--classified', required=True, help='Classified puzzles JSONL')
    parser.add_argument('--cot-file', required=True, help='Existing CoT traces')
    parser.add_argument('--cryptarithm-file', help='Cryptarithm CoT traces')
    parser.add_argument('--cryptarithm-guess-file', help='Cryptarithm guess puzzles')
    parser.add_argument('--equation-guess-file', help='Equation numeric guess puzzles')
    parser.add_argument('--output', required=True, help='Output JSONL')
    parser.add_argument('--balance', action='store_true', help='Balance categories')
    parser.add_argument('--min-assistant-length', type=int, default=30)
    args = parser.parse_args()

    # ── Load classified puzzles ──
    prompt_to_category = {}
    classified_puzzles = {}
    with open(args.classified) as f:
        for line in f:
            p = json.loads(line)
            prompt_to_category[p['prompt'].strip()] = p['category']
            classified_puzzles[p['index']] = p

    print(f"Loaded {len(classified_puzzles)} classified puzzles")

    # ── Load CoT traces + ecology ──
    all_traces = load_jsonl(args.cot_file)
    kaggle_cot = []
    ecology_cot = []
    for t in all_traces:
        meta = t.get('metadata', {})
        if meta.get('source') == 'ecology':
            ecology_cot.append(t)
        else:
            kaggle_cot.append(t)

    print(f"Kaggle CoT: {len(kaggle_cot)}, Ecology: {len(ecology_cot)}")

    # ── Load cryptarithm traces ──
    crypt_traces = load_jsonl(args.cryptarithm_file) if args.cryptarithm_file else []
    print(f"Cryptarithm CoT: {len(crypt_traces)}")

    # ── Load guess puzzles (synthetic, answer-only) ──
    crypt_guess_puzzles = load_jsonl(args.cryptarithm_guess_file) if args.cryptarithm_guess_file else []
    eq_guess_puzzles = load_jsonl(args.equation_guess_file) if args.equation_guess_file else []
    print(f"Cryptarithm guess puzzles: {len(crypt_guess_puzzles)}")
    print(f"Equation guess puzzles: {len(eq_guess_puzzles)}")

    # ── Process by category ──
    per_category = defaultdict(list)
    puzzle_to_cot_used = set()  # Track which prompts already have CoT

    # Track puzzle prompts we've used as answer-only
    answer_only_prompts = set()

    # 1. Process Kaggle CoT traces
    for trace in kaggle_cot:
        msgs = trace.get('messages', [])
        if len(msgs) < 2:
            continue
        user_msg = msgs[1].get('content', '')
        assistant = msgs[-1].get('content', '')

        # Extract prompt and match to category
        if user_msg.startswith('Puzzle '):
            prompt_part = user_msg.split(': ', 1)[1] if ': ' in user_msg else user_msg
            if '\n\nThink step by step' in prompt_part:
                prompt_part = prompt_part.split('\n\nThink step by step')[0]
        else:
            prompt_part = user_msg
        prompt_part = prompt_part.strip()

        cat = prompt_to_category.get(prompt_part)
        if not cat:
            cat = classify_puzzle(prompt_part)
        if cat == 'unknown':
            continue

        config = CATEGORY_CONFIG.get(cat, {})

        # For answer_only categories: skip CoT, generate answer-only instead
        if config.get('mode') == 'answer_only':
            continue  # We'll add these as answer-only later

        # For cot_keep: keep the trace but check quality
        if len(assistant) < args.min_assistant_length:
            continue

        system_prompt = SYSTEM_PROMPTS.get(cat, DEFAULT_SYSTEM)
        enriched = convert_cot_trace(trace, cat, system_prompt)
        per_category[cat].append(enriched)
        puzzle_to_cot_used.add(prompt_part)

    # 2. Add answer-only traces for categories marked as answer_only
    for idx, puzzle in classified_puzzles.items():
        cat = puzzle['category']
        config = CATEGORY_CONFIG.get(cat, {})
        if config.get('mode') == 'answer_only':
            per_category[cat].append(create_answer_only(puzzle))
            answer_only_prompts.add(puzzle['prompt'].strip())

    # 3. Process ecology traces
    for trace in ecology_cot:
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
                'weight': 0,
            }
        }
        per_category['ecology'].append(enriched)

    # 4. Process cryptarithm traces — separate traditional from symbol
    trad_count = 0
    for trace in crypt_traces:
        msgs = trace.get('messages', [])
        if len(msgs) < 2:
            continue
        user_msg = msgs[1].get('content', '')
        assistant = msgs[-1].get('content', '')
        if len(assistant) < args.min_assistant_length:
            continue

        # Traditional cryptarithms have "Each letter = unique digit"
        if 'each letter' in user_msg.lower() or 'cryptarithm' in user_msg.lower():
            system_prompt = SYSTEM_PROMPTS.get('cryptarithm_traditional', DEFAULT_SYSTEM)
            if msgs[0].get('role') == 'system':
                msgs[0]['content'] = system_prompt
            enriched = {
                'messages': msgs,
                'metadata': {
                    'source': 'cryptarithm_traditional',
                    'category': 'cryptarithm_deduce',
                    'subtype': 'traditional',
                    'weight': CATEGORY_CONFIG.get('cryptarithm_deduce', {}).get('weight', 0),
                }
            }
            per_category['cryptarithm_deduce'].append(enriched)
            trad_count += 1

    print(f"\nTraditional cryptarithm traces added: {trad_count}")

    # 5. Process guess puzzles (answer-only) — synthetic from deduce puzzles
    for puzzle in crypt_guess_puzzles:
        per_category['cryptarithm_guess'].append(create_answer_only(puzzle))
    for puzzle in eq_guess_puzzles:
        per_category['equation_numeric_guess'].append(create_answer_only(puzzle))

    print(f"Cryptarithm_guess (answer-only): {len(crypt_guess_puzzles)}")
    print(f"Equation_numeric_guess (answer-only): {len(eq_guess_puzzles)}")

    # ── Statistics ──
    print(f"\n=== Per-Category Distribution ===")
    print(f"{'Category':<30} {'Count':>6} {'Mode':>12} {'Weight':>7}")
    print('-' * 58)
    for cat in sorted(per_category.keys()):
        count = len(per_category[cat])
        config = CATEGORY_CONFIG.get(cat, {})
        mode = config.get('mode', 'unknown')
        weight = config.get('weight', 0)
        print(f"{cat:<30} {count:>6} {mode:>12} {weight:>6.1f}%")

    total = sum(len(v) for v in per_category.values())
    print(f"\nTotal: {total}")

    # ── Balance ──
    if args.balance:
        print(f"\n=== Balancing ===")
        balanced = []
        # Target: weight-adjusted
        total_weight = sum(CATEGORY_CONFIG.get(c, {}).get('weight', 0)
                          for c in per_category.keys() if c != 'ecology')
        
        for cat in sorted(per_category.keys()):
            traces = per_category[cat]
            count = len(traces)
            config = CATEGORY_CONFIG.get(cat, {})
            weight = config.get('weight', 0)

            if cat == 'ecology':
                # Keep ecology but cap at 1000
                if count > 1000:
                    selected = random.sample(traces, 1000)
                    strategy = f"capped {count}→1000"
                else:
                    selected = traces
                    strategy = f"kept {count}"
            elif count < 100:
                # Small categories: oversample to at least 200
                oversampled = list(traces)
                while len(oversampled) < 200:
                    extra = random.sample(traces, min(200 - len(oversampled), count))
                    oversampled.extend(extra)
                selected = oversampled[:200]
                strategy = f"oversampled {count}→{len(selected)}"
            elif count > 1000:
                # Large categories: cap at 1000
                selected = random.sample(traces, 1000)
                strategy = f"capped {count}→1000"
            else:
                selected = traces
                strategy = f"kept {count}"

            balanced.extend(selected)
            print(f"  {cat:<30} {strategy}")

        random.shuffle(balanced)
        final_traces = balanced
    else:
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
    print(f"Total: {len(final_traces)}")

    final_cats = Counter()
    formats = Counter()
    for t in final_traces:
        final_cats[t['metadata']['category']] += 1
        formats[t['metadata'].get('format', 'cot')] += 1

    print(f"Formats: {dict(formats)}")
    print(f"\nCategory breakdown:")
    for cat in sorted(final_cats.keys(), key=lambda c: -final_cats[c]):
        config = CATEGORY_CONFIG.get(cat, {})
        flag = "🔴" if config.get('mode') == 'answer_only' else "🟢" if config.get('match_rate', 100) > 70 else "🟡"
        print(f"  {flag} {cat}: {final_cats[cat]}")

    print(f"\nOutput: {args.output}")


if __name__ == '__main__':
    main()
