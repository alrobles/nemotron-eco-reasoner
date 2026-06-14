#!/usr/bin/env python3
"""
Programmatic solver for bit_manipulation puzzles.

These puzzles provide 8-10 input→output 8-bit binary pairs and ask for the
transformation of a new input. The transformation is one of a fixed set of
bitwise operations.

The solver brute-forces through common operations and their combinations
to find the rule that matches ALL examples, then applies it to the target.

Operations tested:
  - Identity, NOT
  - AND/OR/XOR with constant mask
  - Left/right shift by 1-7 bits (with wrap or zero-fill)
  - Left/right rotate by 1-7 bits
  - Reverse bits
  - Pairs: (shift + XOR), (rotate + XOR), (AND + XOR), (shift + AND)
  - Majority function (bitwise majority of 3 adjacent bits)
  - Choice function

Usage:
    python3 scripts/solve_bit_manipulation.py \
        --input data/bit_manipulation_puzzles.jsonl \
        --output data/bit_solutions.jsonl
"""

import argparse
import itertools
import json
import re
import sys
from typing import Optional


def rotl8(x: int, n: int) -> int:
    """Rotate left by n bits (8-bit)."""
    n = n % 8
    return ((x << n) | (x >> (8 - n))) & 0xFF


def rotr8(x: int, n: int) -> int:
    """Rotate right by n bits (8-bit)."""
    n = n % 8
    return ((x >> n) | (x << (8 - n))) & 0xFF


def rev8(x: int) -> int:
    """Reverse 8 bits."""
    result = 0
    for i in range(8):
        if x & (1 << i):
            result |= 1 << (7 - i)
    return result


def majority(x: int) -> int:
    """Bitwise majority of 3 adjacent bits (wrapping)."""
    result = 0
    for i in range(8):
        a = (x >> i) & 1
        b = (x >> ((i + 1) % 8)) & 1
        c = (x >> ((i + 2) % 8)) & 1
        if a + b + c >= 2:
            result |= 1 << i
    return result


def choice_func(x: int) -> int:
    """Choice function: for each bit, if bit is 1, use next bit, else use previous."""
    result = 0
    for i in range(8):
        if (x >> i) & 1:
            result |= ((x >> ((i + 1) % 8)) & 1) << i
        else:
            result |= ((x >> ((i - 1) % 8)) & 1) << i
    return result


def find_bitwise_rule(pairs: list[tuple[int, int]]) -> Optional[tuple[str, callable, dict]]:
    """
    Find the bitwise rule that maps all inputs to outputs.
    
    Returns (rule_description, function, params) or None if not found.
    """
    inputs = [p[0] for p in pairs]
    outputs = [p[1] for p in pairs]
    
    def check_all(fn) -> bool:
        return all(fn(inp) == out for inp, out in zip(inputs, outputs))
    
    # ── Single operations ──
    
    # Identity
    if check_all(lambda x: x):
        return ("identity (no change)", lambda x: x, {})
    
    # NOT
    if check_all(lambda x: (~x) & 0xFF):
        return ("bitwise NOT (invert all bits)", lambda x: (~x) & 0xFF, {})
    
    # AND/OR/XOR with constant
    for mask in range(256):
        if check_all(lambda x, m=mask: x & m):
            return (f"AND with mask 0x{mask:02X} ({mask:08b})",
                    lambda x, m=mask: x & m, {"mask": mask})
        if check_all(lambda x, m=mask: x | m):
            return (f"OR with mask 0x{mask:02X} ({mask:08b})",
                    lambda x, m=mask: x | m, {"mask": mask})
        if check_all(lambda x, m=mask: x ^ m):
            return (f"XOR with constant 0x{mask:02X} ({mask:08b})",
                    lambda x, m=mask: x ^ m, {"constant": mask})
    
    # Shift left/right with zero-fill
    for n in range(1, 8):
        if check_all(lambda x, n=n: (x << n) & 0xFF):
            return (f"shift left by {n} (zero-fill)", lambda x, n=n: (x << n) & 0xFF, {"shift": n})
        if check_all(lambda x, n=n: x >> n):
            return (f"shift right by {n} (zero-fill)", lambda x, n=n: x >> n, {"shift": n})
    
    # Shift left with wrap
    for n in range(1, 8):
        if check_all(lambda x, n=n: ((x << n) | (x >> (8 - n))) & 0xFF):
            return (f"shift left with wrap by {n}", lambda x, n=n: ((x << n) | (x >> (8 - n))) & 0xFF, {"shift": n})
        if check_all(lambda x, n=n: ((x >> n) | (x << (8 - n))) & 0xFF):
            return (f"shift right with wrap by {n}", lambda x, n=n: ((x >> n) | (x << (8 - n))) & 0xFF, {"shift": n})
    
    # Rotate (same as shift with wrap for 8-bit)
    for n in range(1, 8):
        if check_all(lambda x, n=n: rotl8(x, n)):
            return (f"rotate left by {n}", lambda x, n=n: rotl8(x, n), {"rotate": n})
        if check_all(lambda x, n=n: rotr8(x, n)):
            return (f"rotate right by {n}", lambda x, n=n: rotr8(x, n), {"rotate": n})
    
    # Reverse
    if check_all(rev8):
        return ("reverse all 8 bits", rev8, {})
    
    # Majority / Choice
    if check_all(majority):
        return ("majority of 3 adjacent bits", majority, {})
    if check_all(choice_func):
        return ("choice function (next/prev bit)", choice_func, {})
    
    # ── Two-operation combinations ──
    
    # GF(2) Linear Transform: output[i] = XOR of certain input bits XOR constant
    # For each output bit, try all 256 input subsets and both c values
    linear_per_bit = {}
    for out_bit in range(8):
        for mask in range(256):
            for c in [0, 1]:
                if all((bin(inp & mask).count('1') % 2) ^ c == ((out >> out_bit) & 1)
                       for inp, out in pairs):
                    linear_per_bit[out_bit] = (mask, c)
                    break
            if out_bit in linear_per_bit:
                break
    
    if len(linear_per_bit) == 8:
        masks = [linear_per_bit[b][0] for b in range(8)]
        consts = [linear_per_bit[b][1] for b in range(8)]
        const_byte = sum(c << b for b, c in enumerate(consts))
        
        desc_parts = []
        for b in range(8):
            if consts[b]:
                desc_parts.append(f"out[{b}] = NOT(XOR of {masks[b]:08b})")
            else:
                desc_parts.append(f"out[{b}] = XOR of {masks[b]:08b}")
        desc = "GF(2) linear: " + "; ".join(desc_parts)
        
        def linear_fn(x, masks=masks, consts=consts):
            result = 0
            for b in range(8):
                parity = bin(x & masks[b]).count('1') % 2
                result |= (parity ^ consts[b]) << b
            return result
        
        return (desc, linear_fn, {"masks": masks, "constants": consts})
    
    # Note: bit permutations (8! = 40320 combos per puzzle) are skipped for speed.
    # They would catch some remaining puzzles but are too expensive to brute-force.
    
    # Shift + XOR with constant
    for n in range(1, 8):
        for mask in range(256):
            if check_all(lambda x, n=n, m=mask: (((x << n) & 0xFF) ^ m)):
                return (f"(input << {n}) XOR 0x{mask:02X}",
                        lambda x, n=n, m=mask: ((x << n) & 0xFF) ^ m,
                        {"shift": n, "xor_mask": mask})
            if check_all(lambda x, n=n, m=mask: ((x >> n) ^ m)):
                return (f"(input >> {n}) XOR 0x{mask:02X}",
                        lambda x, n=n, m=mask: (x >> n) ^ m,
                        {"shift": n, "xor_mask": mask})
    
    # Shift + AND with mask
    for n in range(1, 8):
        for mask in range(256):
            if check_all(lambda x, n=n, m=mask: ((x << n) & m) & 0xFF):
                return (f"(input << {n}) AND 0x{mask:02X}",
                        lambda x, n=n, m=mask: ((x << n) & m) & 0xFF,
                        {"shift": n, "and_mask": mask})
            if check_all(lambda x, n=n, m=mask: (x >> n) & m):
                return (f"(input >> {n}) AND 0x{mask:02X}",
                        lambda x, n=n, m=mask: (x >> n) & m,
                        {"shift": n, "and_mask": mask})
    
    # Rotate + XOR
    for n in range(1, 8):
        for mask in range(256):
            if check_all(lambda x, n=n, m=mask: rotl8(x, n) ^ m):
                return (f"rotate_left({n}) XOR 0x{mask:02X}",
                        lambda x, n=n, m=mask: rotl8(x, n) ^ m,
                        {"rotate": n, "xor_mask": mask})
            if check_all(lambda x, n=n, m=mask: rotr8(x, n) ^ m):
                return (f"rotate_right({n}) XOR 0x{mask:02X}",
                        lambda x, n=n, m=mask: rotr8(x, n) ^ m,
                        {"rotate": n, "xor_mask": mask})
    
    # (input << a) XOR (input >> b) — common in Kaggle
    for a in range(1, 8):
        for b in range(1, 8):
            if check_all(lambda x, a=a, b=b: (((x << a) & 0xFF) ^ (x >> b))):
                return (f"(input << {a}) XOR (input >> {b})",
                        lambda x, a=a, b=b: ((x << a) & 0xFF) ^ (x >> b),
                        {"shift_left": a, "shift_right": b})
    
    # (input >> a) XOR (input << b)
    for a in range(1, 8):
        for b in range(1, 8):
            if check_all(lambda x, a=a, b=b: ((x >> a) ^ ((x << b) & 0xFF))):
                return (f"(input >> {a}) XOR (input << {b})",
                        lambda x, a=a, b=b: (x >> a) ^ ((x << b) & 0xFF),
                        {"shift_right": a, "shift_left": b})
    
    # XOR(input, input >> n) — self-XOR shift
    for n in range(1, 8):
        if check_all(lambda x, n=n: x ^ (x >> n)):
            return (f"input XOR (input >> {n})",
                    lambda x, n=n: x ^ (x >> n),
                    {"shift": n})
        if check_all(lambda x, n=n: x ^ ((x << n) & 0xFF)):
            return (f"input XOR (input << {n})",
                    lambda x, n=n: x ^ ((x << n) & 0xFF),
                    {"shift": n})
    
    # (input << a) & mask1 XOR (input >> b) & mask2 — very expensive, try last
    # Limit to masks that appear in the XOR analysis
    # First check: which mask values appear?
    candidate_masks = set()
    for inp, out in pairs:
        candidate_masks.add(inp ^ out)  # XOR shows which bits changed
        candidate_masks.add(out)
        candidate_masks.add(inp)
    
    for a in range(1, 8):
        for b in range(1, 8):
            for m1 in candidate_masks:
                for m2 in candidate_masks:
                    if check_all(lambda x, a=a, b=b, m1=m1, m2=m2:
                                (((x << a) & m1) ^ ((x >> b) & m2)) & 0xFF):
                        return (f"((input << {a}) & 0x{m1:02X}) XOR ((input >> {b}) & 0x{m2:02X})",
                                lambda x, a=a, b=b, m1=m1, m2=m2:
                                    (((x << a) & m1) ^ ((x >> b) & m2)) & 0xFF,
                                {"shift_left": a, "shift_right": b,
                                 "mask_left": m1, "mask_right": m2})
    
    # (input AND mask1) XOR (input AND mask2) — use candidate masks
    for m1 in candidate_masks:
        for m2 in candidate_masks:
            if check_all(lambda x, m1=m1, m2=m2: (x & m1) ^ (x & m2)):
                return (f"(input AND 0x{m1:02X}) XOR (input AND 0x{m2:02X})",
                        lambda x, m1=m1, m2=m2: (x & m1) ^ (x & m2),
                        {"mask1": m1, "mask2": m2})
    
    # NOT + shift
    for n in range(1, 8):
        if check_all(lambda x, n=n: ((~x) & 0xFF) << n & 0xFF):
            return (f"NOT(input) << {n}", lambda x, n=n: ((~x) & 0xFF) << n & 0xFF, {"shift": n})
        if check_all(lambda x, n=n: ((~x) & 0xFF) >> n):
            return (f"NOT(input) >> {n}", lambda x, n=n: ((~x) & 0xFF) >> n, {"shift": n})
    
    return None


def parse_binary(s: str) -> int:
    """Parse an 8-bit binary string to int."""
    return int(s.strip(), 2)


def extract_pairs(prompt: str) -> tuple[list[tuple[int, int]], int]:
    """Extract input-output pairs and target from a bit manipulation prompt."""
    pairs = []
    target = None
    
    lines = prompt.split('\n')
    for line in lines:
        line = line.strip()
        if ' -> ' in line:
            parts = line.split(' -> ')
            left = parts[0].strip()
            right = parts[1].strip()
            # Only match lines where both sides are 8-bit binary
            if re.match(r'^[01]{8}$', left) and re.match(r'^[01]{8}$', right):
                inp = parse_binary(left)
                out = parse_binary(right)
                pairs.append((inp, out))
        elif 'determine the output for:' in line.lower():
            match = re.search(r'([01]{8})', line)
            if match:
                target = parse_binary(match.group(1))
    
    return pairs, target


def format_binary(x: int) -> str:
    """Format int as 8-bit binary string."""
    return f"{x:08b}"


def generate_cot(prompt: str, rule_desc: str, rule_fn: callable, 
                 pairs: list, target: int, params: dict) -> str:
    """Generate a Chain-of-Thought trace describing the solution."""
    cot = []
    
    cot.append("## Bit Manipulation Solution\n")
    cot.append("### Step 1: List input→output pairs\n")
    for i, (inp, out) in enumerate(pairs):
        cot.append(f"  Pair {i+1}: {format_binary(inp)} → {format_binary(out)}")
        cot.append(f"    Input:  {inp:3d} (0x{inp:02X})")
        cot.append(f"    Output: {out:3d} (0x{out:02X})")
        cot.append(f"    XOR:    {format_binary(inp ^ out)} (bits that changed)")
    
    cot.append(f"\n### Step 2: Search for bitwise rule\n")
    cot.append(f"Testing common operations against all {len(pairs)} examples...\n")
    cot.append(f"  Rule found: **{rule_desc}**\n")
    
    # Show verification
    cot.append("### Step 3: Verify against all examples\n")
    for i, (inp, out) in enumerate(pairs):
        computed = rule_fn(inp)
        status = "✓" if computed == out else "✗"
        cot.append(f"  Pair {i+1}: f({format_binary(inp)}) = {format_binary(computed)} "
                   f"(expected {format_binary(out)}) {status}")
    
    # Apply to target
    cot.append(f"\n### Step 4: Apply rule to target\n")
    result = rule_fn(target)
    cot.append(f"  Target: {format_binary(target)} ({target})")
    cot.append(f"  Rule: {rule_desc}")
    cot.append(f"  Result: {format_binary(result)} ({result})")
    
    cot.append(f"\n### Final Answer\n")
    cot.append(f"\\boxed{{{format_binary(result)}}}")
    
    return '\n'.join(cot)


def main():
    parser = argparse.ArgumentParser(description='Solve bit manipulation puzzles')
    parser.add_argument('--input', required=True, help='Input JSONL with puzzles')
    parser.add_argument('--output', required=True, help='Output JSONL with solutions')
    args = parser.parse_args()

    puzzles = []
    with open(args.input) as f:
        for line in f:
            puzzles.append(json.loads(line))

    print(f"Loaded {len(puzzles)} bit manipulation puzzles")

    solved = 0
    failed = 0
    total = 0

    with open(args.output, 'w') as out:
        for puzzle in puzzles:
            total += 1
            pairs, target = extract_pairs(puzzle['prompt'])
            
            if not pairs or target is None:
                failed += 1
                continue

            result = find_bitwise_rule(pairs)
            
            if result:
                rule_desc, rule_fn, params = result
                computed = rule_fn(target)
                
                # Verify
                answer_match = computed == parse_binary(normalize_answer(puzzle['answer']))
                if answer_match:
                    solved += 1
                
                cot = generate_cot(puzzle['prompt'], rule_desc, rule_fn, 
                                   pairs, target, params)
                
                record = {
                    'messages': [
                        {
                            'role': 'system',
                            'content': (
                                "You are an expert in binary arithmetic and bit manipulation. "
                                "For bit manipulation puzzles, systematically test common "
                                "bitwise operations against all examples to find the rule, "
                                "then apply it to the target. Place final answer inside \\\\boxed{}."
                            )
                        },
                        {'role': 'user', 'content': puzzle['prompt']},
                        {'role': 'assistant', 'content': cot},
                    ],
                    'index': puzzle.get('index', total - 1),
                    'metadata': {
                        'engine': 'solver',
                        'model': 'bitwise_brute_force',
                        'category': 'bit_manipulation',
                        'ground_truth': puzzle['answer'],
                        'generated_answer': f"\\\\boxed{{{format_binary(computed)}}}",
                        'answer_match': answer_match,
                        'rule': rule_desc,
                        'params': params,
                    }
                }
                out.write(json.dumps(record) + '\n')
            else:
                failed += 1
            
            if total % 50 == 0:
                print(f"  [{total}/{len(puzzles)}] solved={solved} failed={failed}")

    print(f"\n=== SOLVER RESULTS ===")
    print(f"Total: {total}")
    print(f"Solved: {solved} ({solved/max(total,1)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/max(total,1)*100:.1f}%)")
    print(f"Output: {args.output}")


def normalize_answer(ans: str) -> str:
    ans = ans.strip()
    if ans.startswith('\\boxed{') and ans.endswith('}'):
        ans = ans[7:-1]
    return ans.strip()


if __name__ == '__main__':
    main()
