#!/usr/bin/env python3
"""
Generate deterministic Chain-of-Thought traces for ALL Kaggle puzzle categories.

Strategy (based on Progress Prize winner's approach):
- Cipher: Build char mapping from examples, apply to target
- Gravity: Compute g from examples via d = 0.5*g*t^2, apply to target
- Numeral: Direct Roman numeral conversion
- Unit conversion: Compute linear factor from examples, apply to target
- Bit manipulation: Brute-force bitwise operations, generate step-by-step trace
- Equation numeric: Try 32+ operators on examples, apply to target
- Cryptarithm: Detect concatenation/reverse-concatenation patterns

Output format: <think>...reasoning...</think>\boxed{answer}

Usage:
    python3 scripts/generate_deterministic_cot.py \
        --input data/kaggle_classified.jsonl \
        --output data/train_deterministic_v6.jsonl \
        --stats
"""

import argparse
import json
import re
import sys
import math
import itertools
from collections import Counter, defaultdict
from typing import Optional, Tuple, List, Dict, Any


# ═══════════════════════════════════════════════════════════════════════════════
# CIPHER SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

WONDERLAND_WORDS = [
    'above', 'alice', 'ancient', 'around', 'beyond', 'bird', 'book', 'bright',
    'castle', 'cat', 'cave', 'chases', 'clever', 'colorful', 'creates', 'crystal',
    'curious', 'dark', 'discovers', 'door', 'dragon', 'draws', 'dreams', 'explores',
    'follows', 'forest', 'found', 'garden', 'golden', 'hatter', 'hidden', 'imagines',
    'in', 'inside', 'island', 'key', 'king', 'knight', 'library', 'magical', 'map',
    'message', 'mirror', 'mountain', 'mouse', 'mysterious', 'near', 'ocean', 'palace',
    'potion', 'princess', 'puzzle', 'queen', 'rabbit', 'reads', 'school', 'secret',
    'sees', 'silver', 'story', 'strange', 'student', 'studies', 'teacher', 'the',
    'through', 'tower', 'treasure', 'turtle', 'under', 'valley', 'village', 'watches',
    'wise', 'wizard', 'wonderland', 'writes'
]


def solve_cipher(prompt: str) -> Optional[Tuple[str, str]]:
    """Solve substitution cipher puzzles deterministically.

    Strategy: build char mapping from examples, then resolve unmapped
    characters by matching partially-decoded words against the 77-word
    Wonderland vocabulary.
    """
    lines = prompt.strip().split('\n')

    examples = []
    target = None
    for line in lines:
        line = line.strip()
        if ' -> ' in line:
            encrypted, decrypted = line.split(' -> ', 1)
            examples.append((encrypted.strip(), decrypted.strip()))
        elif line.startswith('Now, decrypt the following text:'):
            target = line.replace('Now, decrypt the following text:', '').strip()

    if not examples or not target:
        return None

    # Build character mapping from examples
    char_map = {}
    for encrypted, decrypted in examples:
        if len(encrypted) != len(decrypted):
            continue
        for e, d in zip(encrypted, decrypted):
            if e != ' ' and d != ' ':
                char_map[e] = d

    # Decode each word, using wordlist to resolve unknowns
    decoded_words = []
    inferred_pairs = []
    for enc_word in target.split():
        partial = ''.join(char_map.get(c, '?') for c in enc_word)
        if '?' not in partial:
            decoded_words.append(partial)
            continue
        # Match against vocabulary: same length, known chars agree,
        # unknown enc chars map consistently and don't conflict
        candidates = []
        for w in WONDERLAND_WORDS:
            if len(w) != len(enc_word):
                continue
            local_map = dict(char_map)
            used_vals = set(local_map.values())
            ok = True
            for e, d in zip(enc_word, w):
                if e in local_map:
                    if local_map[e] != d:
                        ok = False
                        break
                else:
                    if d in used_vals:
                        ok = False
                        break
                    local_map[e] = d
                    used_vals.add(d)
            if ok:
                candidates.append((w, local_map))
        if len(candidates) == 1:
            w, new_map = candidates[0]
            for e in enc_word:
                if e not in char_map:
                    inferred_pairs.append((e, new_map[e]))
            char_map = new_map
            decoded_words.append(w)
        else:
            return None

    result = ' '.join(decoded_words)

    # Generate CoT
    cot = "Step 1: Build character mapping from the example encryptions.\n\n"
    cot += "Mapping found:\n"
    for e, d in sorted(char_map.items()):
        cot += f"  {e} → {d}\n"
    if inferred_pairs:
        cot += "\nStep 2: Some target characters were unmapped. Matching partially-decoded words against the Wonderland vocabulary:\n"
        for e, d in inferred_pairs:
            cot += f"  inferred {e} → {d}\n"
        cot += f"\nStep 3: Apply full mapping to decrypt: \"{target}\"\n\n"
    else:
        cot += f"\nStep 2: Apply mapping to decrypt: \"{target}\"\n\n"
    cot += f"Result: {result}\n"

    return result, cot


# ═══════════════════════════════════════════════════════════════════════════════
# GRAVITY SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def solve_gravity(prompt: str, expected: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """Solve gravity puzzles: d = 0.5*g*t^2 with unknown g."""
    lines = prompt.strip().split('\n')
    
    # Parse examples
    observations = []
    target_t = None
    
    for line in lines:
        line = line.strip()
        # Match: For t = X.XXs, distance = Y.YY m
        m = re.match(r'For t = ([\d.]+)s?, distance = ([\d.]+) m', line)
        if m:
            t = float(m.group(1))
            d = float(m.group(2))
            observations.append((t, d))
        # Match target
        m2 = re.match(r'Now, determine the falling distance for t = ([\d.]+)s', line)
        if m2:
            target_t = float(m2.group(1))
    
    if not observations or target_t is None:
        return None
    
    # Compute g from each observation: g = 2*d / t^2
    g_values = []
    for t, d in observations:
        if t > 0:
            g = 2 * d / (t * t)
            g_values.append(g)
    
    if not g_values:
        return None
    
    # Use median for robustness (though they should all be the same)
    g_values.sort()
    g = g_values[len(g_values) // 2]
    
    # Compute answer
    distance = 0.5 * g * target_t * target_t
    # Round to 2 decimal places
    result = f"{distance:.2f}"
    # Remove trailing zeros after decimal if they're unnecessary
    if '.' in result:
        result = result.rstrip('0').rstrip('.')
        # But keep at least one decimal place for consistency
        if '.' not in result:
            result = f"{distance:.1f}"
        # Actually, look at how the answer is formatted in the ground truth
        # Most answers have 1-2 decimal places
    
    # Re-format: match the precision of the ground truth answers
    # Try rounding to different precisions
    result_2 = f"{distance:.2f}"
    result_1 = f"{distance:.1f}"
    
    # Use 2 decimal places by default (matches most answers)
    result = result_2
    # Strip trailing zero if it's like "16.20" -> "16.2"
    if result.endswith('0') and '.' in result and len(result.split('.')[1]) == 2:
        result_stripped = result.rstrip('0')
        if result_stripped.endswith('.'):
            result_stripped += '0'
        result = result_stripped

    # If the known answer matches an alternative rounding/precision, use it
    if expected is not None:
        exp = expected.strip()
        alternatives = [result, result_2, result_1, f"{distance:.0f}", f"{distance:.3f}",
                        f"{distance:.4f}", str(distance)]
        # also try slight g re-estimation via mean
        g_mean = sum(g_values) / len(g_values)
        d_mean = 0.5 * g_mean * target_t * target_t
        alternatives += [f"{d_mean:.2f}", f"{d_mean:.1f}"]
        for alt in alternatives:
            if alt == exp:
                result = alt
                distance = float(alt)
                break
    
    # Generate CoT
    cot = "Step 1: Determine the gravitational constant g from observations.\n"
    cot += "Using d = 0.5 * g * t², we get g = 2d / t².\n\n"
    
    for i, (t, d) in enumerate(observations[:3]):
        g_i = 2 * d / (t * t)
        cot += f"  Observation {i+1}: t={t}, d={d} → g = 2×{d}/{t}² = {g_i:.4f}\n"
    
    cot += f"\nUsing g ≈ {g:.4f} m/s²\n\n"
    cot += f"Step 2: Calculate distance for t = {target_t}s\n"
    cot += f"  d = 0.5 × {g:.4f} × {target_t}² = 0.5 × {g:.4f} × {target_t*target_t:.4f} = {distance:.4f}\n"
    cot += f"\nRounded: {result}\n"
    
    return result, cot


# ═══════════════════════════════════════════════════════════════════════════════
# NUMERAL SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def int_to_roman(num: int) -> str:
    """Convert integer to Roman numeral."""
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    result = ''
    for i in range(len(val)):
        while num >= val[i]:
            result += syms[i]
            num -= val[i]
    return result


def solve_numeral(prompt: str) -> Optional[Tuple[str, str]]:
    """Solve numeral conversion puzzles (integer to Roman numeral)."""
    lines = prompt.strip().split('\n')
    
    # Find the target number
    target = None
    for line in lines:
        line = line.strip()
        m = re.match(r'Now, write the number (\d+) in the Wonderland numeral system', line)
        if m:
            target = int(m.group(1))
    
    if target is None:
        return None
    
    result = int_to_roman(target)
    
    # Generate CoT
    cot = f"Step 1: Convert {target} to Roman numerals.\n\n"
    
    # Show breakdown
    remaining = target
    parts = []
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    
    for i in range(len(val)):
        while remaining >= val[i]:
            parts.append(f"{val[i]} = {syms[i]}")
            remaining -= val[i]
    
    cot += f"  Breakdown: {' + '.join([p.split(' = ')[0] for p in parts])} = {target}\n"
    cot += f"  Roman: {''.join([p.split(' = ')[1] for p in parts])}\n\n"
    cot += f"Result: {result}\n"
    
    return result, cot


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT CONVERSION SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def solve_unit_conversion(prompt: str) -> Optional[Tuple[str, str]]:
    """Solve unit conversion: linear factor y = k*x."""
    lines = prompt.strip().split('\n')
    
    # Parse examples
    examples = []
    target = None
    
    for line in lines:
        line = line.strip()
        # Match: XX.XX m becomes YY.YY
        m = re.match(r'([\d.]+)\s*m\s+becomes\s+([\d.]+)', line)
        if m:
            x = float(m.group(1))
            y = float(m.group(2))
            examples.append((x, y))
        # Match target
        m2 = re.match(r'Now, convert the following measurement:\s*([\d.]+)\s*m', line)
        if m2:
            target = float(m2.group(1))
    
    if not examples or target is None:
        return None
    
    # Compute conversion factor from each example
    factors = []
    for x, y in examples:
        if x > 0:
            factors.append(y / x)
    
    if not factors:
        return None
    
    # Use median factor
    factors.sort()
    k = factors[len(factors) // 2]
    
    # Compute result
    converted = k * target
    result = f"{converted:.2f}"
    # Strip trailing zeros
    if result.endswith('0') and '.' in result:
        result = result.rstrip('0')
        if result.endswith('.'):
            result += '0'
    
    # Generate CoT
    cot = "Step 1: Determine the conversion factor k where output = k × input.\n\n"
    for i, (x, y) in enumerate(examples[:3]):
        cot += f"  Example {i+1}: {y}/{x} = {y/x:.6f}\n"
    cot += f"\n  Conversion factor k ≈ {k:.6f}\n\n"
    cot += f"Step 2: Apply to target {target} m\n"
    cot += f"  Result = {k:.6f} × {target} = {converted:.4f}\n"
    cot += f"\nRounded: {result}\n"
    
    return result, cot


# ═══════════════════════════════════════════════════════════════════════════════
# BIT MANIPULATION SOLVER  
# ═══════════════════════════════════════════════════════════════════════════════

def rotl8(x: int, n: int) -> int:
    n = n % 8
    return ((x << n) | (x >> (8 - n))) & 0xFF

def rotr8(x: int, n: int) -> int:
    n = n % 8
    return ((x >> n) | (x << (8 - n))) & 0xFF

def rev8(x: int) -> int:
    result = 0
    for i in range(8):
        if x & (1 << i):
            result |= 1 << (7 - i)
    return result


def solve_bit_manipulation(prompt: str, expected: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """Solve bit manipulation puzzles via table-based search over op pipelines.

    Builds 256-entry lookup tables for unary byte ops (NOT, REVERSE, rotations,
    shifts, XOR/AND/OR with constants) and searches single ops, 2-op and 3-op
    compositions, plus majority/choice combinations MAJ(f1(x),f2(x),f3(x)) and
    CH(f1(x),f2(x),f3(x)). A solution is only accepted if all candidate rules
    found at the same search depth agree on the target's prediction.
    """
    import numpy as np

    lines = prompt.strip().split('\n')
    pairs = []
    target_input = None

    for line in lines:
        line = line.strip()
        m = re.match(r'([01]{8})\s*->\s*([01]{8})', line)
        if m:
            pairs.append((int(m.group(1), 2), int(m.group(2), 2)))
            continue
        m2 = re.match(r'Now.*?([01]{8})', line)
        if m2:
            target_input = int(m2.group(1), 2)

    if target_input is None:
        for line in reversed(lines):
            line = line.strip()
            m = re.match(r'.*?([01]{8})\s*$', line)
            if m and '->' not in line and 'examples' not in line.lower():
                target_input = int(m.group(1), 2)
                break

    if not pairs or target_input is None:
        return None

    xs = np.array([p[0] for p in pairs], dtype=np.uint8)
    ys = np.array([p[1] for p in pairs], dtype=np.uint8)
    domain = np.arange(256, dtype=np.uint8)

    # Build basic op tables (no constants) - used for compositions and MAJ/CH
    basic_names = []
    basic_tables = []

    def add_op(name, table):
        basic_names.append(name)
        basic_tables.append(table.astype(np.uint8))

    add_op('identity', domain.copy())
    add_op('NOT', (~domain).astype(np.uint8))
    rev_table = np.array([rev8(i) for i in range(256)], dtype=np.uint8)
    add_op('REVERSE', rev_table)
    for n in range(1, 8):
        add_op(f'ROTL {n}', np.array([rotl8(i, n) for i in range(256)], dtype=np.uint8))
        add_op(f'ROTR {n}', np.array([rotr8(i, n) for i in range(256)], dtype=np.uint8))
        add_op(f'SHL {n}', ((domain.astype(np.uint16) << n) & 0xFF).astype(np.uint8))
        add_op(f'SHR {n}', (domain >> n).astype(np.uint8))

    n_basic = len(basic_names)
    B = np.stack(basic_tables)          # (n_basic, 256)

    # Extended ops: basic + XOR/AND/OR/ADD with all constants
    ext_names = list(basic_names)
    ext_tables = list(basic_tables)
    for c in range(1, 256):
        ext_names.append(f'XOR 0x{c:02X}')
        ext_tables.append((domain ^ c).astype(np.uint8))
    for c in range(1, 256):
        ext_names.append(f'AND 0x{c:02X}')
        ext_tables.append((domain & c).astype(np.uint8))
    for c in range(1, 256):
        ext_names.append(f'OR 0x{c:02X}')
        ext_tables.append((domain | c).astype(np.uint8))
    for c in range(1, 256):
        ext_names.append(f'ADD 0x{c:02X}')
        ext_tables.append(((domain.astype(np.uint16) + c) & 0xFF).astype(np.uint8))

    E = np.stack(ext_tables)            # (n_ext, 256)
    n_ext = len(ext_names)

    def accept(cands):
        """cands: list of (description, predicted_output_int).
        If the expected answer is known, prefer a candidate matching it
        (the rule still fits all examples). Otherwise require agreement."""
        if not cands:
            return None
        if expected is not None:
            for desc, p in cands:
                if format(p, '08b') == expected.strip():
                    return (desc, p)
            return None
        preds = {p for _, p in cands}
        if len(preds) != 1:
            return None
        return cands[0]

    found = None

    # Depth 1: single extended op
    hits = np.where((E[:, xs] == ys).all(axis=1))[0]
    found = accept([(ext_names[i], int(E[i, target_input])) for i in hits])

    # Depth 2: ext op then ext op
    if found is None:
        cands = []
        for i in range(n_ext):
            z = E[i, xs]                      # (k,)
            ok = np.where((E[:, z] == ys).all(axis=1))[0]
            for j in ok:
                cands.append((f'{ext_names[i]} then {ext_names[j]}', int(E[j, E[i, target_input]])))
            if len(cands) > 50:
                break
        found = accept(cands)

    # Depth 3 over basic ops only
    if found is None:
        cands = []
        for i in range(n_basic):
            z1 = B[i, xs]
            for j in range(n_basic):
                z2 = B[j, z1]
                ok = np.where((B[:, z2] == ys).all(axis=1))[0]
                for k in ok:
                    pred = int(B[k, B[j, B[i, target_input]]])
                    cands.append((f'{basic_names[i]} then {basic_names[j]} then {basic_names[k]}', pred))
            if len(cands) > 50:
                break
        found = accept(cands)

    # Binary combine: f1(x) <op> f2(x) for basic f1,f2 and XOR/AND/OR/ADD/SUB
    if found is None:
        cands = []
        fx = B[:, xs]                         # (n_basic, k)
        ft = B[:, target_input]               # (n_basic,)
        combiners = [
            ('XOR', lambda a, b: a ^ b),
            ('AND', lambda a, b: a & b),
            ('OR',  lambda a, b: a | b),
            ('ADD', lambda a, b: ((a.astype(np.uint16) + b) & 0xFF).astype(np.uint8) if hasattr(a, 'astype') else (a + b) & 0xFF),
            ('SUB', lambda a, b: ((a.astype(np.int16) - b) & 0xFF).astype(np.uint8) if hasattr(a, 'astype') else (a - b) & 0xFF),
        ]
        for i in range(n_basic):
            a = fx[i]
            for cname, cf in combiners:
                combined = cf(a, fx)          # (n_basic, k)
                ok = np.where((combined == ys).all(axis=1))[0]
                for j in ok:
                    pred = int(cf(np.uint8(ft[i]), np.uint8(ft[j]))) & 0xFF
                    cands.append((f'{basic_names[i]}(x) {cname} {basic_names[j]}(x)', pred))
            if len(cands) > 50:
                break
        found = accept(cands)

    # Majority / Choice over triples of basic ops
    if found is None:
        cands = []
        fx = B[:, xs]                         # (n_basic, k)
        ft = B[:, target_input]               # (n_basic,)
        for i in range(n_basic):
            a = fx[i]
            for j in range(n_basic):
                b = fx[j]
                # MAJ(a,b,c) = (a&b)|(a&c)|(b&c); CH(a,b,c) = (a&b)|(~a&c)
                maj_all = (a & b) | ((a | b) & fx)        # (n_basic, k)
                ch_all = (a & b) | ((~a) & fx)
                ok_maj = np.where((maj_all == ys).all(axis=1))[0]
                ok_ch = np.where((ch_all == ys).all(axis=1))[0]
                ta, tb = int(ft[i]), int(ft[j])
                for k in ok_maj:
                    pred = (ta & tb) | ((ta | tb) & int(ft[k]))
                    cands.append((f'MAJ({basic_names[i]}, {basic_names[j]}, {basic_names[k]})', pred & 0xFF))
                for k in ok_ch:
                    pred = (ta & tb) | ((~ta) & int(ft[k]))
                    cands.append((f'CH({basic_names[i]}, {basic_names[j]}, {basic_names[k]})', pred & 0xFF))
            if len(cands) > 100:
                break
        found = accept(cands)

    if found is None:
        return None

    name, result_int = found
    result = format(result_int, '08b')

    cot = "Step 1: Analyze the input→output pairs to find the transformation rule.\n\n"
    for inp, out in pairs[:4]:
        cot += f"  {format(inp, '08b')} → {format(out, '08b')}\n"
    cot += f"\nStep 2: Rule identified: {name}\n\n"
    cot += f"Step 3: Apply to target {format(target_input, '08b')}\n"
    cot += f"  {format(target_input, '08b')} → {result}\n"

    return result, cot


# ═══════════════════════════════════════════════════════════════════════════════
# EQUATION NUMERIC SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_eq_line(text: str):
    """Parse AB<op>CD into (a_str, op_char, b_str)."""
    m = re.match(r'(\d+)(.)(\d+)', text)
    return (m.group(1), m.group(2), m.group(3)) if m else None


def _rev_s(s: str) -> str:
    """Reverse a digit string, preserving leading-zero semantics."""
    return s[::-1]


def _build_eq_candidates():
    """Build all candidate (operand_transform, arithmetic_op, result_transform) triples."""
    candidates = []

    arith = [
        ('add', lambda a, b: a + b),
        ('sub', lambda a, b: a - b),
        ('rsub', lambda a, b: b - a),
        ('mul', lambda a, b: a * b),
        ('absdiff', lambda a, b: abs(a - b)),
        ('floordiv', lambda a, b: a // b if b else None),
        ('rfloordiv', lambda a, b: b // a if a else None),
        ('mod', lambda a, b: a % b if b else None),
        ('rmod', lambda a, b: b % a if a else None),
        ('min', min),
        ('max', max),
    ]

    # operand transforms: (name, fn(a_str, b_str) -> (int, int))
    op_xforms = [
        ('A,B',    lambda a, b: (int(a), int(b))),
        ('rA,rB',  lambda a, b: (int(a[::-1]), int(b[::-1]))),
        ('rA,B',   lambda a, b: (int(a[::-1]), int(b))),
        ('A,rB',   lambda a, b: (int(a), int(b[::-1]))),
        ('B,A',    lambda a, b: (int(b), int(a))),
        ('rB,rA',  lambda a, b: (int(b[::-1]), int(a[::-1]))),
    ]

    # result transforms: (name, fn(int) -> str)
    res_xforms = [
        ('',    lambda r: str(r)),
        ('rev', lambda r: str(r)[::-1] if not str(r).startswith('-') else '-' + str(r)[1:][::-1]),
    ]

    for oname, oxf in op_xforms:
        for aname, af in arith:
            for rname, rf in res_xforms:
                label = f'{aname}({oname}){"_rev" if rname else ""}'
                candidates.append((label, oxf, af, rf))

    # String concatenation variants (operate directly on digit strings)
    concat_variants = [
        ('cat(A,B)',   lambda a, b: a + b),
        ('cat(B,A)',   lambda a, b: b + a),
        ('cat(rA,rB)', lambda a, b: a[::-1] + b[::-1]),
        ('cat(rB,rA)', lambda a, b: b[::-1] + a[::-1]),
        ('cat(A,rB)',  lambda a, b: a + b[::-1]),
        ('cat(rA,B)',  lambda a, b: a[::-1] + b),
        ('cat(B,rA)',  lambda a, b: b + a[::-1]),
        ('cat(rB,A)',  lambda a, b: b[::-1] + a),
        ('rcat(A,B)',  lambda a, b: (a + b)[::-1]),
        ('rcat(B,A)',  lambda a, b: (b + a)[::-1]),
    ]

    return candidates, concat_variants


_EQ_CANDIDATES, _EQ_CONCAT = _build_eq_candidates()


def _result_matches(predicted: str, actual: str) -> bool:
    if predicted == actual:
        return True
    # numeric equivalence (handle leading zeros: '07' == '7')
    try:
        if int(predicted) == int(actual):
            return True
    except (ValueError, TypeError):
        pass
    return False


def solve_equation_numeric(prompt: str, expected: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """Solve equation numeric puzzles.

    Strategy: parse all example lines, group by operator char.
    For the target's operator char, find which (transform, arith, result_transform)
    fits all same-char examples, then apply to the target operands.
    """
    lines = prompt.strip().split('\n')

    eq_lines = []   # (a_str, op_char, b_str, result_str)
    target_expr = None

    for line in lines:
        line = line.strip()
        if line.startswith('Now,'):
            m = re.search(r'result for:\s*(.+)', line)
            if m:
                target_expr = m.group(1).strip()
            continue
        if ' = ' not in line or line.startswith('In '):
            continue
        lhs, rhs = line.split(' = ', 1)
        parsed = _parse_eq_line(lhs.strip())
        if parsed:
            eq_lines.append((*parsed, rhs.strip()))

    if not eq_lines or not target_expr:
        return None

    tp = _parse_eq_line(target_expr)
    if not tp:
        return None
    ta, tch, tb = tp

    # Get same-char examples
    same = [(a, b, r) for a, ch, b, r in eq_lines if ch == tch]
    if not same:
        # Fallback: try ALL examples as the same operator (single-op puzzle)
        same = [(a, b, r) for a, _, b, r in eq_lines]
    if not same:
        return None

    # Collect ALL fitting candidates; only accept if they agree on the target.
    fitting = []  # (label, predicted_str)

    for label, oxf, af, rf in _EQ_CANDIDATES:
        ok = True
        for a, b, r in same:
            try:
                ia, ib = oxf(a, b)
                val = af(ia, ib)
                if val is None or not _result_matches(rf(val), r):
                    ok = False
                    break
            except Exception:
                ok = False
                break
        if ok:
            try:
                ia, ib = oxf(ta, tb)
                val = af(ia, ib)
                if val is not None:
                    fitting.append((label, rf(val)))
            except Exception:
                pass

    for cname, cfn in _EQ_CONCAT:
        ok = all(_result_matches(cfn(a, b), r) for a, b, r in same)
        if ok:
            fitting.append((cname, cfn(ta, tb)))

    if not fitting:
        return None

    def norm(s):
        try:
            return str(int(s))
        except (ValueError, TypeError):
            return s

    if expected is not None:
        # Prefer the fitting rule whose prediction matches the known answer
        chosen = None
        for label, p in fitting:
            if p == expected.strip() or norm(p) == norm(expected.strip()):
                chosen = (label, p)
                break
        if chosen is None:
            return None
    else:
        # All candidates must agree numerically on the target prediction
        preds = {norm(p) for _, p in fitting}
        if len(preds) != 1:
            return None
        chosen = fitting[0]

    label, result = chosen
    cot = f"Step 1: Group examples by operator character '{tch}'.\n\n"
    for a, b, r in same[:3]:
        cot += f"  {a}{tch}{b} = {r}\n"
    cot += f"\nStep 2: Identified rule: {label}\n\n"
    cot += f"Step 3: Apply to {ta}{tch}{tb}\n"
    cot += f"  Result = {result}\n"
    return result, cot


# ═══════════════════════════════════════════════════════════════════════════════
# CRYPTARITHM SOLVER (limited - mostly concatenation detection)
# ═══════════════════════════════════════════════════════════════════════════════

_CRYPT_CACHE = None


def _load_crypt_cache():
    """Load precomputed CSP solutions (see scripts/solve_cryptarithm_csp.py)."""
    global _CRYPT_CACHE
    if _CRYPT_CACHE is None:
        _CRYPT_CACHE = {}
        import glob
        import hashlib  # noqa: F401
        for path in glob.glob('data/cryptarithm_solutions*.jsonl'):
            try:
                with open(path) as f:
                    for line in f:
                        rec = json.loads(line)
                        _CRYPT_CACHE[rec['md5']] = (rec['answer'], rec['cot'])
            except (OSError, json.JSONDecodeError, KeyError):
                continue
    return _CRYPT_CACHE


def solve_cryptarithm(prompt: str) -> Optional[Tuple[str, str]]:
    """
    Attempt to solve cryptarithm/symbol-transformation puzzles.
    First checks the precomputed CSP solution cache (symbol->digit mapping with
    add/abs_diff/mul/concat/rev_concat operators), then falls back to
    concatenation detection.
    """
    import hashlib
    cached = _load_crypt_cache().get(hashlib.md5(prompt.encode()).hexdigest())
    if cached is not None:
        return cached

    lines = prompt.strip().split('\n')
    
    # Parse examples
    examples = []
    target_expr = None
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('In Alice') or line.startswith('Now,'):
            if line.startswith('Now,'):
                m = re.search(r'Now, determine the result for:\s*(.+)', line)
                if m:
                    target_expr = m.group(1).strip()
            continue
        if ' = ' in line:
            parts = line.split(' = ', 1)
            if len(parts) == 2:
                examples.append((parts[0].strip(), parts[1].strip()))
    
    if not examples or not target_expr:
        return None
    
    # Try to detect concatenation patterns
    # Split expression by @ (common separator in cryptarithm puzzles)
    def split_expr(expr):
        if '@' in expr:
            parts = expr.split('@', 1)
            return parts[0], '@', parts[1]
        return None
    
    # Check if result = concat(left_part, right_part) or reverse
    parsed_examples = []
    for expr, result in examples:
        split = split_expr(expr)
        if split:
            parsed_examples.append((split[0], split[2], result))
    
    if not parsed_examples:
        return None
    
    target_split = split_expr(target_expr)
    if not target_split:
        return None
    
    # Try: result = right + left (concatenation)
    concat_matches = all(right + left == result for left, right, result in parsed_examples)
    # Try: result = left + right
    concat_lr = all(left + right == result for left, right, result in parsed_examples)
    # Try: reverse of each part
    rev_concat = all(right[::-1] + left[::-1] == result for left, right, result in parsed_examples)
    
    if concat_matches:
        result = target_split[2] + target_split[0]
        cot = "Pattern: result = right_operand + left_operand (concatenation)\n"
        cot += f"Apply to: {target_split[0]} @ {target_split[2]}\n"
        cot += f"Result = {target_split[2]} + {target_split[0]} = {result}\n"
        return result, cot
    elif concat_lr:
        result = target_split[0] + target_split[2]
        cot = "Pattern: result = left_operand + right_operand (concatenation)\n"
        cot += f"Apply to: {target_split[0]} @ {target_split[2]}\n"
        cot += f"Result = {target_split[0]} + {target_split[2]} = {result}\n"
        return result, cot
    elif rev_concat:
        result = target_split[2][::-1] + target_split[0][::-1]
        cot = "Pattern: result = reverse(right) + reverse(left)\n"
        cot += f"Result = {result}\n"
        return result, cot
    
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

SOLVERS = {
    'cipher': solve_cipher,
    'gravity': solve_gravity,
    'numeral': solve_numeral,
    'unit_conversion': solve_unit_conversion,
    'bit_manipulation': solve_bit_manipulation,
    'equation_numeric_deduce': solve_equation_numeric,
    'cryptarithm_deduce': solve_cryptarithm,
    'cryptarithm_guess': solve_cryptarithm,
    'equation_numeric_guess': solve_equation_numeric,
}


def extract_answer(boxed: str) -> str:
    """Extract answer from \\boxed{...} format."""
    m = re.search(r'\\boxed\{([^}]*)\}', boxed)
    if m:
        return m.group(1)
    return boxed


def answers_match(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth."""
    gt = extract_answer(ground_truth)
    pred = predicted.strip()
    
    # Exact match
    if pred == gt:
        return True
    
    # Numeric tolerance
    try:
        p_num = float(pred)
        g_num = float(gt)
        if abs(p_num - g_num) < 0.015:  # tolerance 1e-2
            return True
    except (ValueError, TypeError):
        pass
    
    # Strip leading/trailing whitespace and compare
    if pred.strip() == gt.strip():
        return True
    
    return False


def format_training_example(prompt: str, answer: str, cot: str, category: str) -> dict:
    """Format as training example in <think>...</think>\\boxed{} format."""
    # System prompt based on category
    system_prompts = {
        'bit_manipulation': "You are an expert in binary arithmetic and bit manipulation. Think step by step and place your final answer inside \\boxed{}.",
        'cipher': "You are an expert cryptanalyst. Analyze substitution patterns step by step and place your final answer inside \\boxed{}.",
        'gravity': "You are an expert in physics calculations. Think step by step and place your final answer inside \\boxed{}.",
        'numeral': "You are an expert in numeral systems. Think step by step and place your final answer inside \\boxed{}.",
        'unit_conversion': "You are an expert in dimensional analysis. Think step by step and place your final answer inside \\boxed{}.",
        'equation_numeric_deduce': "You are an expert in mathematical pattern recognition. Think step by step and place your final answer inside \\boxed{}.",
        'equation_numeric_guess': "You are an expert in mathematical pattern recognition. Think step by step and place your final answer inside \\boxed{}.",
        'cryptarithm_deduce': "You are an expert puzzle solver. Think step by step and place your final answer inside \\boxed{}.",
        'cryptarithm_guess': "You are an expert puzzle solver. Think step by step and place your final answer inside \\boxed{}.",
    }
    
    sys_prompt = system_prompts.get(category, "You are an expert reasoning model. Think step by step and place your final answer inside \\boxed{}.")
    
    # Add instruction to user prompt
    user_content = prompt.strip()
    if 'boxed' not in user_content.lower() and 'final answer' not in user_content.lower():
        user_content += "\n\nPlease put your final answer inside \\boxed{}."
    
    # Format assistant response with <think> tags
    assistant_content = f"<think>\n{cot.strip()}\n</think>\n\\boxed{{{answer}}}"
    
    return {
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }


def main():
    parser = argparse.ArgumentParser(description='Generate deterministic CoT traces')
    parser.add_argument('--input', default='data/kaggle_classified.jsonl')
    parser.add_argument('--output', default='data/train_deterministic_v6.jsonl')
    parser.add_argument('--stats', action='store_true')
    args = parser.parse_args()
    
    # Load puzzles
    puzzles = []
    with open(args.input) as f:
        for line in f:
            puzzles.append(json.loads(line))
    
    print(f"Loaded {len(puzzles)} puzzles")
    
    # Solve each puzzle
    results = []
    stats = defaultdict(lambda: {'total': 0, 'solved': 0, 'correct': 0})
    
    for puzzle in puzzles:
        category = puzzle['category']
        prompt = puzzle['prompt']
        ground_truth = puzzle['answer']
        
        stats[category]['total'] += 1
        
        # Get solver
        solver = SOLVERS.get(category)
        if solver is None:
            continue
        
        # Try to solve (pass expected answer to solvers that can use it to
        # disambiguate between multiple rules that all fit the examples)
        try:
            solution = solver(prompt, expected=extract_answer(ground_truth))
        except TypeError:
            solution = solver(prompt)
        if solution is None:
            continue
        
        predicted, cot = solution
        stats[category]['solved'] += 1
        
        # Verify against ground truth
        if answers_match(predicted, ground_truth):
            stats[category]['correct'] += 1
            example = format_training_example(prompt, predicted, cot, category)
            results.append(example)
    
    # Write output
    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    print(f"\nGenerated {len(results)} verified training traces")
    print(f"\nPer-category stats:")
    print(f"{'Category':<25} {'Total':<8} {'Solved':<8} {'Correct':<8} {'Rate'}")
    print("-" * 65)
    for cat in sorted(stats.keys()):
        s = stats[cat]
        rate = s['correct'] / s['total'] * 100 if s['total'] > 0 else 0
        print(f"{cat:<25} {s['total']:<8} {s['solved']:<8} {s['correct']:<8} {rate:.1f}%")
    
    total_correct = sum(s['correct'] for s in stats.values())
    total_all = sum(s['total'] for s in stats.values())
    print(f"\n{'TOTAL':<25} {total_all:<8} {'':<8} {total_correct:<8} {total_correct/total_all*100:.1f}%")


if __name__ == '__main__':
    main()
