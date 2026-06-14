#!/usr/bin/env python3
"""
Generate balanced cryptarithm + reasoning CoT training data.
Uses FORWARD generation (solution → puzzle) to avoid brute-force solving.
Output: JSONL in chat template format for SFTTrainer.
"""

import json, random, string
from collections import defaultdict

random.seed(42)

# ── Forward Cryptarithm Generator ─────────────────────────────────

def forward_gen_cryptarithm(len1=3, len2=3):
    """Generate a cryptarithm FROM a known solution (fast)."""
    for _ in range(100):
        # Pick random numbers
        lo1 = 10**(len1-1)
        hi1 = 10**len1 - 1
        lo2 = 10**(len2-1)
        hi2 = 10**len2 - 1
        
        n1 = random.randint(lo1, hi1)
        n2 = random.randint(lo2, hi2)
        total = n1 + n2
        
        # Get all digits used
        digits_str = str(n1) + str(n2) + str(total)
        unique_digits = set(int(d) for d in digits_str)
        
        if len(unique_digits) > 10:
            continue
        
        # Assign letters to digits
        avail_letters = list(string.ascii_uppercase)
        random.shuffle(avail_letters)
        digit_to_letter = {}
        for d in sorted(unique_digits):
            digit_to_letter[d] = avail_letters.pop(0)
        
        # Build words
        word1 = ''.join(digit_to_letter[int(d)] for d in str(n1))
        word2 = ''.join(digit_to_letter[int(d)] for d in str(n2))
        result = ''.join(digit_to_letter[int(d)] for d in str(total))
        
        # Solution mapping (letter → digit)
        solution = {v: k for k, v in digit_to_letter.items()}
        
        return word1, word2, result, solution, n1, n2, total
    
    return None


def generate_cot_reasoning(word1, word2, result, solution, n1, n2, total):
    """Generate step-by-step constraint propagation reasoning."""
    letters = sorted(set(word1 + word2 + result))
    leading = sorted({word1[0], word2[0], result[0]})
    
    # Column analysis
    max_len = len(result)
    w1 = word1.rjust(max_len)
    w2 = word2.rjust(max_len)
    res = result
    
    col_analysis = []
    carry = 0
    for i in range(max_len - 1, -1, -1):
        col_num = max_len - i
        c1 = w1[i] if w1[i] != ' ' else '-'
        c2 = w2[i] if w2[i] != ' ' else '-'
        cr = res[i]
        
        d1 = solution.get(c1, 0) if c1 != '-' else 0
        d2 = solution.get(c2, 0) if c2 != '-' else 0
        dr = solution[cr]
        
        col_sum = d1 + d2 + carry
        new_carry = col_sum // 10
        
        if c1 != '-' and c2 != '-':
            col_analysis.append(
                f"  Col {col_num}: {c1}({d1}) + {c2}({d2}) + carry({carry}) = {col_sum} → {cr}={col_sum % 10}, carry={new_carry}"
            )
        elif c1 != '-':
            col_analysis.append(
                f"  Col {col_num}: {c1}({d1}) + carry({carry}) = {col_sum} → {cr}={col_sum % 10}, carry={new_carry}"
            )
        else:
            col_analysis.append(
                f"  Col {col_num}: carry({carry}) = {col_sum} → {cr}={col_sum % 10}"
            )
        carry = new_carry
    
    assignment_str = ', '.join(f"{k}={v}" for k, v in sorted(solution.items()))
    
    reasoning = (
        f"STEP 1 — Parse puzzle: {word1} + {word2} = {result}\n"
        f"  Variables: {', '.join(letters)} → unique digits 0-9\n"
        f"  Leading letters ({', '.join(leading)}) ≠ 0\n\n"
        f"STEP 2 — Column analysis (right to left):\n"
        + '\n'.join(col_analysis) + "\n\n"
        f"STEP 3 — Assignment: {assignment_str}\n\n"
        f"STEP 4 — Verify: {n1} + {n2} = {total} ✓\n"
        f"  Digits used: {sorted(set(solution.values()))}\n"
        f"  All distinct: ✓ | No leading zeros: ✓"
    )
    
    return reasoning


def format_chat(system, user, assistant, category, method):
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "category": category,
        "method": method,
    }


# ── Classic Cryptarithms (hardcoded for quality) ──────────────────

CLASSICS = [
    ("SEND", "MORE", "MONEY", {"S":9,"E":5,"N":6,"D":7,"M":1,"O":0,"R":8,"Y":2}, 9567, 1085, 10652),
    ("COCA", "COLA", "OASIS", {"C":1,"O":0,"A":9,"L":7,"S":8,"I":6}, 1019, 1079, 9818),
]

SYS_CRYPTO = "You are an expert mathematical reasoner. Solve problems step by step using constraint propagation and logical deduction. Show your work clearly and verify your answer."
SYS_MATH = "You are an expert mathematical reasoner. Solve equations step by step."
SYS_BITS = "You are an expert in bit manipulation and binary arithmetic."


def main():
    all_examples = []
    
    # ── 1. Cryptarithm deduce (~250) ─────────────────────────────
    print("Generating cryptarithms (forward method)...")
    
    # Classic SEND+MORE=MONEY
    sol = {"S":9,"E":5,"N":6,"D":7,"M":1,"O":0,"R":8,"Y":2}
    reasoning = generate_cot_reasoning("SEND", "MORE", "MONEY", sol, 9567, 1085, 10652)
    all_examples.append(format_chat(
        SYS_CRYPTO,
        "Solve this cryptarithm. Each letter = unique digit (0-9). No leading zeros.\n\nSEND + MORE = MONEY",
        f"{reasoning}\n\nFINAL ANSWER: 9567 + 1085 = 10652",
        "cryptarithm_deduce", "constraint_satisfaction"
    ))
    
    # Generate random cryptarithms
    seen = set()
    configs = [(3,3), (3,3), (3,4), (3,4), (4,3), (4,4), (3,3), (4,3)]
    target = 250
    
    while len([e for e in all_examples if e["category"] == "cryptarithm_deduce"]) < target:
        l1, l2 = random.choice(configs)
        result = forward_gen_cryptarithm(l1, l2)
        if result is None:
            continue
        
        w1, w2, res, sol, n1, n2, total = result
        key = f"{w1}+{w2}={res}"
        if key in seen:
            continue
        seen.add(key)
        
        reasoning = generate_cot_reasoning(w1, w2, res, sol, n1, n2, total)
        all_examples.append(format_chat(
            SYS_CRYPTO,
            f"Solve this cryptarithm. Each letter = unique digit (0-9). No leading zeros.\n\n{w1} + {w2} = {res}",
            f"{reasoning}\n\nFINAL ANSWER: {n1} + {n2} = {total}",
            "cryptarithm_deduce", "constraint_satisfaction"
        ))
    
    print(f"  ✓ {len([e for e in all_examples if e['category'] == 'cryptarithm_deduce'])} cryptarithm_deduce")
    
    # ── 2. Equation numeric (~100) ───────────────────────────────
    print("Generating equation problems...")
    for _ in range(100):
        a = random.randint(2, 15)
        x = random.randint(-20, 20)
        b = random.randint(-50, 50)
        c = a * x + b
        
        eq = f"{a}x + {b} = {c}" if b >= 0 else f"{a}x - {abs(b)} = {c}"
        reasoning = (
            f"STEP 1 — Parse: {eq}\n"
            f"STEP 2 — Isolate x: {a}x = {c} - ({b}) = {c - b}\n"
            f"STEP 3 — Divide: x = {c - b} / {a} = {x}\n"
            f"STEP 4 — Verify: {a} × {x} + {b} = {a*x + b} = {c} ✓"
        )
        all_examples.append(format_chat(
            SYS_MATH, f"Solve for x: {eq}",
            f"{reasoning}\n\nFINAL ANSWER: x = {x}",
            "equation_numeric_deduce", "algebra"
        ))
    print(f"  ✓ 100 equation_numeric_deduce")
    
    # ── 3. Bit manipulation (~100) ───────────────────────────────
    print("Generating bit problems...")
    for _ in range(50):
        n = random.randint(1, 255)
        count = bin(n).count('1')
        binary = bin(n)[2:]
        reasoning = (
            f"STEP 1 — {n} in binary: {binary}\n"
            f"STEP 2 — Count 1-bits: {count}\n"
            f"STEP 3 — Verify: {' + '.join(['1'] * count)} = {count}"
        )
        all_examples.append(format_chat(
            SYS_BITS, f"How many 1-bits in binary representation of {n}?",
            f"{reasoning}\n\nFINAL ANSWER: {count}",
            "bit_manipulation", "binary_arithmetic"
        ))
    
    for _ in range(50):
        n = random.randint(1, 255)
        m = random.randint(1, 255)
        result = n ^ m
        reasoning = (
            f"STEP 1 — Convert:\n  {n:>3d} = {bin(n)[2:].zfill(8)}\n  {m:>3d} = {bin(m)[2:].zfill(8)}\n"
            f"STEP 2 — XOR bit by bit:\n"
            f"  {''.join('1' if a != b else '0' for a, b in zip(bin(n)[2:].zfill(8), bin(m)[2:].zfill(8)))}\n"
            f"STEP 3 — Result: {bin(result)[2:]} = {result}"
        )
        all_examples.append(format_chat(
            SYS_BITS, f"What is {n} XOR {m}?",
            f"{reasoning}\n\nFINAL ANSWER: {result}",
            "bit_manipulation", "binary_arithmetic"
        ))
    print(f"  ✓ 100 bit_manipulation")
    
    # ── 4. Unit conversion (~50) ─────────────────────────────────
    print("Generating unit conversion...")
    conversions = [
        ("km", "miles", 0.621371), ("kg", "pounds", 2.20462),
        ("celsius", "fahrenheit", None), ("meters", "feet", 3.28084),
        ("liters", "gallons", 0.264172), ("hectares", "acres", 2.47105),
    ]
    for _ in range(50):
        unit_from, unit_to, factor = random.choice(conversions)
        val = round(random.uniform(1, 1000), 1)
        
        if unit_from == "celsius":
            result = round(val * 9/5 + 32, 2)
            reasoning = (
                f"STEP 1 — Formula: °F = °C × 9/5 + 32\n"
                f"STEP 2 — Compute: {val} × 9/5 + 32 = {val * 9/5} + 32 = {result}\n"
                f"STEP 3 — Result: {val} °C = {result} °F"
            )
        else:
            result = round(val * factor, 2)
            reasoning = (
                f"STEP 1 — Conversion factor: 1 {unit_from} = {factor} {unit_to}\n"
                f"STEP 2 — Compute: {val} × {factor} = {result}\n"
                f"STEP 3 — Result: {val} {unit_from} = {result} {unit_to}"
            )
        
        all_examples.append(format_chat(
            SYS_MATH, f"Convert {val} {unit_from} to {unit_to}.",
            f"{reasoning}\n\nFINAL ANSWER: {result} {unit_to}",
            "unit_conversion", "arithmetic"
        ))
    print(f"  ✓ 50 unit_conversion")
    
    # ── Shuffle and save ─────────────────────────────────────────
    random.shuffle(all_examples)
    
    cats = defaultdict(int)
    for ex in all_examples:
        cats[ex["category"]] += 1
    
    print(f"\n=== Dataset Summary ===")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")
    print(f"  TOTAL: {len(all_examples)}")
    
    out_path = "/home/ubuntu/cryptarithm_cot.jsonl"
    with open(out_path, "w") as f:
        for ex in all_examples:
            # Only write messages (drop category/method metadata for training)
            f.write(json.dumps({"messages": ex["messages"]}) + "\n")
    
    print(f"\nSaved to {out_path}")
    
    # Also save with metadata for analysis
    meta_path = "/home/ubuntu/cryptarithm_cot_meta.jsonl"
    with open(meta_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
