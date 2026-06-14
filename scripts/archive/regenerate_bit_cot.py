#!/usr/bin/env python3
"""
Regenerate high-quality CoT traces for bit_manipulation puzzles.

The original DeepSeek generation produced 11.2% correct answers because the
generic prompt caused rambling without systematic analysis.

This script uses an optimized system prompt for bit manipulation that forces:
  1. List all input-output pairs as 8-bit binary
  2. Compute XOR between each input-output pair to see which bits change
  3. Try common bitwise operations: NOT, AND, OR, XOR, shift, rotate
  4. Test each hypothesis against ALL examples
  5. Only output the answer once the rule is verified

Usage:
    export DEEPSEEK_API_KEY=*** reumanlab 'cat ~/env/deepseek-token')
    python3 scripts/regenerate_bit_cot.py \
        --input data/bit_manipulation_puzzles.jsonl \
        --output data/cot_bit_v2.jsonl \
        --max-concurrent 8 \
        --test 5
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    import httpx
    USE_HTTPX = True
except ImportError:
    USE_HTTPX = False
    import aiohttp

BIT_MANIPULATION_PROMPT = """You are an expert in binary arithmetic. For bit manipulation puzzles, follow this exact protocol:

PROTOCOL:
1. LIST every input-output pair as 8-bit binary strings
2. COMPUTE XOR(input, output) for each pair — this shows which bits changed
3. TEST common operations one by one:
   - Left/right rotate by N bits
   - Bitwise NOT (invert all bits)
   - AND/OR/XOR with a constant mask
   - Shift left/right by N bits
   - Swap/reverse bit positions
4. For each hypothesis, VERIFY against ALL example pairs
5. Once verified, APPLY to the target input
6. OUTPUT the 8-bit result

Work systematically. Show your XOR analysis. Test each hypothesis explicitly.
Do NOT guess. Place final answer inside \\boxed{}.

Example approach:
- Input: 00010100, Output: 00000000 → XOR: 00010100 (bits 2,4 changed)
- Input: 11110100, Output: 00000110 → XOR: 11110010 (many bits)
- Hypothesis: rotate left by 2...
- Test against all: FAIL
- Hypothesis: XOR with mask 00010100... 
- Test against all: PASS
- Apply to target: ..."""


DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"


def load_api_key():
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        for p in [Path("/home/reumanlab/env/deepseek-token"), Path.home() / "env/deepseek-token"]:
            if p.exists():
                key = p.read_text().strip()
                break
    return key


def extract_boxed(text: str) -> str:
    patterns = [
        r'\\\\boxed\\{([^{}]*(?:\\{[^{}]*\\}[^{}]*)*)\\}',
        r'boxed\\{([^{}]*(?:\\{[^{}]*\\}[^{}]*)*)\\}',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
    return ""


def normalize_answer(ans: str) -> str:
    ans = ans.strip()
    if ans.startswith("\\boxed{") and ans.endswith("}"):
        ans = ans[7:-1]
    return ans.strip()


def answers_match(generated: str, ground_truth: str) -> bool:
    g = normalize_answer(generated)
    t = normalize_answer(ground_truth)
    if g == t:
        return True
    return g.replace(' ', '') == t.replace(' ', '')


async def generate_cot(puzzle: dict, api_key: str, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": BIT_MANIPULATION_PROMPT},
                {"role": "user", "content": puzzle["prompt"]}
            ],
            "max_tokens": 8000,
            "temperature": 0.1,
            "top_p": 0.9,
        }

        for attempt in range(3):
            try:
                if USE_HTTPX:
                    async with httpx.AsyncClient(timeout=180) as client:
                        resp = await client.post(DEEPSEEK_ENDPOINT, headers=headers, json=payload)
                        resp.raise_for_status()
                        data = resp.json()
                else:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            DEEPSEEK_ENDPOINT, headers=headers, json=payload,
                            timeout=aiohttp.ClientTimeout(total=180)
                        ) as resp:
                            data = await resp.json()

                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return {
                    "content": content,
                    "tokens_in": usage.get("prompt_tokens", 0),
                    "tokens_out": usage.get("completion_tokens", 0),
                    "success": True
                }
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {"content": "", "error": str(e), "success": False}


async def main():
    parser = argparse.ArgumentParser(description="Regenerate CoT for bit_manipulation")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--test", type=int, default=0, help="Test N puzzles first, then exit")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    api_key = load_api_key()
    if not api_key:
        print("ERROR: No API key")
        sys.exit(1)

    puzzles = []
    with open(args.input) as f:
        for line in f:
            puzzles.append(json.loads(line))

    end = args.end or len(puzzles)
    puzzles = puzzles[args.start:end]

    if args.test > 0:
        puzzles = puzzles[:args.test]
        print(f"TEST MODE: {len(puzzles)} puzzles")

    print(f"Processing {len(puzzles)} puzzles with {args.max_concurrent} concurrent calls")
    print(f"Model: {DEEPSEEK_MODEL}")

    semaphore = asyncio.Semaphore(args.max_concurrent)
    t0 = time.time()
    total_success = 0
    total_match = 0

    with open(args.output, 'w') as out:
        for i in range(0, len(puzzles), args.max_concurrent):
            batch = puzzles[i:i + args.max_concurrent]
            tasks = [generate_cot(p, api_key, semaphore) for p in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for puzzle, result in zip(batch, results):
                if isinstance(result, Exception):
                    continue
                if not result["success"]:
                    continue

                total_success += 1
                cot_text = result["content"]
                generated = extract_boxed(cot_text)
                ground = normalize_answer(puzzle["answer"])
                match = answers_match(generated, ground)
                if match:
                    total_match += 1

                record = {
                    "messages": [
                        {"role": "system", "content": BIT_MANIPULATION_PROMPT},
                        {"role": "user", "content": puzzle["prompt"]},
                        {"role": "assistant", "content": cot_text},
                    ],
                    "index": puzzle.get("index", i),
                    "metadata": {
                        "engine": "deepseek",
                        "model": DEEPSEEK_MODEL,
                        "category": "bit_manipulation",
                        "ground_truth": puzzle["answer"],
                        "generated_answer": generated,
                        "answer_match": match,
                        "tokens_in": result.get("tokens_in", 0),
                        "tokens_out": result.get("tokens_out", 0),
                    }
                }
                out.write(json.dumps(record) + "\n")

            elapsed = time.time() - t0
            done = i + len(batch)
            rate = total_success / elapsed if elapsed > 0 else 0
            mr = total_match / max(total_success, 1) * 100
            print(f"  [{min(done, len(puzzles))}/{len(puzzles)}] "
                  f"success={total_success} match={total_match} ({mr:.1f}%) "
                  f"rate={rate:.1f}/s elapsed={elapsed:.0f}s")

    elapsed = time.time() - t0
    mr = total_match / max(total_success, 1) * 100
    print(f"\n=== BIT REGENERATION DONE ===")
    print(f"Success: {total_success}/{len(puzzles)}")
    print(f"Match: {total_match} ({mr:.1f}%)")
    print(f"Original match rate: 11.2%")
    print(f"Improvement: {mr - 11.2:+.1f} pp")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    asyncio.run(main())
