#!/usr/bin/env python3
"""
Regenerate high-quality CoT traces for cryptarithm_deduce puzzles.

The original DeepSeek generation produced 2.4% correct answers for
symbol-transformation puzzles because the generic system prompt caused
the model to ramble without systematic analysis.

This script uses an optimized system prompt that forces:
  1. Position-by-position mapping from input to output symbols
  2. Build an explicit substitution/dictionary table
  3. Verify against ALL examples before answering
  4. Short, focused reasoning (no rambling)

Usage:
    export DEEPSEEK_API_KEY=$(ssh reumanlab 'cat ~/env/deepseek-token')
    python3 scripts/regenerate_cryptarithm_cot.py \
        --input data/cryptarithm_deduce_puzzles.jsonl \
        --output data/cot_cryptarithm_v2.jsonl \
        --max-concurrent 8
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

# ── Optimized System Prompt for Cryptarithm Deduce ──

CRYPTARITHM_DEDUCE_PROMPT = """You are an expert in symbolic pattern analysis. For transformation puzzles, follow this exact protocol:

PROTOCOL:
1. LIST every input-output pair from the examples
2. MAP each symbol position: determine what each input symbol becomes in the output
3. CHECK: does the same input symbol always produce the same output? If not, the mapping depends on position
4. BUILD a complete substitution table showing input→output for every symbol
5. VERIFY the table against ALL examples before proceeding
6. APPLY the table to the new input
7. OUTPUT the result

Work concisely. Show the mapping table. Put final answer inside \\boxed{}.

Example approach:
- Example 1: input "]|@<|" → output "\")\\\\" 
  Positions: [0]]→\", [1]|→), [2]@→\\\\, [3]<→\\\\, [4]|→)
- Build mapping: ]→\", |→), @→\\\\, <→\\\\
- Verify on Example 2: input "<)@|<" → ...
- Apply to target: ...

Be systematic. Never guess. Always verify."""


DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"


def load_api_key():
    """Load API key from environment."""
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        # Try reumanlab path
        key_paths = [
            Path("/home/reumanlab/env/deepseek-token"),
            Path.home() / "env/deepseek-token",
        ]
        for p in key_paths:
            if p.exists():
                key = p.read_text().strip()
                break
    return key


def extract_boxed(text: str) -> str:
    """Extract answer from \\boxed{...}."""
    patterns = [
        r'\\\\boxed\\{([^{}]*(?:\\{[^{}]*\\}[^{}]*)*)\\}',
        r'boxed\\{([^{}]*(?:\\{[^{}]*\\}[^{}]*)*)\\}',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            ans = matches[-1].strip()
            m = re.match(r'^\\\\text\\{(.*)\\}$', ans)
            if m:
                ans = m.group(1)
            return ans
    return ""


def normalize_answer(ans: str) -> str:
    """Normalize answer for comparison."""
    ans = ans.strip()
    if ans.startswith("\\boxed{") and ans.endswith("}"):
        ans = ans[7:-1]
    return ans.strip()


def answers_match(generated: str, ground_truth: str) -> bool:
    """Compare answers with tolerance."""
    g = normalize_answer(generated)
    t = normalize_answer(ground_truth)
    if g == t:
        return True
    try:
        gf = float(g)
        tf = float(t)
        return abs(gf - tf) < 0.05
    except (ValueError, TypeError):
        pass
    return g.lower() == t.lower()


async def generate_cot(puzzle: dict, api_key: str, semaphore: asyncio.Semaphore) -> dict:
    """Generate CoT trace using DeepSeek API with optimized prompt."""
    async with semaphore:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": CRYPTARITHM_DEDUCE_PROMPT},
                {"role": "user", "content": puzzle["prompt"]}
            ],
            "max_tokens": 3000,
            "temperature": 0.1,  # Very low temperature for deterministic analysis
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
                    "model": DEEPSEEK_MODEL,
                    "tokens_in": usage.get("prompt_tokens", 0),
                    "tokens_out": usage.get("completion_tokens", 0),
                    "success": True
                }
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {
                    "content": "",
                    "error": str(e),
                    "success": False
                }


def format_sharegpt(puzzle_prompt: str, cot_response: str, ground_truth: str, category: str) -> dict:
    """Format as ShareGPT with category-specific system prompt."""
    return {
        "messages": [
            {
                "role": "system",
                "content": CRYPTARITHM_DEDUCE_PROMPT
            },
            {
                "role": "user",
                "content": puzzle_prompt
            },
            {
                "role": "assistant",
                "content": cot_response
            }
        ]
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Regenerate CoT traces for cryptarithm_deduce puzzles"
    )
    parser.add_argument("--input", required=True, help="Input JSONL with puzzles")
    parser.add_argument("--output", required=True, help="Output JSONL with CoT traces")
    parser.add_argument("--max-concurrent", type=int, default=8, help="Max concurrent API calls")
    parser.add_argument("--batch-size", type=int, default=20, help="Puzzles per batch")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    args = parser.parse_args()

    api_key = load_api_key()
    if not api_key:
        print("ERROR: No DEEPSEEK_API_KEY found. Set the env var or ensure "
              "/home/reumanlab/env/deepseek-token exists.")
        sys.exit(1)

    print(f"API key: {api_key[:10]}...")
    print(f"Model: {DEEPSEEK_MODEL}")

    # Load puzzles
    puzzles = []
    with open(args.input) as f:
        for line in f:
            puzzles.append(json.loads(line))

    end = args.end or len(puzzles)
    puzzles = puzzles[args.start:end]
    print(f"Processing {len(puzzles)} puzzles (indices {args.start} to {end})")

    # Resume support
    done_indices = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                d = json.loads(line)
                done_indices.add(d.get("index", -1))
        print(f"Resuming: {len(done_indices)} already done")

    # Process
    semaphore = asyncio.Semaphore(args.max_concurrent)
    mode = "a" if args.resume else "w"
    output_file = open(args.output, mode)

    total_success = 0
    total_match = 0
    total_processed = 0
    t0 = time.time()

    for batch_start in range(0, len(puzzles), args.batch_size):
        batch_puzzles = []
        for i in range(batch_start, min(batch_start + args.batch_size, len(puzzles))):
            global_idx = args.start + i
            if global_idx in done_indices:
                continue
            batch_puzzles.append((global_idx, puzzles[i]))

        if not batch_puzzles:
            continue

        # Generate CoT for batch
        tasks = []
        for idx, p in batch_puzzles:
            tasks.append(generate_cot(p, api_key, semaphore))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (idx, puzzle), result in zip(batch_puzzles, results):
            total_processed += 1

            if isinstance(result, Exception):
                result = {"content": "", "success": False, "error": str(result)}

            if result["success"]:
                total_success += 1
                cot_text = result["content"]
                generated_answer = extract_boxed(cot_text)
                ground_truth = normalize_answer(puzzle["answer"])
                match = answers_match(generated_answer, ground_truth)

                if match:
                    total_match += 1

                # CRITICAL: Do NOT auto-fix wrong answers with ground truth
                # If the CoT is wrong, write it as-is (but mark it)
                # The downstream consolidation will decide whether to include it

                record = format_sharegpt(
                    puzzle["prompt"], cot_text, puzzle["answer"], puzzle["category"]
                )
                record["index"] = idx
                record["metadata"] = {
                    "engine": "deepseek",
                    "model": DEEPSEEK_MODEL,
                    "category": puzzle["category"],
                    "ground_truth": puzzle["answer"],
                    "generated_answer": generated_answer,
                    "answer_match": match,
                    "tokens_in": result.get("tokens_in", 0),
                    "tokens_out": result.get("tokens_out", 0),
                }
                output_file.write(json.dumps(record) + "\n")
            else:
                # Failed — skip, don't create fallback with auto-fixed answer
                total_match += 0  # Don't count failures

        output_file.flush()

        # Progress
        elapsed = time.time() - t0
        rate = total_processed / elapsed if elapsed > 0 else 0
        match_rate = total_match / max(total_success, 1) * 100
        print(f"  [{total_processed}/{len(puzzles)}] "
              f"success={total_success} match={total_match} ({match_rate:.1f}%) "
              f"rate={rate:.1f}/s elapsed={elapsed:.0f}s")

    output_file.close()
    elapsed = time.time() - t0

    print(f"\n=== REGENERATION DONE ===")
    print(f"Processed: {total_processed}")
    print(f"Success: {total_success} ({total_success/max(total_processed,1)*100:.1f}%)")
    print(f"Answer match: {total_match} ({total_match/max(total_success,1)*100:.1f}%)")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Output: {args.output}")

    # Comparison with original
    if total_success > 0:
        old_match_rate = 2.4  # Original CoT match rate
        new_match_rate = total_match / max(total_success, 1) * 100
        improvement = new_match_rate - old_match_rate
        print(f"\nImprovement: {old_match_rate:.1f}% → {new_match_rate:.1f}% "
              f"({improvement:+.1f} percentage points)")


if __name__ == "__main__":
    asyncio.run(main())
