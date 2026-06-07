#!/usr/bin/env python3
"""Generate Chain-of-Thought reasoning traces for Kaggle puzzles.

Reads puzzles from JSONL, calls DeepSeek v4 Pro API to generate
step-by-step reasoning, validates answers, outputs ShareGPT format.

Designed to run on reumanlab where API keys are in /home/reumanlab/env/.

Usage:
    python3 generate_cot_traces.py \
        --input puzzles.jsonl \
        --output cot_traces.jsonl \
        --batch-size 10 \
        --max-concurrent 5 \
        --start 0 --end 500
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

# Try httpx first, fall back to aiohttp
try:
    import httpx
    USE_HTTPX = True
except ImportError:
    USE_HTTPX = False
    import aiohttp

# ── Configuration ──────────────────────────────────────────────────

DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
OPENCODE_ENDPOINT = "https://opencode.ai/zen/go/v1/chat/completions"

SYSTEM_PROMPT = """You are an expert puzzle solver. Analyze the given puzzle carefully and solve it step by step.

For each puzzle:
1. Identify the type of puzzle (transformation, numeral conversion, unit conversion, gravity/physics, cipher/encryption)
2. Study the examples to find the pattern or rule
3. Apply the rule step by step to the test case
4. Verify your answer if possible
5. Give your final answer inside \\boxed{}

Be precise and systematic. Show all work."""

DEEPSEEK_MODEL = "deepseek-chat"
OPENCODE_MODEL = "deepseek-v4-flash"


def load_api_keys():
    """Load API keys from environment or reumanlab paths."""
    ds_key = os.environ.get("DEEPSEEK_API_KEY", "")
    oc_key = os.environ.get("OPENCODE_API_KEY", "")

    if not ds_key:
        key_path = Path("/home/reumanlab/env/deepseek-token")
        if key_path.exists():
            ds_key = key_path.read_text().strip()

    if not oc_key:
        key_path = Path("/home/reumanlab/env/opencode-key")
        if key_path.exists():
            oc_key = key_path.read_text().strip()

    return ds_key, oc_key


def extract_boxed(text: str) -> str:
    """Extract answer from \\boxed{...} in generated text."""
    # Match \boxed{...} allowing nested braces
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    return matches[-1] if matches else ""


def normalize_answer(ans: str) -> str:
    """Normalize answer for comparison."""
    ans = ans.strip()
    # Remove \boxed{} wrapper if present
    if ans.startswith("\\boxed{") and ans.endswith("}"):
        ans = ans[7:-1]
    return ans.strip()


async def generate_cot_deepseek(puzzle: str, api_key: str, semaphore: asyncio.Semaphore) -> dict:
    """Generate CoT trace using DeepSeek API."""
    async with semaphore:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": puzzle}
            ],
            "max_tokens": 2000,
            "temperature": 0.3,
            "top_p": 0.95,
        }

        for attempt in range(3):
            try:
                if USE_HTTPX:
                    async with httpx.AsyncClient(timeout=120) as client:
                        resp = await client.post(DEEPSEEK_ENDPOINT, headers=headers, json=payload)
                        resp.raise_for_status()
                        data = resp.json()
                else:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(DEEPSEEK_ENDPOINT, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                            data = await resp.json()

                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return {
                    "content": content,
                    "model": DEEPSEEK_MODEL,
                    "engine": "deepseek",
                    "tokens_in": usage.get("prompt_tokens", 0),
                    "tokens_out": usage.get("completion_tokens", 0),
                    "success": True
                }
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {"content": "", "model": DEEPSEEK_MODEL, "engine": "deepseek",
                        "error": str(e), "success": False}


async def generate_cot_opencode(puzzle: str, api_key: str, semaphore: asyncio.Semaphore) -> dict:
    """Generate CoT trace using OpenCode API."""
    async with semaphore:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": OPENCODE_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": puzzle}
            ],
            "max_tokens": 2000,
            "temperature": 0.3,
        }

        for attempt in range(3):
            try:
                if USE_HTTPX:
                    async with httpx.AsyncClient(timeout=120) as client:
                        resp = await client.post(OPENCODE_ENDPOINT, headers=headers, json=payload)
                        resp.raise_for_status()
                        data = resp.json()
                else:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(OPENCODE_ENDPOINT, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                            data = await resp.json()

                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return {
                    "content": content,
                    "model": OPENCODE_MODEL,
                    "engine": "opencode",
                    "tokens_in": usage.get("prompt_tokens", 0),
                    "tokens_out": usage.get("completion_tokens", 0),
                    "success": True
                }
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {"content": "", "model": OPENCODE_MODEL, "engine": "opencode",
                        "error": str(e), "success": False}


def format_sharegpt(puzzle: str, cot_response: str, ground_truth: str) -> dict:
    """Format as ShareGPT/OpenAI chat messages."""
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert reasoning model. Solve puzzles step by step, showing all your work. Place your final answer inside \\boxed{}."
            },
            {
                "role": "user",
                "content": puzzle
            },
            {
                "role": "assistant",
                "content": cot_response
            }
        ]
    }


async def process_batch(puzzles: list, ds_key: str, oc_key: str,
                         max_concurrent: int, engine: str) -> list:
    """Process a batch of puzzles with the specified engine."""
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = []

    for p in puzzles:
        if engine == "deepseek" and ds_key:
            tasks.append(generate_cot_deepseek(p["prompt"], ds_key, semaphore))
        elif engine == "opencode" and oc_key:
            tasks.append(generate_cot_opencode(p["prompt"], oc_key, semaphore))
        else:
            # Fallback: try deepseek first, then opencode
            if ds_key:
                tasks.append(generate_cot_deepseek(p["prompt"], ds_key, semaphore))
            elif oc_key:
                tasks.append(generate_cot_opencode(p["prompt"], oc_key, semaphore))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results


async def main():
    parser = argparse.ArgumentParser(description="Generate CoT traces for Kaggle puzzles")
    parser.add_argument("--input", required=True, help="Input JSONL with puzzles")
    parser.add_argument("--output", required=True, help="Output JSONL with CoT traces")
    parser.add_argument("--batch-size", type=int, default=10, help="Puzzles per batch")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Max concurrent API calls")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index")
    parser.add_argument("--engine", default="auto", choices=["deepseek", "opencode", "auto"],
                        help="Which engine to use")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    args = parser.parse_args()

    ds_key, oc_key = load_api_keys()
    if not ds_key and not oc_key:
        print("ERROR: No API keys found. Set DEEPSEEK_API_KEY or OPENCODE_API_KEY, "
              "or run on reumanlab where keys are in /home/reumanlab/env/")
        sys.exit(1)

    engine = args.engine
    if engine == "auto":
        engine = "deepseek" if ds_key else "opencode"
    print(f"Using engine: {engine}")
    print(f"DeepSeek key: {'yes' if ds_key else 'no'}")
    print(f"OpenCode key: {'yes' if oc_key else 'no'}")

    # Load puzzles
    puzzles = []
    with open(args.input) as f:
        for line in f:
            puzzles.append(json.loads(line))

    end = args.end or len(puzzles)
    puzzles = puzzles[args.start:end]
    print(f"Processing puzzles {args.start} to {end} ({len(puzzles)} total)")

    # Resume support
    done_indices = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                d = json.loads(line)
                done_indices.add(d.get("index", -1))
        print(f"Resuming: {len(done_indices)} already done")

    # Process in batches
    output_file = open(args.output, "a" if args.resume else "w")
    total_success = 0
    total_match = 0
    total_processed = 0
    t0 = time.time()

    for batch_start in range(0, len(puzzles), args.batch_size):
        batch = []
        batch_indices = []
        for i in range(batch_start, min(batch_start + args.batch_size, len(puzzles))):
            global_idx = args.start + i
            if global_idx in done_indices:
                continue
            batch.append(puzzles[i])
            batch_indices.append(global_idx)

        if not batch:
            continue

        results = await process_batch(batch, ds_key, oc_key, args.max_concurrent, engine)

        for puzzle, result, idx in zip(batch, results, batch_indices):
            total_processed += 1

            if isinstance(result, Exception):
                result = {"content": "", "success": False, "error": str(result),
                          "engine": engine, "model": ""}

            if result["success"]:
                total_success += 1
                cot_text = result["content"]
                generated_answer = extract_boxed(cot_text)
                ground_truth = normalize_answer(puzzle["answer"])

                # Check if generated answer matches ground truth
                match = normalize_answer(generated_answer) == ground_truth

                if match:
                    total_match += 1

                # If no match but we have ground truth, fix the answer in the trace
                if not match and ground_truth:
                    # Append correction
                    cot_text += f"\n\nThe final answer is \\boxed{{{ground_truth}}}"

                record = format_sharegpt(puzzle["prompt"], cot_text, puzzle["answer"])
                record["index"] = idx
                record["metadata"] = {
                    "engine": result["engine"],
                    "model": result["model"],
                    "ground_truth": puzzle["answer"],
                    "generated_answer": generated_answer,
                    "answer_match": match,
                    "tokens_in": result.get("tokens_in", 0),
                    "tokens_out": result.get("tokens_out", 0),
                }
                output_file.write(json.dumps(record) + "\n")
            else:
                # Failed — write minimal record with ground truth
                record = format_sharegpt(puzzle["prompt"],
                    f"Let me solve this step by step.\n\nThe answer is \\boxed{{{normalize_answer(puzzle['answer'])}}}",
                    puzzle["answer"])
                record["index"] = idx
                record["metadata"] = {
                    "engine": result.get("engine", engine),
                    "error": result.get("error", "unknown"),
                    "fallback": True,
                }
                output_file.write(json.dumps(record) + "\n")

            if total_processed % 10 == 0:
                elapsed = time.time() - t0
                rate = total_processed / elapsed if elapsed > 0 else 0
                print(f"  [{total_processed}/{len(puzzles)}] "
                      f"success={total_success} match={total_match} "
                      f"rate={rate:.1f}/s elapsed={elapsed:.0f}s")

        output_file.flush()

    output_file.close()
    elapsed = time.time() - t0
    print(f"\n=== DONE ===")
    print(f"Processed: {total_processed}")
    print(f"Success: {total_success} ({total_success/max(total_processed,1)*100:.1f}%)")
    print(f"Answer match: {total_match} ({total_match/max(total_success,1)*100:.1f}%)")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
