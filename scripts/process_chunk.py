#!/usr/bin/env python3
"""
process_chunk.py — Process one chunk of papers with llama.cpp server.
Each Slurm task runs this script with a chunk file as argument.

Usage (Slurm task):
    python3 process_chunk.py chunk_00000.jsonl --model /path/to/model.gguf --output chunk_00000_out.jsonl

Uses llama.cpp server HTTP API (same as Ollama compatible).
Expects llama-server binary available in PATH or via container.
"""

import argparse
import json
import os
import sys
import time
import urllib.request

# System prompt for CoT generation (same as cot_ollama_fleet.py)
SYSTEM = """You are an expert ecological modeler and scientific programmer. Given a paper's title and abstract, generate a Chain-of-Thought reasoning trace.

Output a JSON object with:
{
  "reasoning": "STEP 1 - Problem definition... STEP 2 - Data considerations... STEP 3 - Method... STEP 4 - Implementation... STEP 5 - Interpretation. Use STEP N notation.",
  "code": "Python/R code implementing the core method. 20-50 lines, self-contained, runnable."
}

The reasoning should simulate expert thinking, not just describe the paper."""


def call_llama(server: str, paper: dict, max_tokens: int = 1500) -> dict:
    """Call llama.cpp server API (OpenAI-compatible chat)."""
    user_msg = f"Paper: {paper.get('title','')}\n\nAbstract: {paper.get('abstract','')[:1500]}\n\nGenerate a Chain-of-Thought trace."
    
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    
    url = f"http://{server}/v1/chat/completions"
    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read())
    except Exception as e:
        return {"_error": str(e)}
    
    cot_text = result["choices"][0]["message"]["content"]
    
    # Try to parse JSON from response
    try:
        cot = json.loads(cot_text)
    except json.JSONDecodeError:
        # Try markdown block
        import re
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cot_text, re.DOTALL)
        if m:
            cot = json.loads(m.group(1))
        else:
            # Plain text fallback
            cot = {"reasoning": cot_text[:2000], "code": ""}
    
    reasoning = cot.get("reasoning", "")
    code = cot.get("code", "")
    
    # Build messages format
    return {
        "messages": [
            {"role": "system", "content": SYSTEM.strip()},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\nCODE:\n```python\n{code}\n```"},
        ],
        "source": paper.get("pmid", "unknown"),
        "method": "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("chunk_file", help="Input chunk JSONL")
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", "/models/deepseek-r1-14b-Q4_K_M.gguf"))
    parser.add_argument("--server", default=os.environ.get("LLAMA_SERVER", "localhost:8080"))
    parser.add_argument("--output", help="Output file (default: chunk_file_out.jsonl)")
    parser.add_argument("--delay", type=float, default=0.1)
    args = parser.parse_args()
    
    if not args.output:
        base = os.path.splitext(args.chunk_file)[0]
        args.output = f"{base}_out.jsonl"
    
    # Read papers
    papers = []
    with open(args.chunk_file) as f:
        for line in f:
            if line.strip():
                papers.append(json.loads(line))
    
    print(f"Chunk: {len(papers)} papers → {args.output}", flush=True)
    
    # Wait for server to be ready
    print(f"Waiting for server {args.server}...", flush=True)
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://{args.server}/health", timeout=2)
            print("Server ready!", flush=True)
            break
        except Exception:
            time.sleep(2)
    else:
        print("ERROR: Server not ready after 120s", file=sys.stderr)
        sys.exit(1)
    
    # Process papers
    success = 0
    failed = 0
    t_start = time.time()
    
    with open(args.output, "w") as f:
        for i, paper in enumerate(papers):
            t0 = time.time()
            result = call_llama(args.server, paper)
            elapsed = time.time() - t0
            
            if "_error" in result:
                failed += 1
                print(f"  [{i+1}/{len(papers)}] ERROR: {result['_error'][:80]}", flush=True)
            else:
                success += 1
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  [{i+1}/{len(papers)}] OK ({elapsed:.1f}s)", flush=True)
            
            if (i + 1) % 10 == 0:
                rate = (i + 1) / (time.time() - t_start) * 60
                print(f"  Progress: {i+1}/{len(papers)} ~{rate:.1f}/min", flush=True)
            
            if args.delay > 0:
                time.sleep(args.delay)
    
    elapsed = time.time() - t_start
    print(f"DONE: {success}/{len(papers)} in {elapsed:.0f}s (~{success/max(elapsed,1)*60:.1f}/min)", flush=True)


if __name__ == "__main__":
    main()
