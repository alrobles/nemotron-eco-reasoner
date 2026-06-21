#!/usr/bin/env python3
"""
cot_ollama_fleet.py — Generate CoT training data from ecological papers
using a fleet of Ollama instances serving deepseek-r1:14b on Q6000 GPUs.

Reads papers from a JSONL file, distributes them across N Ollama endpoints
in parallel (ThreadPoolExecutor), and saves CoT traces in Nemotron-compatible
messages format.

Usage (test):
    python3 scripts/cot_ollama_fleet.py --papers pubmed_sdm_papers.jsonl \
        --endpoints endpoints.txt --limit 20 --output cot_test.jsonl

Usage (full):
    python3 scripts/cot_ollama_fleet.py --papers pubmed_sdm_papers.jsonl \
        --endpoints endpoints.txt --output cot_eco_4k.jsonl --delay 0.5

Endpoints file format (one per line):
    r22r20n01:35765
    r22r25n01:49163
    ...

Requires: Python 3.9+ (stdlib only — urllib, json, concurrent.futures)
"""

import argparse
import json
import os
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# ── System prompt for CoT generation ────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert ecological modeler and scientific Python/R programmer.
Given a paper's title and abstract, generate a Chain-of-Thought trace for fine-tuning an AI model on ecological reasoning.

Output ONLY a JSON object with these exact keys:
{
  "paper_pmid": "the PMID",
  "paper_title": "full title",
  "method": "one short method name (e.g. MaxEnt, BRT, Occupancy, N-mixture, GNN, etc.)",
  "context": "Concise technical background: what method, what ecological problem, key innovation. 3-5 sentences.",
  "reasoning": "STEP-by-STEP reasoning through the ecological problem: STEP 1 - Problem definition... STEP 2 - Data considerations... STEP 3 - Method choice and justification... STEP 4 - Implementation details... STEP 5 - Ecological interpretation. Use STEP N notation.",
  "code": "A self-contained Python or R implementation of the core method. Include imports, data simulation (reproducible with seed), model fitting, and visualization. ~30-60 lines. Must be runnable."
}

The reasoning should simulate how an expert ecologist thinks through a problem — not just describe the paper. The code should implement the core statistical/ML method, not just a wrapper."""


def load_endpoints(path: str) -> List[str]:
    """Load Ollama endpoints from a text file (one per line: host:port)."""
    eps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                eps.append(line)
    return eps


def load_papers(path: str, limit: Optional[int] = None, shuffle: bool = False) -> List[Dict[str, object]]:
    """Load papers from a JSONL file."""
    papers = []
    with open(path) as f:
        for line in f:
            papers.append(json.loads(line))
    if shuffle:
        random.shuffle(papers)
    if limit:
        papers = papers[:limit]
    return papers


def call_ollama(endpoint: str, model: str, system: str, user_msg: str,
                max_tokens: int = 2048, temperature: float = 0.3,
                timeout: int = 180) -> Optional[dict]:
    """Call an Ollama endpoint (non-streaming) and return the parsed JSON response.
    
    Returns a dict with:
      - paper_pmid, paper_title, method, context, reasoning, code (from CoT)
      - _reasoning: the raw R1 chain-of-thought
      - _prompt_tokens, _completion_tokens, _endpoint
      - _error: if something went wrong
    """
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }).encode()

    url = f"http://{endpoint}/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        return {"_error": f"HTTP {e.code}: {body}"}
    except Exception as e:
        return {"_error": str(e)}

    choice = result.get("choices", [{}])[0]
    msg = choice.get("message", {})
    content = msg.get("content", "")
    reasoning_raw = msg.get("reasoning", "")
    usage = result.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    # Debug: log raw response size
    print(f"    [DEBUG] content={len(content)}chars reasoning={len(reasoning_raw)}chars", flush=True)

    # Parse the JSON-formatted CoT from the content, trying multiple strategies
    cot = _extract_json(content)
    if cot is None:
        # Try extracting from reasoning if content had no JSON
        cot = _extract_json(reasoning_raw)
    
    if cot is None:
        # Fallback: if model didn't output JSON, use raw content + reasoning as CoT
        # The reasoning field contains the actual chain-of-thought
        if reasoning_raw.strip() or content.strip():
            cot = {
                "paper_pmid": "",
                "paper_title": "",
                "method": "",
                "context": "",
                "reasoning": reasoning_raw[:2000] if reasoning_raw else content[:2000],
                "code": "",
                "_fallback": True,
            }
        else:
            return {"_error": "JSON parse failed: no valid JSON found",
                    "_content": content[:500], "_reasoning": reasoning_raw[:500]}

    # Flatten any nested dict/list fields into strings (model often nests)
    for field in ("method", "context", "reasoning", "code"):
        if field in cot:
            val = cot[field]
            if isinstance(val, dict):
                # Flatten dict fields into readable text
                parts = []
                for k, v in val.items():
                    if isinstance(v, str):
                        parts.append(f"{k}: {v}")
                    elif isinstance(v, list):
                        parts.append(f"{k}: {'; '.join(str(x) for x in v)}")
                cot[field] = " | ".join(parts) if parts else str(val)
            elif isinstance(val, list):
                # Flatten list of step objects
                parts = []
                for item in val:
                    if isinstance(item, dict):
                        parts.append(str(item))
                    else:
                        parts.append(str(item))
                cot[field] = "\n".join(parts) if parts else str(val)
            elif not isinstance(val, str):
                cot[field] = str(val)

    cot["_reasoning"] = reasoning_raw
    cot["_prompt_tokens"] = prompt_tokens
    cot["_completion_tokens"] = completion_tokens
    cot["_endpoint"] = endpoint
    return cot


def _extract_json(text: str) -> Optional[dict]:
    """Try multiple strategies to extract a JSON object from model output."""
    if not text:
        return None
    
    text = text.strip()
    
    # Strategy 1: Direct parse (model output only JSON)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: JSON in markdown code block
    import re
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Find first { to last }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    
    return None


def cot_to_messages(cot: dict, paper: dict) -> dict:
    """Convert a CoT trace to Nemotron-compatible messages format.
    
    Uses the R1 reasoning (chain-of-thought) as the core training signal,
    wrapping it in <think>...</think> tags for the Nemotron training format.
    The structured JSON output follows after the thinking block.
    """
    title = paper.get("title", "Unknown")
    abstract = paper.get("abstract", "")
    pmid = paper.get("pmid", "unknown")

    user_msg = (
        f"Analyze the following ecological research paper and explain the "
        f"reasoning behind its methodology and findings. Provide a step-by-step "
        f"chain-of-thought.\n\n"
        f"Title: {title}\n\n"
        f"Abstract: {abstract[:1500]}"
    )

    # Build assistant response: reasoning CoT + structured output
    reasoning_raw = cot.get("_reasoning", "")
    
    parts = []
    if reasoning_raw:
        parts.append(f"<think>\n{reasoning_raw.strip()}\n</think>")
    
    # Add the structured output fields
    structured = []
    if "context" in cot and isinstance(cot["context"], str):
        structured.append(f"CONTEXT: {cot['context']}")
    if "method" in cot and isinstance(cot["method"], str):
        structured.append(f"METHOD: {cot['method']}")
    if "reasoning" in cot and isinstance(cot["reasoning"], str):
        structured.append(cot["reasoning"])
    if "code" in cot:
        code = cot["code"]
        if isinstance(code, dict):
            # Model returned code as {"language": "python", "code": "..."}
            code_str = code.get("code", "") or code.get("content", "") or json.dumps(code)
        else:
            code_str = str(code)
        if code_str.strip():
            structured.append(f"\nCODE:\n```python\n{code_str}\n```")
    
    parts.append("\n".join(structured))

    assistant = "\n\n".join(parts)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant},
        ],
        "source": f"pubmed:{pmid}",
        "method": cot.get("method", ""),
        "_endpoint": cot.get("_endpoint", ""),
        "_prompt_tokens": cot.get("_prompt_tokens", 0),
        "_completion_tokens": cot.get("_completion_tokens", 0),
    }


def process_paper(args: Tuple[int, dict, str, str, int, float, int]) -> Tuple[int, Optional[dict], Optional[str]]:
    """Process one paper: call Ollama and convert to messages format.
    Returns: (index, messages_dict or None, error_string or None)"""
    idx, paper, endpoint, model, max_tokens, temperature, timeout = args
    title = paper.get("title", "Unknown")[:100]
    abstract = paper.get("abstract", "")[:1500]
    pmid = paper.get("pmid", "?")

    user_msg = (
        f"Paper PMID {pmid}: {title}\n\n"
        f"Abstract: {abstract}\n\n"
        f"Generate a Chain-of-Thought trace for this ecological paper."
    )

    t0 = time.time()
    cot = call_ollama(endpoint, model, SYSTEM_PROMPT, user_msg,
                      max_tokens=max_tokens, temperature=temperature,
                      timeout=timeout)
    elapsed = time.time() - t0

    if cot is None:
        return (idx, None, f"null response")

    if "_error" in cot:
        return (idx, None, f"{endpoint}: {cot['_error']}")

    msg = cot_to_messages(cot, paper)
    print(f"  [{idx}] PMID:{pmid} on {endpoint} ({elapsed:.1f}s) "
          f"tok={cot.get('_prompt_tokens',0)}+{cot.get('_completion_tokens',0)}",
          flush=True)
    return (idx, msg, None)


def main():
    parser = argparse.ArgumentParser(description="Generate CoT traces via Ollama fleet")
    parser.add_argument("--papers", required=True, help="JSONL file with papers")
    parser.add_argument("--endpoints", required=True, help="File with endpoints (host:port per line)")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--limit", type=int, default=0, help="Max papers to process (0=all)")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between submissions per thread (seconds)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.3, help="Generation temperature")
    parser.add_argument("--timeout", type=int, default=180, help="Timeout per generation (seconds)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle papers before processing")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N papers (resume)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    endpoints = load_endpoints(args.endpoints)
    if not endpoints:
        sys.exit("ERROR: No endpoints found")

    papers = load_papers(args.papers, limit=args.limit or None, shuffle=args.shuffle)
    if args.skip > 0:
        papers = papers[args.skip:]
        print(f"  Skipping first {args.skip} papers, {len(papers)} remaining")
    random.seed(args.seed)

    model = "deepseek-r1:14b"
    total = len(papers)
    n_workers = len(endpoints)

    print(f"{'='*60}")
    print(f"OLLAMA FLEET CoT GENERATION")
    print(f"  Papers:     {total}")
    print(f"  Endpoints:  {n_workers} ({', '.join(endpoints)})")
    print(f"  Model:      {model}")
    print(f"  Output:     {args.output}")
    print(f"  Timeout:    {args.timeout}s")
    print(f"{'='*60}")

    # Pre-warm all endpoints (load model into GPU memory)
    print("Pre-warming endpoints...")
    import urllib.request as _ur
    for ep in endpoints:
        try:
            _ur.urlopen(_ur.Request(
                f"http://{ep}/api/generate",
                data=json.dumps({"model": model, "prompt": "test", "stream": False,
                                "options": {"num_predict": 1}}).encode(),
                headers={"Content-Type": "application/json"}), timeout=120)
            print(f"  {ep} ✓")
        except Exception as e:
            print(f"  {ep} ✗ ({e})")
    print(f"Warm-up complete.\n")

    # Prepare tasks: round-robin assignment to endpoints
    tasks = []
    for i, paper in enumerate(papers):
        ep = endpoints[i % n_workers]
        tasks.append((i, paper, ep, model, args.max_tokens, args.temperature, args.timeout))

    failures = []
    t_start = time.time()

    # Dynamic delay per endpoint to avoid overwhelming them
    ep_last = {ep: 0.0 for ep in endpoints}

    # Open output file for incremental writing
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_f = open(args.output, "a")  # append mode for resume safety
    results_count = 0
    
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for task in tasks:
            ep = task[2]
            now = time.time()
            wait = max(0, args.delay - (now - ep_last.get(ep, 0)))
            if wait > 0:
                time.sleep(wait)
            fut = pool.submit(process_paper, task)
            futures[fut] = task
            ep_last[ep] = time.time()

        for i, fut in enumerate(as_completed(futures)):
            task = futures[fut]
            idx = task[0]
            try:
                idx2, msg, err = fut.result()
                if msg:
                    # Write immediately to disk (incremental) + force NFS sync
                    rec = {k: v for k, v in msg.items() if not k.startswith("_")}
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                    if results_count % 5 == 0:
                        os.fsync(out_f.fileno())  # force NFS sync every 5 papers
                    results_count += 1
                else:
                    failures.append((idx, err or "unknown"))
            except Exception as e:
                failures.append((idx, str(e)))

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
                print(f"--- [{i+1}/{total}] {elapsed:.0f}s elapsed, ~{rate:.1f}/min "
                      f"({results_count} ok, {len(failures)} fail)", flush=True)

    out_f.flush()
    os.fsync(out_f.fileno())
    out_f.close()
    elapsed = time.time() - t_start

    # Summary (note: we don't have full token counts since we write incrementally)
    total_tokens_in = 0  # lost with incremental mode
    total_tokens_out = 0  # lost with incremental mode

    print(f"\n{'='*60}")
    print(f"DONE in {elapsed/60:.1f}min")
    print(f"  Success:     {results_count}/{total} ({results_count/max(total,1)*100:.1f}%)")
    print(f"  Failed:      {len(failures)}")
    print(f"  Output:      {args.output}")

    if failures:
        fail_path = args.output.replace(".jsonl", "_failures.jsonl")
        with open(fail_path, "w") as f:
            for idx, err in failures:
                f.write(json.dumps({"index": idx, "error": err}) + "\n")
        print(f"  Failures:    {fail_path}")

    print(f"  Rate:        ~{results_count/max(elapsed,1)*60:.1f} papers/min")


if __name__ == "__main__":
    main()
