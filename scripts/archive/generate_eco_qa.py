#!/usr/bin/env python3
"""
Generate ecological reasoning Q&A pairs from PubMed abstracts using DeepSeek.

For each abstract, DeepSeek generates:
  1. A question that requires ecological reasoning
  2. A step-by-step chain-of-thought answer based on the paper's findings

Output format: {prompt, answer, source} — compatible with train_m1_container.py

Usage:
    python3 generate_eco_qa.py --input eco_abstracts.json --output eco_qa.jsonl --limit 10 --dry-run

Cost estimate: ~500 input tokens + ~400 output tokens per example
  At deepseek-v4-pro pricing (~$0.55/M input, ~$2.19/M output):
  ~$0.0012/example → ~$1.20 per 1000 examples
"""
import argparse
import json
import os
import sys
import time

# DeepSeek API
DEEPSEEK_TOKEN_PATH = "/home/reumanlab/env/deepseek-token"
API_URL = "https://api.deepseek.com/v1/chat/completions"


def get_api_key() -> str:
    with open(DEEPSEEK_TOKEN_PATH) as f:
        return f.read().strip()


def load_abstracts(path: str, limit: int = None) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    return data


def generate_qa(abstract_data: dict, api_key: str, dry_run: bool = False) -> dict | None:
    """Generate a Q&A pair from a PubMed abstract."""
    title = abstract_data.get("title", "Unknown")
    abstract = abstract_data.get("abstract", "")
    pmid = abstract_data.get("pmid", "unknown")

    # Truncate abstract to ~1500 chars to control token usage
    abstract = abstract[:1500]

    system_prompt = (
        "You are an expert ecology professor creating training data for an AI model. "
        "Given a scientific abstract, generate:\n"
        "1. A specific ecological reasoning QUESTION that the paper addresses\n"
        "2. A step-by-step CHAIN-OF-THOUGHT ANSWER that explains the reasoning "
        "using evidence from the paper\n\n"
        "RULES:\n"
        "- The question must require multi-step ecological reasoning\n"
        "- The answer must have 3-5 reasoning steps with evidence\n"
        "- Use concrete data from the abstract (numbers, species, mechanisms)\n"
        "- Output ONLY valid JSON: {\"question\": \"...\", \"answer\": \"...\"}\n"
        "- NO markdown, NO extra text outside the JSON"
    )

    user_message = (
        f"Title: {title}\n\n"
        f"Abstract: {abstract}\n\n"
        f"Generate a question and step-by-step reasoning answer."
    )

    if dry_run:
        return {
            "prompt": f"[DRY RUN] Question about: {title[:80]}...",
            "answer": f"[DRY RUN] Answer for PMID {pmid}",
            "source": f"pubmed:{pmid}",
        }

    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": "deepseek-v4-pro",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,
        "max_tokens": 1500,
    }).encode()

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.read().decode()[:200]}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return None

    content = result["choices"][0]["message"]["content"]
    try:
        qa = json.loads(content)
    except json.JSONDecodeError:
        print(f"  JSON parse error: {content[:200]}", file=sys.stderr)
        return None

    usage = result.get("usage", {})
    return {
        "prompt": qa["question"],
        "answer": qa["answer"],
        "source": f"pubmed:{pmid}",
        "title": title,
        "_tokens_in": usage.get("prompt_tokens", 0),
        "_tokens_out": usage.get("completion_tokens", 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON file with abstracts")
    parser.add_argument("--output", default="eco_qa.jsonl", help="Output JSONL file")
    parser.add_argument("--limit", type=int, default=10, help="Max examples to generate")
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls")
    args = parser.parse_args()

    api_key = get_api_key()
    abstracts = load_abstracts(args.input, args.limit)
    print(f"Loaded {len(abstracts)} abstracts from {args.input}")

    results = []
    total_tokens_in = 0
    total_tokens_out = 0

    for i, abs_data in enumerate(abstracts):
        pmid = abs_data.get("pmid", "?")
        title = abs_data.get("title", "?")[:80]
        print(f"[{i+1}/{len(abstracts)}] PMID:{pmid} — {title}...", end=" ", flush=True)

        qa = generate_qa(abs_data, api_key, dry_run=args.dry_run)
        if qa:
            results.append(qa)
            t_in = qa.pop("_tokens_in", 0)
            t_out = qa.pop("_tokens_out", 0)
            total_tokens_in += t_in
            total_tokens_out += t_out
            print(f"OK ({t_in}+{t_out} tok)")
        else:
            print("FAILED")

        if not args.dry_run and i < len(abstracts) - 1:
            time.sleep(args.delay)

    # Write output
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Cost estimate (deepseek-v4-pro pricing ~Mar 2026)
    cost_in = total_tokens_in * 0.55 / 1_000_000
    cost_out = total_tokens_out * 2.19 / 1_000_000
    total_cost = cost_in + cost_out

    print(f"\n{'='*50}")
    print(f"Generated: {len(results)}/{len(abstracts)} Q&A pairs")
    if not args.dry_run:
        print(f"Tokens: {total_tokens_in:,} in + {total_tokens_out:,} out = {total_tokens_in + total_tokens_out:,}")
        print(f"Cost: ¥{total_cost:.4f} (~${total_cost:.4f})")
        print(f"Per example: ¥{total_cost/len(results):.4f}" if results else "")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
