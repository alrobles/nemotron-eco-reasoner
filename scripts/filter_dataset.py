#!/usr/bin/env python3
"""
filter_dataset.py — Filter and prepare ecoreasoner-cot-20k for training.

Downloads from HuggingFace, filters for quality, normalizes format, and
produces a training-ready JSONL with stratification metadata.

Usage:
    python3 scripts/filter_dataset.py --output data/ecoreasoner_train.jsonl
    python3 scripts/filter_dataset.py --input /path/to/local.jsonl --output data/ecoreasoner_train.jsonl
"""

import argparse
import json
import os
import re
import sys
from collections import Counter


def download_dataset(url: str, output: str) -> str:
    """Download dataset from HuggingFace."""
    import urllib.request
    print(f"Downloading from {url}...", flush=True)
    urllib.request.urlretrieve(url, output)
    print(f"  Saved to {output}", flush=True)
    return output


def normalize_method(method: str) -> str:
    """Normalize method names for stratification."""
    if not method:
        return "other"
    m = method.strip().lower()
    # Consolidate common variants
    if "maxent" in m:
        return "maxent"
    if "brt" in m or "boosted regression" in m:
        return "brt"
    if "glm" in m or "generalized linear" in m:
        return "glm"
    if "gam" in m or "generalized additive" in m:
        return "gam"
    if "random forest" in m:
        return "random_forest"
    if "occupancy" in m:
        return "occupancy"
    if "logistic regression" in m:
        return "logistic_regression"
    if "hmm" in m or "hidden markov" in m:
        return "hmm"
    if "n-mixture" in m or "n_mixture" in m:
        return "n_mixture"
    if "pca" in m or "principal component" in m:
        return "pca"
    if "edna" in m or "metabarcod" in m:
        return "edna"
    if "network" in m:
        return "network_analysis"
    if "linear regression" in m:
        return "linear_regression"
    if "bayesian" in m:
        return "bayesian"
    if "phylo" in m:
        return "phylogenetics"
    if "gnn" in m or "graph neural" in m:
        return "gnn"
    return "other"


def filter_and_normalize(input_path: str, output_path: str,
                         min_think_len: int = 200,
                         min_assistant_len: int = 500,
                         require_think: bool = True) -> dict:
    """Filter dataset for quality and normalize format.

    Returns stats dict.
    """
    stats = Counter()
    records = []

    with open(input_path) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["json_error"] += 1
                continue

            stats["total"] += 1
            msgs = obj.get("messages", [])

            # Must have system + user + assistant
            if len(msgs) < 3:
                stats["too_few_messages"] += 1
                continue

            asst = None
            for m in msgs:
                if m.get("role") == "assistant":
                    asst = m
                    break

            if not asst:
                stats["no_assistant"] += 1
                continue

            content = asst.get("content", "")

            # Check minimum length
            if len(content) < min_assistant_len:
                stats["too_short"] += 1
                continue

            # Check for <think> tags
            has_think = "<think>" in content and "</think>" in content
            if require_think and not has_think:
                stats["no_think_tags"] += 1
                continue

            # Check think content length
            if has_think:
                think_start = content.find("<think>") + 7
                think_end = content.find("</think>")
                think_content = content[think_start:think_end].strip()
                if len(think_content) < min_think_len:
                    stats["think_too_short"] += 1
                    continue

            # Normalize method for stratification
            method = normalize_method(obj.get("method", ""))

            # Build output record
            record = {
                "messages": msgs,
                "source": obj.get("source", "unknown"),
                "method": method,
                "method_raw": obj.get("method", ""),
            }
            records.append(record)
            stats["accepted"] += 1

    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats["output_count"] = len(records)

    # Method distribution
    method_counts = Counter(r["method"] for r in records)
    stats["method_distribution"] = dict(method_counts.most_common(20))

    return stats


def main():
    parser = argparse.ArgumentParser(description="Filter ecoreasoner dataset for training")
    parser.add_argument("--input", help="Local JSONL input (default: download from HF)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--hf-dataset", default="alrobles/ecoreasoner-cot-20k",
                        help="HuggingFace dataset to download")
    parser.add_argument("--hf-file", default="ecoreasoner_cot_20k.jsonl",
                        help="File within the HF dataset")
    parser.add_argument("--min-think-len", type=int, default=200,
                        help="Minimum <think> content length")
    parser.add_argument("--min-assistant-len", type=int, default=500,
                        help="Minimum assistant message length")
    parser.add_argument("--no-require-think", action="store_true",
                        help="Don't require <think> tags")
    args = parser.parse_args()

    input_path = args.input
    if not input_path:
        url = f"https://huggingface.co/datasets/{args.hf_dataset}/resolve/main/{args.hf_file}"
        input_path = f"/tmp/{args.hf_file}"
        if not os.path.exists(input_path):
            download_dataset(url, input_path)
        else:
            print(f"Using cached {input_path}", flush=True)

    print(f"Filtering {input_path} → {args.output}", flush=True)
    stats = filter_and_normalize(
        input_path, args.output,
        min_think_len=args.min_think_len,
        min_assistant_len=args.min_assistant_len,
        require_think=not args.no_require_think,
    )

    print(f"\n{'='*60}")
    print(f"FILTER RESULTS")
    print(f"  Total input:     {stats.get('total', 0)}")
    print(f"  Accepted:        {stats.get('accepted', 0)} ({stats.get('accepted',0)/max(stats.get('total',1),1)*100:.1f}%)")
    print(f"  No <think> tags: {stats.get('no_think_tags', 0)}")
    print(f"  Too short:       {stats.get('too_short', 0)}")
    print(f"  Think too short: {stats.get('think_too_short', 0)}")
    print(f"  Output:          {args.output} ({stats.get('output_count', 0)} records)")
    print(f"\nMethod distribution:")
    for method, count in sorted(stats.get("method_distribution", {}).items(),
                                key=lambda x: -x[1]):
        print(f"  {method:25s} {count:5d}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
