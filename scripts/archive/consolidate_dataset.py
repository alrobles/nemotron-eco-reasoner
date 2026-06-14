#!/usr/bin/env python3
"""Consolidate all CoT traces + ecology data into a unified training dataset.

Combines:
1. DeepSeek-generated CoT traces for Kaggle puzzles
2. Ecology reasoning traces from ecoseek-litdump / HuggingFace
3. Outputs ShareGPT format JSONL ready for SFTTrainer

Usage:
    python3 consolidate_dataset.py \
        --cot-files data/cot_deepseek_batch*.jsonl \
        --ecology-file data/ecology_traces.jsonl \
        --output data/train_cot_unified.jsonl \
        --stats
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path


def load_cot_traces(file_patterns: list) -> list:
    """Load CoT traces from one or more JSONL files."""
    traces = []
    seen_indices = set()
    for pattern in file_patterns:
        for fpath in sorted(glob.glob(pattern)):
            with open(fpath) as f:
                for line in f:
                    d = json.loads(line)
                    idx = d.get("index", len(traces))
                    if idx not in seen_indices:
                        seen_indices.add(idx)
                        traces.append(d)
    return traces


def load_ecology_messages(file_path: str) -> list:
    """Load ecology traces in messages format."""
    traces = []
    if not os.path.exists(file_path):
        return traces
    with open(file_path) as f:
        for line in f:
            d = json.loads(line)
            if "messages" in d:
                traces.append(d)
    return traces


def convert_ecology_to_sharegpt(raw_traces: list) -> list:
    """Convert ecology traces from various formats to ShareGPT."""
    converted = []
    for d in raw_traces:
        if "messages" in d:
            msgs = d["messages"]
            # Ensure system prompt exists
            if msgs and msgs[0]["role"] != "system":
                msgs.insert(0, {
                    "role": "system",
                    "content": "You are an expert reasoning model. Solve problems step by step, showing all your work. Place your final answer inside \\boxed{}."
                })
            converted.append({"messages": msgs, "source": "ecology"})
        elif "context" in d and "reasoning" in d:
            # context/reasoning/code format
            user_content = d["context"]
            assistant_content = d["reasoning"]
            if d.get("code"):
                assistant_content += f"\n\n```python\n{d['code']}\n```"
            converted.append({
                "messages": [
                    {"role": "system", "content": "You are an expert ecological computing assistant. Solve problems step by step with clear reasoning and working code."},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ],
                "source": "ecology"
            })
    return converted


def main():
    parser = argparse.ArgumentParser(description="Consolidate training dataset")
    parser.add_argument("--cot-files", nargs="+", required=True, help="CoT trace JSONL files (glob patterns OK)")
    parser.add_argument("--ecology-file", default=None, help="Ecology traces JSONL")
    parser.add_argument("--output", required=True, help="Output unified JSONL")
    parser.add_argument("--stats", action="store_true", help="Print statistics")
    parser.add_argument("--min-assistant-length", type=int, default=50, help="Min assistant response length")
    args = parser.parse_args()

    # Load CoT traces
    cot_traces = load_cot_traces(args.cot_files)
    print(f"Loaded {len(cot_traces)} CoT traces")

    # Load ecology traces
    eco_traces = []
    if args.ecology_file:
        raw_eco = load_ecology_messages(args.ecology_file)
        eco_traces = convert_ecology_to_sharegpt(raw_eco)
        print(f"Loaded {len(eco_traces)} ecology traces")

    # Filter and write
    total = 0
    skipped = 0
    sources = Counter()
    match_stats = Counter()

    with open(args.output, "w") as f:
        # Write CoT traces
        for trace in cot_traces:
            msgs = trace.get("messages", [])
            if len(msgs) < 2:
                skipped += 1
                continue
            # Check assistant response length
            assistant = msgs[-1].get("content", "")
            if len(assistant) < args.min_assistant_length:
                skipped += 1
                continue

            record = {"messages": msgs}
            if "metadata" in trace:
                meta = trace["metadata"]
                record["metadata"] = {
                    "source": "kaggle_cot",
                    "engine": meta.get("engine", ""),
                    "answer_match": meta.get("answer_match", False),
                }
                match_stats["match" if meta.get("answer_match") else "no_match"] += 1
            sources["kaggle_cot"] += 1
            f.write(json.dumps(record) + "\n")
            total += 1

        # Write ecology traces
        for trace in eco_traces:
            msgs = trace.get("messages", [])
            if len(msgs) < 2:
                skipped += 1
                continue
            assistant = msgs[-1].get("content", "")
            if len(assistant) < args.min_assistant_length:
                skipped += 1
                continue

            record = {"messages": msgs, "metadata": {"source": "ecology"}}
            sources["ecology"] += 1
            f.write(json.dumps(record) + "\n")
            total += 1

    print(f"\n=== Dataset Statistics ===")
    print(f"Total examples: {total}")
    print(f"Skipped: {skipped}")
    print(f"Sources: {dict(sources)}")
    if match_stats:
        print(f"CoT match rate: {match_stats['match']}/{sum(match_stats.values())} "
              f"({match_stats['match']/max(sum(match_stats.values()),1)*100:.1f}%)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
