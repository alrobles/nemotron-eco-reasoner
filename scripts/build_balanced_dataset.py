#!/usr/bin/env python3
"""
build_balanced_dataset.py — Build a balanced training dataset for Nemotron
ecological + biological + scientific reasoning.

Combines multiple sources:
  1. ecoreasoner-cot-20k     — ecological reasoning CoT (filtered for <think>)
  2. ecocoder-scientific-reasoning — structured ecology methods
  3. ecocoder-cot            — paper-based ecology CoT
  4. Brainquiver/reasoning-biology — broad biology reasoning
  5. camel-ai/biology         — biology fundamentals (Q&A → CoT format)
  6. nemotron-reasoning-v3    — physics/science reasoning
  7. nemotron-eco-reasoner-v14 — general reasoning puzzles

Balancing strategy:
  - Cap overrepresented ecological methods (MaxEnt, unspecified)
  - Stratified sampling within ecology
  - Mix: ~40% ecology, ~25% biology, ~20% general reasoning, ~15% physics/science
  - All normalized to chat format with <think> reasoning traces

Usage:
    python3 scripts/build_balanced_dataset.py --output data/balanced_train.jsonl
    python3 scripts/build_balanced_dataset.py --output data/balanced_train.jsonl --target-size 15000
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Optional


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

SYSTEM_ECO = (
    "You are an expert ecological modeler and scientific programmer. "
    "Think step by step inside <think>...</think> tags before answering. "
    "Provide rigorous reasoning grounded in ecological theory, statistical "
    "methods, and reproducible code."
)

SYSTEM_BIO = (
    "You are an expert biologist. Think step by step inside "
    "<think>...</think> tags before answering. Provide thorough, "
    "scientifically accurate reasoning."
)

SYSTEM_SCI = (
    "You are a scientific reasoning assistant. Think step by step inside "
    "<think>...</think> tags before answering. Show your reasoning clearly "
    "and rigorously."
)

SYSTEM_PUZZLE = (
    "You are a helpful assistant that solves puzzles step by step. "
    "Think inside <think>...</think> tags before giving your answer."
)


def normalize_eco_method(method: str) -> str:
    if not method:
        return "unspecified"
    m = method.strip().lower()
    if "maxent" in m: return "maxent"
    if "brt" in m or "boosted regression" in m: return "brt"
    if "glm" in m or "generalized linear" in m: return "glm"
    if "gam" in m or "generalized additive" in m: return "gam"
    if "random forest" in m: return "random_forest"
    if "occupancy" in m: return "occupancy"
    if "logistic regression" in m: return "logistic_regression"
    if "hmm" in m or "hidden markov" in m: return "hmm"
    if "n-mixture" in m or "n_mixture" in m: return "n_mixture"
    if "pca" in m or "principal component" in m: return "pca"
    if "edna" in m or "metabarcod" in m: return "edna"
    if "network" in m: return "network_analysis"
    if "linear regression" in m: return "linear_regression"
    if "bayesian" in m: return "bayesian"
    if "phylo" in m: return "phylogenetics"
    if "sdm" in m or "species distribution" in m: return "sdm"
    if "mark-recapture" in m or "capture-recapture" in m: return "mark_recapture"
    if "distance sampling" in m: return "distance_sampling"
    if "clustering" in m or "k-means" in m: return "clustering"
    if "anova" in m: return "anova"
    if "survival" in m or "cox" in m: return "survival_analysis"
    if "svm" in m or "support vector" in m: return "svm"
    if "cnn" in m or "convolutional" in m: return "cnn"
    if "gnn" in m or "graph neural" in m: return "gnn"
    if "pva" in m or "population viability" in m: return "pva"
    return "other"


def has_think_tags(content: str) -> bool:
    return "<think>" in content and "</think>" in content


def wrap_with_think(reasoning: str, answer: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>\n\n{answer.strip()}"


def to_chat_messages(system: str, user: str, assistant: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


# ---------------------------------------------------------------------------
# Dataset loaders — each returns list of {messages, category, subcategory, source}
# ---------------------------------------------------------------------------

def load_ecoreasoner_cot(max_per_method: int = 500) -> list[dict]:
    """Load ecoreasoner-cot-20k, filter for <think>, cap per method."""
    from datasets import load_dataset
    print("Loading ecoreasoner-cot-20k...", flush=True)
    ds = load_dataset("alrobles/ecoreasoner-cot-20k", split="train")

    by_method = defaultdict(list)
    skipped = 0

    for ex in ds:
        msgs = ex.get("messages", [])
        if len(msgs) < 3:
            skipped += 1
            continue

        asst_content = ""
        for m in msgs:
            if m["role"] == "assistant":
                asst_content = m["content"]
                break

        if not has_think_tags(asst_content):
            skipped += 1
            continue

        if len(asst_content) < 500:
            skipped += 1
            continue

        method = normalize_eco_method(ex.get("method", ""))
        by_method[method].append({
            "messages": msgs,
            "category": "ecology",
            "subcategory": method,
            "source": "ecoreasoner-cot-20k",
        })

    records = []
    for method, examples in by_method.items():
        random.shuffle(examples)
        cap = max_per_method
        records.extend(examples[:cap])

    print(f"  ecoreasoner-cot-20k: {len(records)} accepted, {skipped} skipped "
          f"({len(by_method)} methods, cap={max_per_method})", flush=True)
    return records


def load_ecocoder_scientific(max_examples: Optional[int] = None) -> list[dict]:
    """Load ecocoder-scientific-reasoning (all 1268 examples)."""
    from datasets import load_dataset
    print("Loading ecocoder-scientific-reasoning...", flush=True)
    ds = load_dataset("alrobles/ecocoder-scientific-reasoning", split="train")

    records = []
    for ex in ds:
        msgs = ex.get("messages", [])
        if len(msgs) < 2:
            continue

        asst_content = ""
        for m in msgs:
            if m["role"] == "assistant":
                asst_content = m["content"]
                break

        # Wrap in <think> if missing
        if not has_think_tags(asst_content):
            # The assistant content has [CONTEXT]...[REASONING]...[CODE] structure
            # Wrap the whole thing in think tags
            new_content = f"<think>\n{asst_content}\n</think>"
            new_msgs = []
            for m in msgs:
                if m["role"] == "assistant":
                    new_msgs.append({"role": "assistant", "content": new_content})
                else:
                    new_msgs.append(m)
            msgs = new_msgs

        method_cat = ex.get("method_category", "general")
        records.append({
            "messages": msgs,
            "category": "ecology",
            "subcategory": f"ecocoder_{method_cat}",
            "source": "ecocoder-scientific-reasoning",
        })

    if max_examples and len(records) > max_examples:
        random.shuffle(records)
        records = records[:max_examples]

    print(f"  ecocoder-scientific-reasoning: {len(records)} loaded", flush=True)
    return records


def load_ecocoder_cot(max_examples: Optional[int] = None) -> list[dict]:
    """Load ecocoder-cot (519 paper-based examples)."""
    from datasets import load_dataset
    print("Loading ecocoder-cot...", flush=True)
    ds = load_dataset("alrobles/ecocoder-cot", split="train")

    records = []
    for ex in ds:
        title = ex.get("title", "")
        context = ex.get("context", "")
        reasoning = ex.get("reasoning", "")
        code = ex.get("code", "")

        if not reasoning or len(reasoning) < 100:
            continue

        user_msg = f"Analyze the following ecological research paper and explain the reasoning behind its methodology.\n\nTitle: {title}\n\nContext: {context}"
        assistant_content = wrap_with_think(
            reasoning,
            f"```python\n{code}\n```" if code else reasoning.split("\n")[-1]
        )

        records.append({
            "messages": to_chat_messages(SYSTEM_ECO, user_msg, assistant_content),
            "category": "ecology",
            "subcategory": f"paper_{ex.get('method_type', 'general')}",
            "source": "ecocoder-cot",
        })

    if max_examples and len(records) > max_examples:
        random.shuffle(records)
        records = records[:max_examples]

    print(f"  ecocoder-cot: {len(records)} loaded", flush=True)
    return records


def load_biology_reasoning(max_examples: int = 3000) -> list[dict]:
    """Load Brainquiver/reasoning-biology-finetuning-preview."""
    from datasets import load_dataset
    print("Loading Brainquiver/reasoning-biology...", flush=True)
    ds = load_dataset("Brainquiver/reasoning-biology-finetuning-preview", split="train")

    records = []
    for ex in ds:
        question = ex.get("question", "")
        reasoning = ex.get("reasoning", "")
        answer = ex.get("answer", "")

        if not question or not reasoning or len(reasoning) < 50:
            continue

        assistant_content = wrap_with_think(reasoning, answer)
        records.append({
            "messages": to_chat_messages(SYSTEM_BIO, question, assistant_content),
            "category": "biology",
            "subcategory": "biology_reasoning",
            "source": "brainquiver-biology",
        })

    random.shuffle(records)
    records = records[:max_examples]

    print(f"  brainquiver-biology: {len(records)} sampled (from {len(ds)})", flush=True)
    return records


def load_camel_biology(max_examples: int = 2000) -> list[dict]:
    """Load camel-ai/biology and convert Q&A to CoT format."""
    from datasets import load_dataset
    print("Loading camel-ai/biology (streaming)...", flush=True)

    ds = load_dataset("camel-ai/biology", split="train", streaming=True)

    records = []
    topics_seen = Counter()

    for ex in ds:
        question = ex.get("message_1", "")
        answer = ex.get("message_2", "")
        topic = ex.get("topic;", ex.get("topic", "biology"))
        sub_topic = ex.get("sub_topic", "general")

        if not question or not answer or len(answer) < 200:
            continue

        # Cap per topic for balance
        topic_key = f"{topic}_{sub_topic}"
        if topics_seen[topic_key] >= 100:
            continue
        topics_seen[topic_key] += 1

        # Wrap answer as reasoning
        assistant_content = f"<think>\nLet me think through this biology question step by step.\n\n{answer.strip()}\n</think>\n\n{answer.strip()[:500]}"
        records.append({
            "messages": to_chat_messages(SYSTEM_BIO, question, assistant_content),
            "category": "biology",
            "subcategory": f"camel_{topic}".lower().replace(" ", "_"),
            "source": "camel-ai-biology",
        })

        if len(records) >= max_examples:
            break

    print(f"  camel-ai/biology: {len(records)} loaded ({len(topics_seen)} topics)", flush=True)
    return records


def load_physics_reasoning(max_examples: int = 2000) -> list[dict]:
    """Load physics subset from nemotron-reasoning-v3."""
    from datasets import load_dataset
    print("Loading nemotron-reasoning-v3 (physics)...", flush=True)
    ds = load_dataset("alrobles/nemotron-reasoning-v3", split="train")

    records = []
    for ex in ds:
        source = ex.get("source", "") or ""
        if source != "physics":
            continue

        msgs = ex.get("messages", [])
        if len(msgs) < 3:
            continue

        asst_content = ""
        for m in msgs:
            if m["role"] == "assistant":
                asst_content = m["content"]
                break

        # Wrap in <think> if not present
        if not has_think_tags(asst_content):
            new_msgs = []
            for m in msgs:
                if m["role"] == "assistant":
                    new_msgs.append({
                        "role": "assistant",
                        "content": f"<think>\n{m['content']}\n</think>"
                    })
                else:
                    new_msgs.append(m)
            msgs = new_msgs

        records.append({
            "messages": msgs,
            "category": "physics",
            "subcategory": "physics_reasoning",
            "source": "nemotron-reasoning-v3",
        })

    random.shuffle(records)
    records = records[:max_examples]

    print(f"  nemotron-reasoning-v3 (physics): {len(records)} loaded", flush=True)
    return records


def load_kaggle_puzzles(max_examples: int = 2500) -> list[dict]:
    """Load Kaggle puzzle data for general reasoning."""
    from datasets import load_dataset
    print("Loading nemotron-eco-reasoner-v14 (puzzles)...", flush=True)
    ds = load_dataset("alrobles/nemotron-eco-reasoner-v14", split="train")

    records = []
    for ex in ds:
        msgs = ex.get("messages", [])
        if len(msgs) < 3:
            continue

        asst_content = ""
        for m in msgs:
            if m["role"] == "assistant":
                asst_content = m["content"]
                break

        if not has_think_tags(asst_content):
            continue

        records.append({
            "messages": msgs,
            "category": "reasoning",
            "subcategory": "puzzle",
            "source": "nemotron-eco-reasoner-v14",
        })

    random.shuffle(records)
    records = records[:max_examples]

    print(f"  nemotron-eco-reasoner-v14 (puzzles): {len(records)} loaded", flush=True)
    return records


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_balanced_dataset(
    target_size: int = 15000,
    eco_method_cap: int = 500,
    seed: int = 42,
) -> list[dict]:
    """Build balanced dataset from all sources."""
    random.seed(seed)

    # Target composition
    eco_target = int(target_size * 0.40)       # ~6000
    bio_target = int(target_size * 0.25)       # ~3750
    physics_target = int(target_size * 0.15)   # ~2250
    puzzle_target = int(target_size * 0.20)    # ~3000

    # Load all sources
    eco_main = load_ecoreasoner_cot(max_per_method=eco_method_cap)
    eco_sci = load_ecocoder_scientific()
    eco_cot = load_ecocoder_cot()
    bio_reason = load_biology_reasoning(max_examples=int(bio_target * 0.6))
    bio_camel = load_camel_biology(max_examples=int(bio_target * 0.4))
    physics = load_physics_reasoning(max_examples=physics_target)
    puzzles = load_kaggle_puzzles(max_examples=puzzle_target)

    # Combine ecology sources
    all_eco = eco_main + eco_sci + eco_cot
    random.shuffle(all_eco)
    if len(all_eco) > eco_target:
        all_eco = all_eco[:eco_target]

    # Combine biology sources
    all_bio = bio_reason + bio_camel
    random.shuffle(all_bio)
    if len(all_bio) > bio_target:
        all_bio = all_bio[:bio_target]

    # Trim physics and puzzles
    if len(physics) > physics_target:
        physics = physics[:physics_target]
    if len(puzzles) > puzzle_target:
        puzzles = puzzles[:puzzle_target]

    # Final merge
    all_records = all_eco + all_bio + physics + puzzles
    random.shuffle(all_records)

    return all_records


def compute_stats(records: list[dict]) -> dict:
    """Compute dataset statistics."""
    stats = {
        "total": len(records),
        "by_category": Counter(),
        "by_source": Counter(),
        "by_subcategory": Counter(),
        "has_think": 0,
        "has_code": 0,
        "avg_assistant_len": 0,
    }

    total_asst_len = 0
    for r in records:
        stats["by_category"][r["category"]] += 1
        stats["by_source"][r["source"]] += 1
        stats["by_subcategory"][r["subcategory"]] += 1
        for m in r["messages"]:
            if m["role"] == "assistant":
                c = m["content"]
                total_asst_len += len(c)
                if has_think_tags(c):
                    stats["has_think"] += 1
                if "```" in c:
                    stats["has_code"] += 1

    stats["avg_assistant_len"] = total_asst_len / max(len(records), 1)
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Build balanced training dataset for Nemotron ecological reasoning"
    )
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--target-size", type=int, default=15000,
                        help="Target dataset size (default: 15000)")
    parser.add_argument("--eco-method-cap", type=int, default=500,
                        help="Max examples per ecological method (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    records = build_balanced_dataset(
        target_size=args.target_size,
        eco_method_cap=args.eco_method_cap,
        seed=args.seed,
    )

    # Write output (messages + method for stratified training, metadata for analysis)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for r in records:
            out = {"messages": r["messages"], "method": r["subcategory"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # Write metadata sidecar
    meta_path = args.output.replace(".jsonl", "_meta.jsonl")
    with open(meta_path, "w") as f:
        for r in records:
            out = {
                "category": r["category"],
                "subcategory": r["subcategory"],
                "source": r["source"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # Print stats
    stats = compute_stats(records)

    print(f"\n{'='*70}")
    print(f"BALANCED DATASET BUILT")
    print(f"  Total examples:      {stats['total']}")
    print(f"  With <think> tags:   {stats['has_think']} ({stats['has_think']/max(stats['total'],1)*100:.1f}%)")
    print(f"  With code:           {stats['has_code']} ({stats['has_code']/max(stats['total'],1)*100:.1f}%)")
    print(f"  Avg assistant len:   {stats['avg_assistant_len']:.0f} chars")

    print(f"\n  Category distribution:")
    for cat, count in stats["by_category"].most_common():
        pct = count / stats["total"] * 100
        print(f"    {cat:20s} {count:6d} ({pct:5.1f}%)")

    print(f"\n  Source distribution:")
    for src, count in stats["by_source"].most_common():
        pct = count / stats["total"] * 100
        print(f"    {src:35s} {count:6d} ({pct:5.1f}%)")

    print(f"\n  Top subcategories:")
    for sub, count in stats["by_subcategory"].most_common(25):
        pct = count / stats["total"] * 100
        print(f"    {sub:35s} {count:6d} ({pct:5.1f}%)")

    print(f"\n  Output:  {args.output}")
    print(f"  Meta:    {meta_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
