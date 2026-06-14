#!/usr/bin/env python3
"""
Generate ecological reasoning training dataset from PubMed literature.
Queries the local FTS5 index for ecological concepts and formats
abstracts as chain-of-thought reasoning examples in Nemotron SFT format.

Mixes with Kaggle reasoning data to create a hybrid dataset.
Cross-domain data teaches analogical reasoning — stronger than
math/logic alone.

Usage:
  python3 generate_eco_dataset.py --output eco_reasoning_2k.jsonl --limit 2000
"""

import argparse
import json
import os
import random
import sys
import time

# PubMed search module from ecoseek-litdump — auto-detect location
import os as _os
_LITDUMP_PATHS = [
    "/home/a474r867/work/ecoseek-litdump/scripts",   # cluster
    "/home/reumanlab/ecoseek-litdump/scripts",        # reumanlab
]
for _p in _LITDUMP_PATHS:
    if _os.path.isdir(_p):
        sys.path.insert(0, _p)
        break
from search_pubmed import search as pubmed_search

# ─── Ecological topics for reasoning extraction ────────────────────
ECO_TOPICS = [
    # Species distribution & niche
    "species distribution model climate niche ecological",
    "MaxEnt ecological niche model habitat suitability",
    "species range shift climate change biodiversity",
    "SDM ensemble forecasting uncertainty ecological",

    # Community ecology
    "species coexistence competition niche partitioning",
    "community assembly trait-based ecology functional diversity",
    "ecological network food web trophic interaction",

    # Host-parasite & disease ecology  
    "host parasite coevolution ecological dynamics",
    "disease ecology zoonotic spillover biodiversity",
    "parasite host specificity phylogenetic signal",

    # Macroecology & biogeography
    "macroecological pattern species richness latitude gradient",
    "biogeographic region species turnover beta diversity",
    "island biogeography species-area relationship extinction",

    # Conservation
    "conservation prioritization biodiversity hotspot protected area",
    "extinction risk assessment IUCN Red List ecological traits",
    "invasive species impact native community ecosystem",

    # Methods
    "phylogenetic comparative method ecological trait evolution",
    "joint species distribution model hierarchical Bayesian",
    "occupancy detection probability imperfect sampling",

    # Climate change
    "phenological shift climate warming ecological mismatch",
    "thermal tolerance physiological limit range edge",
]


def format_reasoning_example(abstract: str, title: str, topic: str, pmid: str) -> dict:
    """Format a PubMed abstract as a chain-of-thought reasoning example."""
    # Clean abstract: remove extra whitespace, limit length
    abstract = " ".join(abstract.split())[:1500]

    prompt = (
        f"Analyze the following ecological research context and explain the "
        f"reasoning behind its findings:\n\n"
        f"Title: {title}\n"
        f"Context: {abstract}\n\n"
        f"Question: Based on this study, what ecological mechanisms explain "
        f"the observed patterns? Provide a step-by-step reasoning chain."
    )

    # The "answer" is a chain-of-thought derived from the abstract itself
    # Structure: identify pattern → propose mechanism → cite evidence → conclude
    answer = _generate_chain_of_thought(abstract, title, topic)

    return {"prompt": prompt, "answer": answer, "source": f"pubmed:{pmid}"}


def _generate_chain_of_thought(abstract: str, title: str, topic: str) -> str:
    """Generate a structured chain-of-thought from the abstract.
    
    This creates a reasoning template that the model learns to replicate.
    The key insight: we're training the model to reason about ecological
    data the same way the Kaggle puzzles train it to reason about math.
    """
    sentences = [s.strip() for s in abstract.replace("  ", " ").split(". ") if len(s) > 30]

    if len(sentences) < 3:
        return f"The study '{title}' investigates {topic}. The evidence suggests ecological patterns driven by species interactions and environmental gradients."

    # Build structured reasoning chain
    parts = []
    parts.append(f"Step 1 — Context: {sentences[0]}.")
    
    if len(sentences) > 1:
        parts.append(f"Step 2 — Methods & Evidence: {sentences[1]}.")
    
    if len(sentences) > 2:
        parts.append(f"Step 3 — Key Finding: {sentences[2]}.")
    
    if len(sentences) > 3:
        mechanism = sentences[min(3, len(sentences)-1)]
        parts.append(f"Step 4 — Mechanism: The observed pattern is explained by {mechanism}")

    conclusion = (
        f"Conclusion: The ecological reasoning reveals that {topic} is driven by "
        f"interacting biotic and abiotic factors operating across spatial and "
        f"temporal scales, consistent with established ecological theory."
    )
    parts.append(conclusion)

    return "\n\n".join(parts)


def load_kaggle_data(path: str, max_examples: int = None) -> list[dict]:
    """Load Kaggle training data in prompt→answer format."""
    import csv
    examples = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append({
                "prompt": row.get("prompt", row.get("question", "")),
                "answer": row.get("answer", ""),
                "source": "kaggle",
            })
    if max_examples:
        examples = examples[:max_examples]
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="eco_reasoning_2k.jsonl")
    parser.add_argument("--limit", type=int, default=2000, help="Max eco examples")
    parser.add_argument("--kaggle", help="Path to Kaggle CSV")
    parser.add_argument("--kaggle-limit", type=int, default=5000)
    parser.add_argument("--papers-per-topic", type=int, default=100)
    args = parser.parse_args()

    all_examples = []
    total = 0

    # ── Generate ecological reasoning examples ──────────────────────
    print(f"Querying PubMed for {len(ECO_TOPICS)} topics...")
    for topic in ECO_TOPICS:
        if total >= args.limit:
            break
        try:
            results = pubmed_search(topic, limit=args.papers_per_topic,
                                    year_min=2015, language="eng")
            if not results or "error" in results[0]:
                continue
            for r in results:
                if total >= args.limit:
                    break
                abstract = r.get("abstract", "")
                if len(abstract) < 200:
                    continue  # Too short to learn from
                example = format_reasoning_example(
                    abstract, r.get("title", topic),
                    topic, str(r.get("pmid", "unknown"))
                )
                all_examples.append(example)
                total += 1
        except Exception as exc:
            print(f"  Topic '{topic[:40]}...': error ({exc})")
            continue
        if total % 200 == 0:
            print(f"  {total}/{args.limit} examples generated...")

    print(f"Generated {len(all_examples)} ecological reasoning examples")

    # ── Mix with Kaggle data (if provided) ──────────────────────────
    if args.kaggle and os.path.exists(args.kaggle):
        kaggle_data = load_kaggle_data(args.kaggle, args.kaggle_limit)
        print(f"Loaded {len(kaggle_data)} Kaggle examples")
        # Interleave for diverse training batches
        combined = []
        eco_idx, kag_idx = 0, 0
        while eco_idx < len(all_examples) or kag_idx < len(kaggle_data):
            if eco_idx < len(all_examples):
                combined.append(all_examples[eco_idx])
                eco_idx += 1
            if kag_idx < len(kaggle_data):
                combined.append(kaggle_data[kag_idx])
                kag_idx += 1
        all_examples = combined
        print(f"Combined dataset: {len(all_examples)} total (eco + kaggle interleaved)")

    # ── Write output ────────────────────────────────────────────────
    random.shuffle(all_examples)  # Shuffle for training
    with open(args.output, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(all_examples)} examples to {args.output}")
    print(f"Sources: {len([e for e in all_examples if e['source'].startswith('pubmed')])} PubMed, "
          f"{len([e for e in all_examples if e['source'] == 'kaggle'])} Kaggle")


if __name__ == "__main__":
    main()
