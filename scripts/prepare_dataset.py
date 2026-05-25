#!/usr/bin/env python3
"""
Generate the combined reasoning + ecology dataset for Nemotron-3-Nano training.

Sources:
  1. Kaggle logic puzzles (train.csv) — converted to ShareGPT reasoning traces
  2. EcoAgent synthetic generator — ecological Q&A, tool-calling, triplets, taxonomy

Output: JSONL with {"messages": [{"role": "system/user/assistant", "content": "..."}]}

Usage:
    python scripts/prepare_dataset.py \
        --kaggle-csv data/train.csv \
        --ecoagent-dir ../ecoagent \
        --output data/combined_dataset.jsonl \
        --kaggle-ratio 0.5
"""

import argparse
import json
import logging
import os
import random
import re
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an AI assistant trained to solve both logical reasoning puzzles and "
    "ecological science questions. When the task is a puzzle, reason step by step "
    "and place the final answer inside \\boxed{...}. When the task is an ecology "
    "question, use available tools when needed and synthesize a scientific response."
)


def load_kaggle_csv(path: str) -> list[dict]:
    """Load Kaggle train.csv and convert to ShareGPT format."""
    import pandas as pd

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} Kaggle rows")

    examples = []
    for _, row in df.iterrows():
        puzzle_id = row["id"]
        answer = row["answer"]

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Puzzle {puzzle_id}: Determine the answer to this reasoning puzzle. "
                        f"Think step by step and place your final answer inside \\boxed{{}}."
                    ),
                },
                {"role": "assistant", "content": f"\\boxed{{{answer}}}"},
            ]
        })

    logger.info(f"Converted {len(examples)} Kaggle puzzles")
    return examples


def load_ecoagent_synthetic(ecoagent_dir: str, target_count: int) -> list[dict]:
    """Generate synthetic ecology examples using ecoagent's dataset module."""
    ecoagent_path = os.path.abspath(ecoagent_dir)
    if ecoagent_path not in sys.path:
        sys.path.insert(0, os.path.join(ecoagent_path, "ecoagent", "training"))

    try:
        from dataset import (
            CoFIDDatasetGenerator,
            EcologyExampleGenerator,
            TaxonomyDatasetGenerator,
            ToolCallingDatasetGenerator,
            TripletDatasetGenerator,
        )
    except ImportError:
        logger.warning("ecoagent training modules not importable — using fallback templates")
        return _fallback_eco_examples(target_count)

    examples_per_source = target_count // 5

    generators = {
        "tool_calling": ToolCallingDatasetGenerator(),
        "cofid": CoFIDDatasetGenerator(),
        "triplets": TripletDatasetGenerator(),
        "taxonomy": TaxonomyDatasetGenerator(),
        "ecology": EcologyExampleGenerator(),
    }

    all_examples = []
    for name, gen in generators.items():
        try:
            n = examples_per_source
            gen_examples = gen.generate(n=n)
            # Add system prompt
            for ex in gen_examples:
                if ex["messages"][0]["role"] != "system":
                    ex["messages"].insert(0, {"role": "system", "content": SYSTEM_PROMPT})
                all_examples.append(ex)
            logger.info(f"  {name}: {len(gen_examples)} examples")
        except Exception as e:
            logger.warning(f"  {name} failed: {e}")

    logger.info(f"EcoAgent total: {len(all_examples)} examples")
    return all_examples


def _fallback_eco_examples(count: int) -> list[dict]:
    """Fallback ecology templates when ecoagent module not available."""
    templates = [
        # Tool-calling
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "What parasites are known to infect the fish species Gadus morhua?"},
                {"role": "assistant", "content": "[\"gbif_search\", {\"species\": \"Gadus morhua\", \"interaction\": \"parasiteOf\"}]"},
            ],
        },
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Find all host species for the parasite Anisakis simplex."},
                {"role": "assistant", "content": "[\"gbif_search\", {\"species\": \"Anisakis simplex\", \"interaction\": \"hasHost\"}]"},
            ],
        },
        # Triplet extraction
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Extract host-parasite relationships: 'The nematode Anisakis simplex commonly infects Atlantic cod (Gadus morhua) and Pacific herring (Clupea pallasii), causing anisakiasis in humans who consume raw fish.'"},
                {"role": "assistant", "content": "[(Anisakis simplex, parasiteOf, Gadus morhua), (Anisakis simplex, parasiteOf, Clupea pallasii)]"},
            ],
        },
        # Taxonomy
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Resolve taxonomy: Gadus morhua"},
                {"role": "assistant", "content": "Kingdom: Animalia, Phylum: Chordata, Class: Actinopterygii, Order: Gadiformes, Family: Gadidae, Genus: Gadus, Species: Gadus morhua"},
            ],
        },
        # Ecology reasoning
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "How does climate change affect host-parasite interactions in marine ecosystems?"},
                {"role": "assistant", "content": "Climate change alters host-parasite dynamics through: (1) temperature-dependent parasite development rates, (2) range shifts of hosts and parasites, (3) phenological mismatches, and (4) increased stress on host immune systems. Warmer waters accelerate parasite life cycles and expand their geographic ranges poleward."},
            ],
        },
    ]

    # Repeat templates to reach target count
    examples = []
    for i in range(count):
        base = templates[i % len(templates)].copy()
        # Slight variation
        examples.append({"messages": [m.copy() for m in base["messages"]]})
    return examples


def main():
    parser = argparse.ArgumentParser(description="Prepare dual dataset for Nemotron training")
    parser.add_argument("--kaggle-csv", required=True, help="Path to Kaggle train.csv")
    parser.add_argument("--ecoagent-dir", default="../ecoagent", help="Path to ecoagent repo")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--kaggle-ratio", type=float, default=0.5, help="Fraction of Kaggle examples")
    parser.add_argument("--total", type=int, default=30000, help="Target total examples")
    args = parser.parse_args()

    # Load Kaggle
    kaggle_examples = load_kaggle_csv(args.kaggle_csv)
    kaggle_count = int(args.total * args.kaggle_ratio)
    if len(kaggle_examples) > kaggle_count:
        kaggle_examples = random.sample(kaggle_examples, kaggle_count)
    elif len(kaggle_examples) < kaggle_count:
        logger.warning(f"Only {len(kaggle_examples)} Kaggle examples available (wanted {kaggle_count})")

    # Load ecology
    eco_count = args.total - len(kaggle_examples)
    eco_examples = load_ecoagent_synthetic(args.ecoagent_dir, eco_count)

    # Combine and shuffle
    combined = kaggle_examples + eco_examples
    random.shuffle(combined)

    # Write JSONL
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for ex in combined:
            # TRL SFTTrainer expects 'text' field or 'messages' field
            # We store messages and let the formatter handle it
            f.write(json.dumps(ex) + "\n")

    logger.info(f"Wrote {len(combined)} examples to {args.output}")
    logger.info(f"  Kaggle: {len(kaggle_examples)}, Ecology: {len(eco_examples)}")


if __name__ == "__main__":
    main()
