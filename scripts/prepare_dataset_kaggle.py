#!/usr/bin/env python3
"""
Prepare Kaggle-only reasoning dataset for Nemotron-3-Nano training.

Converts Kaggle train.csv to ShareGPT-format JSONL with reasoning-focused
system prompt. No ecology data — 100% focused on the Kaggle competition.

Usage:
    python scripts/prepare_dataset_kaggle.py \
        --kaggle-csv data/train.csv \
        --output data/kaggle_dataset.jsonl
"""

import argparse
import json
import logging
import os
import random

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an AI assistant specialized in logical reasoning and mathematical puzzles. "
    "For every problem, reason step by step and place your final answer inside \\boxed{...}. "
    "Always show your work before giving the final answer."
)


def load_kaggle_csv(path: str) -> list[dict]:
    """Load Kaggle train.csv and convert to ShareGPT format."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Kaggle reasoning dataset for Nemotron training"
    )
    parser.add_argument("--kaggle-csv", required=True, help="Path to Kaggle train.csv")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--max-examples", type=int, default=0,
        help="Cap total examples (0 = use all available)"
    )
    args = parser.parse_args()

    # Load Kaggle
    kaggle_examples = load_kaggle_csv(args.kaggle_csv)

    # Cap if requested
    if args.max_examples > 0 and len(kaggle_examples) > args.max_examples:
        kaggle_examples = random.sample(kaggle_examples, args.max_examples)
        logger.info(f"Capped to {args.max_examples} examples")

    # Shuffle for good measure
    random.shuffle(kaggle_examples)

    # Write JSONL
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for ex in kaggle_examples:
            f.write(json.dumps(ex) + "\n")

    logger.info(f"Wrote {len(kaggle_examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
