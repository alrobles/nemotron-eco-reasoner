#!/usr/bin/env python3
"""
Prepare Kaggle reasoning dataset for Nemotron-3-Nano training.

Converts Kaggle train.csv to ShareGPT-format JSONL with reasoning-focused
system prompt. The full puzzle prompt from the CSV is used verbatim as the
user message, with the answer wrapped in \\boxed{}.

Usage:
    python scripts/prepare_dataset_kaggle.py \
        --kaggle-csv data/train.csv \
        --output data/kaggle_dataset.jsonl \
        --max-examples 10000
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
    "Show your complete reasoning before giving the final answer."
)


def load_kaggle_csv(path: str, max_examples: int = 0) -> list[dict]:
    """Load Kaggle train.csv and convert to ShareGPT format.
    
    Uses the full puzzle prompt from the 'prompt' column as the user message.
    The answer is wrapped in \\boxed{} format.
    """
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} Kaggle rows from {path}")
    
    # Verify columns
    for col in ["id", "prompt", "answer"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in CSV. Found: {list(df.columns)}")

    examples = []
    skipped = 0
    for _, row in df.iterrows():
        puzzle_id = row["id"]
        prompt_text = str(row["prompt"]).strip()
        answer = str(row["answer"]).strip()
        
        if not prompt_text or not answer:
            skipped += 1
            continue

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Puzzle {puzzle_id}: {prompt_text}\n\n"
                        f"Think step by step and place your final answer inside \\boxed{{}}."
                    ),
                },
                {"role": "assistant", "content": f"\\boxed{{{answer}}}"},
            ]
        })

    if skipped:
        logger.warning(f"Skipped {skipped} rows with empty prompt/answer")
    
    # Cap if requested
    if max_examples > 0 and len(examples) > max_examples:
        examples = random.sample(examples, max_examples)
        logger.info(f"Capped to {max_examples} examples (from {len(df)})")
    
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
    kaggle_examples = load_kaggle_csv(args.kaggle_csv, args.max_examples)

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
