#!/usr/bin/env python3
"""
Kaggle reasoning benchmark evaluation for Nemotron Eco-Reasoner.

Evaluates only reasoning puzzles — extracts \boxed{...} answers and
computes accuracy against expected answers.

Usage:
    python scripts/evaluate.py --adapter checkpoints/final/ --kaggle-csv data/train.csv
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "nvidia/Nemotron-3-Nano-30B-A3B-BF16"


def load_model_and_adapter(adapter_path: str):
    """Load base model + LoRA adapter in BF16."""
    logger.info(f"Loading base model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tokenizer


def generate(model, tokenizer, messages: list[dict], max_new_tokens: int = 512) -> str:
    """Generate a response from the model using chat template."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Strip the input prompt to get only the generated response
    if full_response.startswith(prompt):
        response = full_response[len(prompt):]
    else:
        response = full_response
    return response.strip()


def extract_boxed(text: str) -> str | None:
    """Extract the content inside \\boxed{...}."""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    return None


def evaluate_reasoning(
    model, tokenizer, kaggle_csv: str, system_prompt: str, n: int = 0
) -> dict:
    """Evaluate reasoning accuracy on Kaggle puzzles.

    Args:
        n: Number of puzzles to evaluate (0 = all).
    """
    df = pd.read_csv(kaggle_csv)
    if n > 0 and len(df) > n:
        df = df.sample(n=n, random_state=42)

    correct = 0
    total = 0
    per_sample = []

    for _, row in df.iterrows():
        puzzle_id = row["id"]
        expected = str(row["answer"]).strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Puzzle {puzzle_id}: Determine the answer to this reasoning puzzle. "
                    f"Think step by step and place your final answer inside \\boxed{{}}."
                ),
            },
        ]

        try:
            response = generate(model, tokenizer, messages)
            predicted = extract_boxed(response)
            is_correct = predicted is not None and predicted == expected
            if is_correct:
                correct += 1

            per_sample.append({
                "id": puzzle_id,
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
            })
        except Exception as e:
            logger.error(f"Error on puzzle {puzzle_id}: {e}")
            per_sample.append({
                "id": puzzle_id,
                "expected": expected,
                "predicted": None,
                "correct": False,
                "error": str(e),
            })

        total += 1
        if total % 25 == 0:
            acc = correct / total
            logger.info(f"Progress: {total}/{len(df)} — accuracy {acc:.4f}")

    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"Final reasoning accuracy: {accuracy:.4f} ({correct}/{total})")

    return {
        "reasoning_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_sample": per_sample,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Nemotron Kaggle Reasoner")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter")
    parser.add_argument("--kaggle-csv", required=True, help="Path to Kaggle train.csv")
    parser.add_argument("--n", type=int, default=0, help="Number of puzzles (0=all)")
    parser.add_argument("--output", default=None, help="Save per-sample results as JSON")
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are an AI assistant specialized in logical reasoning and mathematical puzzles. "
            "For every problem, reason step by step and place your final answer inside "
            "\\boxed{...}. Always show your work before giving the final answer."
        ),
        help="System prompt override",
    )
    args = parser.parse_args()

    if not os.path.exists(args.kaggle_csv):
        logger.error(f"Kaggle CSV not found: {args.kaggle_csv}")
        raise SystemExit(1)

    model, tokenizer = load_model_and_adapter(args.adapter)

    results = evaluate_reasoning(
        model, tokenizer, args.kaggle_csv, args.system_prompt, args.n
    )

    # Print summary
    print(f"\n{'='*50}")
    print(f"REASONING ACCURACY: {results['reasoning_accuracy']:.4f}")
    print(f"Correct: {results['correct']} / {results['total']}")
    print(f"{'='*50}")

    # Save per-sample if requested
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved per-sample results to {args.output}")


if __name__ == "__main__":
    main()
