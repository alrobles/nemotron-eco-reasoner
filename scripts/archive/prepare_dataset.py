#!/usr/bin/env python3
"""
Generate training dataset for Nemotron-3-Nano fine-tuning.

Sources:
  1. Kaggle logic puzzles (train.csv) — reformatted to ShareGPT
  2. Ecology synthetic data (optional) — from ecoagent or templates

Output: JSONL with {"messages": [{"role": "system/user/assistant", "content": "..."}]}

Usage:
    # Kaggle only, 5K examples
    python scripts/prepare_dataset.py \\
        --source kaggle \\
        --kaggle-csv data/train.csv \\
        --output data/train_5k.jsonl \\
        --max-examples 5000

    # Mixed Kaggle + ecology
    python scripts/prepare_dataset.py \\
        --source mixed \\
        --kaggle-csv data/train.csv \\
        --ecoagent-dir ../ecoagent \\
        --output data/train_mixed.jsonl \\
        --kaggle-ratio 0.5 \\
        --max-examples 10000
"""

import argparse
import json
import logging
import os
import random
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are Nemotron-3-Nano, an expert reasoning model. "
    "Solve the problem step by step, showing all your work. "
    "Place your final answer inside \\boxed{...}."
)

SYSTEM_ECO = (
    "You are Nemotron-3-Nano, an expert in ecology, evolution, and scientific reasoning. "
    "Solve the problem step by step, showing all your work. "
    "Place your final answer inside \\boxed{...}."
)

# ── Kaggle loader ──

def load_kaggle_csv(csv_path: str, max_examples: int = None):
    """Load Kaggle train.csv and convert to ShareGPT messages."""
    import csv
    examples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Use the actual puzzle prompt, not a placeholder
            prompt = row.get("prompt", "").strip()
            answer = row.get("answer", "").strip()
            if not prompt or not answer:
                continue
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ]
            })
            if max_examples and len(examples) >= max_examples:
                break

    logger.info(f"Loaded {len(examples)} Kaggle examples from {csv_path}")
    return examples


# ── Ecology loader (optional) ──

def load_eco_templates(count: int):
    """Generate synthetic ecology examples from templates (no ecoagent dep)."""
    templates = [
        {
            "user": "Calculate the Shannon diversity index for a community with species abundances: 12, 8, 5, 3, 2.",
            "assistant": r"Step 1: Total individuals = 12+8+5+3+2 = 30.\nStep 2: Proportions: 12/30=0.4, 8/30=0.267, 5/30=0.167, 3/30=0.1, 2/30=0.067.\nStep 3: Shannon H' = -sum(p_i * ln p_i) = -(0.4*ln0.4 + 0.267*ln0.267 + 0.167*ln0.167 + 0.1*ln0.1 + 0.067*ln0.067) ≈ -(0.4×-0.916 + 0.267×-1.321 + 0.167×-1.791 + 0.1×-2.303 + 0.067×-2.703) = -(-0.366 - 0.353 - 0.299 - 0.230 - 0.181) = 1.429.\n\\boxed{1.43}"
        },
        {
            "user": "A population of ferns follows logistic growth with r=0.3/year, K=5000, initial N=100. What is N after 3 years?",
            "assistant": r"Logistic growth: N(t+1) = N(t) + r*N(t)*(1 - N(t)/K).\nYear 1: N1 = 100 + 0.3*100*(1-100/5000) = 100 + 30*0.98 = 129.4.\nYear 2: N2 = 129.4 + 0.3*129.4*(1-129.4/5000) = 129.4 + 38.82*0.974 = 167.2.\nYear 3: N3 = 167.2 + 0.3*167.2*(1-167.2/5000) = 167.2 + 50.16*0.967 = 215.7.\n\\boxed{216}"
        },
        {
            "user": "In a phylogenetic tree, species A and B diverged 15 MYA, A and C 25 MYA. Estimate B-C divergence and explain molecular clock assumption.",
            "assistant": r"Using the molecular clock: divergence time is proportional to genetic distance.\nA-B = 15 MYA, A-C = 25 MYA.\nB-C divergence = 2*(A-MRCA_time) - min_path where MRCA is the most recent common ancestor.\nThe tree topology implies B and C diverged after A split from the (B,C) clade.\nIf A-(B,C) = 22 MYA and B-C split happened later, B-C = 2*25 - (15+25) = 10? Wait, let me redraw.\nThree-taxon statement: d(A,B)=15, d(A,C)=25. By additivity, d(B,C) = d(A,C) + d(A,B) - 2*d(A,X) where X is the A-(B,C) node.\nBut we don't know d(A,X). Simplest: the tree is (A,(B,C)). Then d(B,C) = d(A,C) + d(A,B) - 2*d(A,X).\nWithout d(A,X), we only know d(B,C) ≤ d(A,B) + d(A,C) = 40.\nWith ultrametric tree: d(A,X) = 10, then d(B,C) = 25+15-20 = 20 MYA.\n\\boxed{20 \\text{ MYA (assuming ultrametric tree)}}"
        },
        {
            "user": "A species distribution model for Quercus robur has AUC=0.92. Interpret what this means and identify one limitation.",
            "assistant": r"AUC of 0.92 means the model discriminates between presence and absence locations very well — a randomly chosen presence point ranks higher than a random absence point 92% of the time.\nLimitation: AUC can be inflated by large background extents with trivial absences far from the species range. High AUC ≠ good habitat suitability prediction.\n\\boxed{\\text{Strong discrimination; background extent may inflate AUC}}"
        },
        {
            "user": "Compute the Bray-Curtis dissimilarity between two plots: Plot A = [5, 10, 0, 3], Plot B = [2, 8, 4, 1].",
            "assistant": r"Bray-Curtis = sum(|a_i - b_i|) / sum(a_i + b_i).\n|5-2|=3, |10-8|=2, |0-4|=4, |3-1|=2. Sum diffs = 11.\nSum totals = (5+2)+(10+8)+(0+4)+(3+1) = 7+18+4+4 = 33.\nBC = 11/33 = 0.333.\n\\boxed{0.333}"
        },
        {
            "user": "In a metapopulation with 8 patches, colonization rate c=0.4, extinction rate e=0.15. What fraction of patches is occupied at equilibrium?",
            "assistant": r"Levins model: dp/dt = c*p*(1-p) - e*p. At equilibrium: 0 = c*p*(1-p) - e*p = p*(c*(1-p) - e).\nNonzero solution: c*(1-p) = e → 1-p = e/c → p = 1 - e/c.\np = 1 - 0.15/0.4 = 1 - 0.375 = 0.625.\n\\boxed{0.625}"
        },
        {
            "user": "A lake has 1200 J energy in phytoplankton, 180 J in zooplankton, and 25 J in small fish. Calculate trophic efficiency between each level.",
            "assistant": r"Trophic efficiency = (energy at level N) / (energy at level N-1) × 100%.\nPhyto→Zoo: 180/1200 = 15%.\nZoo→Fish: 25/180 ≈ 13.9%.\nBoth within typical 10-20% range for aquatic ecosystems.\n\\boxed{15\\% \\text{ and } 13.9\\%}"
        },
    ]
    
    result = []
    for i in range(min(count, len(templates) * 10)):
        t = templates[i % len(templates)]
        result.append({
            "messages": [
                {"role": "system", "content": SYSTEM_ECO},
                {"role": "user", "content": t["user"]},
                {"role": "assistant", "content": t["assistant"]},
            ]
        })
    logger.info(f"Generated {len(result)} ecology template examples")
    return result


def load_ecoagent_synthetic(ecoagent_dir: str, count: int):
    """Try to load from ecoagent, fall back to templates."""
    # Try ecoagent
    sys.path.insert(0, os.path.abspath(ecoagent_dir))
    try:
        from ecoagent.training.synthetic import SyntheticGenerator
        gen = SyntheticGenerator()
        examples = gen.generate(count)
        logger.info(f"Loaded {len(examples)} ecoagent synthetic examples")
        return examples
    except (ImportError, AttributeError):
        logger.warning("ecoagent synthetic unavailable, using template fallback")
        return load_eco_templates(count)


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Prepare Nemotron training dataset")
    parser.add_argument("--source", default="mixed", choices=["kaggle", "mixed"],
                        help="Data source: kaggle-only or mixed")
    parser.add_argument("--kaggle-csv", help="Path to Kaggle train.csv")
    parser.add_argument("--ecoagent-dir", default="../ecoagent", help="Path to ecoagent repo")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--kaggle-ratio", type=float, default=0.5,
                        help="Fraction Kaggle when source=mixed")
    parser.add_argument("--max-examples", type=int, default=5000,
                        help="Total examples to generate")
    args = parser.parse_args()

    if args.source == "kaggle":
        if not args.kaggle_csv:
            parser.error("--kaggle-csv required with --source kaggle")
        examples = load_kaggle_csv(args.kaggle_csv, args.max_examples)
    else:
        # Mixed: Kaggle + ecology
        if not args.kaggle_csv:
            parser.error("--kaggle-csv required")
        kaggle_all = load_kaggle_csv(args.kaggle_csv)
        kaggle_count = int(args.max_examples * args.kaggle_ratio)
        eco_count = args.max_examples - kaggle_count
        
        kaggle_samples = random.sample(kaggle_all, min(kaggle_count, len(kaggle_all)))
        eco_samples = load_eco_templates(eco_count)
        examples = kaggle_samples + eco_samples
        random.shuffle(examples)

    # Write JSONL
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    logger.info(f"Wrote {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
