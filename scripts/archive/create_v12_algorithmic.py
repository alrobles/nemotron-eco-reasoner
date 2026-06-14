"""Generate v12 dataset using huikang's algorithmic reasoning traces.

v12 strategy:
- Use ALL 9,500 pre-generated algorithmic CoT from tonghuikang/nemotron
- Format in our training JSONL format (messages=[system, user, assistant])
- The CoT is deterministic (code-generated), not LLM-hallucinated
- This is THE key differentiator: winner scored 0.877 with this approach

Source: /home/ubuntu/repos/huikang-nemotron/reasoning/*.txt (9,500 files)
Output: data/train_deterministic_v12.jsonl
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

# Paths
HUIKANG_DIR = Path("/home/ubuntu/repos/huikang-nemotron")
TRAIN_CSV = HUIKANG_DIR / "train.csv"
PROBLEMS_INDEX = HUIKANG_DIR / "problems.jsonl"
REASONING_DIR = HUIKANG_DIR / "reasoning"
OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "train_deterministic_v12.jsonl"

# System prompt — keep it simple and consistent with our pipeline
SYSTEM_PROMPT = "You are a helpful assistant that solves puzzles step by step."

# Prompt suffix used by competition eval (must match inference format)
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

# Max sequence length filter (seq3072 training)
MAX_CHARS = 12000  # rough filter: ~3072 tokens * ~4 chars/token


def extract_answer_from_reasoning(reasoning_text: str) -> str:
    """Extract the answer from \\boxed{...} in reasoning text."""
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", reasoning_text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()
    return ""


def main() -> None:
    # Load problem prompts from train.csv
    prompts: dict[str, str] = {}
    answers: dict[str, str] = {}
    with open(TRAIN_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["id"]
            prompts[pid] = row["prompt"]
            answers[pid] = row["answer"]

    # Load problem categories
    problem_cats: dict[str, str] = {}
    with open(PROBLEMS_INDEX) as f:
        for line in f:
            d = json.loads(line)
            problem_cats[d["id"]] = d["category"]

    # Process all reasoning files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stats: dict[str, dict[str, int]] = {}
    examples: list[dict] = []
    skipped_no_prompt = 0
    skipped_no_answer = 0
    skipped_too_long = 0

    reasoning_files = sorted(REASONING_DIR.glob("*.txt"))
    print(f"Found {len(reasoning_files)} reasoning files")

    for reasoning_path in reasoning_files:
        pid = reasoning_path.stem

        # Must have prompt from train.csv
        if pid not in prompts:
            skipped_no_prompt += 1
            continue

        # Read reasoning
        reasoning_text = reasoning_path.read_text().rstrip("\n")

        # Extract answer from reasoning's \boxed{}
        reasoning_answer = extract_answer_from_reasoning(reasoning_text)
        if not reasoning_answer:
            # Fallback to train.csv answer
            reasoning_answer = answers.get(pid, "")
        if not reasoning_answer:
            skipped_no_answer += 1
            continue

        # Format assistant response (matches inference format)
        assistant_content = (
            f"<think>\n{reasoning_text}\n</think>\n\\boxed{{{reasoning_answer}}}"
        )

        # Format user prompt (with suffix, matching eval)
        user_content = prompts[pid] + PROMPT_SUFFIX

        # Check length
        total_chars = len(SYSTEM_PROMPT) + len(user_content) + len(assistant_content)
        if total_chars > MAX_CHARS:
            skipped_too_long += 1
            continue

        # Build training example
        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        }
        examples.append(example)

        # Track stats
        cat = problem_cats.get(pid, "unknown")
        if cat not in stats:
            stats[cat] = {"count": 0, "total_chars": 0}
        stats[cat]["count"] += 1
        stats[cat]["total_chars"] += len(assistant_content)

    # Write output
    with open(OUTPUT_FILE, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Report
    print(f"\n{'='*60}")
    print(f"v12 Dataset Generated: {OUTPUT_FILE}")
    print(f"{'='*60}")
    print(f"Total examples: {len(examples)}")
    print(f"Skipped (no prompt): {skipped_no_prompt}")
    print(f"Skipped (no answer): {skipped_no_answer}")
    print(f"Skipped (too long):  {skipped_too_long}")
    print(f"\nPer-category breakdown:")
    for cat in sorted(stats.keys()):
        s = stats[cat]
        avg_chars = s["total_chars"] // s["count"] if s["count"] else 0
        print(f"  {cat:30s}: {s['count']:5d} examples (avg {avg_chars:5d} chars)")
    print(f"\nTotal file size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
