"""Generate the v13 dataset: verified, deterministic, length-safe CoT.

Design (see DESIGN_V13.md for the full rationale). The three levers over v12:

  1. VERIFICATION FILTER. Keep ONLY problems the winner's algorithmic solver
     verified as correct (problems.jsonl status == "rule_found"). v12 mechanically
     converted all 9,500 traces, including ~1,167 where the solver failed
     (status "rule_unknown"/"hypothesis_formed") -- those teach WRONG answers,
     mostly in cryptarithm (which is only ~8% solvable and which over-represented
     itself in v9 -> the 0.46 regression).

  2. LENGTH SAFETY against the real competition limits:
        max_model_len = 8192   (prompt + completion)
        max_tokens    = 7680   (generation budget)
     The winner's deterministic CoT for bit_manipulation (~7000 tok) and equation
     (~5900 tok) does NOT fit in our previous seq3072/4096 training -> bit_manip
     (the winner's #1 differentiator) was being TRUNCATED entirely, so the model
     never saw the \\boxed{} answer. This is the most likely cause of our 0.67
     ceiling. v13 keeps only examples whose completion <= 7680 tok and whose
     total <= 8192 tok, and is meant to be TRAINED AT seq_len=8192.

  3. SELF-CONSISTENT ANSWERS. The final \\boxed{} uses the answer the reasoning
     actually derives (matches corpus.py); for rule_found that is within the
     competition tolerance of the gold answer.

Format matches our proven v8 SFT data (system/user/assistant, <think> baked into
the assistant turn), which scored 0.67 -- we change the CoT CONTENT and the
verification/length filtering, not the wrapper.

Run: cd <repo> && uv run --with tokenizers python3 scripts/create_v13_verified.py
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from tokenizers import Tokenizer

# --- paths -----------------------------------------------------------------
HUIKANG_DIR = Path("/home/ubuntu/repos/huikang-nemotron")
TRAIN_CSV = HUIKANG_DIR / "train.csv"
PROBLEMS_INDEX = HUIKANG_DIR / "problems.jsonl"
REASONING_DIR = HUIKANG_DIR / "reasoning"
TOKENIZER_JSON = HUIKANG_DIR / "tokenizer.json"

OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "train_deterministic_v13.jsonl"

# --- format (identical to v8 / v12 wrapper) --------------------------------
SYSTEM_PROMPT = "You are a helpful assistant that solves puzzles step by step."
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

# --- competition length limits ---------------------------------------------
MAX_COMPLETION_TOKENS = 7680  # vLLM max_tokens at inference (generation budget)
MAX_TOTAL_TOKENS = 8192  # vLLM max_model_len (prompt + completion)
CHAT_OVERHEAD = 24  # chat-template special tokens (im_start/im_end/role markers)

TOK = Tokenizer.from_file(str(TOKENIZER_JSON))


def n_tok(text: str) -> int:
    return len(TOK.encode(text, add_special_tokens=False).ids)


def extract_boxed(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    non_empty = [m.strip() for m in matches if m.strip()]
    if non_empty:
        return non_empty[-1]
    return matches[-1].strip() if matches else ""


def top_category(sub: str) -> str:
    if sub.startswith("cryptarithm"):
        return "cryptarithm"
    if sub.startswith("equation_numeric"):
        return "equation_numeric"
    return sub


def main() -> None:
    prompts: dict[str, str] = {}
    answers: dict[str, str] = {}
    with open(TRAIN_CSV, newline="") as f:
        for row in csv.DictReader(f):
            prompts[row["id"]] = row["prompt"]
            answers[row["id"]] = row["answer"]

    status: dict[str, str] = {}
    sub_cat: dict[str, str] = {}
    for line in PROBLEMS_INDEX.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        status[d["id"]] = d.get("status", "?")
        sub_cat[d["id"]] = d["category"]

    sys_tokens = n_tok(SYSTEM_PROMPT)

    examples: list[dict] = []
    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "compl_tok": 0, "max_tok": 0}
    )
    skipped = {
        "not_rule_found": 0,
        "no_prompt": 0,
        "no_reasoning_file": 0,
        "no_answer": 0,
        "too_long": 0,
    }

    for pid, st in status.items():
        if st != "rule_found":
            skipped["not_rule_found"] += 1
            continue
        if pid not in prompts:
            skipped["no_prompt"] += 1
            continue
        rf = REASONING_DIR / f"{pid}.txt"
        if not rf.exists():
            skipped["no_reasoning_file"] += 1
            continue

        reasoning = rf.read_text().rstrip("\n")
        answer = extract_boxed(reasoning) or answers.get(pid, "")
        if not answer:
            skipped["no_answer"] += 1
            continue

        assistant = f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}"
        user = prompts[pid] + PROMPT_SUFFIX

        compl_tok = n_tok(assistant)
        total_tok = sys_tokens + n_tok(user) + compl_tok + CHAT_OVERHEAD
        if compl_tok > MAX_COMPLETION_TOKENS or total_tok > MAX_TOTAL_TOKENS:
            skipped["too_long"] += 1
            continue

        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            }
        )
        cat = top_category(sub_cat[pid])
        s = stats[cat]
        s["count"] += 1
        s["compl_tok"] += compl_tok
        s["max_tok"] = max(s["max_tok"], total_tok)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("=" * 70)
    print(f"v13 dataset written: {OUTPUT_FILE}")
    print("=" * 70)
    print(f"Total examples: {len(examples)}")
    print("Skipped:")
    for k, v in skipped.items():
        print(f"  {k:18s} {v}")
    print(
        f"\n{'category':18s} {'n':>5s} {'%':>6s} {'avg_compl':>10s} {'max_total':>10s}"
    )
    tot = len(examples)
    for cat in sorted(stats):
        s = stats[cat]
        avg = s["compl_tok"] // s["count"] if s["count"] else 0
        print(
            f"{cat:18s} {s['count']:5d} {100 * s['count'] / tot:5.1f}% "
            f"{avg:10d} {s['max_tok']:10d}"
        )


if __name__ == "__main__":
    main()
