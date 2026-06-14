"""Verify the v13 dataset against gold answers and competition length limits.

Checks:
  - valid JSONL, exactly [system, user, assistant] per row
  - assistant turn has a non-empty trailing \\boxed{...}
  - the boxed answer matches the gold answer in train.csv using the winner's
    compare_answer logic (binary -> exact, numeric -> rel_tol 1e-2, else string)
  - completion <= 7680 tokens and total <= 8192 tokens (competition limits)

Run from huikang dir: uv run --with tokenizers python3 <path>/verify_v13.py
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

from tokenizers import Tokenizer

HUIKANG_DIR = Path("/home/ubuntu/repos/huikang-nemotron")
DATA = Path(
    "/home/ubuntu/repos/nemotron-eco-reasoner/data/train_deterministic_v13.jsonl"
)
TOK = Tokenizer.from_file(str(HUIKANG_DIR / "tokenizer.json"))

SYSTEM_PROMPT = "You are a helpful assistant that solves puzzles step by step."
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)
MAX_COMPLETION_TOKENS = 7680
MAX_TOTAL_TOKENS = 8192
CHAT_OVERHEAD = 24


def n_tok(text: str) -> int:
    return len(TOK.encode(text, add_special_tokens=False).ids)


def extract_boxed(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    non_empty = [m.strip() for m in matches if m.strip()]
    if non_empty:
        return non_empty[-1]
    return matches[-1].strip() if matches else ""


def compare_answer(stored: str, predicted: str) -> bool:
    stored, predicted = stored.strip(), predicted.strip()
    if re.fullmatch(r"[01]+", stored):
        return predicted.lower() == stored.lower()
    try:
        return math.isclose(float(stored), float(predicted), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored.lower()


def main() -> None:
    gold: dict[str, str] = {}
    prompt_to_id: dict[str, str] = {}
    with open(HUIKANG_DIR / "train.csv", newline="") as f:
        for row in csv.DictReader(f):
            gold[row["id"]] = row["answer"]
            prompt_to_id[row["prompt"]] = row["id"]

    sys_tok = n_tok(SYSTEM_PROMPT)
    n = 0
    bad_shape = 0
    no_box = 0
    answer_mismatch: list[str] = []
    over_compl = 0
    over_total = 0
    max_total = 0
    max_compl = 0
    matched_to_gold = 0

    with open(DATA) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            d = json.loads(line)
            ms = d.get("messages", [])
            if len(ms) != 3 or [m["role"] for m in ms] != [
                "system",
                "user",
                "assistant",
            ]:
                bad_shape += 1
                continue
            assistant = ms[2]["content"]
            user = ms[1]["content"]
            box = extract_boxed(assistant)
            if not box:
                no_box += 1
                continue

            # recover problem id by stripping the suffix from the user prompt
            base_user = (
                user[: -len(PROMPT_SUFFIX)] if user.endswith(PROMPT_SUFFIX) else user
            )
            pid = prompt_to_id.get(base_user)
            if pid is not None:
                matched_to_gold += 1
                if not compare_answer(gold[pid], box):
                    if len(answer_mismatch) < 10:
                        answer_mismatch.append(f"{pid}: gold={gold[pid]!r} box={box!r}")

            ctok = n_tok(assistant)
            ttok = sys_tok + n_tok(user) + ctok + CHAT_OVERHEAD
            max_total = max(max_total, ttok)
            max_compl = max(max_compl, ctok)
            if ctok > MAX_COMPLETION_TOKENS:
                over_compl += 1
            if ttok > MAX_TOTAL_TOKENS:
                over_total += 1

    print(f"rows:                 {n}")
    print(f"bad message shape:    {bad_shape}")
    print(f"missing \\boxed{{}}:     {no_box}")
    print(f"matched to gold id:   {matched_to_gold}")
    print(f"answer mismatches:    {len(answer_mismatch)}")
    for m in answer_mismatch:
        print(f"    {m}")
    print(f"completion > 7680:    {over_compl}")
    print(f"total > 8192:         {over_total}")
    print(f"max completion tok:   {max_compl}")
    print(f"max total tok:        {max_total}")
    ok = (
        bad_shape == 0
        and no_box == 0
        and not answer_mismatch
        and over_compl == 0
        and over_total == 0
    )
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
