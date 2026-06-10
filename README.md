# Nemotron Eco-Reasoner — Kaggle Nemotron Reasoning Challenge

**Dataset preparation pipeline for fine-tuning Nemotron-3-Nano on Alice's Wonderland puzzles.**

Repository: `alrobles/nemotron-eco-reasoner`
HuggingFace: [`alrobles/ecocoder-nemotron-kaggle`](https://huggingface.co/datasets/alrobles/ecocoder-nemotron-kaggle)

---

## Quick Reference (for Devin / new contributors)

### The problem

Kaggle evaluation has 9 puzzle categories. Some categories have terrible CoT traces (LLMs can't solve them in one pass). The old pipeline appended correct answers to wrong reasoning — this is harmful.

### The solution

1. **Classify** every puzzle into its 9 evaluation categories
2. **Programmatic solver** for bit_manipulation (brute-force bitwise operations)
3. **Answer-only** format for categories where LLM can't generate correct CoT (symbol-transformation puzzles)
4. **Traditional cryptarithm CoT** from `cryptarithm_cot.jsonl` (250 high-quality traces)
5. **Category-aware consolidation** with balancing

### Dataset files (use the latest)

| File | Traces | Status |
|------|--------|--------|
| `data/train_cot_unified_v5.jsonl` | 7,083 | **CURRENT** — solver + answer-only strategy |
| `data/train_cot_unified.jsonl` | 7,076 | Original (DeepSeek CoT, no categories) |

### Quick commands

```bash
# 1. Classify puzzles
python3 scripts/classify_puzzles.py \
    --input data/kaggle_puzzles_raw.jsonl \
    --output data/kaggle_classified.jsonl

# 2. Run bit manipulation solver (replaces noisy CoT with verified correct traces)
python3 scripts/solve_bit_manipulation.py \
    --input data/bit_manipulation_puzzles.jsonl \
    --output /tmp/bit_solutions.jsonl

# 3. Combine solver + original CoT for bit_manipulation
python3 scripts/combine_bit.py

# 4. Generate guess puzzles (synthetic from deduce)
python3 scripts/generate_guess_puzzles.py

# 5. Final consolidation
python3 scripts/consolidate_final.py \
    --classified data/kaggle_classified.jsonl \
    --cot-file data/train_cot_unified.jsonl \
    --cryptarithm-file data/cryptarithm_cot.jsonl \
    --cryptarithm-guess-file data/cryptarithm_guess_puzzles.jsonl \
    --equation-guess-file data/equation_numeric_guess_puzzles.jsonl \
    --output data/train_cot_unified_v5.jsonl \
    --balance

# 6. Upload to HuggingFace
export HF_TOKEN=$(cat ~/env/hf-token)
hf upload alrobles/ecocoder-nemotron-kaggle \
    data/train_cot_unified_v5.jsonl train_cot_unified_v5.jsonl \
    --repo-type dataset
```

---

## 9 Evaluation Categories

| # | Category | Weight | Eval Acc | Strategy | Train Traces |
|---|----------|--------|----------|----------|-------------|
| 1 | unit_conversion | 18.0% | 98.0% | CoT (91% match) | 817 |
| 2 | bit_manipulation | 17.8% | 84.0% | **Solver** (35% match) | 822 |
| 3 | cipher | 17.1% | 97.5% | CoT (72% match) | 831 |
| 4 | gravity | 16.7% | 98.0% | CoT (99% match) | 850 |
| 5 | numeral | 15.7% | 98.0% | CoT (100% match) | 847 |
| 6 | cryptarithm_deduce | 7.5% | 18.3% | Mixed* | 667 |
| 7 | equation_numeric_deduce | 5.1% | 87.5% | CoT (33% match) | 416 |
| 8 | cryptarithm_guess | 1.5% | 71.4% | Answer-only | 417 |
| 9 | equation_numeric_guess | 0.7% | 42.9% | Answer-only | 416 |

\* cryptarithm_deduce: 250 traditional cryptarithm CoT (verified correct) + 417 symbol-transformation answer-only

---

## Key Findings

### 1. CoT auto-fix is HARMFUL

Original pipeline: DeepSeek generates CoT → 97.6% wrong for cryptarithm_deduce → script appends `\nThe final answer is \boxed{correct}` at end.

**Result**: Model learns to write 37+ steps of wrong reasoning + correct answer. This corrupts training.

**Fix**: For categories where LLM can't generate correct CoT, use answer-only format (or programmatic solver).

### 2. DeepSeek v4 Pro cannot solve these puzzle types

Tried regenerating CoT with specialized prompts for:
- **cryptarithm_deduce**: 0% match rate (symbol-transformation = program synthesis)
- **bit_manipulation**: 0% match rate (105-step search, model gives up)

These are inductive reasoning / search tasks. LLMs cannot do systematic search in a single forward pass.

### 3. Programmatic solver works for bit_manipulation

`scripts/solve_bit_manipulation.py` brute-forces ~30 bitwise operation types (shift, XOR, AND, OR, rotate, GF(2) linear transforms, 2-op combinations).

```
Result: 580/822 puzzles have a discoverable rule
        → 266/822 correct (32.4%) — 3x better than DeepSeek (11.2%)
        → 314/822 overfit (rule fits examples but fails on target)
        → 242/822 no rule found (non-linear operations)
```

### 4. Symbol-transformation puzzles are program synthesis

The cryptarithm_deduce puzzles are NOT substitution ciphers. They involve:
- Unknown operators (`*`, `+`, `-`, `@`, `|`) with secret semantics
- Expression evaluation over a large symbol alphabet
- This is program synthesis over an unknown DSL — NP-hard in general

**Current strategy**: 250 traditional cryptarithm CoT (verified correct) + 417 answer-only.

---

## Scripts Reference

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `classify_puzzles.py` | Classify 5,000 puzzles into 9 categories | `kaggle_puzzles_raw.jsonl` | `kaggle_classified.jsonl` |
| `solve_bit_manipulation.py` | **Solver**: brute-force bitwise operations | `bit_manipulation_puzzles.jsonl` | Solutions JSONL |
| `consolidate_final.py` | **Final** consolidator with category strategy | Multiple sources | `train_cot_unified_v5.jsonl` |
| `consolidate_dataset_v2.py` | Intermediate consolidator (deprecated by final) | — | — |
| `regenerate_bit_cot.py` | DeepSeek bit CoT regeneration | — | **FAILED** (0% match) |
| `regenerate_cryptarithm_cot.py` | DeepSeek cryptarithm CoT regeneration | — | **FAILED** (0% match) |
| `generate_cot_traces.py` | Original DeepSeek CoT generation | Puzzles JSONL | CoT traces |
| `prepare_dataset_kaggle.py` | Kaggle CSV → ShareGPT | `train.csv` | JSONL |
| `submit_kaggle.py` | Package LoRA adapter for submission | Adapter checkpoint | `submission.zip` |
| `train_unsloth.py` | Unsloth QLoRA training | Dataset JSONL | LoRA adapter |

---

## Data Files Reference

```
data/
├── kaggle_puzzles_raw.jsonl              # 5,000 raw puzzles (prompt + answer)
├── kaggle_classified.jsonl               # 5,000 puzzles with category labels
├── bit_manipulation_puzzles.jsonl         # 822 bit manipulation puzzles
├── cryptarithm_deduce_puzzles.jsonl       # 417 symbol-transformation puzzles
├── cryptarithm_guess_puzzles.jsonl        # 417 synthetic (1-2 examples, from deduce)
├── equation_numeric_guess_puzzles.jsonl   # 416 synthetic (1-2 examples)
├── cryptarithm_cot.jsonl                  # 500 CoT traces (250 traditional cryptarithm)
├── train_cot_unified.jsonl                # 7,076 original traces (DeepSeek CoT + ecology)
├── train_cot_unified_v5.jsonl             # 🏆 FINAL: 7,083 traces, all 9 categories
├── ecology_chat.jsonl                     # 2,076 ecology reasoning traces
└── kaggle_5k_train.jsonl                 # Duplicate of puzzles_raw (legacy)
```

---

## CoT Quality by Category (v5 dataset)

```
Category                    Traces   CoT Verified   Format        Quality
─────────────────────────────────────────────────────────────────────────
numeral                       847     847 (100%)     CoT           🟢 Perfect
gravity                       850     838 (99%)      CoT           🟢 Excellent
unit_conversion               817     741 (91%)      CoT           🟢 Excellent
cipher                        831     602 (72%)      CoT           🟢 Good
bit_manipulation              822     287 (35%)      CoT           🟡 Improved (was 11%)
equation_numeric_deduce       416     136 (33%)      CoT           🟡 Noisy
cryptarithm_deduce            667     250 (trad)     CoT + answer  🔴 Mixed
cryptarithm_guess             417     0              answer-only   🔴 Answer-only
equation_numeric_guess        416     0              answer-only   🔴 Answer-only
ecology                      1000     —              CoT           🟢 Ecology
─────────────────────────────────────────────────────────────────────────
TOTAL                        7083    3701 (63%)      —              —
```

---

## Category Strategy Decision Tree

```
Puzzle category?
├── gravity / numeral / unit_conversion / cipher
│   → Keep existing CoT (high quality, >70% match rate)
│
├── bit_manipulation
│   → Run solver (solve_bit_manipulation.py)
│   → Use solver traces where correct (35% match)
│   → Fall back to original CoT for the rest
│
├── equation_numeric_deduce
│   → Keep existing CoT (33% match, best available)
│
├── cryptarithm_deduce
│   → 250 traditional cryptarithm CoT (cryptarithm_cot.jsonl)
│   → 417 symbol-transformation → answer-only
│
├── cryptarithm_guess / equation_numeric_guess
│   → Generate synthetic puzzles (reduce examples from deduce variants)
│   → answer-only format (LLM can't solve)
│
└── ecology
    → Keep existing traces (high quality)
```

---

## Environment

- **DeepSeek API key**: `~/env/deepseek-token` (on reumanlab)
- **HuggingFace token**: `~/env/hf-token` (local)
- **Python**: 3.12+ with `httpx`, `pandas` (for dataset prep)
- **HF CLI**: `hf` (not deprecated `huggingface-cli`)

---

## License

Apache 2.0 — see [LICENSE](LICENSE)
