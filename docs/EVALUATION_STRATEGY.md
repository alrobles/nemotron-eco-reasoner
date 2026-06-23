# Evaluation Strategy: Ecological & Scientific Reasoning

## Overview

This document defines the evaluation framework for assessing reasoning quality
in models fine-tuned on ecological and biological science data. It combines
external validated benchmarks with our own domain-specific internal metrics.

---

## 1. External Benchmarks

### 1.1 General Scientific Reasoning

| Benchmark | Size | Level | Format | What it tests |
|-----------|------|-------|--------|---------------|
| **GPQA Diamond** | ~500 | Graduate/expert | MC + CoT | Deep STEM knowledge (bio, chem, physics) |
| **MMLU-Pro** (bio/science) | ~2K+ | College+ | MC | Biology, environmental science subsets |
| **ARC-Challenge** | 2,590 | Moderate | MC | Reasoning over recall |
| **HumanityLastExam (HLE)** | 2,500+ | Research-level | Open | Frontier STEM knowledge |

### 1.2 Ecological / Biodiversity

| Benchmark | Size | Level | Format | What it tests |
|-----------|------|-------|--------|---------------|
| **CURIE** (ICLR 2025) | 580 pairs, 429 papers | Expert | Long-context + figures | Scientific reasoning over real papers; **includes biodiversity as a discipline** |
| **ScienceAgentBench** | 102 tasks | Expert | Agent-based | End-to-end scientific workflows across 4 disciplines |

### 1.3 Discovery & Agent-Based

| Benchmark | What it evaluates |
|-----------|------------------|
| **LLM-SRBench** (ICML 2025) | Equation discovery from data (anti-memorization) |
| **ReplicatorBench** | Replicating published scientific studies end-to-end |
| **DISCOVERYWORLD** | Agents doing scientific discovery in simulation |

---

## 2. Internal Metric: EcoReason-Eval

### 2.1 Design

Five-dimensional evaluation specific to ecological reasoning quality:

| Dimension | What it measures | Evaluation method |
|-----------|-----------------|-------------------|
| **Factual Accuracy** | Are ecological facts correct? | LLM-as-judge vs PubMed ground truth |
| **Reasoning Chain** | Are step 1->N logically valid and causal? | Step-level coherence scoring (CaSE-style) |
| **Code Correctness** | Does generated R/Python code execute? | Sandbox execution (pass/fail + output check) |
| **Method Appropriateness** | Is the statistical method correct for the problem? | Exact match vs method in the original paper |
| **Ecological Grounding** | Does response reference real ecological concepts? | Keyword/concept extraction + coverage score |

### 2.2 Scoring

Each dimension scores 0-1. Composite score:

```
EcoReason = 0.25 * factual + 0.20 * reasoning + 0.20 * code + 0.20 * method + 0.15 * grounding
```

Weights emphasize factual accuracy (hard to recover from wrong facts) and balance
reasoning/code/method equally, with ecological grounding as a softer signal.

### 2.3 Eval Set Construction

- **200 examples** stratified from `ecoreasoner-cot-20k` by method
- Each example has: user question, reference answer (from dataset), method label
- Run model inference, then score each dimension
- Compare: fine-tuned model vs base Nemotron-3-Nano-30B vs DeepSeek-R1

### 2.4 LLM-as-Judge Prompt (Factual Accuracy)

```
You are an expert ecologist evaluating AI-generated responses.

Given:
- QUESTION: {question}
- REFERENCE ANSWER: {reference}
- MODEL ANSWER: {model_output}

Score the MODEL ANSWER on factual accuracy (0-1):
- 1.0: All ecological facts are correct and consistent with the reference
- 0.7: Minor inaccuracies that don't affect conclusions
- 0.4: Some significant errors but overall approach is sound
- 0.0: Fundamentally incorrect ecological claims

Return JSON: {"score": <float>, "justification": "<brief>"}
```

---

## 3. Key Preprints (2025)

### DeepResearch^Eco (arXiv:2505.14279)
Recursive multi-agent LLM workflow for ecological synthesis. Uses GPT-4 as judge
across rigor, domain expertise, and concept integration. Finding: deep recursion
(111 sources) increases conceptual coverage ~25%. Relevant for our multi-step
reasoning evaluation.

### APEF (arXiv:2505.13794)
LLM-based policy extraction for ecological modeling evaluation. Extracts eval
criteria from expert feedback for GPP/CO2 flux models. Relevant for our
"method appropriateness" dimension.

### FineLogic / CaSE (2025)
Step-level reasoning evaluation frameworks. FineLogic decomposes reasoning chains
into individual logical steps and scores each. CaSE evaluates causal structure.
Both inform our "reasoning chain" dimension.

---

## 4. Implementation Plan

### Phase 1: Baseline (immediate)
1. Build eval set (200 examples from ecoreasoner-cot-20k, stratified by method)
2. Run base Nemotron-3-Nano-30B on eval set
3. Run fine-tuned adapter (best from ablation study) on eval set
4. LLM-as-judge scoring (factual accuracy + method appropriateness)
5. Report: base vs fine-tuned on 5 dimensions

### Phase 2: Code Execution (next sprint)
1. Sandbox environment for R/Python code execution
2. Automated code extraction from model outputs
3. Pass/fail + output validation
4. Integration with EcoReason composite score

### Phase 3: External Benchmarks (validation)
1. CURIE biodiversity subset — long-context ecological reasoning
2. MMLU-Pro biology — knowledge baseline
3. GPQA Diamond — graduate-level scientific reasoning
4. Compare our fine-tuned model against published baselines

---

## 5. Dataset for Training

The balanced training dataset (`data/balanced_train.jsonl`) targets both
reasoning capability and biological science focus:

| Category | Count | Percentage | Sources |
|----------|-------|------------|---------|
| Ecology | 6,000 | 40.4% | ecoreasoner-cot-20k, ecocoder-scientific-reasoning, ecocoder-cot |
| Biology | 3,750 | 25.3% | Brainquiver/reasoning-biology, camel-ai/biology |
| Reasoning | 3,000 | 20.2% | nemotron-eco-reasoner-v14 (puzzles) |
| Physics | 2,100 | 14.1% | nemotron-reasoning-v3 |
| **Total** | **14,850** | **100%** | 7 sources |

Key properties:
- **100% have `<think>` reasoning traces** (wrapped if originally missing)
- **45.7% include code** (R/Python)
- **Avg assistant response: 4,910 chars** (substantial reasoning depth)
- Ecology capped at 500/method to prevent MaxEnt dominance
- Biology split: 60% Brainquiver (reasoning) + 40% camel-ai (fundamentals)

Build command:
```bash
python3 scripts/build_balanced_dataset.py \
    --output data/balanced_train.jsonl \
    --target-size 15000 \
    --eco-method-cap 500 \
    --seed 42
```
