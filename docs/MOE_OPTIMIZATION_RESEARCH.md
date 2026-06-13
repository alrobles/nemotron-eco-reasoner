# MoE Optimization Research — Nemotron LoRA Fine-Tuning

**Date**: June 13, 2026  
**Competition**: NVIDIA Nemotron Model Reasoning Challenge (deadline June 15)  
**Goal**: Maximize learning signal per optimizer step for 128-expert MoE fine-tuning

## Implemented (Batch 5, step 800→1000)

### MoE Weight Tying
Ties LoRA parameters across all 128 experts. SUM gradients across expert dimension, broadcast. Ties LoRA_A for w1-type (gate_up_proj, up_proj, gate_proj) and LoRA_B for w2-type (down_proj). 5934 parameters tied.

Reference: huikang's 0.86 Kaggle solution (discussion #689915)

### Liger Kernel
`use_liger_kernel=True` in SFTConfig. Fused linear + cross-entropy kernel. -60% memory on CE loss. Mathematically exact.

## Research Findings — Next Round

### MoE-Sieve (arXiv:2603.24044, Mar 2026)
**Selective Expert Tuning** — only tune top-25% most-routed experts per layer.

Key metrics for 128-expert models:
- Per-layer routing CV is 4-5× higher than global CV
- Top 25% experts capture 37-53% of tokens
- 70-73% parameter reduction with NO quality loss
- Cold experts inject gradient NOISE — removing them helps
- Profile with 10% of data recovers same experts (Jaccard ≥ 0.94)
- Synergizes with weight tying: sieve → tie only hot experts

**Pipeline**: profile v11 → count per-layer expert activations → select top-32 → fine-tune with weight tying

### DR-LoRA (arXiv:2601.04823, Jan 2026)
Dynamic rank allocation — start r=8, grow hot experts to r=32. NOT recommended for our setup (1000 steps too short, requires custom training loop).

### Router Freeze-then-Unfreeze
Freeze router during warmup (~20-40% steps), unfreeze with 0.1-0.5× LR. <10 lines of code. High impact, very easy.

### Shared Expert Differential Treatment
Nemotron has 1 shared expert per MoE layer (processes ALL tokens). Options:
- Higher LoRA rank on shared expert (64 vs 32)
- Lower LR on shared expert (more gradient signal = needs less LR)
- OR: freeze routed experts, only tune shared + attention

### Rejected
- **lm_head LoRA**: Overfitting risk, no evidence for reasoning, PEFT/Unsloth exclude by default
- **Cut Cross-Entropy (Apple CCE)**: Requires model forward patching, not supported for Nemotron model_type
- **GRPO two-stage**: Too complex for available time
- **Expert dropout**: Opposite philosophy to MoE-Sieve

## Decision Matrix

| Method | Impact | Difficulty | Status |
|--------|--------|-----------|--------|
| MoE weight tying | HIGH | Medium | ✓ Batch 5 |
| Liger Kernel | Medium | 1 flag | ✓ Batch 5 |
| MoE-Sieve (top-25%) | HIGH | Easy | → Next |
| Router freeze→unfreeze | High | Very easy | → Next |
| Shared expert diff treatment | High | Easy | → Next |
| Expert-specific LR | Medium | Medium | → Future |
