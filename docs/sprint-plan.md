# Sprint Plan — Kaggle Nemotron Competition
## Jun 6-15, 2026

### Deadline
- **Entry:** Jun 8 (2 days)
- **Final:** Jun 15 (9 days)

### Submission: LoRA adapter (rank ≤ 32) for Nemotron-3-Nano-30B-A3B-BF16

---

## What Went Well ✓

1. **MoE monkey-patch breakthrough** — After 40+ failures, found the dtype-safe `index_add_` + top-k aggregation fix for Nemotron's hybrid Mamba-2/MoE architecture
2. **Q6000 recipe validated** — 29 GPUs available, zero queue, seq=128/rank=16/fp16 fits in 24GB
3. **A100 loss convergence** — Best run reached loss 4.51 (epoch 2.15) from 18.88 starting point
4. **Dep isolation** — `PYTHONUSERBASE=/tmp` + `--no-deps` pattern prevents torch version conflicts
5. **Multiple checkpoints saved** — 19 adapter checkpoints across 8 Q6000 + 2 A100 runs

## What Went Wrong ✗

1. **A100 TIMEOUT** — Best run (loss 4.51) killed at 5h walltime with ~8.7h remaining. No checkpoint chain.
2. **No completed training** — Zero runs finished all 500 steps. All adapters are intermediate checkpoints.
3. **Dataset fragmentation** — 4 different JSONL files with inconsistent schemas (`prompt/answer` vs `messages`)
4. **Q6000 seq_len=128 too short** — Limits model's ability to reason over long contexts
5. **MI210 still not working** — ROCm + Unsloth compatibility unresolved
6. **Monitor offline** — Cron job stopped Jun 2, no automated tracking
7. **No evaluation** — Never ran `evaluate.py` to measure actual accuracy

## Fixes Implemented (v18)

### 1. Checkpoint Chain (CRITICAL)
- `train_unsloth.py` v18: Added `RESUME_CHECKPOINT` env var support
- SIGUSR1 handler saves emergency checkpoint 120s before walltime kill
- New chain scripts: `train_a100_chain.slurm`, `train_q6k_chain.slurm`
- Array jobs `0-4%1`: each auto-resumes from latest checkpoint
- Total potential training: 5 × 5h = 25h continuous per GPU

### 2. Configurable Hyperparams
- All params via env vars: `MAX_SEQ_LEN`, `LORA_RANK`, `LORA_ALPHA`, `MAX_STEPS`, `USE_FP16`, `SAVE_STEPS`
- Same script works for both A100 (seq=2048, bf16) and Q6000 (seq=128, fp16)

### 3. Dataset Unification
- New unified ecology dataset: 1,586 CoT traces in chat template format
- Uploaded to HuggingFace: `alrobles/ecocoder-scientific-reasoning`
- Needs sync to HPC for training

---

## Three-Front GPU Strategy

| Front | GPU | Count | VRAM | Config | Speed | Status |
|-------|-----|-------|------|--------|-------|--------|
| **A100** | A100 40GB | 1-2 | 40GB | seq=2048, rank=32, bf16 | ~97s/step | ✓ Working, needs chain |
| **Q6000** | RTX Q6000 | 29 | 24GB | seq=128, rank=16, fp16 | ~250s/step | ✓ Working, needs chain |
| **MI210** | MI210 | 72 | 68.7GB | TBD (ROCm) | TBD | ✗ Not working yet |

### A100 Plan
- Submit `train_a100_chain.slurm` — 5 × 5h = 25h max
- Best config: seq=2048, rank=32, 500 steps on `train_final.jsonl` (8,771 examples)
- Expected: Complete 500 steps → final adapter → package submission

### Q6000 Plan
- Submit `train_q6k_chain.slurm` — 5 × 5h = 25h max per chain
- Can run N chains in parallel (29 GPUs available)
- Lower quality (seq=128) but massive parallelism
- Use as backup/ensemble candidate

### MI210 Plan (stretch goal)
- 72 GPUs × 68.7 GB = 4.9 TB total VRAM
- Blocked on: ROCm torch + Unsloth compatibility
- Alternative: Use standard PEFT QLoRA (without Unsloth) — slower but works on ROCm
- Potential game-changer if unlocked

---

## Sprint Timeline

### Day 0 (Jun 6) — NOW
- [x] Audit repo + cluster status
- [x] Fix train_unsloth.py v18 (resume, SIGUSR1, configurable)
- [x] Create A100 + Q6000 chain scripts
- [ ] Push changes to HPC
- [ ] Submit A100 chain job
- [ ] Submit Q6000 chain jobs (×N parallel)
- [ ] Sync unified dataset to HPC

### Day 1 (Jun 7)
- [ ] Monitor training progress
- [ ] Run evaluate.py on best checkpoint
- [ ] Package preliminary submission.zip
- [ ] Debug MI210 Unsloth/ROCm (if time)

### Day 2 (Jun 8) — ENTRY DEADLINE
- [ ] Select best adapter (lowest loss, highest accuracy)
- [ ] Package final submission.zip
- [ ] Submit to Kaggle
- [ ] Continue training for Jun 15 final

### Days 3-9 (Jun 9-15) — FINAL DEADLINE
- [ ] Continue chain training on all fronts
- [ ] Hyperparameter sweep (rank, seq_len, lr)
- [ ] Ensemble evaluation
- [ ] Final submission optimization

---

## Dataset Strategy for HPC

### Current (on HPC)
```
train_merged.jsonl     5,500 examples  {prompt, answer}  ← OLD
train_expanded.jsonl  10,500 examples  {prompt, answer}  ← EXPANDED  
train_final.jsonl      8,771 examples  {prompt, answer}  ← FILTERED
kaggle_5k_train.jsonl  5,000 examples  {prompt, answer}  ← KAGGLE ONLY
ecocoder_cot.jsonl       906 examples  {messages}        ← OLD ECOLOGY
```

### Target
```
train_sprint.jsonl    ~6,586 examples  {messages}  ← UNIFIED
  = 5,000 Kaggle puzzles (chat template with reasoning system prompt)
  + 1,586 ecology CoT traces (from HuggingFace unified dataset)
```

The `messages` format (system/user/assistant) preserves system prompts and is compatible with SFTTrainer's chat template tokenization.
