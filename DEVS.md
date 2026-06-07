# Devin — Nemotron Training Handoff

## Current State (Jun 7, 2026 ~17:30 CDT)

**Running on cluster:**
- `gpu_monitor.py` (job 22444337, kbs) — scanning GPUs every 60s → `~/scratch/gpu_state.json`
- `orch_v2.py` (job 22444349, kbs) — adaptive orchestrator, submits `nem_unified.slurm`

**Key files:**
- `hpc/nem_unified.slurm` — UNIFIED template (auto-detects GPU, sets seq/rank)
- `scripts/orch_v2.py` — orchestrator with exponential OOM backoff
- `scripts/gpu_monitor.py` — cluster GPU scanner
- `scripts/smart_swarm.py` — per-node smart submitter
- `docs/ADAPTIVE_SWARM.md` — full architecture docs

**Training:**
- Model: Nemotron-3-Nano-30B-A3B-BF16 (4-bit QLoRA)
- Data: `data/train_final_v2.jsonl` (7086 examples)
- Container: `nemotron-cuda.sif` (torch 2.5.1+cu124)
- Checkpoints: `outputs/unified/` (shared across GPU types)
- Best loss so far: 5.59 (V6, rank=4, seq=48, Q6000 swarm)

## Pitfalls (CRITICAL)

1. **DONT pip install torch** — use containers torch. PyPI torch breaks on mixed cuDNN nodes.
2. **--mem >= 62G** for Nemotron 30B. Model loading spikes to 50-55GB RAM.
3. **/tmp contamination**: use `$SLURM_JOB_ID` in pip path, `rm -rf` before `mkdir`.
4. **PeftModel.from_pretrained()** for resume. torch.load() breaks on torch 2.5.1.
5. **Constant LR** (no cosine reset). Cosine schedule resets every batch → loss plateau.
6. **OOM backoff**: exponential (5min → 10min → 20min → ... → 1h max).

## Cluster GPU Inventory

| GPU | VRAM | Idle | Config |
|-----|------|------|--------|
| Pro6000 Blackwell | 96 GB | 5 | seq=2048, rank=32, bf16 |
| Q6000 | 24 GB | 18 | seq=48, rank=4, fp16 |
| V100 | 32 GB | 2 | seq=64, rank=8, fp16 |
| A100 | 40 GB | 0 (busy) | seq=1024, rank=16, bf16 |

## Useful Commands

```bash
# Status
squeue -u a474r867
tail -f ~/scratch/orch_v2.log
tail -f ~/scratch/gpu_monitor.log

# Loss tracking
grep -h "train_loss" ~/scratch/nem_unified_*.out | tail -10
grep -h "SUCCESS" ~/scratch/nem_unified_*.out | wc -l

# GPU state
cat ~/scratch/gpu_state.json | python3 -m json.tool

# Kill all our training
scancel -u a474r867 --name=nem

# Resubmit orchestrator
sbatch hpc/orch_v2.slurm
```

## Phase 2 Plan (when loss plateaus at ~5)

1. Wait for batch completion
2. Load best checkpoint → `merge_and_unload()`
3. Submit Pro6000-only jobs with seq=2048, rank=32
4. Target loss ~2-3
