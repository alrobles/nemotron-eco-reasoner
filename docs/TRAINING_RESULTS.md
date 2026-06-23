# Nemotron-3-Nano-30B Ecological Reasoning — Training Results

## Summary

Five ablation configurations trained for 500 steps on PRO6000 Blackwell GPUs (95GB)
using the `ecoreasoner-cot-20k` dataset (14,156 filtered examples). Training used
a two-phase approach: aggressive Phase 1 (lr=2e-4, no gradient clipping, steps 0-350)
followed by precise Phase 2 "nudge" (lr=5e-6, gradient clipping=1.0, steps 350-500).

All techniques adapted from the [3rd place Kaggle solution](https://github.com/YS-L/nvidia-nemotron-reasoning-3rd-place-solution) (score 0.900).

## Final Results (Step 500)

| Config | Phase 1 Loss | Phase 2 Loss | Improvement | Adapter Size | Job IDs |
|--------|-------------|-------------|-------------|-------------|---------|
| **default** | 3.594 | **3.574** | -0.020 | 4.0 GB | 23127571 → 23143968 |
| no NEFTune | 3.594 | 3.575 | -0.019 | 4.0 GB | 23125561 → 23144125 |
| long nudge | 3.594 | 3.575 | -0.019 | 4.0 GB | 23125563 → 23144126 |
| high LR | **3.568** | 3.579 | -0.011 | 4.0 GB | 23127572 → 23144127 |
| no lm_head | 3.596 | 3.575 | -0.021 | 3.3 GB | 23129050 → 23144128 |

**Best overall: default** (lr=2e-4, NEFTune=5.0, lm_head included, NUDGE_FRAC=0.3)

## Ablation Analysis

### Default (all techniques enabled)
- LR: 2e-4, NEFTune alpha: 5.0, lm_head: yes, NUDGE_FRAC: 0.3
- Best Phase 2 loss (3.574) — all Kaggle 3rd-place techniques working together
- Phase 2 nudge started at global step 350 (step 100 of Phase 2 job)

### No NEFTune (alpha=0)
- Identical to default except NEFTune embedding noise disabled
- Near-identical loss (3.575 vs 3.574) — NEFTune had minimal effect at this scale
- May show difference in downstream evaluation (robustness, not loss)

### Long Nudge (NUDGE_FRAC=0.5)
- Phase 2 nudge started at global step 250 (immediately when Phase 2 job began)
- Entire Phase 2 ran at low lr=5e-6 with gradient clipping
- Similar loss to default — the extra nudge steps didn't improve over shorter nudge

### High LR (5e-4)
- Best Phase 1 loss (3.568) — faster initial convergence
- Worst Phase 2 loss (3.579) — overshooting made refinement harder
- Conclusion: higher LR helps early but hurts precision in the nudge phase

### No lm_head
- LoRA applied to q/k/v/o/gate/up/down/in/out but NOT lm_head
- Smaller adapter (3.3 GB vs 4.0 GB) — 17.5% size reduction
- Near-identical loss (3.575) — lm_head LoRA adds size without clear loss benefit
- 12,008 tensors loaded (vs 12,011 for others)

## Training Configuration

```
Model:          NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (30B total / ~3.5B active, MoE)
Dataset:        ecoreasoner-cot-20k → 14,156 examples (filtered)
GPU:            PRO6000 Blackwell (95GB VRAM, sm_120)
Venv:           torch 2.12.0.dev+cu128, Python 3.11
Sequence length: 2048
Batch size:     1 per GPU, gradient accumulation 4
LoRA rank/alpha: 32/32
LoRA targets:   q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj,
                in_proj, out_proj [+ lm_head for most configs]
LoRA precision: fp32 (LORA_FP32=1)
Loss:           completion-only (prompt tokens masked)
Phase 1:        lr=2e-4, no gradient clipping, steps 0-350
Phase 2:        lr=5e-6, gradient clipping=1.0, steps 350-500
NEFTune:        alpha=5.0 (except no_neftune config)
Peak VRAM:      ~87 GB / 96 GB available
Training time:  ~130-150 min per phase per job (~31s/step)
```

## Checkpoints

All ablations have checkpoints at steps 50, 100, 150, 200, 250, and 500:

```
/home/a474r867/scratch/nemotron-eco-reasoner/outputs/
├── eco_v2_twophase/       (default)
├── eco_v2_no_neftune/     (NEFTune disabled)
├── eco_v2_long_nudge/     (NUDGE_FRAC=0.5)
├── eco_v2_high_lr/        (lr=5e-4)
└── eco_v2_no_lmhead/      (lm_head excluded from LoRA)
    └── checkpoint-{50,100,150,200,250,500}/
        └── adapter_model.safetensors (3.3-4.0 GB each)
```

## Key Technical Fix: PEFT Adapter Resume

Phase 2 required resuming training from Phase 1 checkpoints. The native PEFT
`resume_from_checkpoint` path crashed with:
```
TypeError: WeightConverter.__init__() got unexpected keyword argument 'distributed_operation'
```

**Root cause**: PEFT's `save_pretrained` strips the adapter name (`.default.`)
from parameter keys when saving. When loading into a fresh model, parameters
include `.default.`, causing a key mismatch.

**Fix** (PR #51, merged): `_expand_key()` function in `eco_train_v2.py` generates
all candidate key variants (with/without `.default.`, with/without `base_model.model.`
prefix) to robustly match checkpoint keys to model parameters:

```python
def _expand_key(k):
    variants = [k]
    for lora_part in ("lora_A.", "lora_B."):
        if lora_part in k and ".default." not in k:
            variants.append(k.replace(lora_part, f"{lora_part}default."))
    result = []
    for v in variants:
        result.append(v)
        if v.startswith("base_model.model."):
            result.append(v[len("base_model.model."):])
        else:
            result.append(f"base_model.model.{v}")
    return result
```

Result: 12,011/12,011 tensors loaded successfully for all ablations.

## Infrastructure Notes

- **PRO6000 Blackwell** requires `torch nightly+cu128` (standard PyTorch <2.7 lacks sm_100/sm_120 kernels)
- **Blackwell venv**: `/home/a474r867/scratch/nemotron-blackwell-venv/`
- **Model cache**: `/home/a474r867/scratch/nemotron-model-cache/` (avoids re-downloading 60GB)
- **Memory management**: checkpoint loaded to CPU first, then copied tensor-by-tensor to GPU to avoid OOM
- **Port collisions**: `MASTER_PORT` randomized per job to avoid DDP conflicts on shared nodes
- **DataLoader**: `num_workers=0` required (shared memory `/dev/shm` too small on compute nodes)

## Next Steps

1. **Evaluate adapters** on held-out ecological reasoning tasks
2. **Compare with base model** to quantify fine-tuning benefit
3. **Test adapter merging** — combine best aspects of different ablations
4. **Scale training** — more steps, larger batch size, or curriculum learning
5. **Deploy via Hermes** for interactive ecological reasoning queries
