# Milestone: BF16 LoRA fine-tuning of Nemotron-3-Nano-30B on AMD MI210 (and NVIDIA Blackwell) without compiled Mamba kernels

## TL;DR

We fine-tuned **NVIDIA-Nemotron-3-Nano-30B-A3B-BF16** (30B-param hybrid Mamba-2 + MoE
+ Attention model) with LoRA in **full BF16 precision** on **2× AMD MI210 (ROCm)** —
no `bitsandbytes`, no `unsloth`, and **no `mamba_ssm` / `causal_conv1d` CUDA kernels**.
The same code path also runs on **NVIDIA Blackwell RTX PRO 6000** (sm_100), which has
no pre-built Mamba wheels either. The resulting LoRA adapter (rank 32) is **portable
across any BF16-capable GPU** because training never depends on a vendor-specific
compiled kernel.

To our knowledge this is the first reported LoRA fine-tune of this model on AMD data
center GPUs, and it opens the ~23 MI210 nodes on the cluster (plus any ROCm box) to
this competition's training.

## Why it does not work out of the box

Nemotron-3-Nano is a hybrid architecture (52 layers: 23 Mamba-2 + 23 MoE + 6
Attention). Its `modeling_nemotron_h.py` imports the fused RMSNorm kernel from
`mamba_ssm`:

```python
try:
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
except ImportError:
    raise ImportError("mamba-ssm is required by the Mamba model but cannot be imported")
```

Two environments cannot satisfy that import:

1. **AMD MI210 (ROCm)** — `mamba_ssm` and `causal_conv1d` ship CUDA kernels only; there
   is no ROCm build. `bitsandbytes` (and therefore `unsloth`'s 4-bit path) is also
   CUDA-only, so the usual QLoRA recipe is impossible on AMD.
2. **NVIDIA Blackwell (sm_100+)** — the pre-built `mamba_ssm` wheels do not include
   `sm_100` kernels, so the import also fails (or the kernels crash at run time).

So the model raises `ImportError` at load on both, and there is no quantized fallback
on AMD.

## The fix: a pure-torch `rmsnorm_fn` fallback

`rmsnorm_fn` is just gated RMS normalization. We replace the hard `raise` with a small
pure-PyTorch implementation that runs on any backend (CUDA, ROCm, CPU). It is patched
directly into the model snapshot's `modeling_nemotron_h.py` at job start (idempotent —
a no-op if already patched), and the stale `transformers_modules` cache is cleared so
the patched source is re-imported:

```python
except ImportError:
    def rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5, group_size=None, norm_before_gate=False):
        dt = x.dtype
        x = x.float()
        if z is not None and not norm_before_gate:        # gate before norm
            x = x * torch.nn.functional.silu(z.float())
        if group_size is not None and group_size != x.shape[-1]:
            s = x.shape
            xg = x.view(*s[:-1], s[-1] // group_size, group_size)
            xg = xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + eps)
            x = xg.view(s)
        else:
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        out = x * weight.float()
        if bias is not None:
            out = out + bias.float()
        if z is not None and norm_before_gate:            # gate after norm
            out = out * torch.nn.functional.silu(z.float())
        return out.to(dt)
```

It reproduces the kernel's semantics: optional grouped normalization, the SiLU gate
applied either before or after the norm (`norm_before_gate`), and float32 accumulation
cast back to the input dtype. It is slower than the fused Triton kernel but correct, and
the Mamba mixer's other ops (`causal_conv1d`, selective scan) already have torch
fallbacks in the modeling code, so RMSNorm was the only missing piece.

## Memory strategy

The base model is ~60 GB in BF16. LoRA freezes the base weights (no optimizer state for
them), so the budget is base weights + LoRA params + optimizer state for LoRA +
activations.

- **2× MI210 (64 GB each):** `device_map="auto"` with an explicit
  `max_memory={0:"46GiB", 1:"46GiB"}` cap. Without the cap, `auto` packs ~50 GB onto
  GPU0; since backward activations land on the same GPU as their layers, GPU0 then OOMs
  on the first backward. Capping weights at 46 GiB/GPU leaves headroom for activations.
  Also set `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` and `packing=False`.
- **1× Blackwell PRO 6000 (96 GB):** the whole model fits on one card with
  `device_map={"": 0}`; no split needed.

Other knobs shared by both paths: `gradient_checkpointing=True`,
`per_device_train_batch_size=1`, `optim="adamw_torch"`, `packing=False`,
`TORCH_COMPILE_DISABLE=1`.

## LoRA config (Kaggle-legal)

```python
LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
           target_modules={q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj,out_proj})
```

Rank 32 keeps the adapter within the competition's rank ≤ 32 limit. (Note: the BF16
path uses `lora_alpha=64`, a 2:1 alpha:rank ratio; the CUDA QLoRA path in
`nem_chained.slurm` uses `lora_alpha=32`. Adapters trained with different alpha should
not be averaged/merged together.)

## Results

- **Smoke (2× MI210, 10 steps):** ~**39.5 s/step** — comparable to QLoRA on a single
  PRO 6000 (~50 s/step) — loss 4.7 → 4.0, adapter saved. First confirmation the path
  trains end-to-end on AMD.
- **Full training (2× MI210, BF16 v8):** healthy convergence (loss down to ~1.3–1.4,
  token-accuracy ~89%), checkpoints + resume working via the chained-job pattern.
- **Blackwell PRO 6000 (BF16):** same code path, model loads in ~15 s from warm cache,
  trains without the compiled kernels.

## Portability of the adapter

Because neither path links a vendor-specific compiled kernel during training, the LoRA
adapter (`adapter_model.safetensors` + `adapter_config.json`) is a plain PEFT adapter
that loads on any BF16-capable GPU for inference/eval — AMD or NVIDIA. This is what
makes the AMD nodes usable for this competition without changing the submission format.

## Reproduce

- `hpc/nem_mi210_smoke.slurm` — 10-step smoke on 2× MI210.
- `hpc/nem_mi210_train.slurm` — full chained BF16 training on 2× MI210
  (`--gres=gpu:mi210:2`, auto-resubmit `STEPS_PER_JOB` → `TARGET_TOTAL`).
- `hpc/nem_bf16_train.slurm` — same BF16 path on 1× Blackwell PRO 6000
  (`--gres=gpu:pro6000:1`).

Container: ROCm Apptainer image (`nemotron-rocm.sif`) for MI210; CUDA image
(`nemotron-blackwell.sif`) for Blackwell. Both run the identical Python body and the
identical `rmsnorm_fn` patch.

Env knobs (all paths): `DATA_PATH`, `OUT_PATH`, `SEQ_LEN`, `GRAD_ACCUM`, `LR`,
`TARGET_TOTAL`, `STEPS_PER_JOB`.

## Why it matters

- Unlocks ~23 AMD MI210 nodes (and any ROCm box) for fine-tuning a model whose stock
  code hard-requires CUDA-only Mamba kernels.
- Enables **full-precision BF16** LoRA instead of 4-bit QLoRA, removing NF4 quantization
  noise from every forward/backward — a cleaner-gradient training signal.
- The `rmsnorm_fn` fallback is a one-function, dependency-free patch that other
  Nemotron-H / hybrid-Mamba users on AMD or Blackwell can drop in directly.
