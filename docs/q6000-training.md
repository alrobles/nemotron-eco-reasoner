# Q6000 Nemotron Training

## Status: VALIDATED (Jun 6, 2026)

4 of 5 Q6000 v9 jobs successfully loaded and started training Nemotron-3-Nano-30B-A3B.

## Recipe

```bash
#SBATCH --gres=gpu:q6000:1
#SBATCH --mem=128G
export APPTAINER_WRITABLE_TMPFS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install deps (ALL pinned, no upgrades)
pip install --user --no-cache-dir -q "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1"
pip install --user --no-cache-dir -q mamba-ssm causal-conv1d
pip install --user --no-cache-dir -q --no-deps "datasets==4.3.0" bitsandbytes unsloth_zoo unsloth

# Training: Unsloth 4-bit QLoRA, seq_len=512, fp16 (no bf16 on Turing)
model = FastLanguageModel.from_pretrained(M, max_seq_length=512, load_in_4bit=True)
```

## GPU Comparison

| GPU | VRAM | Count | Availability | Speed |
|---|---|---|---|---|
| A100 80GB | 80 GB | 18 | 516-job queue | 96s/it |
| Q6000 | 24 GB | 29 | Instant | ~250s/it |

## Key Breakthroughs

1. `APPTAINER_WRITABLE_TMPFS=1` — prevents squashfuse timeout
2. `torch==2.5.1` pinned — prevents mamba-ssm symbol mismatch
3. `--no-deps` on unsloth/bitsandbytes/unsloth_zoo — prevents torchao/torch 2.10
4. `max_seq_length=512` — fits in 23.46 GB (v9 OOM at 1024)
5. `fp16=True` (not bf16) — Turing doesn't support bfloat16
6. MoE monkey-patch required (same as A100 recipe)
