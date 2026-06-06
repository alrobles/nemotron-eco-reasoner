# Q6000 v15 — Winning Recipe for Nemotron-3-Nano-30B-A3B

## Breakthrough: Jun 6, 2026 04:15 CST

8 Q6000 GPUs training simultaneously with zero OOM.

## Recipe
```bash
#SBATCH --gres=gpu:q6000:1
#SBATCH --mem=128G
export APPTAINER_WRITABLE_TMPFS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install (ALL --no-deps to prevent torchao)
pip install --user --no-cache-dir -q --no-deps "torch==2.5.1"
pip install --user --no-cache-dir -q mamba-ssm causal-conv1d
pip install --user --no-cache-dir -q --no-deps "datasets==4.3.0" bitsandbytes unsloth_zoo
pip install --user --no-cache-dir -q --no-deps unsloth

# Training
max_seq_length=128, r=16, lora_alpha=16, fp16=True
```

## Key Numbers
- 441M trainable params (1.38% of 32B)
- VRAM: 20.7 GB used / 20.8 GB reserved (2.6 GB margin)
- 10500 training examples
- 8+ GPUs training simultaneously

## Lessons Learned (15 iterations)
1. torch==2.5.1 --no-deps (NO torchvision → torchao chain)
2. seq 128 required (2048→1024→512→256→128)
3. LoRA rank 16 (32→16 saves 2.5GB)
4. expandable_segments helps fragmentation
5. fp16 (not bf16, Turing limitation)
6. APPTAINER_WRITABLE_TMPFS=1 prevents squashfuse

## Scaling
29 Q6000 GPUs available, zero queue.
Combined: 29 × 24 GB = 696 GB VRAM.
A100 40GB: 1 training, 18 losses, converging ~5.5-6.0.
