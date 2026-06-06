# Bitácora de Mejora — Nemotron en KU-HPC

## Sesión Jun 5-6, 2026

### Camino de destrabado (cada fallo → solución)

| # | Qué falló | Causa raíz | Solución |
|---|---|---|---|
| 1 | Torch 2.10 rompe mamba-ssm | pip install unsloth sube torch | --no-deps en unsloth |
| 2 | ~/.local contamina todo | Python carga user site-packages primero | PYTHONUSERBASE=/tmp aislado |
| 3 | datasets 4.5 recursion | Unsloth incompatible | Pin datasets==4.3.0 |
| 4 | torchao → torch.int1 | torchvision jala torchao como dep | --no-deps en torch (sin torchvision) |
| 5 | bitsandbytes faltante | --no-deps saltó el dep | Instalación explícita |
| 6 | MoE dtype bf16 vs fp32 | Nemotron bug en cuantización | Monkey-patch 23 capas decoder |
| 7 | MoE double .view() | Reshape duplicado caller+patch | Quitar .view() del patch |
| 8 | MoE top-k sin agrupar | Nemotron rutea a 6 expertos | Sum(dim=1) para agregar |
| 9 | Bash quoting roto | Comillas simples en bash -c | Heredoc << 'EOF' |
| 10 | Lustre buffering | Python stdout no flushea | PYTHONUNBUFFERED=1 |
| 11 | Squashfuse timeout (Q6000) | Modelo tarda >60s en cargar | APPTAINER_WRITABLE_TMPFS=1 |
| 12 | OOM seq 2048 (Q6000 24GB) | 24GB no alcanza | Bajar seq: 2048→1024→512→256→128 |
| 13 | OOM en LM head | Capa final explota | seq 128 + rank 16 |
| 14 | OOM en backward | Gradientes no caben | rank 16 + expandable_segments |
| 15 | MI210 Python 3.9 | str\|Path no soportado | Python 3.11 del sistema |
| 16 | MI210 sin torch | venv --system-site-packages no jala | pip install torch --index-url rocm6.2 |

### Estado actual (Jun 6 05:00 CST)

| GPU | Cantidad | Estado | Config |
|---|---|---|---|
| A100 40GB | 1 | TIMEOUT 5h, 18 losses, loss 5.25 | seq 2048, rank 32 |
| Q6000 | 20 | ENTRENANDO | v15: seq 128, rank 16 |
| Q6000 | 10 | ENTRENANDO | v16: seq 64, rank 16 |
| MI210 | 1 | PROBANDO | v4: Python 3.11 + torch ROCm |
| A100 DDP | 2 | PENDIENTE | 516-job queue |

### Receta ganadora Q6000
```bash
APPTAINER_WRITABLE_TMPFS=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install --no-deps torch==2.5.1  # sin torchvision!
pip install mamba-ssm causal-conv1d
pip install --no-deps datasets==4.3.0 bitsandbytes unsloth_zoo unsloth
# Training: seq 128, rank 16, fp16, 10500 ejemplos
```

### Próximos hitos
- [ ] MI210 funcional (68.7 GB VRAM × 72 GPUs = 4.6 TB)
- [ ] A100 DDP multi-GPU
- [ ] Ensemble multi-arquitectura
- [ ] DeepSeek-V3 en Q6000×24
