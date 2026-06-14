# Hermes Overnight Experiment Grid — 14 Jun 2026

## Instrucciones para Hermes

Ejecuta estos comandos en secuencia vía `ssh kuhpc`. Cada `sbatch` lanza un job independiente en el cluster. Los jobs se auto-encadenan (250 steps → resubmit automático hasta target).

---

## Paso 0: Pull v12 dataset al cluster

```bash
ssh kuhpc "cd /home/a474r867/scratch/nemotron-eco-reasoner && git pull origin master"
```

Verificar que `data/train_deterministic_v12.jsonl` existe (~52MB).

---

## Paso 1: Malla de Experimentos v12

El dataset v12 tiene **9,482 ejemplos con CoT algorítmico determinístico** (del winner tonghuikang, 0.877). Esta es la pieza clave para superar 0.67.

### Experimento A: v12 QLoRA seq3072 (PRINCIPAL — replica config de 0.67)
```bash
ssh kuhpc "cd /home/a474r867/scratch/nemotron-eco-reasoner && sbatch --job-name=nem-v12-3072 --export=ALL,DATA_PATH=/home/a474r867/scratch/nemotron-eco-reasoner/data/train_deterministic_v12.jsonl,OUT_PATH=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/v12_seq3072,TARGET_TOTAL=500,STEPS_PER_JOB=250,SEQ_LEN=3072,RANK=32,GRAD_ACCUM=4,LR=1e-4,TAG=v12-3072 hpc/nem_chained.slurm"
```

### Experimento B: v12 QLoRA seq4096 (más contexto, 0% truncation)
```bash
ssh kuhpc "cd /home/a474r867/scratch/nemotron-eco-reasoner && sbatch --job-name=nem-v12-4096 --export=ALL,DATA_PATH=/home/a474r867/scratch/nemotron-eco-reasoner/data/train_deterministic_v12.jsonl,OUT_PATH=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/v12_seq4096,TARGET_TOTAL=500,STEPS_PER_JOB=250,SEQ_LEN=4096,RANK=32,GRAD_ACCUM=4,LR=1e-4,TAG=v12-4096 hpc/nem_chained.slurm"
```

### Experimento C: v12 QLoRA seq3072 lr=5e-5 (learning rate más bajo para no sobreajustar)
```bash
ssh kuhpc "cd /home/a474r867/scratch/nemotron-eco-reasoner && sbatch --job-name=nem-v12-lr5e5 --export=ALL,DATA_PATH=/home/a474r867/scratch/nemotron-eco-reasoner/data/train_deterministic_v12.jsonl,OUT_PATH=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/v12_lr5e5,TARGET_TOTAL=500,STEPS_PER_JOB=250,SEQ_LEN=3072,RANK=32,GRAD_ACCUM=4,LR=5e-5,TAG=v12-lr5e5 hpc/nem_chained.slurm"
```

### Experimento D: v12 QLoRA seq3072 grad_accum=8 (effective batch 8, más estable)
```bash
ssh kuhpc "cd /home/a474r867/scratch/nemotron-eco-reasoner && sbatch --job-name=nem-v12-ga8 --export=ALL,DATA_PATH=/home/a474r867/scratch/nemotron-eco-reasoner/data/train_deterministic_v12.jsonl,OUT_PATH=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/v12_ga8,TARGET_TOTAL=500,STEPS_PER_JOB=250,SEQ_LEN=3072,RANK=32,GRAD_ACCUM=8,LR=1e-4,TAG=v12-ga8 hpc/nem_chained.slurm"
```

---

## Paso 2: Control con v8 (baseline probado, 0.67)

### Experimento E: v8 QLoRA seq3072 (reproducción exacta del 0.67)
```bash
ssh kuhpc "cd /home/a474r867/scratch/nemotron-eco-reasoner && sbatch --job-name=nem-v8-repro --export=ALL,DATA_PATH=/home/a474r867/scratch/nemotron-eco-reasoner/data/train_deterministic_v8.jsonl,OUT_PATH=/home/a474r867/scratch/nemotron-eco-reasoner/outputs/v8_repro,TARGET_TOTAL=500,STEPS_PER_JOB=250,SEQ_LEN=3072,RANK=32,GRAD_ACCUM=4,LR=1e-4,TAG=v8-repro hpc/nem_chained.slurm"
```

---

## Paso 3: Monitoreo

Cada 30 minutos, verificar estado:
```bash
ssh kuhpc "squeue -u a474r867 --format='%.8i %.12j %.8T %.6M %.6D %R' | head -20"
```

Cuando un job termine (desaparezca de squeue), verificar loss:
```bash
ssh kuhpc "for d in /home/a474r867/scratch/nemotron-eco-reasoner/outputs/v12_*/manifest-*.json; do echo \"--- \$d ---\"; cat \$d 2>/dev/null; done"
```

---

## Paso 4: Preparar submissions (cuando checkpoints estén listos)

Para cada checkpoint exitoso, crear submission.zip:
```bash
# Ejemplo para v12_seq3072 checkpoint-250
ssh kuhpc "cd /home/a474r867/scratch/nemotron-eco-reasoner/outputs/v12_seq3072 && BEST=\$(ls -td checkpoint-*/ | grep -v sigusr1 | head -1) && echo \"Best: \$BEST\" && cd \$BEST && zip -r /home/a474r867/scratch/nemotron-eco-reasoner/submission_v12_3072.zip adapter_config.json adapter_model.safetensors"
```

Luego copiar a reumanlab para submit:
```bash
scp a474r867@hpc.crc.ku.edu:/home/a474r867/scratch/nemotron-eco-reasoner/submission_v12_3072.zip ~/submission_v12_3072.zip
kaggle competitions submit -c nvidia-nemotron-model-reasoning-challenge -f ~/submission_v12_3072.zip -m "v12 algorithmic CoT seq3072 ckpt-250"
```

---

## Prioridad de Submit (Kaggle deadline: 15 Jun 2026)

1. **v12_seq3072 ckpt-250** — principal (misma config que 0.67, pero con CoT algorítmico)
2. **v12_seq4096 ckpt-250** — más contexto para bit_manipulation (8.8K chars avg)
3. **v12_seq3072 ckpt-500** — si ckpt-250 mejora, ver si más steps ayuda
4. **v12_lr5e5 ckpt-500** — lr conservador, menos overfitting
5. **v8_repro ckpt-500** — control para confirmar que v8 sigue en 0.67

---

## Notas Importantes

- **v12 = 9,482 ejemplos** con CoT del winner (tonghuikang, 0.877 score)
- **v8 = 9,771 ejemplos** con CoT genérico LLM (nuestro baseline 0.67)
- La diferencia clave: CoT algorítmico vs CoT heurístico
- Los jobs se auto-encadenan (250→500 steps automáticamente)
- Si un job falla por OOM: la secuencia se detiene. Verificar que seq4096 no excede 48GB
- Para seq4096 puede necesitar `--mem=62G`: si falla, re-lanzar con más memoria

---

## GPUs Disponibles

| GPU | Partición | VRAM | GRES flag |
|-----|-----------|------|-----------|
| RTX PRO 6000 Blackwell | sixhour | 102GB | `pro6000` |
| AMD MI210 | sixhour | 64GB | `mi210` |
| A100 | sixhour | 80GB | `a100` |
| A40 | sixhour | 48GB | `a40` |

PRO6000 es la más rápida para este modelo (supports BF16 + flash attention). Lanzar en PRO6000 primero.
