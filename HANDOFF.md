# Nemotron-3-Nano-30B Training — Handoff para Hermes/reumanlab

## Estado Actual (12 Jun 2026, ~17:00 UTC)

### Situación Crítica
**Los 3 últimos submissions a Kaggle bajaron del baseline 0.64:**
- test submission v1: **0.46**
- BF16 PRO6000 ckpt-250 loss=7.5093: **0.57**
- MI210 V8 BF16 ckpt-500 loss=1.27 acc=89.9%: **0.54**

### Causa Raíz Identificada
El dataset v9 estaba **masivamente desbalanceado** vs la distribución del test de Kaggle:

| Categoría         | Kaggle test | v9 (malo)  | v10 (fix)  |
|-------------------|-------------|------------|------------|
| cryptarithm       | 16.7%       | **44.5%**  | 16.7%      |
| bit_manipulation  | 16.4%       | **22.0%**  | 16.7%      |
| gravity           | 17.0%       | 7.8%       | 16.7%      |
| cipher            | 16.6%       | 8.6%       | 16.7%      |
| numeral           | 16.9%       | 8.7%       | 16.7%      |
| unit_conversion   | 16.3%       | 8.4%       | 16.7%      |

Los modelos sobreajustaron a cryptarithm (44.5%) perdiendo capacidad en gravity/cipher/numeral/unit_conversion que son el 67% del test.

### Fix Aplicado
**Dataset v10 balanceado** (`data/train_balanced_v10.jsonl`):
- 4,896 examples, exactamente 816 por categoría (16.7% uniforme)
- Prioriza traces de v6 (probados con score 0.64)
- Suplementa con v9 solo para cryptarithm (v6 tenía solo 271) y bit_manipulation
- PR #26 merged a master

### Jobs Activos (12 Jun 2026)

| Job ID | Nombre | Script | Dataset | GPU | Estado |
|--------|--------|--------|---------|-----|--------|
| **22604245** | nem-v10 | nem_chained.slurm (QLoRA) | v10 balanced | PRO6000 Blackwell 102GB | RUNNING |
| **22605231** | nem-v6repro | nem_chained.slurm (QLoRA) | v6 original | PRO6000 | PENDING |
| **22615506** | nem-v10-bf16 | nem_bf16_train.slurm (BF16) | v10 balanced | PRO6000 | PENDING |
| **22615768** | nem-v10-mi210 | nem_mi210_train.slurm (BF16) | v10 balanced | 2x MI210 | PENDING |

### Configuraciones de Training

**QLoRA (nem_chained.slurm) — RECOMENDADO, produjo el 0.64:**
- 4-bit quantization + LoRA rank=32, alpha=32
- seq_len=2048, lr=1e-4, constant_with_warmup
- grad_accum=8, batch_size=1
- Container: nemotron-blackwell.sif
- 250 steps/job, auto-resubmit hasta 500

**BF16 (nem_bf16_train.slurm):**
- Full precision, LoRA rank=32, alpha=64 (scaling factor 2x!)
- seq_len=2048, lr=1e-4, grad_accum=4
- Container: nemotron-blackwell.sif

**MI210 (nem_mi210_train.slurm):**
- BF16, LoRA rank=32, 2x MI210 GPUs
- Container: nemotron-rocm.sif
- seq_len=2048, lr=1e-4, grad_accum=4

---

## Kaggle — 6 Categorías del Test (5,000 puzzles)

```
gravity           850 (17.0%)
numeral           847 (16.9%)
cryptarithm_deduce 833 (16.7%)
cipher            831 (16.6%)
bit_manipulation  822 (16.4%)
unit_conversion   817 (16.3%)
```

Cada categoría es ~16.7%. El dataset de entrenamiento DEBE reflejar esta distribución.

---

## Paths en el Cluster (hpc.crc.ku.edu)

```bash
# Repo
R=/home/a474r867/scratch/nemotron-eco-reasoner

# Modelo pre-cacheado (NO descargar)
M=/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848

# SIF containers
$R/nemotron-blackwell.sif   # PRO6000 Blackwell (15G)
$R/nemotron-cuda.sif        # A100/L40/A40/Q8000/V100 (7G)
$R/nemotron-rocm.sif        # MI210 AMD (21G)
$R/nemotron-mi210.sif       # MI210 alternativo (12G)

# Datasets
$R/data/train_balanced_v10.jsonl      # ← USAR ESTE (4,896 ex, balanceado)
$R/data/train_deterministic_v6.jsonl  # Baseline original (4,230 ex, 0.64 score)
$R/data/train_deterministic_v9.jsonl  # DESBALANCEADO, NO USAR
$R/data/kaggle_5k_train.jsonl         # Eval data (5,000 puzzles con answers)

# Training outputs
$R/outputs/balanced_v10/     # ← v10 QLoRA (job 22604245)
$R/outputs/v6_repro/         # ← v6 QLoRA reproducción (job 22605231)
$R/outputs/bf16_v10/         # ← v10 BF16 (job 22615506)
$R/outputs/mi210_v10/        # ← v10 MI210 (job 22615768)
$R/outputs/deterministic_v6/ # Anterior (v6 QLoRA, 500 steps, eval 0.429)
$R/outputs/deterministic_v9/ # Anterior desbalanceado
$R/outputs/bf16_v9/          # Anterior desbalanceado
$R/outputs/mi210_v8/         # Anterior (loss=1.27, Kaggle=0.54)
```

---

## Cómo Operar

### 1. Monitorear jobs
```bash
ssh kuhpc "squeue -u a474r867"
```

### 2. Ver progreso de un job
```bash
ssh kuhpc "tail -20 /home/a474r867/scratch/nem_chain_JOBID.out"
ssh kuhpc "tail -20 /home/a474r867/scratch/nem_chain_JOBID.err"
# Para BF16:
ssh kuhpc "tail -20 /home/a474r867/scratch/nem_bf16_JOBID.out"
# Para MI210:
ssh kuhpc "tail -20 /home/a474r867/scratch/nem_mi210_JOBID.out"
```

### 3. Ver loss de un job
```bash
ssh kuhpc "grep -E '(loss|step|Step|DONE|MANIFEST)' /home/a474r867/scratch/nem_chain_JOBID.out | tail -20"
```

### 4. Ver manifest (resumen final de un job)
```bash
ssh kuhpc "cat /home/a474r867/scratch/nemotron-eco-reasoner/outputs/balanced_v10/manifest-JOBID.json"
```

### 5. Lanzar nuevo job QLoRA v10
```bash
ssh kuhpc "cd /home/a474r867/scratch/nemotron-eco-reasoner && sbatch --job-name=nem-v10 --export=ALL,DATA_PATH=$R/data/train_balanced_v10.jsonl,OUT_PATH=$R/outputs/balanced_v10,TARGET_TOTAL=500,STEPS_PER_JOB=250 hpc/nem_chained.slurm"
```

### 6. Lanzar BF16 v10
```bash
ssh kuhpc "cd /home/a474r867/scratch/nemotron-eco-reasoner && sbatch --job-name=nem-v10-bf16 --export=ALL,DATA_PATH=$R/data/train_balanced_v10.jsonl,OUT_PATH=$R/outputs/bf16_v10,TARGET_TOTAL=500,STEPS_PER_JOB=200 hpc/nem_bf16_train.slurm"
```

### 7. Lanzar MI210 v10
```bash
ssh kuhpc "cd /home/a474r867/scratch/nemotron-eco-reasoner && sbatch --job-name=nem-v10-mi210 --export=ALL,DATA_PATH=$R/data/train_balanced_v10.jsonl,OUT_PATH=$R/outputs/mi210_v10,TARGET_TOTAL=500,STEPS_PER_JOB=200 hpc/nem_mi210_train.slurm"
```

### 8. Preparar submission.zip
```bash
# Encontrar el mejor checkpoint (menor loss)
ssh kuhpc "for d in /home/a474r867/scratch/nemotron-eco-reasoner/outputs/balanced_v10/checkpoint-*/; do echo \$d; done"

# Crear zip desde el checkpoint
ssh kuhpc "cd /home/a474r867/scratch/nemotron-eco-reasoner && python3 scripts/submit_kaggle.py outputs/balanced_v10/checkpoint-STEP/"
```

### 9. Subir a Kaggle (desde reumanlab main, ya autenticado)
```bash
kaggle competitions submit \
  -c nvidia-nemotron-model-reasoning-challenge \
  -f ~/submission.zip \
  -m "v10 balanced QLoRA ckpt-250"
```

---

## Pitfalls CRÍTICOS

1. **NO pip install torch** — destruye el torch del container
2. **--mem >= 62G** para Nemotron 30B (48G mínimo con QLoRA)
3. **/tmp contamination** — usar $SLURM_JOB_ID en pip path
4. **PeftModel.from_pretrained()** para resume (no torch.load)
5. **Constant LR** (no cosine reset) — cosine resetea cada batch
6. **OOM backoff**: exponencial (5→10→20→...→1h max)
7. **NO usar dataset v9** directamente — está desbalanceado, solo v10 o v6
8. **BF16 alpha=64 vs QLoRA alpha=32** — configs diferentes, no mezclar checkpoints

---

## Estrategia de Submission (Deadline: 15 Jun 2026)

1. **Prioridad 1**: Cuando job 22604245 (v10 QLoRA) termine ckpt-250 → submit inmediato
2. **Prioridad 2**: Si ckpt-250 > baseline 0.64 → dejar correr a ckpt-500
3. **Prioridad 3**: Comparar v10 QLoRA vs v10 BF16 vs v10 MI210
4. **Reservar 2 submissions finales** para los mejores variantes
5. **Si todos bajan**: volver a v6 puro con QLoRA 200 steps (el baseline probado)

### Checkpoints a evaluar (por orden de prioridad)
- `outputs/balanced_v10/checkpoint-250` (QLoRA v10, el más prometedor)
- `outputs/v6_repro/checkpoint-200` (reproducción del baseline 0.64)
- `outputs/bf16_v10/checkpoint-200` (BF16 v10)
- `outputs/mi210_v10/checkpoint-200` (MI210 v10)

### Evaluación offline antes de submit
```bash
ssh kuhpc "cd /home/a474r867/scratch/nemotron-eco-reasoner && sbatch --export=ALL,CKPT_PATH=$R/outputs/balanced_v10/checkpoint-250 hpc/nem_eval.slurm"
```

---

## Conexión SSH

Hermes en reumanlab-alpha ahora puede conectarse al HPC:
```bash
ssh kuhpc "comando"
# Funciona directamente — key copiada desde reumanlab main
```

### Kaggle CLI
Autenticado en reumanlab main (OAuth). Para submissions, copiar submission.zip al reumanlab main y usar `kaggle competitions submit`.

---

## Checklist para Continuar

- [ ] Monitorear job 22604245 (v10 QLoRA) — esperar ckpt-250 y revisar loss
- [ ] Monitorear jobs 22605231, 22615506, 22615768 cuando arranquen
- [ ] Cuando ckpt-250 esté listo: preparar submission.zip y subir a Kaggle
- [ ] Comparar loss de v10 vs v6-repro para validar que el balanceo ayuda
- [ ] Si score > 0.64: dejar correr a ckpt-500 y evaluar overfitting
- [ ] Si score <= 0.64: investigar format mismatch o probar lr=5e-5
- [ ] Reservar 2 submissions finales para deadline Jun 15

---

*Generado por Devin — sesión fcb86a4c871646ada4868c1b432ea608 — 12 Jun 2026*
