# Nemotron-3-Nano-30B Training — Handoff (14 Jun 2026)

## Estado Actual

### Mejor Score: 0.67 (v8_seq3072 ckpt-500)
### Warmstart subiendo: v8 500→700 seq=4096 (resultado pendiente)
### v12 LISTO para entrenar — dataset pushed, PR #29 merged

---

## Análisis de la Barrera 0.67

**Problema diagnosticado:** No es hyperparams ni dataset size — es **calidad del Chain-of-Thought**.

| Enfoque | CoT | Score |
|---------|-----|-------|
| Nuestro v8 (genérico) | "Hypothesis: reverse bit order" → verify → apply | 0.67 |
| Nuestro v11 (más genérico) | Más ejemplos del mismo estilo | 0.54 |
| Winner tonghuikang | Solvers algorítmicos per-category (1005 líneas para bit_manipulation!) | 0.877 |

**Conclusión:** Más datos genéricos = PEOR. Datos algorítmicos determinísticos = MEJOR.

---

## Dataset v12 — LA APUESTA PRINCIPAL

**Fuente:** `tonghuikang/nemotron` (winner, 0.877 score, repo público)
**Script:** `scripts/create_v12_algorithmic.py`
**Output:** `data/train_deterministic_v12.jsonl` (9,482 examples, 52MB)

### Per-category reasoning approach:
| Categoría | Método algorítmico | Avg chars | Nuestro score actual |
|-----------|-------------------|-----------|---------------------|
| bit_manipulation | Exhaustive bit-op matching (8 ops × 8 bits) | 8,812 | 12.5% (1/8) |
| equation_numeric | 32-operator systematic deduction | 10,552 | 25% (2/8) |
| cipher | Character-by-character mapping | 5,331 | 75% (6/8) |
| gravity | Long division/multiplication + truncate_3dp | 4,180 | 0% eval (PASA Kaggle) |
| unit_conversion | Long multiplication/division chains | 3,145 | 87.5% (7/8) |
| cryptarithm | Letter-to-digit constraint solving | 1,371 | 0% (0/8) |
| numeral | Roman numeral greedy decomposition | 303 | 100% (8/8) |

### Diferencia clave vs v8:
```
v8 bit_manipulation CoT: "Hypothesis: reverse the bit order" (heurístico, 624 chars)
v12 bit_manipulation CoT: Enumera TODAS las operaciones posibles por bit,
    encuentra runs consistentes, selecciona la regla correcta (8,812 chars)
```

---

## Malla de Experimentos (OVERNIGHT)

Ver `HERMES_OVERNIGHT_GRID.md` para el prompt completo de Hermes.

| Exp | Dataset | seq_len | LR | grad_accum | Output dir |
|-----|---------|---------|-----|-----------|------------|
| A | v12 | 3072 | 1e-4 | 4 | v12_seq3072 |
| B | v12 | 4096 | 1e-4 | 4 | v12_seq4096 |
| C | v12 | 3072 | 5e-5 | 4 | v12_lr5e5 |
| D | v12 | 3072 | 1e-4 | 8 | v12_ga8 |
| E | v8 | 3072 | 1e-4 | 4 | v8_repro (control) |

Todos con: QLoRA rank=32, alpha=32, PRO6000 Blackwell, 500 steps auto-chained.

---

## Paths en el Cluster (hpc.crc.ku.edu)

```bash
R=/home/a474r867/scratch/nemotron-eco-reasoner

# Datasets (en orden de prioridad)
$R/data/train_deterministic_v12.jsonl   # ← v12 algorítmico (9,482 ex) *** USAR ***
$R/data/train_deterministic_v8.jsonl    # ← v8 baseline (9,771 ex, 0.67 score)
$R/data/train_balanced_v10.jsonl        # v10 balanceado (4,896 ex)
$R/data/train_deterministic_v6.jsonl    # v6 original (4,230 ex)

# NO USAR:
$R/data/train_deterministic_v9.jsonl    # DESBALANCEADO
$R/data/train_deterministic_v11.jsonl   # DAÑINO (0.54 score)

# Training outputs (nuevos)
$R/outputs/v12_seq3072/    # Exp A
$R/outputs/v12_seq4096/    # Exp B
$R/outputs/v12_lr5e5/      # Exp C
$R/outputs/v12_ga8/        # Exp D
$R/outputs/v8_repro/       # Exp E (control)

# Training outputs (anteriores)
$R/outputs/v8_seq3072/            # MEJOR hasta ahora (0.67, ckpt-500)
$R/outputs/v8_seq4096_warm/       # Warmstart 500→700 (subiendo a Kaggle)
$R/outputs/balanced_v10/          # v10 QLoRA
$R/outputs/deterministic_v6/      # v6 original
```

---

## Cómo Operar

### Monitorear jobs
```bash
ssh kuhpc "squeue -u a474r867 --format='%.8i %.12j %.8T %.6M %.6D %R'"
```

### Ver loss de un job
```bash
ssh kuhpc "grep -E '(loss|DONE|SUCCESS|BATCH)' /home/a474r867/scratch/nem_chain_JOBID.out | tail -20"
```

### Ver manifest (resumen final)
```bash
ssh kuhpc "cat $R/outputs/v12_seq3072/manifest-*.json"
```

### Crear submission
```bash
ssh kuhpc "cd $R/outputs/v12_seq3072 && BEST=\$(ls -td checkpoint-*/ | grep -v sigusr1 | head -1) && cd \$BEST && zip -r $R/submission_v12.zip adapter_config.json adapter_model.safetensors"
```

### Submit a Kaggle (desde reumanlab main)
```bash
scp a474r867@hpc.crc.ku.edu:$R/submission_v12.zip ~/submission_v12.zip
kaggle competitions submit -c nvidia-nemotron-model-reasoning-challenge -f ~/submission_v12.zip -m "v12 algorithmic CoT seq3072"
```

---

## Pitfalls CRÍTICOS

1. **NO pip install torch** — destruye el torch del container
2. **--mem >= 48G** para QLoRA, **62G** para seq4096
3. **NO usar dataset v9 ni v11** — ambos dañan el score
4. **v12 bit_manipulation avg 8.8K chars** → seq3072 trunca ~5%, seq4096 no trunca
5. **Jobs se auto-encadenan** (250→500 steps). No relanzar manualmente si aún están en squeue
6. **BF16 alpha=64 vs QLoRA alpha=32** — no mezclar checkpoints entre configs

---

## Estrategia de Submission (Deadline: 15 Jun 2026, ~24h restantes)

1. **Warmstart v8 seq4096** — ya subiendo, resultado pendiente
2. **v12_seq3072 ckpt-250** → submit apenas termine (~4-5h de training)
3. **v12_seq4096 ckpt-250** → submit si seq3072 no mejora (más contexto para bit_manipulation)
4. **v12 ckpt-500** → si ckpt-250 mejora, dejar correr y submit ckpt-500
5. **Fallback:** v8_repro ckpt-500 como control

### Resultado esperado:
- Si el CoT algorítmico es lo que falta → v12 debería saltar de 0.67 a 0.75+
- El salto debería venir principalmente de bit_manipulation (12.5% → ~75%+) y equation (25% → ~60%+)
- Si v12 NO mejora → el problema es formato de respuesta o el training loop, no los datos

---

## Conexión SSH

Hermes en reumanlab-alpha → SSH a kuhpc funciona directamente.
```bash
ssh kuhpc "comando"
```

Kaggle CLI autenticado en reumanlab main (OAuth).

---

*Actualizado por Devin — 14 Jun 2026*
