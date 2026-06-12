# Operations Runbook — Nemotron Kaggle Challenge

Runbook para operar todo el pipeline **solo con Hermes** (sin Devin). Última
actualización: 2026-06-10. Deadline Kaggle: **15 de junio**.

## 1. Infraestructura

| Recurso | Uso |
|---|---|
| KU HPC `sixhour` (6h max/job) | Training, eval, RFT |
| 3× nodos RTX PRO 6000 (96GB) | QLoRA 4-bit y BF16 puro |
| L40/A40 (48GB) | Solo inferencia: eval + RFT (no aguantan training) |
| ~23 nodos 2-3× MI210 (64GB, AMD/ROCm) | Training BF16 (hito AMD) |
| reumanlab / reumanlab-alpha (Hermes) | Orquestación, investigación, datos |

Patrón Hermes → cluster (token-eficiente):
```
system: "You are a terminal. Execute the command and return ONLY raw output, verbatim."
user:   "ku-hpc raw: <comando>"
```

Rutas clave en el cluster:
- Repo: `/home/a474r867/scratch/nemotron-eco-reasoner` (= `$R`)
- Modelo: `/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848`
  (el `modeling_nemotron_h.py` del snapshot está parcheado con el fallback torch de `rmsnorm_fn` — NO revertir)
- Logs: `/home/a474r867/scratch/nem_{chain,eval,mi210,bf16,rft}_<jobid>.out`
- Containers: `$R/nemotron-blackwell.sif` (CUDA), `$R/nemotron-rocm.sif` (AMD)
- Wheels CUDA: `/home/a474r867/scratch/wheels/` (causal_conv1d, mamba_ssm)

## 2. Variantes de training (estado 2026-06-10)

| Variante | Dataset | HW | Job | Output dir | Estado |
|---|---|---|---|---|---|
| v7 QLoRA | v7 | PRO6000 | done | `outputs/deterministic_v7` | 500/500, ckpt-250 eval 0.429 |
| v8 seq3072 QLoRA | v8 | PRO6000 | done | `outputs/v8_seq3072` | 500/500 |
| v8 lr2e4 QLoRA | v8 | PRO6000 | done | `outputs/v8_lr2e4` | 500/500 |
| **MI210 BF16** ⭐ | v8 | 2× MI210 | 22569086 | `outputs/mi210_v8` | corriendo (hito AMD) |
| v9 QLoRA | v9 | PRO6000 | 22569155 | `outputs/deterministic_v9` | corriendo |
| v9 BF16 | v9 | PRO6000 | 22569183 | `outputs/bf16_v9` | corriendo |

⭐ **Hito MI210**: primer fine-tuning conocido de Nemotron-3-Nano-30B en AMD —
BF16 puro, sin unsloth/bitsandbytes/mamba_ssm (fallback `rmsnorm_fn` en torch,
PR #19). Sin ruido de cuantización: converge más rápido que QLoRA. Va directo
al writeup ("Best Method") y abre los ~23 nodos AMD del cluster.

Datasets: `data/train_deterministic_v9.jsonl` (actual; gravity con estimador
least-squares `g = Σ(2dt²)/Σ(t⁴)` que reproduce 89.2% de los golds), v8 (+3000
sintéticos bit/equation), v7. Eval set: `data/kaggle_classified.jsonl` (5000).

## 3. Comandos (todo vía `ku-hpc raw:`)

### Monitoreo
```bash
squeue -u a474r867 --format="%i %T %j %M %N" --noheader
tail -30 /home/a474r867/scratch/nem_bf16_<jobid>.out      # loss lines
grep -E "^== |BATCH DONE|TARGET_REACHED|NEED_RESUBMIT|RESUBMITTED" /home/a474r867/scratch/nem_*_<jobid>.out
ls $R/outputs/<variant>/   # checkpoints cada 50 pasos
```
Los trainings son encadenados: se auto-resubmiten hasta TARGET_TOTAL=500.
Si un chain muere sin resubmit, relanzar el MISMO sbatch (hace resume solo).

### Lanzar trainings
```bash
cd $R && git pull   # siempre antes
# QLoRA (PRO6000):
sbatch --job-name=nem-v9 --export=ALL,DATA_PATH=$R/data/train_deterministic_v9.jsonl,OUT_PATH=$R/outputs/deterministic_v9,TARGET_TOTAL=500,STEPS_PER_JOB=250 hpc/nem_chained.slurm
# BF16 PRO6000 (sin cuantización; default v9):
sbatch hpc/nem_bf16_train.slurm
# BF16 2x MI210 (default v8; cambiar DATA_PATH/OUT_PATH para v9):
sbatch --export=ALL,DATA_PATH=$R/data/train_deterministic_v9.jsonl,OUT_PATH=$R/outputs/mi210_v9 hpc/nem_mi210_train.slurm
```

### Eval por checkpoint (L40, ~3-4h con N_PER_CAT=8)
```bash
sbatch --export=ALL,ADAPTER=$R/outputs/<variant>/checkpoint-<N>,N_PER_CAT=8 hpc/nem_eval.slurm
# Resultados: líneas "== <cat>: X/N" en el log + <adapter>/eval_results.json
```
Ojo gravity: el eval usa match exacto; misses por centésimas probablemente
cuentan como acierto en Kaggle (gold 2 decimales sin cero final).

### Rejection sampling (RFT)
```bash
sbatch --export=ALL,ADAPTER=$R/outputs/deterministic_v7/checkpoint-250,SHARD=0,NUM_SHARDS=2 hpc/nem_rft.slurm
sbatch --export=ALL,ADAPTER=$R/outputs/deterministic_v7/checkpoint-250,SHARD=1,NUM_SHARDS=2 hpc/nem_rft.slurm
# Traces aceptados: $R/data/rft/shard<i>.jsonl (mismo schema que train_*.jsonl)
# Para reentrenar: cat data/train_deterministic_v9.jsonl data/rft/shard*.jsonl > data/train_v10.jsonl
```

### Submission a Kaggle
El adapter de un checkpoint es legal si LoRA rank ≤ 32 (todos los nuestros: r=32).
```bash
cd $R/outputs/<variant>/checkpoint-<N> && zip -j /home/a474r867/scratch/submission.zip adapter_model.safetensors adapter_config.json
```
Subir el zip a Kaggle (5 submissions/día, 2 finales). Baseline a batir: 0.64
(v6 ckpt-200). Elegir checkpoint según evals (cuidado overfitting: comparar
ckpt-250 vs ckpt-500; epoch>3 en 500 pasos).

## 4. Decisiones pendientes (qué haría Devin)
1. Recoger evals ckpt-500 vs ckpt-250 → si 500 < 250, hay overfitting → usar ckpt temprano.
2. Evaluar `outputs/mi210_v8/checkpoint-250` y `outputs/bf16_v9/checkpoint-250` en cuanto existan (candidatos más fuertes: BF16 sin ruido de cuantización + dataset v9).
3. Integrar traces RFT en v10 y reentrenar la mejor configuración.
4. Submission diaria con el mejor ckpt disponible; reservar los 2 finales para las mejores variantes BF16.

## 5. Bugs conocidos / trampas
- `--gres=gpu:1` genérico puede caer en nodos AMD → siempre pinear tipo (`pro6000`, `l40`, `mi210`).
- Multi-GPU NVIDIA: usar `device_map={"":0}` (índice físico ≠ ordinal visible).
- L40/A40 NO aguantan training (OOM); solo eval/inferencia.
- seq 4096 OOM incluso en PRO6000 (path torch puro de Mamba) → max 3072.
- MoE dispatch necesita el patch denso (bf16/fp32 mismatch) en eval/RFT — ya está en los scripts.
- MI210: `packing=False`, `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`, `max_memory={0:"46GiB",1:"46GiB"}` (PR #21) o hay OOM en backward.
- Hermes puede resumir en vez de dar output verbatim → para logs largos usar relay GitHub (`alrobles/test`).
- Reportes de investigación de los Hermes: `alrobles/test` bajo `research/`.
