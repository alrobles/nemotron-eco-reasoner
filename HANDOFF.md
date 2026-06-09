# Nemotron-3-Nano-30B Training — Handoff para Hermes/reumanlab

## Estado Actual (7 Jun 2026, 23:30 UTC)

### ✅ Completado
1. **5,000 CoT traces** generados via DeepSeek API (65.3% exact match)
2. **Dataset unificado**: `data/train_cot_unified.jsonl` — 7,076 traces (5K CoT + 2K ecology)
3. **Orquestador v3** (`scripts/orch_v3.py`) — targeting inteligente por GPU + OOM backoff
4. **SLURM template** (`hpc/nem_unified.slurm`) — auto-detecta GPU y configura seq/rank/precision
5. **Blackwell .def** corregido (`hpc/nemotron-blackwell.def`) — rm stale dirs, non-fatal builds
6. **MI210 .def** corregido (`hpc/nemotron-mi210-rocm.def`) — ROCm 7.2.2, non-fatal verification
7. **OOM fix**: Q6000 ahora usa seq=128, rank=4, grad_accum=4 (antes 256/4/8 → OOM)
8. **Investigación de entrenamiento paralelo** — reporte completo entregado

### 🔨 En Progreso
- **3 jobs corriendo** en Q6000 (22447637, 22447638, 22447639) — Unsloth cargado, training iniciando
- **12 jobs pendientes** en cola (V100 + Q6000)
- **nemotron-blackwell.sif** subiendo al cluster (~2% al momento de este doc)

### ⏳ Pendiente
- Construir + subir `nemotron-mi210.sif`
- Upload dataset a HuggingFace (`alrobles/nemotron-reasoning-cot`)
- Lanzar orch_v3 en modo continuo
- Evaluar adapters y subir el mejor a Kaggle

---

## Estructura del Repo

```
nemotron-eco-reasoner/
├── data/
│   └── train_cot_unified.jsonl          # 7,076 training traces (32MB)
├── hpc/
│   ├── nem_unified.slurm               # Template SLURM principal (auto-config por GPU)
│   ├── nemotron-blackwell.def           # Apptainer def para Blackwell Pro6000 (sm_120)
│   ├── nemotron-mi210-rocm.def          # Apptainer def para MI210 (ROCm 7.2.2)
│   ├── rebuild_blackwell_sif.sh         # Script alternativo para build Blackwell
│   ├── nem_blackwell_test.slurm         # Test job para verificar SIF Blackwell
│   └── nem_mi210_test.slurm            # Test job para verificar SIF MI210
├── scripts/
│   ├── orch_v3.py                       # Orquestador adaptativo (targeting + OOM backoff)
│   ├── train_unsloth.py                 # Script de training standalone
│   ├── generate_cot_traces.py           # Generador de CoT via DeepSeek API
│   └── consolidate_dataset.py           # Consolida batches en dataset unificado
└── outputs/
    └── unified/                         # Checkpoints de training (en cluster)
```

**Branch activo:** `devin/1780776984-checkpoint-chain-sprint`

---

## Paths en el Cluster (hpc.crc.ku.edu)

```
# Repo
/home/a474r867/scratch/nemotron-eco-reasoner/

# Modelo pre-cacheado (NO descargar de nuevo)
/home/a474r867/scratch/nemotron-model-cache/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848

# SIF containers
/home/a474r867/scratch/nemotron-eco-reasoner/nemotron-cuda.sif       # ✅ Funciona (Q6000/Q8000/V100)
/home/a474r867/scratch/nemotron-eco-reasoner/nemotron-blackwell.sif  # ⏳ Subiendo
/home/a474r867/scratch/nemotron-eco-reasoner/nemotron-mi210.sif      # ⏳ Pendiente build

# Training outputs
/home/a474r867/scratch/nemotron-eco-reasoner/outputs/unified/

# Job logs
/home/a474r867/scratch/nem_unified_*.out
/home/a474r867/scratch/nem_unified_*.err
```

---

## Cómo Operar (Comandos Exactos)

### 1. Sync del repo en el cluster
```bash
cd /home/a474r867/scratch/nemotron-eco-reasoner && git pull
```

### 2. Lanzar el enjambre (una vez)
```bash
cd /home/a474r867/scratch/nemotron-eco-reasoner
python3 scripts/orch_v3.py --once --target-jobs 20
```

### 3. Lanzar el enjambre (continuo, cada 90s)
```bash
cd /home/a474r867/scratch/nemotron-eco-reasoner
nohup python3 scripts/orch_v3.py --target-jobs 20 --interval 90 > ~/scratch/orch_v3_live.log 2>&1 &
```

### 4. Monitorear jobs
```bash
squeue -u a474r867 --format="%i %T %N %j %M" --noheader
```

### 5. Ver loss de un job
```bash
grep -E "(loss|TRAINING|DONE)" /home/a474r867/scratch/nem_unified_JOBID.out
```

### 6. Construir SIF MI210 (en reumanlab-terminal con sudo)
```bash
cd /path/to/nemotron-eco-reasoner
git checkout devin/1780776984-checkpoint-chain-sprint && git pull
sudo apptainer build /tmp/nemotron-mi210.sif hpc/nemotron-mi210-rocm.def
scp /tmp/nemotron-mi210.sif a474r867@hpc.crc.ku.edu:/home/a474r867/scratch/nemotron-eco-reasoner/nemotron-mi210.sif
```

### 7. Construir SIF Blackwell (si necesitas reconstruir)
```bash
sudo apptainer build /tmp/nemotron-blackwell.sif hpc/nemotron-blackwell.def
scp /tmp/nemotron-blackwell.sif a474r867@hpc.crc.ku.edu:/home/a474r867/scratch/nemotron-eco-reasoner/nemotron-blackwell.sif
```

### 8. Upload dataset a HuggingFace
```bash
export HF_TOKEN=<tu token con write access>
huggingface-cli upload alrobles/nemotron-reasoning-cot data/train_cot_unified.jsonl train_cot_unified.jsonl --repo-type dataset
```

### 9. Evaluar el mejor adapter (después de que los jobs terminen)
```bash
# Listar todos los checkpoints con su loss
for m in /home/a474r867/scratch/nemotron-eco-reasoner/outputs/unified/manifest-*.json; do
  python3 -c "import json; d=json.load(open('$m')); print(f\"{d['jobid']:>10} {d['tag']:>6} loss={d['final_loss']}  {d['gpu']} seq={d['seq']} r={d['rank']}\")"
done | sort -t= -k2 -n
```

### 10. Subir adapter a Kaggle
```bash
# El adapter con loss más bajo está en outputs/unified/checkpoint-JOBID-stepN/
# Contiene: adapter_model.safetensors, adapter_config.json
# Subir a Kaggle como dataset o modelo según las instrucciones del competition
```

---

## Auto-Config por GPU (en nem_unified.slurm)

| GPU | VRAM | seq_len | rank | grad_accum | precision | Tag |
|-----|------|---------|------|------------|-----------|-----|
| Q6000/V100 | <40GB | 128 | 4 | 4 | fp16 | LIGHT |
| Q8000 | 40-55GB | 512 | 8 | 8 | fp16 | MEDIUM |
| MI210 | 55-90GB | 1024 | 16 | 8 | bf16 | HEAVY |
| Blackwell Pro6000 | 90GB+ | 2048 | 32 | 8 | bf16 | BEAST |

La detección es automática — el script lee `torch.cuda.get_device_properties(0).total_memory` y configura todo solo.

---

## Orquestador v3 — Cómo Funciona

1. **Escanea** nodos disponibles via `sinfo`
2. **Identifica** tipo de GPU por nombre (q6000, pro6000, mi210, v100, etc.)
3. **Selecciona SIF** apropiado (cuda/blackwell/mi210) — SOLO si el .sif existe en disco
4. **Apunta** jobs a nodos específicos con `--nodelist`
5. **Asigna** `--mem` basado en RAM libre del nodo
6. **Rastrea OOM** — si un nodo mata un job, lo evita con backoff exponencial (5→10→20 min)
7. **Prioriza** GPUs con más VRAM (Blackwell primero, luego MI210, luego V100/Q6000)

**Estado persistente:** `~/scratch/gpu_state.json`
**Log:** `~/scratch/orch_v3.log`

---

## Estrategia de Entrenamiento Paralelo (Resumen)

### Lo que funciona HOY (Fase 1 — Enjambre Independiente)
- Cada GPU entrena un LoRA adapter independiente
- Diferentes shuffling del dataset → exploración de loss landscape
- Al final: seleccionar el adapter con loss más bajo
- Tolerante a fallos (un job muere, los otros siguen)

### Lo que NO se puede hacer
- ❌ Mezclar CUDA + ROCm en el mismo job distribuido (NCCL ≠ RCCL)
- ❌ DDP entre GPUs de diferente VRAM (el más chico es el cuello de botella)
- ❌ FSDP con Mamba-2 tiene issues conocidos (GitHub #36982)

### Futuro (post-Kaggle)
- DDP intra-nodo (2+ GPUs iguales en mismo nodo → torchrun)
- DeepSpeed ZeRO-2 para Q6000 (más seq_len sin OOM)
- Megatron Expert Parallel en Blackwell (5 GPUs × 96GB)

---

## Variables de Entorno Críticas

```bash
# SIEMPRE en jobs SLURM (ya están en nem_unified.slurm)
export HF_HUB_OFFLINE=1          # No intentar descargar modelos
export TRANSFORMERS_OFFLINE=1     # No intentar descargar tokenizers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1    # Evitar torch.compile (incompatible con Mamba)
export OMP_NUM_THREADS=1          # Evitar overhead de OpenMP
export PYTHONUNBUFFERED=1         # Ver output en tiempo real
```

---

## Problemas Conocidos y Soluciones

### OOM en Q6000 (24GB)
**Solución:** seq=128, rank=4, grad_accum=4. Si sigue fallando, reducir grad_accum a 2.

### SIF Blackwell no compila mamba-ssm
**Esperado:** Los kernels CUDA sm_120 no se compilan sin GPU durante el build. El SIF se construye igual. En runtime, mamba-ssm se importa del pip install (si hay wheels compatibles) o se puede compilar on-the-fly en el nodo.

### Intel oneMKL error durante build MI210
**Solución:** Todas las verificaciones de Python son `|| true` (non-fatal). El error es por PyTorch ROCm intentando cargar MKL sin GPU — no afecta el SIF final.

### Jobs marcados COMPLETED pero sin output
**Causa:** Si el training termina antes de producir output (OOM durante load). Revisar `.err` para ver el error real.

### git clone falla en SIF build
**Causa:** Directorio existe de build anterior. **Solución:** Ya hay `rm -rf` antes de cada `git clone` en los .def.

---

## Deadlines

| Competencia | Deadline | Qué subir |
|-------------|----------|-----------|
| **Kaggle Nemotron** | June 8 entry / June 15 final | LoRA adapter (max rank 32) |
| **Google Hackathon** | June 11, 2026 | ADK agent + Phoenix MCP |
| **Ebbe Nielsen** | June 26, 2026 | EcoSeek platform completo |

---

## Checklist para Continuar

- [ ] Verificar que los 3 Q6000 jobs producen loss decreciente
- [ ] Cuando `nemotron-blackwell.sif` termine de subir → probar con `sbatch hpc/nem_blackwell_test.slurm`
- [ ] Construir y subir `nemotron-mi210.sif`
- [ ] Lanzar `orch_v3.py` en modo continuo
- [ ] Después de ~100 steps: revisar manifests y seleccionar mejor adapter
- [ ] Upload dataset a HuggingFace
- [ ] Submit a Kaggle

---

*Generado por Devin — sesión 51512abbb626437e8eacf640c0f6e10f*
