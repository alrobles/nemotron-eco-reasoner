# Ecoreasoner — Estrategias de Generación CoT

Documento comparativo entre las dos arquitecturas probadas para generar
Chain-of-Thought training data con DeepSeek R1 14B en el cluster KU HPC.

---

## Estrategia A: Fleet v3 (Ollama como servicio)

### Arquitectura
```
┌─────────────┐    ┌──────────────┐
│ cot_fleet   │───▶│ Ollama serve │  (Q6000, 5-9 nodos)
│  .slurm     │    │ Puerto fijo  │
│  (1 job)    │    └──────────────┘
│  con pool   │    ┌──────────────┐
│  threading  │───▶│ Ollama serve │  ... × N nodos
└─────────────┘    └──────────────┘
```

### Cómo funciona
1. Jobs Ollama iniciados manualmente (sbatch launch_mi210_ollama.slurm)
2. Cada job expone HTTP en node:puerto
3. Un solo Slurm job (cot_fleet_v2.slurm) con ThreadPoolExecutor
4. Lee papers de JSONL, distribuye entre endpoints vía HTTP POST
5. Endpoints viven ~5h55m (sixhour), se renuevan al expirar

### Resultados reales
| Métrica | Valor |
|----------|-------|
| Endpoints máximos | 9-11 (Q6000 + MI210) |
| Velocidad | ~23 papers/min |
| Batch 1 (SDM) | 1,329 papers en ~60 min |
| Batch 2 (multi) | 276 papers en ~12 min |
| **Rendimiento** | **~2.5-3 papers/min/GPU** |
| Paper loss | 417 (login node crash), 173 (restart) |
| Mantenimiento | Alto — renovar puertos cada 5h, archivo endpoints_q6000.txt |

### Problemas
- Ollama overhead HTTP → pérdida de velocidad
- Endpoints efímeros → requiere monitoreo constante
- Login node mata procesos Python → pérdida de datos (resuelto con Slurm)
- Escritura incremental parcheada después del crash
- No escala automáticamente

### Ventajas
- Carga de modelo persistente (6s warm reload vs 6min cold)
- Reutiliza el mismo modelo en memoria
- Más simple de depurar (logs centralizados)

---

## Estrategia B: Tsunami (llama.cpp directo, job array)

### Arquitectura
```
Slurm Job Array (1-173%20)
  │
  ├── Task 1  → [llama-server] → process_chunk.py → chunk_00000.jsonl (50 papers)
  ├── Task 2  → [llama-server] → process_chunk.py → chunk_00001.jsonl (50 papers)
  ├── ...
  └── Task 173 → [llama-server] → process_chunk.py → chunk_00172.jsonl (50 papers)
       ↑
  Slurm scheduler decide cuáles 20 corren simultáneamente
```

### Cómo funciona
1. `gen_manifest.py` — divide papers en chunks de 50
2. `submit_tsunami.slurm` — job array, 20 concurrentes máximo
3. Cada task:
   - Arranca llama-server vía apptainer (ollama.sif)
   - Espera health check
   - Procesa su chunk con `process_chunk.py`
   - Mata el server al terminar
4. `autoscaler.sh` (opcional) — monitorea y lanza más si hay slots

### Resultados reales
| Métrica | Valor |
|----------|-------|
| Papers totales | 8,642 (todas las fuentes) |
| Chunks | 173 × 50 papers |
| Concurrentes | 20 (configurable con %N) |
| Cold start | ~6 min por GPU (carga modelo desde NFS) |
| Por GPU | ~50 papers/task, ~1-2 min/paper |
| **Estimación** | **~25-50 papers/min con 20 GPUs** |
| **Rendimiento estimado** | **~2.5 papers/min/GPU** (similar a Fleet) |
| Paper loss | 0 (cada chunk es atómico, si falla se re-ejecuta) |

### Ventajas
- Sin HTTP overhead (localhost llama-server)
- Sin gestión de endpoints
- Slurm scheduler maneja la concurrencia automáticamente
- Si hay 50 GPUs libres a las 3 AM → escala a 50
- Cada task es autónomo, si falla Slurm lo reencola
- Output atómico por chunk

### Desventajas
- Cold start 6 min por GPU (frente a 6s warm reload)
- El modelo se carga y descarga en cada task
- NFS puede ser cuello de botella si muchos tasks arrancan a la vez

---

## Comparación directa

| Dimensión | Fleet v3 | Tsunami |
|-----------|----------|---------|
| Overhead | HTTP (Ollama) | Directo (llama.cpp) |
| Carga modelo | 6s (warm) | 6 min (cold) |
| Rendimiento/GPU | ~2.5-3 papers/min | ~2.5 papers/min (est) |
| Escalabilidad | Manual | Automática (Slurm) |
| Fiabilidad | Media (endpoints mueren) | Alta (Slurm reencola) |
| Mantenimiento | Alto | Cero |
| Paper loss | 590/3,500 (17%) | 0 (atómico) |

---

## Veredicto preliminar

**Tsunami es superior** para producción, pero Fleet fue útil para prototipado rápido.
Tsunami gana en:
- Cero gestión manual
- Escalado automático
- Sin pérdida de datos
- Sin HTTP overhead (llama.cpp directo es más rápido que Ollama HTTP)

Fleet fue bueno para:
- Iterar rápido el prompt y el formato de salida
- Probar el modelo antes de escalar
- Depurar problemas de compatibilidad (ROCm, CUDA)

---

## Archivos en el repo

```
scripts/
  cot_ollama_fleet.py   — Fleet (Ollama HTTP, legacy)
  process_chunk.py      — Tsunami (llama.cpp directo)
  gen_manifest.py       — Generador de chunks
  autoscaler.sh         — Auto-escalador para tsunami
  harvest_*.py          — Harvesters (PubMed, GBIF, arXiv, bioRxiv, ecoevorxiv)
  endpoints_q6000.txt   — Lista de endpoints (Fleet, legacy)

hpc/
  cot_fleet_v2.slurm    — Fleet Slurm job (legacy)
  submit_tsunami.slurm  — Tsunami Slurm job array
  gen_manifest.slurm    — Genera manifiesto
  harvest_*.slurm       — Harvesters
```

## Datasets generados

| Batch | Método | Papers | Output |
|-------|--------|--------|--------|
| SDM | Fleet | 1,329 | cot_eco_full.jsonl |
| Multi-DB | Fleet | 465 | cot_eco_batch2.jsonl |
| Tsunami | Tsunami | 8,642 | chunk_*_out.jsonl → pending |

---

Última actualización: 2026-06-21 00:15 CDT
