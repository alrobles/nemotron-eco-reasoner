# Nemotron Adaptive Swarm

Sistema de entrenamiento distribuido multi-arquitectura para Nemotron-3-Nano-30B en KU HPC.

## Arquitectura

```
KBS (sin walltime):
  gpu_monitor.py → escanea GPUs cada 60s → gpu_state.json
  orch_v2.py     → lee JSON, submit adaptativo, backoff OOM

SIXHOUR (entrenamiento, 6h walltime):
  nem_unified.slurm → auto-detect GPU, config óptima
  outputs/unified/  → checkpoints compartidos
```

## Componentes

| Archivo | Ubicación | Función |
|---------|-----------|---------|
| `hpc/nem_unified.slurm` | Cluster | Template unificado, auto-detect GPU |
| `scripts/orch_v2.py` | Cluster | Orquestador adaptativo con backoff |
| `scripts/gpu_monitor.py` | Cluster | Monitor de GPUs en tiempo real |
| `scripts/smart_swarm.py` | Cluster | Submit inteligente por nodo |
| `hpc/orch_v2.slurm` | Cluster | Launcher kbs para orquestador |
| `hpc/gpu_monitor.slurm` | Cluster | Launcher kbs para monitor |

## Auto-config por GPU

| GPU | VRAM | seq | rank | bf16 |
|-----|------|-----|------|------|
| Pro6000 Blackwell | 96 GB | 2048 | 32 | ✅ |
| A100/A40/L40 | 40-48 GB | 1024 | 16 | ✅ |
| V100 | 32 GB | 64 | 8 | ❌ |
| Q6000 | 24 GB | 48 | 4 | ❌ |

## Despliegue

```bash
# 1. Monitor (una vez)
sbatch hpc/gpu_monitor.slurm

# 2. Orquestador (una vez)
sbatch hpc/orch_v2.slurm

# 3. Verificar
squeue -u $USER
tail -f ~/scratch/orch_v2.log
```

## Lecciones

- `--mem >= 62G` obligatorio para Nemotron 30B
- `/tmp` se contamina entre jobs → usar `$SLURM_JOB_ID`
- NO reinstalar torch con pip → usar el del container
- PeftModel.from_pretrained() para resume (no torch.load)
- LR constante evita meseta por scheduler reset
- Backoff exponencial en OOM evita loops mortales
