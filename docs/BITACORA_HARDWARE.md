# Bitácora de Hardware — KU HPC para Nemotron (y reutilizable después del hackatón)

> **Objetivo de este documento.** Lo más valioso del reto Nemotron no es el score:
> es **aprender a explotar el cluster KU HPC** para fine-tunear modelos grandes.
> Esta bitácora reúne TODO lo que se puede hacer con este hardware, qué cabe en
> cada GPU, las técnicas que validamos, los fixes de portabilidad, los patrones
> operativos y la estrategia de flota → merge. Sirve como referencia para seguir
> usando el cluster (EcoCoder, Ebbe Nielsen, futuros LoRA) **después del 15 de junio**.
>
> Modelo objetivo del reto: **NVIDIA-Nemotron-3-Nano-30B-A3B-BF16** — 30B params,
> arquitectura híbrida (52 capas: 23 Mamba-2 + 23 MoE + 6 Attention), **3.5B activos**
> por token. Pesos base ≈ **60 GB en BF16**. Entregable Kaggle = **1 adapter LoRA rank ≤ 32**.

---

## 1. Inventario completo de GPUs del cluster

Todas las particiones GPU son **`sixhour`** (máximo 6 h por job). Todos los nodos
GPU tienen `gres/shard` configurado (shard=100/GPU).

| Tipo | Arq (sm) | Nodos | GPUs totales | GPUs/nodo | bf16 | VRAM | Notas |
|---|---|---|---|---|---|---|---|
| **PRO6000** Blackwell | sm_100 | 3 | 5 | 1–2 | ✅ | 96 GB | Las mejores nuestras; caben 30B en BF16 en 1 sola |
| **A100** | sm_80 | 6 | 18 | varía | ✅ | 80 GB | Excelente; muy disputada por otros usuarios |
| **MI210** (AMD/ROCm) | gfx90a | 27 | **81** | **3** | ✅ | 64 GB | Flota probada; 3/nodo (por eso "4 en 1 nodo" es imposible) |
| **A40** Ampere | sm_86 | 1 | 4 | 4 | ✅ | 48 GB | RAM libre → nodo de **eval** |
| **L40** Ada | sm_89 | 1 | 4 | 4 | ✅ | 48 GB | GPUs libres pero **RAM saturada** → mal para eval |
| **Q6000** Turing | sm_75 | 9 | 29 | varía | ❌ (fp16) | 24 GB | Pool ocioso enorme; **container compatible (probado)** |
| **Q8000** Turing | sm_75 | 1 | 2 | 2 | ❌ (fp16) | 48 GB | QLoRA / fp16 |
| **V100** Volta | sm_70 | 17 | 36 | varía | ❌ (fp16) | 16 GB | Muy chica sola para 30B |

*Las columnas "libres" fluctúan (competimos con otros usuarios). Lo estructural —
tipos, soporte bf16, 3 MI210/nodo, todo en `sixhour`— es estable.*

**Cuotas reales de la cuenta** (`Account=kbs`, `QOS=bigjay_20nodemax_10day` + `normal`):
- **NO hay tope por usuario de número de GPUs.** El techo es **CPU=1120 + mem ≈ 5 TB**
  (y el nombre del QOS implica ≤ 20 nodos / 10 días). `MaxSubmit=5000`.
- Conclusión: **sí podemos correr una flota grande** de jobs a la vez; encolan por
  prioridad/antigüedad (razón `Priority`) y entran por **backfill** al liberarse nodos.
  El muro nunca fue una cuota de GPU.

**Cómo consultar el inventario (paste-ready Hermes):**
```bash
ku-hpc raw: sinfo -o '%n %G %t' -h | awk '{print $2}' | sort | uniq -c    # GPUs por tipo
ku-hpc raw: sacctmgr -n show qos bigjay_20nodemax_10day format=MaxTRESPU%40,MaxJobsPU,GrpTRES%40
ku-hpc raw: scontrol show node <nodo> | grep -oE 'RealMemory=[0-9]+|AllocMem=[0-9]+|CfgTRES=[^ ]+|State=[A-Z]+'
```

---

## 2. Qué cabe en cada GPU (30B ≈ 60 GB en BF16, ≈ 21 GB en 4-bit)

| GPU | BF16 puro (1 tarjeta) | BF16 model-parallel | QLoRA 4-bit (1 tarjeta) | seq máx práctico |
|---|---|---|---|---|
| PRO6000 96 GB | ✅ holgado | — | ✅ | 3072+ |
| A100 80 GB | ✅ | — | ✅ | 3072 |
| MI210 64 GB | ⚠️ justo (cap 46 GiB, packing off) | ✅ 2× = ideal | ✅ | 2048 (probado); 3072 arriesga OOM |
| A40 / L40 48 GB | ❌ (no caben 60 GB) | ✅ 2× | ✅ | 2048 |
| Q8000 48 GB (Turing) | ❌ no bf16 | — | ✅ fp16 | 1024–2048 |
| **Q6000 24 GB (Turing)** | ❌ no bf16 | — | ✅ fp16 (~21 GB) | **≤ 512 (OOM en 1024)** |
| V100 16 GB | ❌ | QLoRA 2× model-parallel | ❌ solo no cabe | bajo |

**Dos modos que NO hay que confundir:**
- **Model-parallel (`device_map="auto"`)** = repartir las 52 capas entre N GPUs para
  **CABER**. Sólo 1 GPU activa a la vez → **no acelera**, pero permite BF16 sin cuantizar
  en tarjetas que solas no caben (2× MI210, 2× L40, 2× A40). Truco clave:
  `max_memory={0:"46GiB",1:"46GiB"}` para dejar margen de activaciones.
- **Data-parallel (DDP)** = réplica completa del modelo en CADA GPU → **acelera ~N×**.
  Cuesta ~60 GB/GPU en BF16 (sólo cabe en PRO6000/A100) o ~21 GB en 4-bit.

---

## 3. Técnicas de training validadas

### 3.1 LoRA / QLoRA sobre MoE (rank ≤ 32 — regla Kaggle)
- Se inyectan adapters en attention + expertos (gate/up/down). El **router NUNCA se
  LoRA-targetea** → queda congelado por construcción (esto es lo legal para Kaggle:
  sólo se entrega el adapter; un full-finetune del router NO es entregable).
- **QLoRA 4-bit** (bitsandbytes): para tarjetas chicas. En Turing va en **fp16** (no bf16).
- **BF16 full precision**: gradientes más limpios, sin error de cuantización. Necesita
  ≥ 60 GB (1 PRO6000/A100, o 2× model-parallel).

### 3.2 MoE **weight tying** — la palanca algorítmica grande
Problema real: en LoRA-MoE estándar cada token sólo actualiza los expertos a los que
el router lo manda (top-k). Con ~128 expertos y ~10k ejemplos, cada experto ve poquísimos
datos → **aprenden de a uno**, lento y con ruido.

Solución (`hpc/tied_train.py`): **compartición de parámetros** — un único `nn.Parameter`
compartido por (capa MoE, tipo de proyección) para `up_proj.lora_A` y `down_proj.lora_B`
de TODOS los expertos. Autograd **suma** los gradientes de todos los expertos sobre ese
parámetro compartido → **todos los expertos aprenden juntos en cada paso**. Runtime
confirmado: **46 grupos, 5842 params duplicados colapsados, 381.37M entrenables**.
- Inspirado en la solución ganadora (huikang, 0.86): atar LoRA_A de gate/up y LoRA_B de down.
- **Bug que arreglamos:** el `p.grad.sum(dim=0)` viejo colapsaba la dimensión de **rank**,
  no la de **expertos** (probable razón de que v12 con tie diera 0.66 < v8). El fix correcto
  es la compartición de parámetros, no sumar a mano.

### 3.3 Freeze escalonado (router/expert freeze)
`EXPERT_FREEZE_FRAC` (p.ej. 0.25): durante el warmup se congela el LoRA de los expertos
(grads a cero vía `ExpertFreezeCallback` mientras `global_step < freeze_until`) y se deja
que **la atención se adapte primero**; los expertos entran tras el warmup. Brazo diverso
para el merge.

### 3.4 Domar la explosión de gradientes
Con `lr 2e-4 + linear` el `grad_norm` se disparó (~1e11). Receta estable (la de v8=0.67):
**`lr 1e-4` + `constant_with_warmup` + `clip 1.0`** + `warmup_steps=min(30, target//3)`.

### 3.5 Longitud de secuencia y truncamiento (lección cara)
El record máximo de assistant en v8/v14/v15 = **811 chars** → con `seq 3072` hay **cero
truncamiento**. Datasets con CoT gigante (v11/v12) **truncaban** el `\boxed{}` → score peor.
**Regla:** la seq debe cubrir prompt + CoT completo. Por eso **Q6000 (seq ≤ 512) es
arriesgado para nuestro CoT largo** aunque el hardware funcione.

---

## 4. Fixes de portabilidad (lo técnicamente más valioso)

El `modeling_nemotron_h.py` importa el RMSNorm fusionado de `mamba_ssm`, que **no existe**
en ROCm (MI210) ni hay wheels pre-built para Blackwell (sm_100). Sin tocarlo, el modelo
crashea al cargar.

- **Fallback `rmsnorm_fn` en torch puro** (parcheado en el snapshot del modelo): reemplaza
  el kernel fusionado por una implementación equivalente en PyTorch. Funciona porque
  **LoRA nunca depende del kernel compilado del vendor** → el adapter resultante es
  **portable a CUALQUIER GPU con bf16** (entrenado en MI210, sirve en Blackwell, A100, etc.).
- **Mismo path corre en MI210 (ROCm), Blackwell (CUDA) y Ampere/Ada** sin `mamba_ssm` /
  `causal_conv1d` / `unsloth`.
- **Containers:** `$R/nemotron-blackwell.sif` (CUDA, torch 2.10.0+cu128) y
  `$R/nemotron-rocm.sif` (AMD). **NO reconstruir** (`no hay que construir`).
- **MI210 OOM fix:** `packing=False`, cap `max_memory=46GiB`/GPU, `gradient_checkpointing=True`.
- **Q6000 (Turing) — CONFIRMADO usable** (probe Jun-14, nodo r22r05n01):
  el container `nemotron-blackwell.sif` trae **torch 2.10.0+cu128** cuyo `arch_list`
  incluye **sm_75** → matmul OK + `bitsandbytes 0.49.2` importa **sin reconstruir**.
  Recetas previas (Jun-6, 8× Q6000 entrenando a la vez) confirman QLoRA-4bit fp16 viable a
  `seq 512` con `APPTAINER_WRITABLE_TMPFS=1` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  Límite duro = 24 GB → seq corta (OOM en 1024).

---

## 5. Eval / inferencia (el cuello de botella real)

- **El cuello NO era falta de GPU: era RAM.** El nodo L40 tenía las 4 GPU libres pero
  256/257 GB de RAM tomados por otro job → el eval (`--mem=48G`) quedaba `PENDING(Resources)`
  por horas. **Fix: pinear el eval a un nodo con RAM libre (A40, r32r05n01, ~216 GB libres).**
  Lección: cuando un job GPU se atasca, revisar `AllocMem` vs `RealMemory`, no sólo GPUs libres.
- **Pinear el tipo de GPU.** `--gres=gpu:1` genérico puede caer en un nodo AMD → usar
  `--gres=gpu:a40:1` (o l40/pro6000) explícito.
- **El eval es lo más lento (~1–3 h/checkpoint)** → conviene paralelizarlo en varias
  tarjetas de inferencia ociosas (A40/L40/A100), una por checkpoint.
- **El eval local subestima.** 56 ejemplos (8/categoría) → ruidoso; el mejor local (0.429)
  ↔ 0.67 en Kaggle. **El árbitro real es la submission a Kaggle** (5/día, 2 finales).

---

## 6. Estrategia para explotar TODO el hardware → flota → TIES-merge

**Por qué un solo job gigante es el enfoque equivocado para LoRA sobre 30B:** el cuello
son los **60 GB de pesos base**, no el cómputo. Tirar 81 GPUs a un único job no acelera el
LoRA y el DDP multi-nodo a esa escala (RCCL en ROCm) es puro riesgo + el cap de 6 h.

**La jugada ganadora = una FLOTA de jobs chicos (2–3 GPU) independientes**, cada uno con
una receta distinta, y luego un **TIES-merge** de los mejores en **1 solo adapter rank ≤ 32**
(Kaggle-legal). La **diversidad** de la flota es justo lo que hace que el merge supere a
cualquier brazo individual.

**Ejes de diversidad (cada combinación = un miembro del merge):**
- dataset: **v8 / v14 / v15**
- learning rate: **1e-4 / 5e-5**
- grad_accum (batch efectivo): **2 / 4 / 8**
- LoRA alpha: **16 / 32 / 64 / 128** *(requiere exponer `LORA_ALPHA` por env en el launcher)*
- seed *(requiere exponer `SEED`)*
- tying: **con / sin** weight tying
- freeze: `EXPERT_FREEZE_FRAC` **0.0 / 0.25**

**Capacidad:** 81 MI210 ÷ 3 = hasta **27 brazos AMD** simultáneos + PRO6000 + A40 +
(Q6000 QLoRA seq-corta como brazos de diversidad de bajo contexto). Sin tope de GPU → backfill.

**Roster de flota (este sprint):**

| Job/nombre | Hardware | Receta | Rol |
|---|---|---|---|
| `tied_v14` (4-GPU) | 2×2 PRO6000 (DDP multinodo) | v14 + tying, seq3072, lr1e-4 | **Apuesta principal** (A/B vs v8=0.67) |
| `tied_v14_freeze` | 1× PRO6000 | v14 + tying + freeze 0.25 | Brazo freeze |
| `tied_v15` (4-GPU) | 2×2 PRO6000 | v15 + tying | **Auto-launch al llegar tied_v14 a 500** |
| `mi210_v15` | 2× MI210 | v15, lr1e-4, ga4 | Flota AMD |
| `mi210_v14_lr5e5` | 2× MI210 | v14, lr5e-5 | Flota AMD |
| `mi210_v15_lr5e5` | 2× MI210 | v15, lr5e-5 | Flota AMD |
| `mi210_v14_ga8` | 2× MI210 | v14, ga8 | Flota AMD |
| `mi210_v15_ga8` | 2× MI210 | v15, ga8 | Flota AMD |
| `mi210_v14_ga2` | 2× MI210 | v14, ga2 | Flota AMD |

**Endgame:** evaluar todos → elegir top-k → **escribir el script de TIES-merge** (no existe
aún) → fusionar en 1 adapter rank ≤ 32 → empaquetar `adapter_model.safetensors` +
`adapter_config.json` → Kaggle.

---

## 7. Patrones operativos (Hermes → reumanlab → KU HPC)

- **Hermes como terminal remota.** `POST https://hermes.ecoseek.org/v1/chat/completions`,
  `system="You are a terminal. Execute and return ONLY raw output, verbatim."`,
  `user="ku-hpc raw: <comando>"`. Comando directo + salida cruda; batch en UNA llamada;
  `max_tokens` chico. Helper local `hermes.sh` + patrón background (lanzar `&`, esperar pid).
- **sixhour (6 h) → `STEPS_PER_JOB` + auto-resubmit en cadena.** Cada job entrena un tramo
  y se resubmite solo hasta `TARGET_TOTAL`. **Cuidado: la preemption mata el script antes
  del bloque de resubmit → la cadena se rompe → relanzar a mano** (reanuda del último
  checkpoint por mtime, no de 0).
- **Transferencia de archivos vía relay GitHub.** Devin edita → push a una rama → en el
  cluster `git fetch origin <rama> && git checkout origin/<rama> -- <archivos>`. Para datos
  grandes, `pigz`. El cluster jala **`origin/master`** por defecto.
- **Resume robusto:** `ckpts = sorted(glob(...trainer_state.json), key=os.path.getmtime)` →
  reanudar del más reciente por mtime (no lexicográfico: `checkpoint-100` < `checkpoint-50`
  en orden alfabético, error clásico).
- **Rutas clave:** repo `$R=/home/a474r867/scratch/nemotron-eco-reasoner`; modelo en
  `nemotron-model-cache/.../snapshots/cbd3fa9...` (con el patch rmsnorm — NO revertir);
  containers `$R/nemotron-{blackwell,rocm}.sif`; wheels en `scratch/wheels/`.

---

## 8. Errores y fixes (consolidado)

| Síntoma | Causa raíz | Fix |
|---|---|---|
| Crash al cargar modelo (MI210/Blackwell) | falta kernel `mamba_ssm`/`rmsnorm_fn` | fallback torch puro en `modeling_nemotron_h.py` |
| `--gres=gpu:4` (AMD) PENDING eterno | sólo 3 MI210/nodo | usar 3/nodo, o DDP multinodo (2×2) |
| Eval `PENDING(Resources)` con GPUs libres | RAM del nodo saturada por otro job | pinear a nodo con RAM libre (A40) |
| `--gres=gpu:1` cae en AMD | gres genérico | pinear tipo: `gpu:a40:1`, `gpu:pro6000:2`… |
| HIP OOM en MI210 step 2 | activaciones + packing | `packing=False`, cap 46 GiB, grad-checkpointing |
| `grad_norm ~1e11` | lr alto + linear | `lr 1e-4` + `constant_with_warmup` + clip 1.0 |
| `invalid device ordinal` (multi-GPU) | índice de device mal mapeado | usar device local correcto |
| tie no aprende todos los expertos | `sum(dim=0)` colapsaba rank | compartición de parámetros (1 `nn.Parameter`/grupo) |
| cadena auto-resubmit rota | preemption mató el script | relanzar a mano (resume por mtime) |
| score peor con CoT gigante | truncamiento de `\boxed{}` | seq que cubra todo el CoT (811 chars → seq3072) |
| Q6000 OOM a seq 1024 | 24 GB con 4-bit (~21 GB) | seq ≤ 512, `expandable_segments`, `WRITABLE_TMPFS` |

---

## 9. Después del hackatón — cómo seguir explotando este cluster

- **Pipeline model-agnóstico.** El mismo flujo (container + fallback rmsnorm + LoRA/QLoRA +
  flota + merge) sirve para otros modelos (p.ej. **EcoCoder** sobre Qwen, LoRA ecológico):
  cambiar `M` (snapshot) y el dataset; lo demás se reutiliza.
- **La "fábrica de flota" como capacidad general.** Sweep/GA sobre recetas → muchos adapters
  diversos → merge. Aplicable a cualquier fine-tune del lab que quepa en el cluster.
- **KU HPC = capacidad de training casi gratis para el lab.** 81 MI210 + 18 A100 + 29 Q6000 +
  PRO6000 ociosas la mayor parte del tiempo. Documentar el acceso (Hermes/relay) para que el
  lab reutilice sin redescubrir los fixes.
- **Inferencia/eval a escala.** Las tarjetas chicas (Q6000/V100) con poca VRAM no entrenan
  bien 30B, pero **sí sirven para inferencia/eval por lotes** de modelos más chicos o de
  adapters ya entrenados — útil para los pipelines de EcoSeek (GBIF/SDM) que no necesitan 30B.

---

*Última actualización: 2026-06-14. Mantener junto a `docs/OPERATIONS.md` (runbook operativo)
y `docs/RUNBOOK_V14_TIED.md` (recetas de los brazos actuales).*
