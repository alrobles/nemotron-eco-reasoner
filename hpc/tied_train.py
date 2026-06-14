#!/usr/bin/env python3
"""
tied_train.py — v14 + MoE weight tying + (router-frozen) staged expert freeze.

Multi-node / multi-GPU DDP training, launched under SLURM `srun` (one process per
GPU). No torchrun needed: the distributed env (RANK / LOCAL_RANK / WORLD_SIZE /
MASTER_ADDR / MASTER_PORT) is taken from the SLURM_* variables exported by the
launcher. Each rank loads a full BF16 replica of the model on its local GPU.

MoE weight tying (the lever):
    In Nemotron-H every MoE layer has `self.experts = nn.ModuleList([NemotronHMLP])`
    where each expert is up_proj -> act -> down_proj (no gate). With per-expert LoRA,
    each token only updates the experts the router selected -> cold experts barely
    learn ("one at a time").

    We tie, PER MoE LAYER, the "shared-space-facing" LoRA factor across all experts
    by making them the SAME nn.Parameter object (huikang's 0.86 trick):
      * up_proj.lora_A   (reads the shared hidden state)   -> tied across experts
      * down_proj.lora_B (writes the shared hidden state)  -> tied across experts
      * up_proj.lora_B / down_proj.lora_A                  -> free per expert
    Because the tied factor is a single shared Parameter, autograd SUMS the gradient
    contributions of every expert into it automatically -> "all experts learn
    together" every step. Gradient accumulation, gradient clipping and DDP all-reduce
    all behave normally (no N-fold amplification like the old grad-sum hack).

    The resulting adapter is a *standard* PEFT LoRA adapter (the tied factor simply
    has identical values across experts), so it loads with the normal inference path
    and is Kaggle-legal (rank <= 32, only the adapter ships).

Router freeze (Kaggle-legal interpretation):
    The router is NEVER a LoRA target in this setup -> it is frozen by construction
    (and only LoRA weights ship to Kaggle, so a full-param router finetune would be
    illegal anyway). Additionally, EXPERT_FREEZE_FRAC>0 keeps the expert LoRA frozen
    for the first fraction of steps (attention/mamba adapt first), implemented by
    zeroing the expert-LoRA grads after all-reduce (DDP-safe).
"""
import os, re, time, glob, json, math
from collections import defaultdict
import torch

# ----------------------------------------------------------------------------- env
def _envint(*names, default=0):
    for n in names:
        v = os.environ.get(n)
        if v is not None and v != "":
            return int(v)
    return default

RANK       = _envint("RANK", "SLURM_PROCID", default=0)
LOCAL_RANK = _envint("LOCAL_RANK", "SLURM_LOCALID", default=0)
WORLD_SIZE = _envint("WORLD_SIZE", "SLURM_NTASKS", default=1)
os.environ["RANK"] = str(RANK)
os.environ["LOCAL_RANK"] = str(LOCAL_RANK)
os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
IS_MAIN = RANK == 0

def log(*a):
    if IS_MAIN:
        print(*a, flush=True)

torch.cuda.set_device(LOCAL_RANK)

M           = os.environ["M"]
DATA        = os.environ["DATA"]
OUT         = os.environ["OUT"]
TARGET_TOTAL  = _envint("TARGET_TOTAL", default=500)
STEPS_PER_JOB = _envint("STEPS_PER_JOB", default=200)
SEQ_LEN     = _envint("SEQ_LEN", default=3072)
GRAD_ACCUM  = _envint("GRAD_ACCUM", default=2)
RANK_LORA   = _envint("LORA_RANK", default=32)
ALPHA_LORA  = _envint("LORA_ALPHA", default=32)
LR          = float(os.environ.get("LR", "1e-4"))
LR_SCHED    = os.environ.get("LR_SCHED", "constant_with_warmup")
FREEZE_FRAC = float(os.environ.get("EXPERT_FREEZE_FRAC", "0.0"))

log(f"=== TIED TRAIN | world={WORLD_SIZE} rank={RANK} local={LOCAL_RANK} "
    f"master={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']} ===")
log(f"  data={DATA} seq={SEQ_LEN} lr={LR} sched={LR_SCHED} grad_accum={GRAD_ACCUM} "
    f"r={RANK_LORA} alpha={ALPHA_LORA} freeze_frac={FREEZE_FRAC} target={TARGET_TOTAL}")

# ----------------------------------------------------------------------------- model
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    M, trust_remote_code=True, dtype=torch.bfloat16,
    device_map={"": LOCAL_RANK}, low_cpu_mem_usage=True)
model.config.use_cache = False
log(f"MODEL LOADED in {time.time()-t0:.0f}s on cuda:{LOCAL_RANK} "
    f"{torch.cuda.get_device_name(LOCAL_RANK)}")

# ----------------------------------------------------------------------------- LoRA
from peft import LoraConfig, get_peft_model
target = sorted({n.split(".")[-1] for n, m in model.named_modules()
                 if isinstance(m, torch.nn.Linear)} &
                {"q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj", "in_proj", "out_proj"})
log(f"LoRA target modules: {target}")
peft_config = LoraConfig(
    r=RANK_LORA, lora_alpha=ALPHA_LORA, target_modules=target,
    lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, peft_config)
model.enable_input_require_grads()   # needed for gradient checkpointing on a PEFT model

# ------------------------------------------------------------- MoE weight tying init
# Group expert LoRA factors per (layer, proj, factor) and share a single Parameter.
#   * experts.<E>.up_proj.lora_A   -> tie
#   * experts.<E>.down_proj.lora_B -> tie
EXPERT_RE = re.compile(r"\.experts\.\d+\.")

def _is_tie_target(mod_name):
    if ".experts." not in mod_name:
        return False
    if mod_name.endswith("up_proj.lora_A.default"):
        return True
    if mod_name.endswith("down_proj.lora_B.default"):
        return True
    return False

groups = defaultdict(list)  # group_key -> [nn.Linear modules sharing weight]
for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.Linear) and _is_tie_target(name):
        key = EXPERT_RE.sub(".experts.#.", name)
        groups[key].append(mod)

tied_groups = 0
tied_eliminated = 0
shapes_seen = set()
with torch.no_grad():
    for key, mods in sorted(groups.items()):
        if len(mods) < 2:
            continue
        shapes = {tuple(m.weight.shape) for m in mods}
        # safety: every expert factor in a group must share the same shape
        if len(shapes) != 1:
            log(f"  !! SKIP group {key}: mixed shapes {shapes}")
            continue
        canonical = mods[0].weight
        mean = torch.stack([m.weight.data for m in mods], dim=0).mean(dim=0)
        canonical.data.copy_(mean)
        for m in mods[1:]:
            m.weight = canonical            # share the *same* Parameter object
            tied_eliminated += 1
        tied_groups += 1
        shapes_seen.add((key.split('.experts.#.')[-1], tuple(canonical.shape), len(mods)))

log("=== MoE TIE DIAGNOSTIC ===")
for proj, shp, n in sorted(shapes_seen):
    log(f"  {proj:32s} shape={shp} experts/group={n}")
log(f"MoE tied: {tied_groups} groups, {tied_eliminated} duplicate params collapsed "
    f"into shared Parameters")

# sanity: trainable param count after dedup
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"Trainable params after tying: {trainable/1e6:.2f}M")

# expert-LoRA params (for the staged freeze); identified by name on the PeftModel
expert_lora_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and ".experts." in n and ".lora_" in n]

# ----------------------------------------------------------------------------- data
from datasets import Dataset
ds = Dataset.from_json(DATA)
def fmt(ex):
    ms = ex.get("messages") or [{"role": "user", "content": ex.get("prompt", "?")},
                                {"role": "assistant", "content": ex.get("answer", "?")}]
    return {"text": tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=False)}
ds = ds.map(fmt, remove_columns=ds.column_names)
log(f"{len(ds)} training records")

# ----------------------------------------------------------------------------- resume
ckpts = sorted(glob.glob(os.path.join(OUT, "checkpoint-*", "trainer_state.json")),
               key=os.path.getmtime)
RESUME = os.path.dirname(ckpts[-1]) if ckpts else None
global_step = json.load(open(ckpts[-1])).get("global_step", 0) if ckpts else 0
batch_steps = min(STEPS_PER_JOB, TARGET_TOTAL - global_step)
target_step = global_step + batch_steps
log(f"RESUME step {global_step} -> {target_step}" if RESUME else f"FRESH START -> {target_step}")

# ----------------------------------------------------------------- staged freeze cb
from transformers import TrainerCallback
class ExpertFreezeCallback(TrainerCallback):
    """Keep expert-LoRA frozen for the first FREEZE_FRAC of steps by zeroing their
    grads AFTER DDP all-reduce / clipping (DDP-safe: params stay registered)."""
    def __init__(self, params, freeze_until):
        self.params = params
        self.freeze_until = freeze_until
    def on_pre_optimizer_step(self, args, state, control, **kw):
        if state.global_step < self.freeze_until:
            for p in self.params:
                if p.grad is not None:
                    p.grad.zero_()
        return control

freeze_until = int(math.ceil(TARGET_TOTAL * FREEZE_FRAC)) if FREEZE_FRAC > 0 else 0
callbacks = []
if freeze_until > 0:
    callbacks.append(ExpertFreezeCallback(expert_lora_params, freeze_until))
    log(f"Staged expert-LoRA freeze ON: frozen until step {freeze_until}")
else:
    log("Staged expert-LoRA freeze OFF (router still frozen by construction)")

# ----------------------------------------------------------------------------- train
from trl import SFTTrainer, SFTConfig
warmup_steps = min(30, max(1, target_step // 10))
cfg = SFTConfig(
    output_dir=OUT, max_steps=target_step,
    per_device_train_batch_size=1, gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR, lr_scheduler_type=LR_SCHED, warmup_steps=warmup_steps,
    logging_steps=10, bf16=True, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch", max_length=SEQ_LEN, dataset_text_field="text", packing=False,
    save_steps=50, save_total_limit=15, report_to="none",
    adam_beta1=0.9, adam_beta2=0.95, adam_epsilon=1e-8,
    weight_decay=0.0, max_grad_norm=1.0, use_liger_kernel=True,
    ddp_find_unused_parameters=True,      # MoE: some experts unused in a given step
    dataloader_num_workers=2,
)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                     processing_class=tok, callbacks=callbacks)

log(f"TRAINING [TIED]: {global_step} -> {target_step} | eff_batch="
    f"{1*GRAD_ACCUM*WORLD_SIZE} (1 x ga{GRAD_ACCUM} x {WORLD_SIZE} gpu)")
t0 = time.time()
trainer.train(resume_from_checkpoint=RESUME)
elapsed = (time.time() - t0) / 60
final_loss = next((e["loss"] for e in reversed(trainer.state.log_history) if "loss" in e), "?")
log(f"DONE {elapsed:.1f}min loss={final_loss}")

if IS_MAIN:
    model.save_pretrained(os.path.join(OUT, f"checkpoint-{target_step}"))
    print(f"Adapter saved to checkpoint-{target_step}", flush=True)
    if target_step < TARGET_TOTAL:
        print(f"NEED_RESUBMIT: target_step={target_step} target_total={TARGET_TOTAL}", flush=True)
    else:
        print(f"TARGET_REACHED: {target_step}/{TARGET_TOTAL}", flush=True)
    print("SUCCESS", flush=True)
