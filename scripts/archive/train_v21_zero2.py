import os, sys, signal, types, torch, time, json, glob
os.environ['TORCH_COMPILE_DISABLE'] = '1'
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
torch.cuda.set_device(local_rank)
def log(msg):
    if local_rank == 0: print(msg, flush=True)
t0 = time.time()
MODEL_PATH = os.environ["MODEL_PATH"]
DATA_PATH = os.environ["DATA_PATH"]
OUT_PATH = os.environ["OUT_PATH"]
SEQ_LEN = int(os.environ.get("SEQ_LEN", "48"))
RANK = int(os.environ.get("LORA_RANK", "8"))
gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
log(f"GPU[{local_rank}/{world_size}]: {gpu} ({vram:.1f}GB) | seq={SEQ_LEN} rank={RANK}")
from unsloth import FastLanguageModel
from datasets import load_dataset
log("Loading model...")
model, tok = FastLanguageModel.from_pretrained(MODEL_PATH, max_seq_length=SEQ_LEN, load_in_4bit=True, trust_remote_code=True)
patched = 0
for module in model.modules():
    if not hasattr(module, 'moe') or not hasattr(module, 'experts'): continue
    if not callable(module.moe): continue
    def mp(mod):
        def pm(_self, h, ti, tw):
            h = h.view(-1, h.size(-1)); ft = ti.view(-1)
            h = h.repeat_interleave(ti.shape[-1], dim=0)
            fh = torch.zeros_like(h); dt = fh.dtype
            for i, el in enumerate(mod.experts):
                m = (ft == i).nonzero(as_tuple=True)[0]
                if m.numel() == 0: continue
                eh = h[m]; eo = el(eh); w = tw.view(-1)[m]; wo = eo * w.unsqueeze(-1)
                if wo.dtype != dt: wo = wo.to(dt)
                fh.index_add_(0, m, wo)
            tk = ti.shape[-1]; fh = fh.view(-1, tk, fh.size(-1)).sum(dim=1)
            return fh
        return pm
    module.moe = types.MethodType(mp(module), module); patched += 1
log(f"MoE patched: {patched} layers")
model = FastLanguageModel.get_peft_model(model, r=RANK, lora_alpha=RANK,
    target_modules=['q_proj','k_proj','v_proj','o_proj','in_proj','out_proj','up_proj','down_proj'],
    use_gradient_checkpointing='unsloth')
if local_rank == 0: model.print_trainable_parameters()
ds = load_dataset("json", data_files={"train": DATA_PATH}, split="train")
def fmt(ex):
    ms = ex.get("messages")
    if not ms: ms = [{"role":"user","content":ex.get("prompt","?")},{"role":"assistant","content":ex.get("answer","?")}]
    return {"text": tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=False)}
ds = ds.map(fmt)
alloc = torch.cuda.memory_allocated() / 1e9; reserved = torch.cuda.memory_reserved() / 1e9
log(f"VRAM[r{local_rank}]: {alloc:.1f}G/{reserved:.1f}G")
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(model=model, args=SFTConfig(
    output_dir=OUT_PATH, max_steps=500, per_device_train_batch_size=1,
    gradient_accumulation_steps=8, learning_rate=2e-4, max_seq_length=SEQ_LEN,
    warmup_steps=50, logging_steps=10, save_steps=100, save_total_limit=5,
    bf16=False, fp16=True, remove_unused_columns=False, report_to="none",
    dataloader_num_workers=0, packing=True,
    deepspeed=os.environ.get("DS_CONFIG", None), local_rank=local_rank),
    train_dataset=ds, processing_class=tok)
log(f"TRAINING v21 ZeRO-2 x{world_size}...")
trainer.train()
if local_rank == 0:
    elapsed = (time.time() - t0) / 60
    print(f"DONE {elapsed:.1f}min")
    trainer.save_model(OUT_PATH); tok.save_pretrained(OUT_PATH)
    json.dump({"model":MODEL_PATH,"data":DATA_PATH,"seq_len":SEQ_LEN,"rank":RANK,
        "gpus":world_size,"elapsed_min":elapsed,"gpu":gpu,
        "framework":"unsloth-4bit-qlora-q6000-zero2-v21"},
        open(os.path.join(OUT_PATH,"manifest.json"),"w"), indent=2)
    print("SUCCESS")
