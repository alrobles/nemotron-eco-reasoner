#!/usr/bin/env python3
"""One-shot generation diagnostic: prints the RAW decoded text for a few prompts
so we can see whether the model closes </think> and emits \\boxed{} within the
token budget, or whether it loops / never stops. NOT for PRs.

Usage (in the eval container):
    MODEL_PATH=... python3 scripts/diag_gen.py --adapter <ckpt> \
        --data data/kaggle_classified.jsonl --cats numeral,cipher \
        --max-new-tokens 6000
"""
import argparse
import json
import os
import re
import sys
import time
import types
from collections import defaultdict

import torch


def get_cache_cls():
    for name, mod in list(sys.modules.items()):
        if name.endswith("modeling_nemotron_h") and hasattr(
            mod, "HybridMambaAttentionDynamicCache"
        ):
            return mod.HybridMambaAttentionDynamicCache
    return None


def make_cache(cache_cls, model, batch_size):
    if cache_cls is None:
        return None
    fp = next((p for p in model.parameters() if p.is_floating_point()), None)
    dtype = fp.dtype if fp is not None else torch.bfloat16
    device = fp.device if fp is not None else getattr(model, "device", "cuda")
    for args, kwargs in (
        ((model.config, batch_size), {"dtype": dtype, "device": device}),
        ((model.config, batch_size), {}),
    ):
        try:
            return cache_cls(*args, **kwargs)
        except TypeError:
            continue
    return None


def patch_moe(model):
    patched = 0
    for module in model.modules():
        if not hasattr(module, "moe") or not hasattr(module, "experts"):
            continue
        if not callable(module.moe):
            continue

        def mp(mod):
            def pm(_self, h, ti, tw):
                h = h.view(-1, h.size(-1))
                ft = ti.view(-1)
                h = h.repeat_interleave(ti.shape[-1], dim=0)
                fh = torch.zeros_like(h)
                dt = fh.dtype
                for i, el in enumerate(mod.experts):
                    m = (ft == i).nonzero(as_tuple=True)[0]
                    if m.numel() == 0:
                        continue
                    eh = h[m]
                    eo = el(eh)
                    w = tw.view(-1)[m]
                    wo = eo * w.unsqueeze(-1)
                    if wo.dtype != dt:
                        wo = wo.to(dt)
                    fh.index_add_(0, m, wo)
                tk = ti.shape[-1]
                return fh.view(-1, tk, fh.size(-1)).sum(dim=1)

            return pm

        module.moe = types.MethodType(mp(module), module)
        patched += 1
    print(f"MoE patched: {patched} layers", flush=True)


SYS_PROMPT = (
    "You are an expert puzzle solver. Think step by step and place "
    "your final answer inside \\boxed{}."
)


def build_msgs(prompt):
    return [
        {"role": "system", "content": SYS_PROMPT},
        {
            "role": "user",
            "content": prompt + "\n\nPlease put your final answer inside \\boxed{}.",
        },
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", default="data/kaggle_classified.jsonl")
    ap.add_argument("--cats", default="numeral,cipher")
    ap.add_argument("--max-new-tokens", type=int, default=6000)
    ap.add_argument("--seq-len", type=int, default=2048)
    args = ap.parse_args()

    model_path = os.environ.get(
        "MODEL_PATH", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    )
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        model_path,
        max_seq_length=args.seq_len,
        load_in_4bit=True,
        trust_remote_code=True,
        device_map={"": 0},
    )
    patch_moe(model)
    model.load_adapter(args.adapter)
    FastLanguageModel.for_inference(model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cache_cls = get_cache_cls()
    print(f"KV cache class: {cache_cls.__name__ if cache_cls else 'auto'}", flush=True)

    # Show what the chat template actually produces (reasoning priming?).
    sample_ids = tok.apply_chat_template(
        [build_msgs("PROMPT_PLACEHOLDER")],
        tokenize=False,
        add_generation_prompt=True,
    )
    print("===== CHAT TEMPLATE RENDER =====", flush=True)
    print(repr(sample_ids)[:1500], flush=True)
    print("===== /CHAT TEMPLATE =====", flush=True)

    by_cat = defaultdict(list)
    with open(args.data) as f:
        for line in f:
            r = json.loads(line)
            by_cat[r["category"]].append(r)

    for cat in args.cats.split(","):
        cat = cat.strip()
        if not by_cat.get(cat):
            print(f"[{cat}] no examples", flush=True)
            continue
        r = by_cat[cat][0]
        enc = tok.apply_chat_template(
            [build_msgs(r["prompt"])],
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        plen = enc["input_ids"].shape[1]
        cache = make_cache(cache_cls, model, 1)
        gkw = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=True,
        )
        if cache is not None:
            gkw["past_key_values"] = cache
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], **gkw
            )
        dt = time.time() - t0
        gen_ids = out[0][plen:]
        n_gen = int(gen_ids.shape[0])
        raw = tok.decode(gen_ids, skip_special_tokens=False)
        clean = tok.decode(gen_ids, skip_special_tokens=True)
        boxed = re.findall(r"\\boxed\{([^}]*)\}", clean)
        print(f"\n########## CATEGORY={cat} ##########", flush=True)
        print(
            f"prompt_tokens={plen} gen_tokens={n_gen} hit_cap={n_gen>=args.max_new_tokens} "
            f"gen_s={dt:.1f} has_think_close={'</think>' in raw} "
            f"n_boxed={len(boxed)} gold={r.get('answer')!r}",
            flush=True,
        )
        print("----- RAW HEAD (first 600 chars) -----", flush=True)
        print(raw[:600], flush=True)
        print("----- RAW TAIL (last 1200 chars) -----", flush=True)
        print(raw[-1200:], flush=True)
        print("########## /END ##########\n", flush=True)


if __name__ == "__main__":
    main()
