#!/usr/bin/env python3
"""Merge several LoRA adapters into ONE rank-32 adapter (Kaggle-legal).

Every adapter here is a low-rank delta on the SAME base model:
    delta_W = (alpha / r) * B @ A          (A: [r, in], B: [out, r])
so combining adapters is combining their deltas. The catch for this competition
is max_lora_rank=32: summing two rank-32 deltas gives rank up to 64, which is
illegal. The `svd`/`ties` modes recompress the combined delta back to rank 32 by
construction (keep the 32 largest singular values), so the output is always legal.

Modes (the ladder from the merging research):
  * ckpt_avg : element-wise average of A and B across adapters trained in the SAME
               run (same basin) -> stays rank 32, no SVD. Lowest risk denoise.
  * svd      : weighted soup. delta = sum_i w_i * scale_i * B_i @ A_i, then SVD
               truncate to rank 32. For adapters from different runs/seeds/data.
  * ties     : TIES-style. Trim each delta to its top-`density` magnitude entries,
               elect the majority sign per weight, keep only agreeing deltas, then
               SVD truncate to rank 32. Robust when adapters interfere (opposite
               signs on the same weights).

The merge runs on CPU from the .safetensors alone (no 30B base load needed).

Usage:
  python scripts/merge_adapters.py --mode svd --rank 32 --alpha 32 \
      --out outputs/merge_svd \
      --adapter outputs/tied_v14/checkpoint-500:0.5 \
      --adapter outputs/mi210_v14/checkpoint-200:0.5
  # weight after ':' is optional (defaults to equal weights)
"""
import argparse
import json
import os

import torch
from safetensors.torch import load_file, save_file


def find_adapter_file(path):
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        f = os.path.join(path, name)
        if os.path.exists(f):
            return f
    raise FileNotFoundError(f"no adapter weights in {path}")


def load_adapter(path):
    """Return (state_dict, scaling=alpha/r, r) for an adapter checkpoint dir."""
    cfg = json.load(open(os.path.join(path, "adapter_config.json")))
    r = int(cfg["r"])
    alpha = float(cfg["lora_alpha"])
    f = find_adapter_file(path)
    if f.endswith(".safetensors"):
        sd = load_file(f)
    else:
        sd = torch.load(f, map_location="cpu")
    return sd, alpha / r, r, cfg


def lora_pairs(sd):
    """Yield (module_prefix, A_key, B_key) for every LoRA pair in a state dict."""
    for k in sd:
        if ".lora_A." in k and k.endswith(".weight"):
            bk = k.replace(".lora_A.", ".lora_B.")
            if bk in sd:
                prefix = k.split(".lora_A.")[0]
                yield prefix, k, bk


def factor_delta(delta, rank, dtype):
    """SVD-truncate a [out, in] delta to rank, return (A[rank,in], B[out,rank])
    such that B @ A == U S V^T (singular values folded in). Output scaling is 1.
    """
    delta32 = delta.to(torch.float32)
    U, S, Vh = torch.linalg.svd(delta32, full_matrices=False)
    r = min(rank, S.shape[0])
    U, S, Vh = U[:, :r], S[:r], Vh[:r, :]
    sqrt_s = torch.sqrt(S)
    B = (U * sqrt_s.unsqueeze(0)).to(dtype).contiguous()
    A = (sqrt_s.unsqueeze(1) * Vh).to(dtype).contiguous()
    return A, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ckpt_avg", "svd", "ties"], required=True)
    ap.add_argument(
        "--adapter",
        action="append",
        required=True,
        help="checkpoint_dir[:weight]; repeat for each adapter",
    )
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32, help="output lora_alpha")
    ap.add_argument("--density", type=float, default=0.2, help="TIES keep fraction")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    specs = []
    for a in args.adapter:
        if ":" in a and not a[a.rfind(":") + 1 :].startswith("/"):
            path, w = a.rsplit(":", 1)
            specs.append((path, float(w)))
        else:
            specs.append((a, 1.0))
    # normalise weights
    tot = sum(w for _, w in specs)
    specs = [(p, w / tot) for p, w in specs]
    print(f"mode={args.mode} out_rank={args.rank} out_alpha={args.alpha}")
    for p, w in specs:
        print(f"  adapter {p}  weight={w:.4f}")

    loaded = [(load_adapter(p), w) for p, w in specs]
    base_sd = loaded[0][0][0]
    base_cfg = loaded[0][0][3]
    new_sd = {}

    if args.mode == "ckpt_avg":
        scales = {round(s, 6) for (_, s, _, _), _ in loaded}
        if len(scales) != 1:
            raise SystemExit(
                f"ckpt_avg needs identical alpha/r across adapters, got {scales}; "
                "use --mode svd for adapters with different alpha"
            )
        args.alpha = int(round(loaded[0][0][1] * args.rank))  # keep input scaling
        # element-wise weighted average of A and B (same-run, same shapes)
        for prefix, ak, bk in lora_pairs(base_sd):
            A = sum(w * sd[ak].to(torch.float32) for (sd, _, _, _), w in loaded)
            B = sum(w * sd[bk].to(torch.float32) for (sd, _, _, _), w in loaded)
            new_sd[ak] = A.to(base_sd[ak].dtype).contiguous()
            new_sd[bk] = B.to(base_sd[bk].dtype).contiguous()
        # also carry through any non-LoRA tensors (rare) unchanged
    else:
        for prefix, ak, bk in lora_pairs(base_sd):
            dtype = base_sd[ak].dtype
            deltas = []
            for (sd, scale, _, _), w in loaded:
                if ak not in sd or bk not in sd:
                    continue
                A = sd[ak].to(torch.float32)
                B = sd[bk].to(torch.float32)
                deltas.append((w * scale, B @ A))  # [out, in]
            if not deltas:
                continue
            if args.mode == "svd":
                delta = sum(c * d for c, d in deltas)
            else:  # ties
                stacked = torch.stack([d for _, d in deltas], dim=0)  # [n,out,in]
                coeffs = torch.tensor([c for c, _ in deltas]).view(-1, 1, 1)
                # trim: per-adapter keep top-density magnitude, zero the rest
                n = stacked.shape[0]
                flat = stacked.abs().reshape(n, -1)
                k = max(1, int(args.density * flat.shape[1]))
                thr = flat.kthvalue(flat.shape[1] - k + 1, dim=1).values.view(-1, 1)
                mask = (stacked.abs().reshape(n, -1) >= thr).reshape_as(stacked)
                trimmed = stacked * mask
                signed = trimmed * coeffs
                # elect sign from the summed signed magnitude
                agg_sign = torch.sign(signed.sum(dim=0))  # [out,in]
                keep = (torch.sign(trimmed) == agg_sign.unsqueeze(0)).float()
                delta = (signed * keep).sum(dim=0)
            A, B = factor_delta(delta, args.rank, dtype)
            new_sd[ak] = A
            new_sd[bk] = B

    os.makedirs(args.out, exist_ok=True)
    save_file(new_sd, os.path.join(args.out, "adapter_model.safetensors"))
    out_cfg = dict(base_cfg)
    out_cfg["r"] = args.rank
    out_cfg["lora_alpha"] = args.alpha
    json.dump(out_cfg, open(os.path.join(args.out, "adapter_config.json"), "w"), indent=2)
    n_pairs = sum(1 for _ in lora_pairs(new_sd))
    print(f"saved {len(new_sd)} tensors ({n_pairs} LoRA pairs) -> {args.out}")
    print(f"out r={args.rank} alpha={args.alpha} scale={args.alpha / args.rank}")


if __name__ == "__main__":
    main()
