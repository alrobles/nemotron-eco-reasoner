#!/usr/bin/env python3
"""SVD denoising for LoRA adapters.

Takes a trained rank-32 LoRA adapter and removes noise by zeroing out
the weakest singular values. Uses QR decomposition for efficiency.

Based on the technique from the Kaggle Nemotron competition (0.86 LB).

Usage:
  python scripts/denoise_adapter.py \
      --input outputs/v8_seq3072/checkpoint-500 \
      --output outputs/v8_denoised_08 \
      --keep-ratio 0.8
"""
import argparse
import json
import os
import shutil
import zipfile

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def denoise_lora(input_dir, output_dir, keep_ratio=0.8):
    """Apply SVD denoising to a LoRA adapter, keeping top keep_ratio singular values."""

    # Find adapter files
    safetensors_path = os.path.join(input_dir, "adapter_model.safetensors")
    config_path = os.path.join(input_dir, "adapter_config.json")

    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"No adapter_model.safetensors in {input_dir}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No adapter_config.json in {input_dir}")

    # Load weights
    print(f"Loading adapter from {input_dir}...")
    state_dict = load_file(safetensors_path)
    config = json.load(open(config_path))

    print(f"Applying SVD denoising (keep_ratio={keep_ratio})...")
    modified_state_dict = {}

    # Group pairs of A and B matrices
    lora_pairs = []
    standalone_keys = []

    for key in state_dict.keys():
        if 'lora_A' in key:
            b_key = key.replace('lora_A', 'lora_B')
            if b_key in state_dict:
                lora_pairs.append((key, b_key))
            else:
                standalone_keys.append(key)
        elif 'lora_B' in key:
            a_key = key.replace('lora_B', 'lora_A')
            if a_key not in state_dict:
                standalone_keys.append(key)
        else:
            standalone_keys.append(key)

    # Copy standalone keys as-is
    for key in standalone_keys:
        modified_state_dict[key] = state_dict[key]

    # Process LoRA pairs with SVD denoising
    for a_key, b_key in tqdm(lora_pairs, desc="Denoising LoRA Pairs"):
        A = state_dict[a_key]
        B = state_dict[b_key]

        orig_A_shape = A.shape
        orig_B_shape = B.shape
        orig_dtype = A.dtype

        # Cast to float32 for SVD
        A_float = A.to(torch.float32)
        B_float = B.to(torch.float32)

        # Flatten to 2D (handles Conv1d/Conv2d weights)
        r = A_float.shape[0]  # Rank is first dim of A
        A_flat = A_float.view(r, -1)
        B_flat = B_float.view(-1, r)

        # Efficient SVD via QR decomposition
        # Instead of SVD on (B @ A) which could be huge,
        # use QR to reduce to an [r, r] problem
        Q_A, R_A = torch.linalg.qr(A_flat.T)
        Q_B, R_B = torch.linalg.qr(B_flat)

        # Inner matrix SVD on [r, r]
        M = R_B @ R_A.T  # [r, r]
        U_m, S, Vh_m = torch.linalg.svd(M)  # S is [r]

        # Zero out the weakest singular values
        k = max(1, int(r * keep_ratio))
        S_new = S.clone()
        S_new[k:] = 0.0  # Drop the (r - k) weakest components

        # Reconstruct denoised matrices
        sqrt_S = torch.diag(torch.sqrt(S_new))
        B_new_flat = Q_B @ U_m @ sqrt_S
        A_new_flat = sqrt_S @ Vh_m @ Q_A.T

        # Convert back to original shapes and dtypes
        modified_state_dict[a_key] = A_new_flat.view(orig_A_shape).to(orig_dtype)
        modified_state_dict[b_key] = B_new_flat.view(orig_B_shape).to(orig_dtype)

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    out_safetensors = os.path.join(output_dir, "adapter_model.safetensors")
    out_config = os.path.join(output_dir, "adapter_config.json")

    print(f"Saving denoised adapter to {output_dir}...")
    save_file(modified_state_dict, out_safetensors)
    shutil.copy2(config_path, out_config)

    print(f"Done! Denoised {len(lora_pairs)} LoRA pairs (kept top {k}/{r} components)")
    return output_dir


def package_submission(adapter_dir, output_zip):
    """Package adapter as submission.zip for Kaggle."""
    safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    config_path = os.path.join(adapter_dir, "adapter_config.json")

    print(f"Creating submission zip: {output_zip}")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        zf.write(safetensors_path, "adapter_model.safetensors")
        zf.write(config_path, "adapter_config.json")

    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"Submission zip: {size_mb:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SVD denoise a LoRA adapter")
    ap.add_argument("--input", required=True, help="Input adapter directory")
    ap.add_argument("--output", required=True, help="Output adapter directory")
    ap.add_argument("--keep-ratio", type=float, default=0.8,
                    help="Fraction of singular values to keep (default: 0.8)")
    ap.add_argument("--zip", action="store_true",
                    help="Also create submission.zip")
    args = ap.parse_args()

    out_dir = denoise_lora(args.input, args.output, args.keep_ratio)

    if args.zip:
        zip_path = os.path.join(args.output, "submission.zip")
        package_submission(out_dir, zip_path)
