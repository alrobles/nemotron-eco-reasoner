#!/usr/bin/env python3
"""SVD denoising for LoRA adapters — NumPy-only version (no torch needed).

Takes a trained rank-32 LoRA adapter and removes noise by zeroing out
the weakest singular values. Uses QR decomposition for efficiency.

Dependencies: numpy, safetensors (pip install numpy safetensors)

Usage:
  python scripts/denoise_adapter_np.py \
      --input outputs/v8_seq3072/checkpoint-500 \
      --output outputs/v8_denoised_08 \
      --keep-ratio 0.8
"""
import argparse
import json
import os
import shutil
import struct
import zipfile

import numpy as np

# Try safetensors numpy API; fall back to manual parsing
try:
    from safetensors.numpy import load_file, save_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


def _load_safetensors_manual(path):
    """Load safetensors file without the safetensors package."""
    with open(path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header_json = f.read(header_size)
        header = json.loads(header_json)
        data_start = 8 + header_size
        tensors = {}
        for name, info in header.items():
            if name == '__metadata__':
                continue
            dtype_str = info['dtype']
            shape = info['shape']
            offsets = info['data_offsets']
            np_dtype = {
                'F32': np.float32, 'F16': np.float16,
                'BF16': np.float16,  # read as float16, we'll upcast
                'I32': np.int32, 'I64': np.int64,
                'U8': np.uint8,
            }.get(dtype_str, np.float32)
            byte_size = offsets[1] - offsets[0]
            f.seek(data_start + offsets[0])
            raw = f.read(byte_size)
            if dtype_str == 'BF16':
                # Convert bfloat16 to float32
                arr = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
                arr32 = np.zeros(shape, dtype=np.float32)
                arr32.view(np.uint32)[:] = arr.astype(np.uint32) << 16
                tensors[name] = arr32
            else:
                tensors[name] = np.frombuffer(raw, dtype=np_dtype).reshape(shape).copy()
    return tensors


def _save_safetensors_manual(tensors, path):
    """Save tensors in safetensors format without the safetensors package."""
    header = {}
    tensor_data = []
    offset = 0
    for name, arr in tensors.items():
        if arr.dtype == np.float32:
            dtype_str = 'F32'
        elif arr.dtype == np.float16:
            dtype_str = 'F16'
        else:
            dtype_str = 'F32'
            arr = arr.astype(np.float32)
        raw = arr.tobytes()
        header[name] = {
            'dtype': dtype_str,
            'shape': list(arr.shape),
            'data_offsets': [offset, offset + len(raw)]
        }
        tensor_data.append(raw)
        offset += len(raw)
    header_json = json.dumps(header).encode('utf-8')
    # Pad to 8-byte alignment
    padding = (8 - len(header_json) % 8) % 8
    header_json += b' ' * padding
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(header_json)))
        f.write(header_json)
        for raw in tensor_data:
            f.write(raw)


def load_adapter(path):
    if HAS_SAFETENSORS:
        return load_file(path)
    return _load_safetensors_manual(path)


def save_adapter(tensors, path):
    if HAS_SAFETENSORS:
        save_file(tensors, path)
    else:
        _save_safetensors_manual(tensors, path)


def denoise_lora(input_dir, output_dir, keep_ratio=0.8):
    """Apply SVD denoising to a LoRA adapter."""
    safetensors_path = os.path.join(input_dir, "adapter_model.safetensors")
    config_path = os.path.join(input_dir, "adapter_config.json")

    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"No adapter_model.safetensors in {input_dir}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No adapter_config.json in {input_dir}")

    print(f"Loading adapter from {input_dir}...")
    state_dict = load_adapter(safetensors_path)
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

    n_pairs = len(lora_pairs)
    print(f"Processing {n_pairs} LoRA pairs...")

    for i, (a_key, b_key) in enumerate(lora_pairs):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{n_pairs}] {a_key.split('.lora_A')[0][-40:]}")

        A = state_dict[a_key]
        B = state_dict[b_key]

        orig_A_shape = A.shape
        orig_B_shape = B.shape
        orig_dtype = A.dtype

        # Cast to float32 for SVD
        A_float = A.astype(np.float32)
        B_float = B.astype(np.float32)

        # Flatten to 2D
        r = A_float.shape[0]  # Rank is first dim of A
        A_flat = A_float.reshape(r, -1)
        B_flat = B_float.reshape(-1, r)

        # Efficient SVD via QR decomposition
        Q_A, R_A = np.linalg.qr(A_flat.T)
        Q_B, R_B = np.linalg.qr(B_flat)

        # Inner matrix SVD on [r, r]
        M = R_B @ R_A.T  # [r, r]
        U_m, S, Vh_m = np.linalg.svd(M, full_matrices=False)

        # Zero out the weakest singular values
        k = max(1, int(r * keep_ratio))
        S_new = S.copy()
        S_new[k:] = 0.0

        # Reconstruct denoised matrices
        sqrt_S = np.diag(np.sqrt(S_new))
        B_new_flat = Q_B @ U_m @ sqrt_S
        A_new_flat = sqrt_S @ Vh_m @ Q_A.T

        # Convert back to original shapes and dtypes
        modified_state_dict[a_key] = A_new_flat.reshape(orig_A_shape).astype(orig_dtype)
        modified_state_dict[b_key] = B_new_flat.reshape(orig_B_shape).astype(orig_dtype)

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    out_safetensors = os.path.join(output_dir, "adapter_model.safetensors")
    out_config = os.path.join(output_dir, "adapter_config.json")

    print(f"Saving denoised adapter to {output_dir}...")
    save_adapter(modified_state_dict, out_safetensors)
    shutil.copy2(config_path, out_config)

    print(f"Done! Denoised {n_pairs} LoRA pairs (kept top {k}/{r} components)")
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
    ap = argparse.ArgumentParser(description="SVD denoise a LoRA adapter (numpy-only)")
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
