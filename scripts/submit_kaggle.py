#!/usr/bin/env python3
"""
Package a LoRA adapter for Kaggle submission.

The Kaggle challenge expects a submission.zip containing:
  - adapter_config.json
  - adapter_model.safetensors

Usage:
    python scripts/submit_kaggle.py --adapter checkpoints/final/ --output submission.zip
"""

import argparse
import json
import logging
import os
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def package_submission(adapter_path: str, output_path: str):
    """Create submission.zip from adapter checkpoint."""
    adapter_dir = Path(adapter_path)

    required = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [f for f in required if not (adapter_dir / f).exists()]
    if missing:
        logger.error(f"Missing required files: {missing}")
        logger.info("Looking for sharded safetensors...")
        # Check for sharded adapter
        shards = list(adapter_dir.glob("adapter_model-*-of-*.safetensors"))
        if shards:
            logger.info(f"Found {len(shards)} sharded adapter files — merging not yet supported")
        raise FileNotFoundError(f"Missing: {missing}")

    logger.info(f"Packaging adapter from {adapter_path}")
    logger.info(f"  - adapter_config.json ({os.path.getsize(adapter_dir / 'adapter_config.json')} bytes)")
    logger.info(f"  - adapter_model.safetensors ({os.path.getsize(adapter_dir / 'adapter_model.safetensors')} bytes)")

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(adapter_dir / "adapter_config.json", "adapter_config.json")
        zf.write(adapter_dir / "adapter_model.safetensors", "adapter_model.safetensors")

        # Include any additional shards
        for shard in sorted(adapter_dir.glob("adapter_model-*-of-*.safetensors")):
            zf.write(shard, shard.name)
            logger.info(f"  - {shard.name}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Submission saved: {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Package Kaggle submission")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--output", default="submission.zip", help="Output zip path")
    args = parser.parse_args()

    package_submission(args.adapter, args.output)


if __name__ == "__main__":
    main()
