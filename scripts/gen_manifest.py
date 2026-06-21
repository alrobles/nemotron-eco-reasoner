#!/usr/bin/env python3
"""
gen_manifest.py — Split paper JSONL files into chunks for Slurm job array processing.
Each chunk = N papers, each Slurm task = 1 chunk, processed by llama.cpp directly.

Usage:
    python3 gen_manifest.py --input cot_generation/all_papers.jsonl --chunk-size 50 --output manifest.txt
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file with papers")
    parser.add_argument("--chunk-size", type=int, default=50, help="Papers per chunk")
    parser.add_argument("--output", default="manifest.txt")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        sys.exit(f"ERROR: Input file not found: {args.input}")
    
    # Read all papers
    papers = []
    with open(args.input, "r") as f:
        for line in f:
            if line.strip():
                papers.append(line.strip())
    
    total = len(papers)
    chunks = (total + args.chunk_size - 1) // args.chunk_size
    
    print(f"Manifest: {total} papers → {chunks} chunks of {args.chunk_size}")
    
    # Write manifest: one file per chunk
    with open(args.output, "w") as f:
        for i in range(chunks):
            start = i * args.chunk_size
            end = min(start + args.chunk_size, total)
            chunk_file = f"chunk_{i:05d}.jsonl"
            f.write(f"{chunk_file}\n")
            
            # Write chunk file
            with open(chunk_file, "w") as cf:
                for j in range(start, end):
                    cf.write(papers[j] + "\n")
    
    print(f"Wrote {args.output} with {chunks} chunks")
    print(f"Job array: sbatch --array=1-{chunks} submit_tsunami.slurm")
    print(f"First chunk: chunk_00000.jsonl ({min(args.chunk_size, total)} papers)")


if __name__ == "__main__":
    main()
