#!/bin/bash
# cot_fleet.slurm — Run cot_ollama_fleet.py as a Slurm job
# Talks to Ollama endpoints on compute nodes and writes incrementally.
#
# Usage:
#   sbatch --export=ALL,PAPERS=...,SKIP=400 hpc/cot_fleet.slurm
#   sbatch --export=ALL,PAPERS=pubmed_sdm_papers.jsonl,SKIP=0 hpc/cot_fleet.slurm

#SBATCH --job-name=cot-fleet
#SBATCH --partition=sixhour
#SBATCH --time=05:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/a474r867/scratch/cot_fleet_%j.out
#SBATCH --error=/home/a474r867/scratch/cot_fleet_%j.err

PAPERS=${PAPERS:-/home/a474r867/scratch/ecocoder_harvest/pubmed_sdm_papers.jsonl}
ENDPOINTS=/home/a474r867/scratch/cot_generation/endpoints_q6000.txt
OUTPUT=/home/a474r867/scratch/cot_generation/cot_eco_full.jsonl
SKIP=${SKIP:-0}

SCRIPT=/home/a474r867/scratch/cot_generation/cot_ollama_fleet.py

echo "=== COT FLEET ==="
echo "  Papers:   $PAPERS"
echo "  Output:   $OUTPUT"
echo "  Skip:     $SKIP"
echo "  Date:     $(date)"
echo "  Node:     $(hostname)"

# Ensure output dir exists
mkdir -p "$(dirname "$OUTPUT")"

python3 "$SCRIPT" \
    --papers "$PAPERS" \
    --endpoints "$ENDPOINTS" \
    --output "$OUTPUT" \
    --delay 0.5 \
    --timeout 240 \
    --shuffle \
    --skip "$SKIP"

EXIT=$?
echo "Exit: $EXIT at $(date)"
exit $EXIT
