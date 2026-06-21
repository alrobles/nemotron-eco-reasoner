#!/bin/bash
# harvest_gbif.slurm — Query GBIF Literature FTS5 for ecological papers
# GBIF has 61,561 papers that cite biodiversity data.
#
# Usage:
#   sbatch hpc/harvest_gbif.slurm
#   sbatch --export=ALL,LIMIT=500,WORKERS=12 hpc/harvest_gbif.slurm

#SBATCH --job-name=harvest-gbif
#SBATCH --partition=sixhour
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=6
#SBATCH --output=/home/a474r867/scratch/harvest_gbif_%j.out
#SBATCH --error=/home/a474r867/scratch/harvest_gbif_%j.err

LIMIT=${LIMIT:-500}
WORKERS=${WORKERS:-10}
OUTPUT=/home/a474r867/scratch/cot_generation/gbif_papers.jsonl
SCRIPT=/home/a474r867/scratch/cot_generation/harvest_gbif.py

echo "=== HARVEST GBIF ==="
echo "  Limit:   $LIMIT papers/topic"
echo "  Workers: $WORKERS"
echo "  Output:  $OUTPUT"
echo "  Date:    $(date)"

python3 $SCRIPT --limit $LIMIT --workers $WORKERS --output $OUTPUT

EXIT=$?
echo "Exit: $EXIT at $(date)"
exit $EXIT
