#!/bin/bash
# harvest_eco.slurm — Parallel PubMed FTS5 harvest for ecological papers
# Runs harvest_papers.py on a dedicated compute node to avoid login-node contention.
#
# Usage:
#   sbatch hpc/harvest_eco.slurm
#   sbatch --export=ALL,LIMIT=500,WORKERS=16 hpc/harvest_eco.slurm
#
# Output: /home/a474r867/scratch/cot_generation/eco_papers_extra.jsonl

#SBATCH --job-name=harvest-eco
#SBATCH --partition=sixhour
#SBATCH --time=05:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/a474r867/scratch/harvest_eco_%j.out
#SBATCH --error=/home/a474r867/scratch/harvest_eco_%j.err

LIMIT=${LIMIT:-300}
WORKERS=${WORKERS:-8}
OUTPUT=/home/a474r867/scratch/cot_generation/eco_papers_extra.jsonl
SCRIPT=/home/a474r867/scratch/cot_generation/harvest_papers.py

echo "=== HARVEST ECO PARALLEL ==="
echo "  Limit:   $LIMIT papers/topic"
echo "  Workers: $WORKERS"
echo "  Output:  $OUTPUT"
echo "  Date:    $(date)"

module load python/3.11 2>/dev/null || true

python3 $SCRIPT --limit $LIMIT --workers $WORKERS --output $OUTPUT

EXIT=$?
echo "Exit: $EXIT at $(date)"
exit $EXIT
