#!/bin/bash
# harvest_arxiv.slurm — Query arXiv API for ecological papers
# arXiv categories: q-bio.PE, q-bio.QM, stat.AP, cs.LG
#
# Usage:
#   sbatch hpc/harvest_arxiv.slurm
#   sbatch --export=ALL,LIMIT=100,WORKERS=5 hpc/harvest_arxiv.slurm

#SBATCH --job-name=harvest-arxiv
#SBATCH --partition=sixhour
#SBATCH --time=01:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/a474r867/scratch/harvest_arxiv_%j.out
#SBATCH --error=/home/a474r867/scratch/harvest_arxiv_%j.err

LIMIT=${LIMIT:-100}
WORKERS=${WORKERS:-5}
OUTPUT=/home/a474r867/scratch/cot_generation/arxiv_papers.jsonl
SCRIPT=/home/a474r867/scratch/cot_generation/harvest_arxiv.py

echo "=== HARVEST ARXIV ==="
echo "  Limit:   $LIMIT papers/query"
echo "  Workers: $WORKERS"
echo "  Output:  $OUTPUT"
echo "  Date:    $(date)"

python3 $SCRIPT --limit $LIMIT --workers $WORKERS --output $OUTPUT

EXIT=$?
echo "Exit: $EXIT at $(date)"
exit $EXIT
