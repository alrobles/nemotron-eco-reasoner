#!/bin/bash
# harvest_ecoevorxiv.slurm — Harvest ecoevorxiv preprints
#SBATCH --job-name=harvest-ecorxiv
#SBATCH --partition=sixhour
#SBATCH --time=01:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/a474r867/scratch/harvest_ecoevorxiv_%j.out
#SBATCH --error=/home/a474r867/scratch/harvest_ecoevorxiv_%j.err

python3 /home/a474r867/scratch/cot_generation/harvest_ecoevorxiv.py \
    --limit 50 \
    --output /home/a474r867/scratch/cot_generation/ecoevorxiv_papers.jsonl
EXIT=$?
echo "Exit: $EXIT at $(date)"
exit $EXIT
