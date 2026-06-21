#!/bin/bash
#SBATCH --job-name=harvest-biorxiv
#SBATCH --partition=sixhour
#SBATCH --time=02:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/a474r867/scratch/harvest_biorxiv_%j.out
#SBATCH --error=/home/a474r867/scratch/harvest_biorxiv_%j.err
python3 /home/a474r867/scratch/cot_generation/harvest_biorxiv.py --limit 100 --output /home/a474r867/scratch/cot_generation/biorxiv_papers.jsonl
echo "Exit: $? at $(date)"
