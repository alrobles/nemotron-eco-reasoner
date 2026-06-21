#!/bin/bash
#SBATCH --job-name=harvest-biorxiv
#SBATCH --partition=sixhour
#SBATCH --time=04:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/a474r867/scratch/harvest_biorxiv_v2_%j.out
#SBATCH --error=/home/a474r867/scratch/harvest_biorxiv_v2_%j.err
python3 /home/a474r867/scratch/cot_generation/harvest_biorxiv.py \
    --limit 200 --output /home/a474r867/scratch/cot_generation/biorxiv_papers.jsonl
echo "Exit: $? at $(date)"
