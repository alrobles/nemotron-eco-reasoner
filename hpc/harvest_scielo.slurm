#!/bin/bash
#SBATCH --job-name=harvest-scielo
#SBATCH --partition=sixhour
#SBATCH --time=02:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/a474r867/scratch/harvest_scielo_%j.out
#SBATCH --error=/home/a474r867/scratch/harvest_scielo_%j.err
python3 /home/a474r867/scratch/cot_generation/harvest_scielo.py --limit 100 --output /home/a474r867/scratch/cot_generation/scielo_papers.jsonl
echo "Exit: $? at $(date)"
