#!/bin/bash
#SBATCH --job-name=nem-orch2
#SBATCH --partition=kbs
#SBATCH --time=60-00:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/a474r867/scratch/orch_v2_%j.out
#SBATCH --error=/home/a474r867/scratch/orch_v2_%j.err
echo "=== Orchestrator V2 ==="
python3 /home/a474r867/scratch/orch_v2.py
