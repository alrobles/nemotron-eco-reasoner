#!/bin/bash
# GPU Cluster Monitor — kbs partition
#SBATCH --job-name=gpu-mon
#SBATCH --partition=kbs
#SBATCH --time=60-00:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/a474r867/scratch/gpu_monitor_%j.out
#SBATCH --error=/home/a474r867/scratch/gpu_monitor_%j.err

echo "=== GPU Cluster Monitor ==="
echo "Started: $(date)"
python3 /home/a474r867/scratch/gpu_monitor.py
