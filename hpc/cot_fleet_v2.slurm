#!/bin/bash
# cot_fleet_v2.slurm — Phase 2: process ALL papers with 7 endpoints
# Concatenates PubMed extra + GBIF + arXiv, then runs CoT generation.
#
# Usage: sbatch hpc/cot_fleet_v2.slurm

#SBATCH --job-name=cot-fleet2
#SBATCH --partition=sixhour
#SBATCH --time=05:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/a474r867/scratch/cot_fleet2_%j.out
#SBATCH --error=/home/a474r867/scratch/cot_fleet2_%j.err

DIR=/home/a474r867/scratch/cot_generation
SCRIPT=$DIR/cot_ollama_fleet.py
ENDPOINTS=$DIR/endpoints_q6000.txt

# Concatenate all new papers
echo "=== CONCATENATING PAPERS ==="
cat $DIR/eco_papers_extra.jsonl $DIR/gbif_papers.jsonl $DIR/arxiv_papers.jsonl > $DIR/all_new_papers.jsonl
TOTAL=$(wc -l < $DIR/all_new_papers.jsonl)
echo "  Total new papers: $TOTAL"

# Kill old cot-fleet if running
scancel -n cot-fleet 2>/dev/null || true
sleep 3

echo "=== STARTING COT FLEET v2 ==="
echo "  Papers:   $TOTAL"
echo "  Endpoints: 7"
echo "  Date:     $(date)"

python3 "$SCRIPT" \
    --papers "$DIR/all_new_papers.jsonl" \
    --endpoints "$ENDPOINTS" \
    --output "$DIR/cot_eco_batch2.jsonl" \
    --delay 0.4 \
    --timeout 240 \
    --shuffle

EXIT=$?
echo "Exit: $EXIT at $(date)"
exit $EXIT
