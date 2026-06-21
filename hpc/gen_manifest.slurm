#!/bin/bash
#SBATCH --job-name=gen-manifest
#SBATCH --partition=sixhour
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/a474r867/scratch/gen_manifest_%j.out
#SBATCH --error=/home/a474r867/scratch/gen_manifest_%j.err

DIR=/home/a474r867/scratch/cot_generation
SCRIPT=$DIR/gen_manifest.py
OUTPUT=$DIR/all_eco_papers.jsonl
MANIFEST=$DIR/manifest.txt

echo "=== GEN MANIFEST ==="
echo "Concatenating all paper sources..."
cat $DIR/eco_papers_extra.jsonl $DIR/gbif_papers.jsonl $DIR/arxiv_papers.jsonl $DIR/ecoevorxiv_papers.jsonl $DIR/biorxiv_papers.jsonl 2>/dev/null > $OUTPUT

TOTAL=$(wc -l < $OUTPUT)
echo "Total papers: $TOTAL"

echo "Generating manifest (50 papers/chunk)..."
cd $DIR
python3 $SCRIPT --input $OUTPUT --chunk-size 50 --output $MANIFEST

CHUNKS=$(wc -l < $MANIFEST)
echo "Manifest: $CHUNKS chunks"
echo "Launch command: sbatch --array=1-$CHUNKS%20 $DIR/submit_tsunami.slurm"
echo "Done at $(date)"
