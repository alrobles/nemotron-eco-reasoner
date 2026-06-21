#!/bin/bash
# ingest_s2orc.slurm — Download S2ORC index, filter ecology, extract text
# S2ORC: 8.1M papers, ~50GB compressed index
# Downloads index, streams through Python filter, downloads matching papers
#
# SBATCH --job-name=s2orc-ingest
#SBATCH --partition=sixhour
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/a474r867/scratch/s2orc_ingest_%j.out
#SBATCH --error=/home/a474r867/scratch/s2orc_ingest_%j.err

INDEX_URL="https://ai2-s3-research.s3.amazonaws.com/s2orc/2023-11/full/v0/paper_locations.txt.gz"
WORK_DIR=/home/a474r867/scratch/s2orc
OUTPUT=/home/a474r867/scratch/cot_generation/s2orc_papers.jsonl
SCRIPT=/home/a474r867/scratch/cot_generation/ingest_s2orc.py

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "=== S2ORC INGEST ==="
echo "  Index:    $INDEX_URL"
echo "  Output:   $OUTPUT"
echo "  Limit:    20000 papers"
echo "  Date:     $(date)"

# Download index (stream, don't save to disk fully)
echo "Downloading index..."
wget -q -O - "$INDEX_URL" | zcat | head -n 5000000 | \
    python3 "$SCRIPT" --output "$OUTPUT" --limit 20000 --delay 0.05

EXIT=$?
echo "Exit: $EXIT at $(date)"
exit $EXIT
