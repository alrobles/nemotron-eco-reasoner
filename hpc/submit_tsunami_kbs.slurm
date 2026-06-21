#!/bin/bash
# submit_tsunami_kbs.slurm — KBS partition version (no time limit)
#SBATCH --job-name=cot-tsunami-kbs
#SBATCH --partition=kbs
#SBATCH --time=0
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=/home/a474r867/scratch/cot_tsunami_kbs_%A_%a.out
#SBATCH --error=/home/a474r867/scratch/cot_tsunami_kbs_%A_%a.err

WORK_DIR=/home/a474r867/scratch/cot_generation
MANIFEST=$WORK_DIR/manifest.txt
SCRIPT=$WORK_DIR/process_chunk.py
OLLAMA_SIF="/home/a474r867/work/ollama/ollama.sif"

CHUNK_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
[ -z "$CHUNK_FILE" ] && { echo "ERROR: No chunk $SLURM_ARRAY_TASK_ID"; exit 1; }
OUTPUT_FILE="${CHUNK_FILE%.jsonl}_out.jsonl"

PORT=11434
echo "=== TASK ${SLURM_ARRAY_TASK_ID} (KBS) ==="
echo "  Chunk: $CHUNK_FILE ($(wc -l < $CHUNK_FILE) papers)"

apptainer instance start --nv --bind "/home/a474r867/scratch/ollama-models:/root/.ollama" \
    "$OLLAMA_SIF" ollama_kbs_${SLURM_ARRAY_TASK_ID} ollama serve
trap "apptainer instance stop ollama_kbs_${SLURM_ARRAY_TASK_ID} 2>/dev/null" EXIT

for i in $(seq 1 90); do
    curl -s "http://127.0.0.1:${PORT}/api/tags" >/dev/null 2>&1 && { echo "Ready after ${i}s!"; break; }
    sleep 2
done

python3 "$SCRIPT" "$CHUNK_FILE" --server "127.0.0.1:${PORT}" --output "$OUTPUT_FILE" --delay 0.05
EXIT=$?
echo "Exit: $EXIT at $(date)"
exit $EXIT
