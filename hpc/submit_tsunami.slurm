#!/bin/bash
# submit_tsunami.slurm — V2: Uses ollama serve (GPU-tested) instead of llama-server
# Each task: start ollama serve → process chunk → kill
#SBATCH --job-name=cot-tsunami
#SBATCH --partition=sixhour
#SBATCH --time=05:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=/home/a474r867/scratch/cot_tsunami_%A_%a.out
#SBATCH --error=/home/a474r867/scratch/cot_tsunami_%A_%a.err

WORK_DIR=/home/a474r867/scratch/cot_generation
MANIFEST=$WORK_DIR/manifest.txt
SCRIPT=$WORK_DIR/process_chunk.py
OLLAMA_SIF="/home/a474r867/work/ollama/ollama.sif"

CHUNK_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
[ -z "$CHUNK_FILE" ] && { echo "ERROR: No chunk $SLURM_ARRAY_TASK_ID"; exit 1; }
OUTPUT_FILE="${CHUNK_FILE%.jsonl}_out.jsonl"

echo "=== TASK ${SLURM_ARRAY_TASK_ID} ==="
echo "  Node: $(hostname) | GPU: $(nvidia-smi -L 2>/dev/null || rocm-smi --showproductname 2>/dev/null || echo unknown)"
echo "  Chunk: $CHUNK_FILE ($(wc -l < $CHUNK_FILE) papers)"

# ── Start ollama serve (GPU-tested, unlike bare llama-server) ──
PORT=11434
echo "Starting ollama serve on port $PORT..."
apptainer instance start \
    --nv \
    --bind "/home/a474r867/scratch/ollama-models:/root/.ollama" \
    "$OLLAMA_SIF" \
    ollama_tsunami_${SLURM_ARRAY_TASK_ID} \
    ollama serve

# Cleanup
cleanup() {
    echo "Stopping ollama..."
    apptainer instance stop ollama_tsunami_${SLURM_ARRAY_TASK_ID} 2>/dev/null
    echo "Cleanup done at $(date)"
}
trap cleanup EXIT

# ── Wait for server ──
echo "Waiting for ollama..."
for i in $(seq 1 90); do
    if curl -s "http://127.0.0.1:${PORT}/api/tags" >/dev/null 2>&1; then
        echo "Ollama ready after ${i}s!"
        break
    fi
    sleep 2
done

# ── Process chunk ──
echo "Processing chunk..."
python3 "$SCRIPT" \
    "$CHUNK_FILE" \
    --server "127.0.0.1:${PORT}" \
    --output "$OUTPUT_FILE" \
    --delay 0.05

EXIT=$?
echo "Exit: $EXIT at $(date)"
exit $EXIT
