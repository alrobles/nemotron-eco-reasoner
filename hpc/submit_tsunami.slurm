#!/bin/bash
# submit_tsunami.slurm — Massive parallel CoT generation via Slurm job array
# Each task: start llama-server → process chunk → kill server
# Spawn up to 20 concurrent jobs, Slurm queues the rest automatically.
#
# Papers per task = chunk_size * papers_per_task
# Total tasks = number of chunks in manifest
#
# Usage:
#   sbatch --array=1-N%20 submit_tsunami.slurm
#   Or with custom concurrency: sbatch --array=1-N%30 submit_tsunami.slurm

#SBATCH --job-name=cot-tsunami
#SBATCH --partition=sixhour
#SBATCH --time=05:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=/home/a474r867/scratch/cot_tsunami_%A_%a.out
#SBATCH --error=/home/a474r867/scratch/cot_tsunami_%A_%a.err

# ── Config ──────────────────────────────────────────────────
WORK_DIR=/home/a474r867/scratch/cot_generation
MANIFEST=$WORK_DIR/manifest.txt
SCRIPT=$WORK_DIR/process_chunk.py

# Model paths (pick one based on GPU type)
# Q6000: Q4_K_M fits in 24GB
# MI210: Q4_K_M or Q6_K fits in 64GB
MODEL_PATH=${MODEL_PATH:-/home/a474r867/scratch/ollama-models/blobs/sha256-6e9f90f02bb3b39b59e81916e8cfce9deb45aeaeb9a54a5be4414486b907dc1e}
MODEL_NAME="deepseek-r1-14b-Q4_K_M.gguf"

# Ollama container (has llama-server at /usr/lib/ollama/llama-server)
OLLAMA_SIF="/home/a474r867/work/ollama/ollama.sif"
LLAMA_BIN="/usr/lib/ollama/llama-server"
LLAMA_PORT=8080
LLAMA_HOST=127.0.0.1

# ── Get chunk file for this task ────────────────────────────
CHUNK_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
if [ -z "$CHUNK_FILE" ]; then
    echo "ERROR: No chunk file for task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi
OUTPUT_FILE="${CHUNK_FILE%.jsonl}_out.jsonl"

echo "=== TASK ${SLURM_ARRAY_TASK_ID} ==="
echo "  Node:     $(hostname)"
echo "  GPU:      $(nvidia-smi -L 2>/dev/null || rocm-smi --showproductname 2>/dev/null || echo 'unknown')"
echo "  Chunk:    $CHUNK_FILE"
echo "  Papers:   $(wc -l < $CHUNK_FILE)"
echo "  Output:   $OUTPUT_FILE"
echo "  Date:     $(date)"

# ── Start llama-server via apptainer ────────────────────────
echo "Starting llama-server on port $LLAMA_PORT..."
apptainer exec --nv --bind "/home/a474r867/scratch/ollama-models:/models" "$OLLAMA_SIF" \
    $LLAMA_BIN \
    --model "/models/blobs/sha256-6e9f90f02bb3b39b59e81916e8cfce9deb45aeaeb9a54a5be4414486b907dc1e" \
    --host "$LLAMA_HOST" \
    --port "$LLAMA_PORT" \
    --n-gpu-layers 99 \
    --ctx-size 8192 \
    --batch-size 512 \
    --threads 4 \
    --no-webui \
    --log-disable \
    &
SERVER_PID=$!

# Cleanup on exit
cleanup() {
    echo "Shutting down llama-server (PID $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    echo "Cleanup done at $(date)"
}
trap cleanup EXIT

# ── Wait for server to be ready ─────────────────────────────
echo "Waiting for server readiness..."
for i in $(seq 1 60); do
    if curl -s "http://${LLAMA_HOST}:${LLAMA_PORT}/health" >/dev/null 2>&1; then
        echo "Server ready after ${i}s!"
        break
    fi
    sleep 2
done

# ── Process chunk ───────────────────────────────────────────
echo "Processing chunk..."
python3 "$SCRIPT" \
    "$CHUNK_FILE" \
    --server "${LLAMA_HOST}:${LLAMA_PORT}" \
    --output "$OUTPUT_FILE" \
    --delay 0.05

EXIT=$?

echo "=== TASK ${SLURM_ARRAY_TASK_ID} DONE ==="
echo "  Exit:     $EXIT"
echo "  Date:     $(date)"

exit $EXIT
