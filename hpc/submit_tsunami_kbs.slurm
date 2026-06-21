#!/bin/bash
# submit_tsunami_kbs.slurm — Same as submit_tsunami but for kbs partition (NO time limit)
# Use this if sixhour jobs die at walltime.

#SBATCH --job-name=cot-tsunami-kbs
#SBATCH --partition=kbs
#SBATCH --time=0  # No limit
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=/home/a474r867/scratch/cot_tsunami_kbs_%A_%a.out
#SBATCH --error=/home/a474r867/scratch/cot_tsunami_kbs_%A_%a.err

WORK_DIR=/home/a474r867/scratch/cot_generation
MANIFEST=$WORK_DIR/manifest.txt
SCRIPT=$WORK_DIR/process_chunk.py
OLLAMA_SIF="/home/a474r867/work/ollama/ollama.sif"
LLAMA_BIN="/usr/lib/ollama/llama-server"
LLAMA_PORT=8080
LLAMA_HOST=127.0.0.1

CHUNK_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
[ -z "$CHUNK_FILE" ] && { echo "ERROR: No chunk $SLURM_ARRAY_TASK_ID"; exit 1; }
OUTPUT_FILE="${CHUNK_FILE%.jsonl}_out.jsonl"

echo "=== TASK ${SLURM_ARRAY_TASK_ID} (KBS) ==="
echo "  Node: $(hostname) | GPU: $(nvidia-smi -L 2>/dev/null || rocm-smi --showproductname 2>/dev/null || echo unknown)"
echo "  Chunk: $CHUNK_FILE ($(wc -l < $CHUNK_FILE) papers)"

# Start llama-server
apptainer exec --nv --bind "/home/a474r867/scratch/ollama-models:/models" "$OLLAMA_SIF" \
    $LLAMA_BIN --model "/models/blobs/sha256-6e9f90f02bb3b39b59e81916e8cfce9deb45aeaeb9a54a5be4414486b907dc1e" \
    --host "$LLAMA_HOST" --port "$LLAMA_PORT" --n-gpu-layers 99 --ctx-size 8192 \
    --batch-size 512 --threads 4 --no-webui --log-disable &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null" EXIT

# Wait for server
for i in $(seq 1 90); do
    curl -s "http://${LLAMA_HOST}:${LLAMA_PORT}/health" >/dev/null 2>&1 && { echo "Server ready after ${i}s"; break; }
    sleep 2
done

# Process
python3 "$SCRIPT" "$CHUNK_FILE" --server "${LLAMA_HOST}:${LLAMA_PORT}" --output "$OUTPUT_FILE" --delay 0.05

EXIT=$?
echo "Exit: $EXIT at $(date)"
exit $EXIT
