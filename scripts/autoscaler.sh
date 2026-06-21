#!/bin/bash
# autoscaler.sh — Dynamic scaling for CoT tsunami jobs
# Monitors running jobs and launches new ones up to MAX_ALLOWED.
# Run as: nohup bash autoscaler.sh > autoscaler.log 2>&1 &
#
# USAGE:
#   1. Generate manifest:   python3 gen_manifest.py --input all_papers.jsonl
#   2. Launch initial batch: sbatch --array=1-500%20 submit_tsunami.slurm
#   3. Start autoscaler:    nohup bash autoscaler.sh > autoscaler.log 2>&1 &

USER=${USER:-a474r867}
JOB_NAME="cot-tsunami"
MAX_ALLOWED=${MAX_ALLOWED:-30}
CHECK_INTERVAL=${CHECK_INTERVAL:-180}  # seconds between checks
TEMPLATE="/home/a474r867/scratch/cot_generation/submit_tsunami.slurm"
MANIFEST="/home/a474r867/scratch/cot_generation/manifest.txt"

if [ ! -f "$TEMPLATE" ]; then
    echo "ERROR: Template not found: $TEMPLATE"
    exit 1
fi

TOTAL_CHUNKS=$(wc -l < "$MANIFEST")
echo "=== AUTOSCALER START ==="
echo "  User:     $USER"
echo "  Job name: $JOB_NAME"
echo "  Max conc: $MAX_ALLOWED"
echo "  Interval: ${CHECK_INTERVAL}s"
echo "  Chunks:   $TOTAL_CHUNKS"
echo "  Date:     $(date)"

while true; do
    RUNNING=$(squeue -u "$USER" -t RUNNING --name="$JOB_NAME" --noheader 2>/dev/null | wc -l)
    PENDING=$(squeue -u "$USER" -t PENDING --name="$JOB_NAME" --noheader 2>/dev/null | wc -l)
    TOTAL=$((RUNNING + PENDING))
    FREE=$((MAX_ALLOWED - TOTAL))
    
    TIMESTAMP=$(date '+%H:%M:%S')
    
    if [ "$TOTAL" -lt "$MAX_ALLOWED" ]; then
        # Calculate how many to launch
        LAUNCH=$FREE
        if [ "$LAUNCH" -gt "$TOTAL_CHUNKS" ]; then
            LAUNCH=$TOTAL_CHUNKS
        fi
        
        # Launch via job array continuation
        # Start from current total + 1
        NEXT_START=$((TOTAL + 1))
        NEXT_END=$((TOTAL + LAUNCH))
        
        if [ "$NEXT_START" -le "$TOTAL_CHUNKS" ]; then
            echo "[$TIMESTAMP] Running=$RUNNING Pending=$PENDING Free=$FREE → Launching $LAUNCH ($NEXT_START-$NEXT_END)"
            sbatch --array=${NEXT_START}-${NEXT_END} "$TEMPLATE" 2>/dev/null
        fi
    else
        echo "[$TIMESTAMP] At capacity: Running=$RUNNING Pending=$PENDING (limit=$MAX_ALLOWED)"
    fi
    
    sleep "$CHECK_INTERVAL"
done
