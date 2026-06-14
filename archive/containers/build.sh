#!/bin/bash
# Build Nemotron Apptainer image and push to KU-HPC cluster
set -euo pipefail

IMAGE="nemotron-rocm.sif"
WORKDIR="/home/reumanlab/work/Github/nemotron-eco-reasoner/containers"
CLUSTER_SCRATCH="/home/a474r867/scratch/nemotron-eco-reasoner"

echo "=== Build ==="
cd "$WORKDIR"
apptainer build "$IMAGE" nemotron.def

echo "=== Push to cluster ==="
scp "$IMAGE" "a474r867@hpc.crc.ku.edu:$CLUSTER_SCRATCH/"

echo "=== Done ==="
echo "Image: $CLUSTER_SCRATCH/$IMAGE"
echo ""
echo "Test it:"
echo "  apptainer exec --rocm $CLUSTER_SCRATCH/$IMAGE python -c 'import torch; print(torch.cuda.is_available())'"
