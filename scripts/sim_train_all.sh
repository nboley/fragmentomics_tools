#!/bin/bash
# Train all 3 losses on a simulation store (regime B by default).
# Runs on GPU; designed for batch-run --gpu a10g.
#
# Usage:
#   export PATH=/home/nathanboley/miniconda3/envs/biomarker_env/bin:$PATH
#   bash scripts/sim_train_all.sh [STORE_PATH] [RUNS_ROOT]
set -euo pipefail

STORE="${1:-/efs/analytics/nathanboley/background_model/simulation_v2/stores/sim_store_B.zarr}"
RUNS="${2:-/efs/analytics/nathanboley/background_model/simulation_v2/runs}"
REPO="/home/nathanboley/src/fragmentomics_tools"

export PATH=/home/nathanboley/miniconda3/envs/biomarker_env/bin:$PATH
cd "$REPO"
export PYTHONPATH="$REPO"

mkdir -p "$RUNS"

for LOSS in multinomial dirichlet_multinomial nb_offset; do
    echo "=== Training $LOSS ==="
    python -m background_model.train \
        --loss "$LOSS" \
        --run-name "sim_v2_B_${LOSS}" \
        --store "$STORE" \
        --runs-root "$RUNS" \
        --max-epochs 200 \
        --patience 15 \
        --batch-size 8 \
        --lr 1e-4 \
        --num-workers 2 \
        --seed 1337 \
        --min-N 0
    echo "=== $LOSS done ==="
done

echo "All 3 losses trained."
