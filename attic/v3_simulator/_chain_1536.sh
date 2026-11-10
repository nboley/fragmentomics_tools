#!/bin/bash
# One-off chain (NOT a committed pipeline): wait for the running 1536 simulation
# to finish, then build its store, submit the KEN GPU run, and compute its OWN
# oracle anchors in parallel.
#
# Owner instruction 2026-09-26: "launch the Ken run on the smaller region size
# whenever the simulator finishes."
#
# Refuses to proceed if the simulation did not exit 0 -- a KEN run on a partial
# store would look like a result rather than a failure.
set -u

WORKTREE=/home/nathanboley/src/fragmentomics_tools/.claude/worktrees/background-model-work
ENVBIN=/home/nathanboley/miniconda3/envs/biomarker_env/bin
BASE=/efs/analytics/nathanboley/background_model/simulation_v4/1536
SIMDIR=$BASE/A
STORE=$BASE/stores/sim_store_v4_1536_A.zarr
ORACLE=$BASE/oracle_v4_1536_A.json
RUNS=$BASE/runs
SIMJOB=/home/nathanboley/.config/claude-mcp/jobs/db34cdf3a742

export PATH=$ENVBIN:$PATH
export PYTHONPATH=$WORKTREE
cd "$WORKTREE" || exit 1

echo "[chain] waiting for the 1536 simulation (job db34cdf3a742) to exit"
while [ ! -f "$SIMJOB/exit_code" ]; do sleep 60; done
RC=$(cat "$SIMJOB/exit_code")
echo "[chain] simulation exited rc=$RC"
if [ "$RC" != "0" ]; then
    echo "[chain] ABORT: simulation did not succeed; not building a store from partial output"
    exit 1
fi
if [ ! -f "$SIMDIR/ground_truth.json" ]; then
    echo "[chain] ABORT: $SIMDIR/ground_truth.json missing despite rc=0"
    exit 1
fi

# Geometry: region_len 1536 = tile_size 1024 + 2*jitter 256 (zero margin).
mkdir -p "$BASE/stores" "$RUNS"
echo "[chain] building store"
nice -n 5 python scripts/sim_build_store.py \
    --sim-dir "$SIMDIR" --out "$STORE" \
    --tile-size 1024 --jitter 256 --workers 6 || exit 1

# GPU run first so the A10G is not idle while the oracle runs on CPU.
# Hyperparameters identical to the 2560 run so the two are comparable;
# --min-N 0 is mandatory at this density (per-track median ~2).
echo "[chain] submitting KEN to AWS Batch"
batch-run submit --gpu a10g --timeout 21600 --user nathanboley -- \
  "cd /tmp && PYTHONPATH=$WORKTREE micromamba run -p /home/nathanboley/miniconda3/envs/biomarker_env python -m background_model.train --loss multinomial --model ken --run-name v4_1536_A_ken --store $STORE --runs-root $RUNS --max-epochs 60 --batch-size 64 --lr 5e-4 --num-workers 4 --seed 1337 --min-N 0 --precision bf16-mixed --n-kernels 128 --num-residual-layers 1" \
  > "$BASE/ken_submit.log" 2>&1 &
SUBMIT_PID=$!

# This store's OWN anchors.  The 2560 store's 7.143705 / 7.228059 DO NOT apply.
echo "[chain] computing oracle anchors"
nice -n 10 python scripts/sim_oracle.py \
    --sim-dir "$SIMDIR" --store "$STORE" --out "$ORACLE"
echo "[chain] oracle done"

wait $SUBMIT_PID
echo "[chain] KEN submit returned; see $BASE/ken_submit.log"
