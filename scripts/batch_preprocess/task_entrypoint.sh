#!/usr/bin/env bash
# Phase A entrypoint for ONE Batch array task (index -> one sample shard).
#
# Runs inside the biomarker container. The bootstrap (Batch job-def command)
# has already cloned the repo to $REPO_DIR and set PYTHONPATH is done here.
# Required env:
#   S3_BASE   e.g. s3://fragmentomics.kariusdx.com/nboley/bg-preprocess
#   REPO_DIR  path to the cloned repo (set by bootstrap)
#   REF       reference name (default hg38)
#   Index comes from AWS_BATCH_JOB_ARRAY_INDEX (array) or INDEX (single pilot job)
set -euo pipefail

: "${S3_BASE:?S3_BASE required}"
: "${REPO_DIR:?REPO_DIR required (set by bootstrap)}"
REF="${REF:-hg38}"
IDX="${AWS_BATCH_JOB_ARRAY_INDEX:-${INDEX:?INDEX or AWS_BATCH_JOB_ARRAY_INDEX required}}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

command -v aws >/dev/null 2>&1 || pip install --quiet awscli
# Pilot c882e52a failed with ModuleNotFoundError: zarr — image omni/biomarker:0.2.2
# lacks zarr. Pin 2.18.3 (matches biomarker_env; zarr_format=2 output).
pip install --quiet 'zarr==2.18.3'

WORKDIR="${WORKDIR:-/tmp/bgwork}"
INPUTS="$WORKDIR/inputs"
SHARDS="$WORKDIR/shards"
mkdir -p "$INPUTS/h5" "$SHARDS"

# Ship only background_model onto PYTHONPATH so the container's INSTALLED
# fragmentomics_tools / fragments_h5 (with compiled extensions) are used.
mkdir -p "$WORKDIR/bgcode"
cp -r "$REPO_DIR/background_model" "$WORKDIR/bgcode/"
export PYTHONPATH="$WORKDIR/bgcode"
SCRIPTS="$REPO_DIR/scripts/batch_preprocess"

# Small shared inputs needed by Phase A (NOT the 3 GB FASTA; only its .fai).
for f in ibd_quiescent.tsv training_tiles.bed blacklist_encode_v2.bed draw50.tsv hg38.fa.fai; do
  aws s3 cp "$S3_BASE/inputs/$f" "$INPUTS/$f"
done

# Resolve library + config hash8 for this index.
python "$SCRIPTS/resolve_index.py" --inputs-dir "$INPUTS" --index "$IDX" > "$WORKDIR/meta.txt"
LIBRARY="$(cut -d' ' -f1 "$WORKDIR/meta.txt")"
HASH8="$(cut -d' ' -f2 "$WORKDIR/meta.txt")"
SHARD_KEY="$S3_BASE/shards_${HASH8}/${LIBRARY}.npz"
echo "[phaseA] index=$IDX library=$LIBRARY hash8=$HASH8"

# Resume: skip if this shard already exists in S3.
if aws s3 ls "$SHARD_KEY" >/dev/null 2>&1; then
  echo "[phaseA] RESUME: $SHARD_KEY exists, skipping."
  exit 0
fi

# Download this sample's h5.
aws s3 cp "$S3_BASE/inputs/h5/${LIBRARY}.h5" "$INPUTS/h5/${LIBRARY}.h5"

# Run the worker under /usr/bin/time -v when available (process-level RSS/CPU
# to the log); the driver also writes a precise worker-only timing.json.
TIMEBIN=""
command -v /usr/bin/time >/dev/null 2>&1 && TIMEBIN="/usr/bin/time -v"
$TIMEBIN python "$SCRIPTS/run_phase_a_one.py" \
  --inputs-dir "$INPUTS" --index "$IDX" --shard-dir "$SHARDS" --ref "$REF"

# Upload shard + timing (resume-safe: only on success).
aws s3 cp "$SHARDS/${LIBRARY}.npz" "$SHARD_KEY"
aws s3 cp "$SHARDS/${LIBRARY}.timing.json" "$S3_BASE/shards_${HASH8}/${LIBRARY}.timing.json"
echo "[phaseA] DONE library=$LIBRARY -> $SHARD_KEY"
