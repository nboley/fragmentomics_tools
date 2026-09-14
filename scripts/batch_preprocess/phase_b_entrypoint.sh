#!/usr/bin/env bash
# Phase B entrypoint (single Batch job): assemble all shards -> zarr store.
#
# Required env:
#   S3_BASE   s3://.../bg-preprocess
#   REPO_DIR  cloned repo (set by bootstrap)
#   REF       reference name (default hg38)
set -euo pipefail

: "${S3_BASE:?S3_BASE required}"
: "${REPO_DIR:?REPO_DIR required (set by bootstrap)}"
REF="${REF:-hg38}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

command -v aws >/dev/null 2>&1 || pip install --quiet awscli
# Pilot c882e52a failed with ModuleNotFoundError: zarr — image omni/biomarker:0.2.2
# lacks zarr. Pin 2.18.3 (matches biomarker_env; zarr_format=2 output).
pip install --quiet 'zarr==2.18.3'

WORKDIR="${WORKDIR:-/tmp/bgwork}"
INPUTS="$WORKDIR/inputs"
SHARDS="$WORKDIR/shards"
OUT="$WORKDIR/out"
mkdir -p "$INPUTS/h5" "$SHARDS" "$OUT"

mkdir -p "$WORKDIR/bgcode"
cp -r "$REPO_DIR/background_model" "$WORKDIR/bgcode/"
export PYTHONPATH="$WORKDIR/bgcode"
SCRIPTS="$REPO_DIR/scripts/batch_preprocess"

# All inputs, including the FASTA (needed for the sequence array).
for f in ibd_quiescent.tsv training_tiles.bed blacklist_encode_v2.bed draw50.tsv hg38.fa hg38.fa.fai; do
  aws s3 cp "$S3_BASE/inputs/$f" "$INPUTS/$f"
done

# Resolve hash8 (index 0 is arbitrary; hash8 is index-independent).
python "$SCRIPTS/resolve_index.py" --inputs-dir "$INPUTS" --index 0 > "$WORKDIR/meta.txt"
HASH8="$(cut -d' ' -f2 "$WORKDIR/meta.txt")"
echo "[phaseB] hash8=$HASH8"

# Pull all per-sample shards.
aws s3 cp --recursive "$S3_BASE/shards_${HASH8}/" "$SHARDS/" --exclude "*" --include "*.npz"
echo "[phaseB] shards downloaded: $(ls -1 "$SHARDS"/*.npz 2>/dev/null | wc -l)"

python "$SCRIPTS/run_phase_b.py" \
  --inputs-dir "$INPUTS" --shard-dir "$SHARDS" --output-dir "$OUT" --ref "$REF"

STORE_NAME="bg_store_${HASH8}.zarr"
CONFIG_JSON="bg_store_${HASH8}.config.json"

# Upload the store (recursive) + sidecar config. Recursive keeps the store
# directly usable via `aws s3 cp --recursive` sync-back to EFS.
aws s3 cp --recursive "$OUT/${STORE_NAME}" "$S3_BASE/store/${STORE_NAME}/"
aws s3 cp "$OUT/${CONFIG_JSON}" "$S3_BASE/store/${CONFIG_JSON}"
echo "[phaseB] DONE store -> $S3_BASE/store/${STORE_NAME}/"
