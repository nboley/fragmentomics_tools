#!/usr/bin/env bash
# Stage all preprocess inputs from EFS/repo -> S3. Must run on a host with EFS
# access (AWS Batch has NO EFS mounts). Idempotent: re-running only uploads
# what is missing / size-mismatched.
set -euo pipefail

S3_BASE="${S3_BASE:-s3://fragmentomics.kariusdx.com/nboley/bg-preprocess}"
REPO="${REPO:-/home/nathanboley/src/fragmentomics_tools}"
GENOME="${GENOME:-/efs/analytics/nathanboley/data_resources/genome}"
AWS="${AWS:-aws}"
PYBIN="${PYBIN:-python3}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
SCRIPTS="$REPO/scripts/batch_preprocess"

echo "== small inputs (BEDs, sheets, .fai) =="
$AWS s3 cp "$GENOME/hg38.fa.fai"                                       "$S3_BASE/inputs/hg38.fa.fai"
$AWS s3 cp "$REPO/data/region_sets/training_tiles.bed"                 "$S3_BASE/inputs/training_tiles.bed"
$AWS s3 cp "$REPO/data/region_sets/exclusion_4_blacklist_encode_v2.bed" "$S3_BASE/inputs/blacklist_encode_v2.bed"
$AWS s3 cp "$REPO/data/sample_sheets/ibd_quiescent.tsv"               "$S3_BASE/inputs/ibd_quiescent.tsv"
$AWS s3 cp "$REPO/data/sample_sheets/draw50.tsv"                      "$S3_BASE/inputs/draw50.tsv"

echo "== FASTA hg38.fa (3.3 GB; skip if already present) =="
# Exact-match check: a plain `aws s3 ls .../hg38.fa` prefix-matches hg38.fa.fai
# too, so grep for a line ending in the exact object name.
if $AWS s3 ls "$S3_BASE/inputs/hg38.fa" 2>/dev/null | grep -q ' hg38.fa$'; then
  echo "hg38.fa present, skipping"
else
  $AWS s3 cp "$GENOME/hg38.fa" "$S3_BASE/inputs/hg38.fa"
fi

echo "== 50 per-sample h5s (keyed by library) =="
$PYBIN "$SCRIPTS/stage_h5.py" \
  --draw "$REPO/data/sample_sheets/draw50.tsv" \
  --s3-base "$S3_BASE" --region "$AWS_DEFAULT_REGION"

echo "staging complete: $S3_BASE/inputs/"
