#!/bin/bash
# Batch shard script: count cut-site hexamers for one sample.
# Submitted as an 88-element array job; each shard reads its sample from
# the batch_remaining_88.tsv manifest via $AWS_BATCH_JOB_ARRAY_INDEX.
#
# Provenance is pinned at SUBMIT time (the values below are literals baked
# in by the submitter) so that mid-run commits cannot pollute the recorded
# origin.  git is absent in the container, so nothing here calls it.

set -euo pipefail

# ── provenance (baked in by submitter, do not edit) ──────────────────────
COMMIT_SHA="f98f66517b212e1f9b54335e0383b3ef3d9135e7"
SCRIPT_SHA="221529fa0554ba5432980b4aa61a18567292c5db"
REGIONS_BED_MD5="761b711c1c98034d164f5f77b22481f0"

# ── paths ────────────────────────────────────────────────────────────────
PYTHON=/home/nathanboley/miniconda3/envs/biomarker_env/bin/python
REPO=/home/nathanboley/src/fragmentomics_tools/.claude/worktrees/background-model-work
SCRIPT="${REPO}/scripts/count_cut_site_hexamers.py"
REGIONS_BED=/home/nathanboley/src/fragmentomics_tools/data/region_sets/quiet_v2_pad1200_repeats_removed_tile2560.bed
FASTA=/efs/analytics/nathanboley/data_resources/genome/hg38.fa
SAMPLE_TSV=/efs/analytics/nathanboley/background_model/cut_site_hexamers/batch_remaining_88.tsv
OUTPUT_DIR=/efs/analytics/nathanboley/background_model/cut_site_hexamers

# ── resolve shard index ──────────────────────────────────────────────────
IDX="${AWS_BATCH_JOB_ARRAY_INDEX}"
echo "=== Shard ${IDX} starting at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "COMMIT_SHA=${COMMIT_SHA}"
echo "SCRIPT_SHA=${SCRIPT_SHA}"

if ! [[ "${IDX}" =~ ^[0-9]+$ ]]; then
    echo "FATAL: AWS_BATCH_JOB_ARRAY_INDEX='${IDX}' is not an integer" >&2
    exit 1
fi

# ── read sample from manifest (0-indexed, skip header) ───────────────────
LINE=$(awk -v idx="${IDX}" 'NR == idx + 2' "${SAMPLE_TSV}")
if [ -z "${LINE}" ]; then
    echo "FATAL: no line at index ${IDX} in ${SAMPLE_TSV}" >&2
    exit 1
fi

SAMPLE_NAME=$(echo "${LINE}" | cut -f2)
H5_PATH=$(echo "${LINE}" | cut -f4)

echo "SAMPLE_NAME=${SAMPLE_NAME}"
echo "H5_PATH=${H5_PATH}"

if [ ! -f "${H5_PATH}" ]; then
    echo "FATAL: h5 not found: ${H5_PATH}" >&2
    exit 1
fi

OUTPUT="${OUTPUT_DIR}/${SAMPLE_NAME}.cut_site_hexamers.parquet"

# ── run from /tmp (treat /home as read-only) ─────────────────────────────
cd /tmp
export PYTHONPATH="${REPO}:/home/nathanboley/src/biomarker"

echo "Running counting script..."
"${PYTHON}" "${SCRIPT}" \
    --fragments-h5 "${H5_PATH}" \
    --regions-bed "${REGIONS_BED}" \
    --fasta "${FASTA}" \
    --sample-name "${SAMPLE_NAME}" \
    --output "${OUTPUT}"

echo "=== Shard ${IDX} (${SAMPLE_NAME}) completed at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Output: ${OUTPUT}"
