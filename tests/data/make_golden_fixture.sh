#!/usr/bin/env bash
# Rebuild the golden fragments-h5 fixtures used by tests/test_bg_golden_counts.py.
#
# Runs the REAL production CLI (fragments_h5.main / build-fragments-h5) over the
# committed BAM + FASTA fixtures.  Building the fixture with the production writer
# — rather than our own code — is what makes the golden count test non-circular.
#
# Requirements: the `fragments_h5` package importable by $PYTHON (in this repo's
# biomarker_env it is an editable install).  The `build-fragments-h5` console
# script is NOT on PATH in biomarker_env, so we invoke the module directly.
#
# Usage:  PYTHON=/path/to/python bash tests/data/make_golden_fixture.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python}"
FASTA="$HERE/GRCh38.p12.genome.chr6_99110000_99130000.fa.gz"

# Default PE build (min_mapq default 0 == no-op; duplicates excluded).
"$PY" -m fragments_h5.main \
    "$HERE/small.chr6.bam" \
    "$HERE/golden.small.chr6.frag.h5" \
    --fasta "$FASTA"

# Build WITH --include-duplicates so the coordinate-identical pair in
# test_duplicates.bam enters the h5 and exercises read-time
# RegionFragmentArray.drop_duplicate_fragments() in the golden dedup test.
"$PY" -m fragments_h5.main \
    "$HERE/test_duplicates.bam" \
    "$HERE/golden.test_duplicates.frag.h5" \
    --fasta "$FASTA" \
    --include-duplicates

echo "Built:"
ls -la "$HERE"/golden.*.frag.h5
