#!/usr/bin/env python
"""Phase 0 — pin the CURRENT (pybedtools) interval-algebra behaviour on real data.

Run this BEFORE changing any interval code; run it again after, and diff the
manifest.  A mismatch is a regression.

What is committed:
  - The manifest (counts + digests), in test/fixtures/interval_manifest.tsv.

What is NOT committed:
  - Derived inputs on EFS (the BED6 CTCF file).
  - Full output DataFrames — they are large, regenerable, and live on EFS.

Why real data in addition to the synthetic corner-case tests:
  Synthetic edges catch semantic divergence at boundaries.  Real data
  catches silent divergence at scale — a million real intervals with real
  strand and contig spread, where a subtle row-order or type change only
  shows up because the SHA over the full output differs.

What this captures (the design doc's fixture table):
  - merge_regions             per-dataset
  - join_on_overlap           pairwise (CTCF × blacklist, both directions)
  - drop_overlapping_regions  pairwise
  - overlaps_rdf              pairwise (with and without max_distance)
  - get_overlapping_base_counts  pairwise (captures the strand-collision bug)
  - from_beds_merged          merging both BEDs together

What this deliberately does NOT capture:
  - _get_fragment_coverage_sum: requires a fragment-level BED file (~100M rows)
    not available as a stable fixture.  The synthetic test in
    test_interval_corner_cases.py covers this method instead.
  - nearest, cluster: no current implementation to capture a baseline from.
    These get specification tests in Phase 1.

Note on digest:
  The SHA covers frame content AND row order (via to_csv with index=False).
  Order is part of the behaviour under test: a replacement can agree on the
  row set and still differ on ordering, silently breaking anything that zips
  or positionally indexes.

Note on strand:
  The hg38 blacklist has no strand column; from_bed fills it with ".".
  The CTCF BED6 file is stranded.  Both a strandless and a stranded
  dataset are pinned because the "."-vs-"." divergence is exactly what
  a naive swap gets wrong.

Usage:
  # Generate and print manifest
  PATH=/home/nathanboley/miniconda3/envs/biomarker_env/bin:$PATH \\
    python scripts/capture_interval_fixtures.py

  # Generate and write manifest to file
  PATH=/home/nathanboley/miniconda3/envs/biomarker_env/bin:$PATH \\
    python scripts/capture_interval_fixtures.py \\
      --out test/fixtures/interval_manifest.tsv
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy
import pandas as pd

from fragmentomics_tools.dataframe import RegionDataFrame

EFS = Path("/efs/analytics")
GENOME = EFS / "nathanboley/data_resources/genome"

# Derived inputs live here, not in git: ~1M rows is not a repo artifact.
WORK = Path("/efs/analytics/nathanboley/interval_fixtures")


def _sha(obj) -> str:
    """Digest a frame's *content and order*.

    Order is part of the behaviour under test — bedtools and a replacement
    can agree on the set of rows and still differ on what comes out first, and
    downstream code that zips or positionally indexes would silently break.
    """
    if isinstance(obj, pd.DataFrame):
        payload = obj.to_csv(index=False, sep="\t", na_rep="NA").encode()
    elif isinstance(obj, pd.Series):
        payload = obj.to_csv(index=False, na_rep="NA").encode()
    elif isinstance(obj, dict):
        # get_overlapping_base_counts returns {"counts": array, "max_counts": array}
        parts = []
        for k in sorted(obj.keys()):
            v = obj[k]
            parts.append(f"{k}:{','.join(str(x) for x in v)}")
        payload = "|".join(parts).encode()
    elif isinstance(obj, numpy.ndarray):
        payload = ",".join(str(x) for x in obj).encode()
    else:
        payload = str(obj).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def build_inputs(force=False):
    """Materialise canonical BED6 from the real sources.

    The CTCF source is 7 columns in a non-standard order
    (chrom/start/end/score/name/score2/strand), which from_bed rejects —
    it accepts 3,4,5,6,10,12 columns only.  Reorder to real BED6 with awk
    rather than widening the loader, so the fixture exercises the supported path.
    """
    WORK.mkdir(parents=True, exist_ok=True)
    ctcf_src = EFS / "lily/ssDNA/top_1000_TF/CTCF.hg38.bed"
    ctcf_out = WORK / "ctcf.hg38.bed6.bed"
    if force or not ctcf_out.exists():
        with open(str(ctcf_out), "w") as out_fh:
            subprocess.run(
                ["awk", "-v", "OFS=\t", "{print $1,$2,$3,$5,$4,$7}", str(ctcf_src)],
                stdout=out_fh,
                check=True,
            )
    return {
        "blacklist": (GENOME / "hg38-blacklist.v2.bed.gz", "hg38"),
        "ctcf": (ctcf_out, "hg38"),
    }


def capture(datasets):
    """Run every fixture-worthy operation and record (op, dataset, n, sha)."""
    rows = []

    def record(op, dataset, result, note=""):
        if isinstance(result, dict):
            n = sum(len(v) for v in result.values())
        elif hasattr(result, "__len__"):
            n = len(result)
        else:
            n = 1
        digest = _sha(result)
        rows.append({
            "op": op, "dataset": dataset, "n": n,
            "sha256_16": digest, "note": note,
        })
        print(f"  {op:35} {dataset:25} n={n:<10} {digest}")

    # ── Load ──────────────────────────────────────────────────────────
    loaded = {}
    for name, (path, ref) in datasets.items():
        rdf = RegionDataFrame.from_bed(str(path), ref=ref)
        loaded[name] = rdf
        record("load", name, rdf)

    # ── Per-dataset ops ───────────────────────────────────────────────
    for name, rdf in loaded.items():
        record("sort", name, rdf.sort())
        record("merge_regions", name, rdf.merge_regions())
        record("unique_regions", name, rdf.unique_regions())

    # ── Pairwise: CTCF × blacklist ────────────────────────────────────
    if "blacklist" in loaded and "ctcf" in loaded:
        bl, ctcf = loaded["blacklist"], loaded["ctcf"]

        # join_on_overlap (both directions — order matters for the index)
        record("join_on_overlap", "ctcf_x_blacklist",
               ctcf.join_on_overlap(bl))
        record("join_on_overlap", "blacklist_x_ctcf",
               bl.join_on_overlap(ctcf))

        # drop_overlapping_regions
        record("drop_overlapping_regions", "ctcf_x_blacklist",
               ctcf.drop_overlapping_regions(bl))

        # overlaps_rdf (IntervalTree path, both with and without wiggle)
        record("overlaps_rdf", "ctcf_x_blacklist",
               ctcf.overlaps_rdf(bl))
        record("overlaps_rdf_d10", "ctcf_x_blacklist",
               ctcf.overlaps_rdf(bl, max_distance=10),
               note="max_distance=10")
        record("overlaps_rdf", "blacklist_x_ctcf",
               bl.overlaps_rdf(ctcf))

        # get_overlapping_base_counts
        # This uses the raw blacklist path (bed_file argument), not the loaded RDF.
        # Captures the strand-collision bug faithfully.
        record("get_overlapping_base_counts", "ctcf_x_blacklist",
               ctcf.get_overlapping_base_counts(
                   str(datasets["blacklist"][0])))

    # ── from_beds_merged ──────────────────────────────────────────────
    if len(datasets) >= 2:
        all_paths = [str(p) for p, _ in datasets.values()]
        record("from_beds_merged", "all",
               RegionDataFrame.from_beds_merged(all_paths, ref="hg38"))

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Capture Phase 0 interval-algebra fixtures on real data."
    )
    ap.add_argument(
        "--out", default=None,
        help="manifest output path (default: print to stdout only)",
    )
    ap.add_argument("--force-rebuild-inputs", action="store_true")
    args = ap.parse_args()

    print("Building inputs...")
    datasets = build_inputs(force=args.force_rebuild_inputs)
    for k, (p, _) in datasets.items():
        print(f"  {k:10} {p}")
    print()

    print("Capturing:")
    manifest = capture(datasets)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(str(out_path), sep="\t", index=False)
        print(f"\nWrote {args.out} ({len(manifest)} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
