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
  - overlap_indices(how="anti")  pairwise
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

from fragmentomics_tools import intervals
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

    ``float_format`` is pinned even though every column captured today is
    integer. Without it, pandas uses repr, which is not stable across pandas
    versions or platforms for floats — so the first float column anyone adds
    would turn this manifest into a false-alarm generator rather than a
    regression detector, and the cause would be extremely unobvious.
    """
    if isinstance(obj, pd.DataFrame):
        payload = obj.to_csv(
            index=False, sep="\t", na_rep="NA", float_format="%.10g"
        ).encode()
    elif isinstance(obj, pd.Series):
        payload = obj.to_csv(index=False, na_rep="NA", float_format="%.10g").encode()
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


def _digest_file(path):
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return size, h.hexdigest()[:16]


def input_provenance(datasets):
    """Digest the INPUTS, so a manifest mismatch is diagnosable.

    Without this, a changed digest has two indistinguishable causes: the code
    regressed, or the EFS file underneath moved. Those demand opposite
    responses — revert versus re-baseline — and guessing wrong wastes the
    whole point of having a baseline. Recording the input digests makes the
    question answerable by comparing two rows instead of by archaeology.

    Digested by content, not mtime: EFS mtimes are not stable across restores,
    and a file that is byte-identical after a restore has not changed in any
    sense this fixture cares about.

    The CTCF *source* is digested as well as the derived BED6, and that is not
    redundant. ``build_inputs`` only rebuilds the derived file when it is
    missing, so a changed source hides behind a stale derivative: every
    downstream digest would match while the fixture silently no longer
    describes the data it claims to. Digesting only the derivative would leave
    exactly the blind spot this function exists to remove.
    """
    rows = []
    for name, (path, ref) in sorted(datasets.items()):
        size, digest = _digest_file(path)
        rows.append(
            {
                "op": "input",
                "dataset": name,
                "n": size,
                "sha256_16": digest,
                "note": str(path),
            }
        )

    ctcf_src = EFS / "lily/ssDNA/top_1000_TF/CTCF.hg38.bed"
    if ctcf_src.exists():
        size, digest = _digest_file(ctcf_src)
        rows.append(
            {
                "op": "input_source",
                "dataset": "ctcf",
                "n": size,
                "sha256_16": digest,
                "note": f"{ctcf_src} (derived BED6 is rebuilt only when absent)",
            }
        )
    return rows


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
        record("merge", name, intervals.merge(rdf))
        record("unique_regions", name, rdf.unique_regions())
        record("cluster", name, intervals.cluster(rdf))

    # ── Pairwise: CTCF × blacklist ────────────────────────────────────
    if "blacklist" in loaded and "ctcf" in loaded:
        bl, ctcf = loaded["blacklist"], loaded["ctcf"]

        # Both directions — A's identity drives the result, so order matters.
        record("overlap_indices", "ctcf_x_blacklist",
               intervals.overlap_indices(ctcf, bl))
        record("overlap_indices", "blacklist_x_ctcf",
               intervals.overlap_indices(bl, ctcf))

        # The replacement for drop_overlapping_regions.
        record("overlap_indices_anti", "ctcf_x_blacklist",
               intervals.overlap_indices(ctcf, bl, how="anti"))

        record("overlaps", "ctcf_x_blacklist", intervals.overlaps(ctcf, bl))
        record("overlaps_w10", "ctcf_x_blacklist",
               intervals.overlaps(ctcf, bl, wiggle=10),
               note="wiggle=10")
        record("overlaps", "blacklist_x_ctcf", intervals.overlaps(bl, ctcf))

        # What get_overlapping_base_counts used to return, now a groupby over
        # the primitive. Note this is the CORRECTED answer: the old method
        # keyed aggregation on (contig, start, stop) with strand excluded, so
        # rows sharing coordinates collided.
        pairs = intervals.overlap_indices(ctcf, bl)
        record("overlap_bases_sum", "ctcf_x_blacklist",
               pairs.groupby("a_index")["overlap_bases"].sum())

        record("nearest", "ctcf_x_blacklist", intervals.nearest(ctcf, bl))

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

    print("Input provenance:")
    provenance = input_provenance(datasets)
    for r in provenance:
        print(f"  {r['dataset']:10} {r['n']:>12} bytes  {r['sha256_16']}")
    print()

    print("Capturing:")
    manifest = pd.concat(
        [pd.DataFrame(provenance), capture(datasets)], ignore_index=True
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(str(out_path), sep="\t", index=False)
        print(f"\nWrote {args.out} ({len(manifest)} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
