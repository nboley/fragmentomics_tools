#!/usr/bin/env python
"""Capture real-data orientation facts for fragment arrays.

Pins the strand-flip behaviour of ``RegionFragmentArray.from_fragments_h5``
and the fragment-array coupling in ``SampleAndRegionDataFrame``'s geometry
overrides (``resize_regions``, ``bin_regions_into_windows``).

What is committed:
  - The manifest (counts + digests), in test/fixtures/orientation_manifest.tsv.

Why this exists:
  ``from_fragments_h5`` reverses coordinates and swaps strands for minus-strand
  regions (``is_flipped``).  The correction applier refuses flipped/minus input.
  Getting the ordering wrong silently destroys strand asymmetry.

  Phases 3-4 of the dataframe layering move the four SRDF geometry overrides
  (``expand_regions``, ``resize_regions``, ``_resize_region_boundaries``,
  ``bin_regions_into_windows``) that keep fragment arrays coupled to region
  coordinates.  This manifest detects a coupling break on real data, where
  the synthetic in-process tests have already failed to catch seven defects.

Captured facts:
  Per region (20 regions, all from CTCF sites):
    - load with strand="+", strand="-", strand="." (strandless)
    - fragment count, coordinate digest, strand digest, is_flipped
    - strand symmetry: reverse_strand(strandless) matches minus-strand load
  SRDF coupling:
    - attach fragments to a stranded region set
    - resize_regions (shrink) — fragment arrays must follow
    - bin_regions_into_windows — fragment arrays must subset correctly

Usage:
  PYTHONPATH=/home/nathanboley/src/fragmentomics_tools/.claude/worktrees/f10-test-fix \\
    /home/nathanboley/miniconda3/envs/biomarker_env/bin/python \\
    scripts/capture_orientation_fixtures.py \\
      --out test/fixtures/orientation_manifest.tsv
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fragmentomics_tools.region import Region
from fragmentomics_tools.fragment_array.fragment_array import RegionFragmentArray
from fragmentomics_tools.dataframe import RegionDataFrame, SampleAndRegionDataFrame

EFS = Path("/efs/analytics")
H5_PATH = (
    EFS / "nathanboley/biomarker-projects/data_cache/DC4-16709"
    / "764d5ce67737e478b927fc0ef17f2df1-86-AC-124104-Lib1_DC4-16709_S22.hg38.fragments.h5"
)
CTCF_SRC = EFS / "lily/ssDNA/top_1000_TF/CTCF.hg38.bed"

N_REGIONS = 20
REGION_SIZE = 2048
RNG_SEED = 42


def _sha_array(arr):
    """SHA256 digest of a numpy array, order-sensitive."""
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _sha_str(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def rfa_digest(rfa):
    """Deterministic digest of a RegionFragmentArray's content and order.

    Hashes starts_0, stops_0, fragment_strands (if present), is_flipped,
    region coordinates, and n_fragments — all order-sensitive.
    """
    parts = [
        _sha_array(rfa.starts_0),
        _sha_array(rfa.stops_0),
        str(rfa.is_flipped),
        str(len(rfa.starts_0)),
        f"{rfa.region.chrom}:{rfa.region.start}-{rfa.region.stop}:{rfa.region.strand}",
    ]
    if rfa.fragment_strands is not None:
        parts.append(_sha_array(rfa.fragment_strands))
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def dense_digest(rfa):
    """SHA256 digest of the dense pileup array (fragment_length x position)."""
    dense = rfa.dense_array
    return _sha_array(dense)


def select_regions(ctcf_path, n=N_REGIONS, size=REGION_SIZE, seed=RNG_SEED):
    """Select N CTCF regions, expand to `size` bp, from diverse chromosomes.

    Returns a list of (chrom, start, stop) tuples.  The regions are centred
    on the original CTCF site and expanded symmetrically.
    """
    rng = np.random.RandomState(seed)
    raw = pd.read_csv(
        ctcf_path, sep="\t", header=None,
        names=["chrom", "start", "stop", "name", "score", "score2", "strand"],
        dtype={"chrom": str},
    )
    autosomes = [f"chr{i}" for i in range(1, 23)]
    raw = raw[raw["chrom"].isin(autosomes)].copy()

    selected = []
    per_chrom = n // 4
    target_chroms = ["chr1", "chr2", "chr5", "chr10"]
    for chrom in target_chroms:
        chrom_rows = raw[raw["chrom"] == chrom]
        if len(chrom_rows) < per_chrom:
            continue
        indices = rng.choice(len(chrom_rows), per_chrom, replace=False)
        for idx in sorted(indices):
            row = chrom_rows.iloc[idx]
            mid = (row["start"] + row["stop"]) // 2
            new_start = max(0, mid - size // 2)
            new_stop = new_start + size
            selected.append((row["chrom"], int(new_start), int(new_stop)))

    return selected


def _digest_file(path):
    h = hashlib.sha256()
    file_size = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            file_size += len(chunk)
    return file_size, h.hexdigest()[:16]


def input_provenance():
    rows = []
    for name, path in [("h5", H5_PATH), ("ctcf_source", CTCF_SRC)]:
        file_size, digest = _digest_file(path)
        rows.append({
            "op": "input", "dataset": name, "n": file_size,
            "sha256_16": digest, "note": str(path),
        })
    return rows


def capture():
    """Run all orientation checks and return manifest rows."""
    from fragments_h5 import FragmentsH5

    rows = []

    def record(op, dataset, n, digest, note=""):
        rows.append({
            "op": op, "dataset": dataset, "n": n,
            "sha256_16": digest, "note": note,
        })
        print(f"  {op:40} {dataset:30} n={n:<10} {digest}")

    regions = select_regions(CTCF_SRC)
    print(f"Selected {len(regions)} regions")

    h5 = FragmentsH5(str(H5_PATH), cache_pointers=False)

    # ── Per-region orientation facts ─────────────────────────────────
    for i, (chrom, start, stop) in enumerate(regions):
        region_name = f"region_{i:02d}"

        for strand_label, strand_val in [("plus", "+"), ("minus", "-"), ("strandless", ".")]:
            region = Region(chrom, start, stop, strand_val, ref="hg38")
            rfa = RegionFragmentArray.from_fragments_h5(h5, region)

            record(
                f"load_{strand_label}", region_name, len(rfa.starts_0),
                rfa_digest(rfa),
                note=f"{chrom}:{start}-{stop} strand={strand_val} flipped={rfa.is_flipped}",
            )
            record(
                f"dense_{strand_label}", region_name, rfa.dense_array.size,
                dense_digest(rfa),
                note=f"dense_array shape={rfa.dense_array.shape}",
            )

        # ── Strand symmetry check ────────────────────────────────────
        # Loading strandless and then reversing should match loading minus
        region_sl = Region(chrom, start, stop, ".", ref="hg38")
        rfa_sl = RegionFragmentArray.from_fragments_h5(h5, region_sl)
        rfa_sl_rev = rfa_sl.reverse_strand()

        region_minus = Region(chrom, start, stop, "-", ref="hg38")
        rfa_minus = RegionFragmentArray.from_fragments_h5(h5, region_minus)

        # Compare starts_0 arrays (should be identical)
        starts_match = np.array_equal(rfa_sl_rev.starts_0, rfa_minus.starts_0)
        stops_match = np.array_equal(rfa_sl_rev.stops_0, rfa_minus.stops_0)
        strands_match = (
            rfa_sl_rev.fragment_strands is None and rfa_minus.fragment_strands is None
        ) or (
            rfa_sl_rev.fragment_strands is not None
            and rfa_minus.fragment_strands is not None
            and np.array_equal(rfa_sl_rev.fragment_strands, rfa_minus.fragment_strands)
        )
        flipped_match = rfa_sl_rev.is_flipped == rfa_minus.is_flipped
        symmetry_ok = starts_match and stops_match and strands_match and flipped_match

        sym_digest = _sha_str(
            f"starts={starts_match}|stops={stops_match}|"
            f"strands={strands_match}|flipped={flipped_match}"
        )
        record(
            "symmetry_reverse_strandless_vs_minus", region_name,
            1 if symmetry_ok else 0, sym_digest,
            note=f"starts={starts_match} stops={stops_match} strands={strands_match} flipped={flipped_match}",
        )

    # ── SRDF coupling: resize_regions ────────────────────────────────
    print("\n  SRDF coupling tests...")
    srdf_regions = regions[:10]
    rdf_data = {
        "contig": [r[0] for r in srdf_regions],
        "start": [r[1] for r in srdf_regions],
        "stop": [r[2] for r in srdf_regions],
        "strand": ["-" if i % 2 else "+" for i in range(len(srdf_regions))],
        "sample_id": ["test_sample"] * len(srdf_regions),
        "frag_h5": [h5] * len(srdf_regions),
    }
    srdf = SampleAndRegionDataFrame(pd.DataFrame(rdf_data), ref="hg38")
    srdf = srdf.attach_fragment_arrays(n_workers=1, verbose=0)

    for i, (_, row) in enumerate(srdf.iterrows()):
        rfa = row.fragment_array
        record(
            "srdf_attach", f"srdf_region_{i:02d}", len(rfa.starts_0),
            rfa_digest(rfa),
            note=f"strand={row.strand} flipped={rfa.is_flipped}",
        )

    # Resize (shrink to 1024)
    srdf_resized = srdf.resize_regions(1024)
    for i, (_, row) in enumerate(srdf_resized.iterrows()):
        rfa = row.fragment_array
        record(
            "srdf_resize_1024", f"srdf_region_{i:02d}", len(rfa.starts_0),
            rfa_digest(rfa),
            note=f"strand={row.strand} flipped={rfa.is_flipped} region={row.contig}:{row.start}-{row.stop}",
        )

    # Bin into windows (512bp, valid mode)
    srdf_binned = srdf.bin_regions_into_windows(512, mode="valid")
    for i, (_, row) in enumerate(srdf_binned.iterrows()):
        rfa = row.fragment_array
        record(
            "srdf_bin_512", f"srdf_window_{i:02d}", len(rfa.starts_0),
            rfa_digest(rfa),
            note=f"strand={row.strand} flipped={rfa.is_flipped} region={row.contig}:{row.start}-{row.stop}",
        )

    # ── Non-default index ────────────────────────────────────────────
    print("\n  Non-default index test...")
    srdf_reindexed = srdf.copy()
    srdf_reindexed.index = pd.RangeIndex(100, 100 + len(srdf_reindexed))
    srdf_reindexed_resized = srdf_reindexed.resize_regions(1024)
    for i, (idx, row) in enumerate(srdf_reindexed_resized.iterrows()):
        rfa = row.fragment_array
        record(
            "srdf_reindex_resize", f"srdf_region_{i:02d}", len(rfa.starts_0),
            rfa_digest(rfa),
            note=f"index={idx} strand={row.strand} flipped={rfa.is_flipped}",
        )

    h5.close()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Capture orientation fixtures on real data."
    )
    ap.add_argument(
        "--out", default=None,
        help="manifest output path (default: print to stdout only)",
    )
    args = ap.parse_args()

    print("Input provenance:")
    provenance = input_provenance()
    for r in provenance:
        print(f"  {r['dataset']:15} {r['n']:>12} bytes  {r['sha256_16']}")
    print()

    print("Capturing:")
    manifest = pd.concat(
        [pd.DataFrame(provenance), capture()], ignore_index=True
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(str(out_path), sep="\t", index=False)
        print(f"\nWrote {args.out} ({len(manifest)} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
