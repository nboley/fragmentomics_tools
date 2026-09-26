#!/usr/bin/env python
"""Count deduplicated, MAPQ-filtered fragments per region for IBD samples.

In-region rule
--------------
A fragment is counted in a region if its **midpoint** falls within the
half-open interval [region_start, region_stop).  Midpoint is defined as
``(start + stop) // 2`` (integer division, i.e. floor of the true center).

Choosing midpoint (rather than any-overlap or fragment-start) ensures that
each fragment is assigned to exactly one non-overlapping tile, preventing
double-counting across adjacent tiles.

Fragment filters
----------------
* ``mapq >= --min-mapq`` — **inclusive** (``>=``), applied to
  ``min(mapq_read1, mapq_read2)`` for paired-end data.  This matches the
  background-model store: ``fragment_array.from_fragments_h5`` filters with
  ``mapq_vals >= min_mapq``, so passing ``--min-mapq 10`` reproduces the
  store's ``min_mapq=10`` filter and the density fitted from these counts is
  measured under the same filter as the model's training data.

  NOTE: the ``>=`` convention differs from this script's original hardcoded
  ``mapq > 30`` (which is equivalent to ``>= 31``).  A file produced by the
  old code is therefore NOT reproducible via ``--min-mapq 30``.  The exact
  comparison actually used is written into each output's ``#`` header, so any
  file is self-describing regardless of which convention produced it.
* Deduplication by unique ``(start, stop)`` coordinate pairs — same semantics
  as ``FragmentArray.drop_duplicate_fragments()``.
* **No fragment-length restriction.** All fragment lengths pass through so that
  banding can be decided downstream from raw counts.

Output
------
One gzipped TSV per sample at
``<output_dir>/<sample_name>.region_counts.tsv.gz``
with header:
    ``sample\\tregion_set\\tcontig\\tstart\\tstop\\tcount``

Sharding
--------
Accepts ``--shard-index`` / ``--shard-count`` for AWS Batch array jobs.
Pass ``$AWS_BATCH_JOB_ARRAY_INDEX`` as the shard index.
"""
import argparse
import gzip
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fragments_h5 import FragmentsH5

logger = logging.getLogger(__name__)

# ── region set paths (relative to repo root) ────────────────────────────
REGION_SET_NAMES = [
    "quiet_v2_pad1200_repeats_kept_tile2560",
    "quiet_v2_pad1200_repeats_kept_tile1536",
    "quiet_v2_pad1200_repeats_removed_tile2560",
    "quiet_v2_pad1200_repeats_removed_tile1536",
]


# ── shard helpers ────────────────────────────────────────────────────────

def partition_samples(samples: list, shard_index: int, shard_count: int) -> list:
    """Return the subset of *samples* that belong to *shard_index*.

    Raises ValueError for out-of-range or inconsistent arguments.
    """
    if shard_count < 1:
        raise ValueError(f"shard_count must be >= 1, got {shard_count}")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            f"shard_index {shard_index} out of range for shard_count {shard_count}"
        )
    if shard_count > len(samples):
        raise ValueError(
            f"shard_count ({shard_count}) exceeds number of samples ({len(samples)})"
        )
    return [s for i, s in enumerate(samples) if i % shard_count == shard_index]


# ── fragment filtering ───────────────────────────────────────────────────

def filter_and_dedup(
    starts: np.ndarray,
    stops: np.ndarray,
    mapqs: np.ndarray,
    min_mapq: int,
) -> tuple:
    """Apply the ``mapq >= min_mapq`` filter and coordinate-pair deduplication.

    The comparison is **inclusive** (``>=``), matching the background-model
    store's ``from_fragments_h5`` (``mapq_vals >= min_mapq``).  A fragment
    whose ``min(mapq_read1, mapq_read2)`` equals ``min_mapq`` is admitted.

    Parameters
    ----------
    starts, stops : int32 arrays — genomic coordinates from FragmentsH5
    mapqs : (N, 2) int32 array — paired-end MAPQ values
    min_mapq : int — inclusive lower bound on min(read1, read2) MAPQ

    Returns
    -------
    (filtered_starts, filtered_stops) after filtering and dedup.
    """
    # MAPQ filter: min of paired-end reads must be >= min_mapq (inclusive),
    # matching from_fragments_h5's `mapq_vals >= min_mapq`.
    if mapqs.ndim == 2:
        min_mapq_vals = mapqs.min(axis=1)
    elif len(mapqs) == 2 * len(starts):
        min_mapq_vals = np.minimum(mapqs[::2], mapqs[1::2])
    else:
        min_mapq_vals = mapqs.ravel()
    mask = min_mapq_vals >= min_mapq
    starts = starts[mask]
    stops = stops[mask]

    # Dedup: unique by (start, stop) — same as drop_duplicate_fragments()
    _, idx = np.unique(np.array([starts, stops]), axis=1, return_index=True)
    starts = starts[idx]
    stops = stops[idx]

    return starts, stops


# ── midpoint counting ────────────────────────────────────────────────────

def count_midpoints_in_regions(
    frag_midpoints_sorted: np.ndarray,
    region_starts: np.ndarray,
    region_stops: np.ndarray,
) -> np.ndarray:
    """Count how many midpoints fall in each [start, stop) region.

    Parameters
    ----------
    frag_midpoints_sorted : sorted int array of fragment midpoints
    region_starts, region_stops : int arrays (same length), half-open intervals

    Returns
    -------
    int array of counts, one per region.
    """
    left = np.searchsorted(frag_midpoints_sorted, region_starts, side="left")
    right = np.searchsorted(frag_midpoints_sorted, region_stops, side="left")
    return right - left


# ── BED loading ──────────────────────────────────────────────────────────

def load_bed_regions(bed_path: str) -> pd.DataFrame:
    """Load a 3-column BED file into a DataFrame with contig/start/stop."""
    df = pd.read_csv(
        bed_path, sep="\t", header=None,
        names=["contig", "start", "stop"],
        dtype={"contig": str, "start": np.int64, "stop": np.int64},
    )
    return df


# ── main logic ───────────────────────────────────────────────────────────

def count_fragments_for_sample(
    sample_name: str,
    h5_path: str,
    region_sets: dict,
    output_dir: str,
    min_mapq: int,
) -> str:
    """Count fragments per region for one sample, write gzipped TSV.

    Parameters
    ----------
    sample_name : identifier for the sample
    h5_path : path to the .fragments.h5 file
    region_sets : {set_name: DataFrame with contig/start/stop}
    output_dir : directory for output files
    min_mapq : inclusive lower bound on min(read1, read2) MAPQ (``>=``)

    Returns
    -------
    Path to the output file.
    """
    out_path = os.path.join(output_dir, f"{sample_name}.region_counts.tsv.gz")
    h5 = FragmentsH5(h5_path, cache_pointers=False)

    # Collect all contigs that appear in any region set
    all_contigs = set()
    for df in region_sets.values():
        all_contigs.update(df["contig"].unique())
    all_contigs = sorted(all_contigs)

    # Pre-group regions by contig for each set
    regions_by_contig = {}
    for set_name, df in region_sets.items():
        regions_by_contig[set_name] = {
            contig: sub.reset_index(drop=True)
            for contig, sub in df.groupby("contig")
        }

    with gzip.open(out_path, "wt") as fh:
        # Header with provenance comment.  Record the ACTUAL comparison used
        # (inclusive `>=` on min(read1,read2)) so any reader can tell which
        # convention produced this file.  This differs from the original
        # hardcoded `mapq > 30` (== `>= 31`), which is not reproducible here.
        fh.write(
            f"# mapq_filter: min(mapq_read1,mapq_read2) >= {min_mapq} "
            "(inclusive), dedup: by (start,stop), "
            "in_region: midpoint in [start,stop), no fragment-length filter\n"
            "# matches bg-model store from_fragments_h5 (mapq_vals >= min_mapq); "
            "pass --min-mapq 10 to match the store's min_mapq=10\n"
        )
        fh.write("sample\tregion_set\tcontig\tstart\tstop\tcount\n")

        for contig in all_contigs:
            # Check if this contig exists in the h5
            if contig not in h5.contig_lengths:
                logger.warning("Contig %s not in h5 file, skipping", contig)
                continue

            t0 = time.monotonic()

            # Fetch ALL fragments for this contig
            starts, stops, supp = h5.fetch_array(contig, return_mapqs=True)
            t_fetch = time.monotonic() - t0

            if len(starts) == 0:
                midpoints_sorted = np.array([], dtype=np.int64)
            else:
                # Filter and dedup
                starts, stops = filter_and_dedup(
                    starts, stops, supp["mapq"], min_mapq
                )
                # Compute midpoints and sort
                midpoints = (starts.astype(np.int64) + stops.astype(np.int64)) // 2
                midpoints_sorted = np.sort(midpoints)

            t_filter = time.monotonic() - t0 - t_fetch

            # Count per region set — bulk write via DataFrame
            for set_name in region_sets:
                cdf = regions_by_contig[set_name].get(contig)
                if cdf is None:
                    continue
                counts = count_midpoints_in_regions(
                    midpoints_sorted,
                    cdf["start"].values,
                    cdf["stop"].values,
                )
                # Build output block as a DataFrame for fast to_csv
                out_df = pd.DataFrame({
                    "sample": sample_name,
                    "region_set": set_name,
                    "contig": contig,
                    "start": cdf["start"].values,
                    "stop": cdf["stop"].values,
                    "count": counts,
                })
                out_df.to_csv(fh, sep="\t", header=False, index=False)

            t_total = time.monotonic() - t0
            logger.info(
                "%s %s: %d frags -> %d after filter+dedup  "
                "(fetch %.1fs, filter %.1fs, total %.1fs)",
                sample_name, contig, len(supp["mapq"]),
                len(starts), t_fetch, t_filter, t_total,
            )

    h5.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sample-sheet", required=True,
        help="TSV with columns: sample_name, path (at minimum)",
    )
    parser.add_argument(
        "--region-sets-dir", required=True,
        help="Directory containing the region BED files",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for per-sample gzipped TSVs",
    )
    parser.add_argument(
        "--min-mapq", type=int, required=True,
        help="Inclusive MAPQ lower bound (>=) on min(mapq_read1, mapq_read2). "
             "Pass 10 to match the background-model store (min_mapq=10). "
             "This is >=, so 10 admits MAPQ==10 (unlike the old hardcoded >30).",
    )
    parser.add_argument(
        "--shard-index", type=int, default=None,
        help="This shard's index (0-based). Pass $AWS_BATCH_JOB_ARRAY_INDEX.",
    )
    parser.add_argument(
        "--shard-count", type=int, default=None,
        help="Total number of shards.",
    )
    parser.add_argument(
        "--max-regions", type=int, default=None,
        help="Limit regions per BED file (for testing only).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.info(
        "MAPQ filter: min(mapq_read1,mapq_read2) >= %d (inclusive)", args.min_mapq
    )

    # ── load sample sheet ────────────────────────────────────────────
    sheet = pd.read_csv(args.sample_sheet, sep="\t")
    if "sample_name" not in sheet.columns or "path" not in sheet.columns:
        sys.exit("Sample sheet must have 'sample_name' and 'path' columns")
    samples = list(sheet.itertuples(index=False))

    # ── shard validation ─────────────────────────────────────────────
    if (args.shard_index is None) != (args.shard_count is None):
        sys.exit("--shard-index and --shard-count must both be set or both omitted")

    if args.shard_index is not None:
        try:
            sample_indices = [
                i for i in range(len(samples))
                if i % args.shard_count == args.shard_index
            ]
            # Validate first via the helper (which raises on bad input)
            partition_samples(
                list(range(len(samples))), args.shard_index, args.shard_count,
            )
        except ValueError as exc:
            sys.exit(f"Shard validation error: {exc}")
        samples = [samples[i] for i in sample_indices]
        logger.info(
            "Shard %d/%d: processing %d samples",
            args.shard_index, args.shard_count, len(samples),
        )

    if not samples:
        sys.exit("No samples to process (empty shard or empty sheet)")

    # ── load region sets ─────────────────────────────────────────────
    region_sets = {}
    for name in REGION_SET_NAMES:
        bed_path = os.path.join(args.region_sets_dir, f"{name}.bed")
        if not os.path.exists(bed_path):
            sys.exit(f"Region BED not found: {bed_path}")
        df = load_bed_regions(bed_path)
        if args.max_regions is not None:
            df = df.head(args.max_regions)
        region_sets[name] = df
        logger.info("Loaded %s: %d regions", name, len(df))

    total_regions = sum(len(df) for df in region_sets.values())
    logger.info("Total regions across all sets: %d", total_regions)

    # ── ensure output dir ────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # ── process each sample ──────────────────────────────────────────
    for row in samples:
        t0 = time.monotonic()
        logger.info("Processing %s ...", row.sample_name)
        if not os.path.exists(row.path):
            logger.error("H5 file not found: %s", row.path)
            sys.exit(1)
        out = count_fragments_for_sample(
            row.sample_name, row.path, region_sets, args.output_dir,
            args.min_mapq,
        )
        elapsed = time.monotonic() - t0
        logger.info("Finished %s in %.1fs -> %s", row.sample_name, elapsed, out)


if __name__ == "__main__":
    main()
