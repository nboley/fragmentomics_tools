"""1-sample throughput measurement hook.

Run this on a single sample before committing to a full preprocessing run.
Measures wall time and memory for Phase A processing of one sample across
all tiles, then prints a summary with extrapolated full-run estimates.

Usage:
    python -m background_model.measure_throughput \\
        --h5 /path/to/sample.frag.h5 \\
        --region-bed /path/to/regions.bed \\
        --fasta /path/to/genome.fa \\
        [--blacklist-bed /path/to/blacklist.bed] \\
        [--ref hg38]
"""

import argparse
import logging
import os
import resource
import tempfile
import time

import numpy as np

from background_model.config import PlumbingConfig
from background_model.preprocess import _worker_process_sample, build_tiles

log = logging.getLogger(__name__)


def measure_one_sample(
    h5_path: str,
    region_bed: str,
    fasta: str,
    blacklist_bed: str = "",
    ref: str = "hg38",
    tile_size: int = 16_384,
) -> dict:
    """Measure throughput for one sample.

    Returns a dict with timing and memory stats.
    """
    config = PlumbingConfig(
        sample_sheet="",  # not used for single-sample measurement
        region_beds={"train_pool": region_bed},
        blacklist_bed=blacklist_bed,
        fasta=fasta,
        tile_size=tile_size,
    )

    tiles = build_tiles(config.region_beds, config.tile_size, config.jitter, config.rf_budget, ref)

    sample_row = {
        "library": "throughput_test",
        "h5_path": h5_path,
    }

    with tempfile.TemporaryDirectory() as shard_dir:
        mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t0 = time.perf_counter()

        _worker_process_sample(sample_row, tiles, config, shard_dir, ref)

        elapsed = time.perf_counter() - t0
        mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # Check shard size
        shard_path = os.path.join(shard_dir, "throughput_test.npz")
        shard_size_mb = os.path.getsize(shard_path) / (1024 * 1024) if os.path.exists(shard_path) else 0

    return {
        "n_tiles": len(tiles),
        "elapsed_sec": elapsed,
        "sec_per_tile": elapsed / max(1, len(tiles)),
        "peak_rss_kb": mem_after,
        "rss_delta_kb": mem_after - mem_before,
        "shard_size_mb": shard_size_mb,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure 1-sample preprocessing throughput")
    parser.add_argument("--h5", required=True, help="Fragment h5 file path")
    parser.add_argument("--region-bed", required=True, help="Region BED file")
    parser.add_argument("--fasta", required=True, help="Reference FASTA path")
    parser.add_argument("--blacklist-bed", default="", help="Blacklist BED path")
    parser.add_argument("--ref", default="hg38", help="Reference genome name")
    parser.add_argument("--tile-size", type=int, default=16_384)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    stats = measure_one_sample(
        args.h5, args.region_bed, args.fasta,
        args.blacklist_bed, args.ref, args.tile_size,
    )

    print("\n=== Throughput Measurement (1 sample) ===")
    print(f"  Tiles:          {stats['n_tiles']}")
    print(f"  Wall time:      {stats['elapsed_sec']:.1f} s")
    print(f"  Per tile:       {stats['sec_per_tile']*1000:.1f} ms")
    print(f"  Peak RSS:       {stats['peak_rss_kb']/1024:.0f} MB")
    print(f"  Shard size:     {stats['shard_size_mb']:.1f} MB")

    # Extrapolate for 50 samples / 8 workers
    wall_50 = stats["elapsed_sec"] * 50 / 8
    print(f"\n  Est. 50 samples / 8 workers: {wall_50/60:.0f} min wall")


if __name__ == "__main__":
    main()
