#!/usr/bin/env python
"""Profile the fragment simulator per-region pipeline (single-worker).

Measures wall time broken down by stage and runs cProfile for function-level
attribution.  Uses 1000 regions by default for a statistically meaningful
measurement without burning excessive time.

Usage:
    export PATH=/home/nathanboley/miniconda3/envs/biomarker_env/bin:$PATH
    PYTHONPATH=/home/nathanboley/src/biomarker \
      python scripts/profile_simulator.py --n-regions 1000
"""
from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from background_model.simulator.capture import fit_and_build  # noqa: E402
from background_model.simulator.precompute import precompute_region  # noqa: E402
from background_model.simulator.sampler import (  # noqa: E402
    draw_fragments_for_region,
    target_count_for_region,
)
from background_model.simulator.weights import (  # noqa: E402
    L_MAX,
    L_MIN,
    MAX_FL_HALF,
    NHEX,
    HexamerTables,
    build_region_weights,
    generative_domain_size,
)


DEFAULT_REGION_SET = (
    "/home/nathanboley/src/fragmentomics_tools/data/region_sets/"
    "quiet_v2_pad1200_repeats_removed_tile2560.bed"
)
DEFAULT_FASTA = "/efs/analytics/nathanboley/data_resources/genome/hg38.fa"
DEFAULT_SAMPLE = "RD-56670"


def build_hexamer_tables(seed: int, dynamic_range: float) -> HexamerTables:
    rng = np.random.default_rng(seed)
    sigma = np.log(dynamic_range) / (2 * 1.6448536269514722)
    tables = []
    for _ in range(4):
        w = np.exp(rng.normal(0.0, sigma, size=NHEX))
        tables.append(w / w.max())
    return HexamerTables(*tables)


def load_regions(bed_path, n_regions):
    from fragmentomics_tools.dataframe import RegionDataFrame
    rdf = RegionDataFrame.from_bed(bed_path, ref="hg38")
    if n_regions is not None:
        rdf = rdf.iloc[:n_regions]
    return rdf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-regions", type=int, default=1000)
    ap.add_argument("--region-set-bed", default=DEFAULT_REGION_SET)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--sample", default=DEFAULT_SAMPLE)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--cprofile", action="store_true",
                    help="Run cProfile on the hot loop")
    ap.add_argument("--cprofile-regions", type=int, default=500,
                    help="Number of regions for the cProfile run")
    args = ap.parse_args()

    print("=== Setup ===")
    t0 = time.perf_counter()
    rdf = load_regions(args.region_set_bed, args.n_regions)
    region_len = int((rdf.stop - rdf.start).unique()[0])
    target = target_count_for_region(region_len)
    print(f"Loaded {len(rdf)} regions, region_len={region_len}, target={target}")

    predict_lut, marginal_fl = fit_and_build(args.sample)
    hex_tables = build_hexamer_tables(42, 4.0)
    print(f"Setup: {time.perf_counter() - t0:.1f}s")

    import pysam
    fasta = pysam.FastaFile(args.fasta)

    master_ss = np.random.SeedSequence(args.seed)
    child_seeds = master_ss.spawn(len(rdf))

    # === Timed per-stage measurement ===
    print(f"\n=== Per-stage timing ({len(rdf)} regions, single worker) ===")

    t_precompute = 0.0
    t_weights = 0.0
    t_sample = 0.0
    t_total = 0.0

    # Also verify invariants on first few regions
    n_verify = min(5, len(rdf))

    for i, row in enumerate(rdf.itertuples(index=False)):
        contig = row.contig
        gstart = int(row.start)
        gstop = int(row.stop)
        rng = np.random.default_rng(child_seeds[i])

        t_r0 = time.perf_counter()

        t_s = time.perf_counter()
        pc = precompute_region(contig, gstart, gstop, "", fasta=fasta,
                               pad=MAX_FL_HALF)
        t_precompute += time.perf_counter() - t_s

        t_s = time.perf_counter()
        rw = build_region_weights(
            hex_fwd=pc.hex_fwd, hex_rc=pc.hex_rc, cum_gc=pc.cum_gc,
            hex_tables=hex_tables, marginal_fl=marginal_fl,
            predict_lut=predict_lut, region_len=region_len, valid=pc.valid,
            pad=MAX_FL_HALF,
        )
        t_weights += time.perf_counter() - t_s

        t_s = time.perf_counter()
        starts, stops, strands = draw_fragments_for_region(
            hex_fwd=pc.hex_fwd, hex_rc=pc.hex_rc, cum_gc=pc.cum_gc,
            valid=pc.valid, hex_tables=hex_tables, marginal_fl=marginal_fl,
            predict_lut=predict_lut, region_len=region_len,
            n_fragments=target, rng=rng, region_weights=rw,
            pad=MAX_FL_HALF,
        )
        t_sample += time.perf_counter() - t_s

        t_total += time.perf_counter() - t_r0

        if i < n_verify:
            s_plus = float(rw.w_plus.sum())
            s_minus = float(rw.w_minus.sum())
            assert abs(1.0 - (s_plus + s_minus)) < 1e-12, \
                f"region {i}: sum={s_plus+s_minus}"
            assert abs(0.5 - s_plus) < 1e-12
            assert abs(0.5 - s_minus) < 1e-12

    fasta.close()

    n = len(rdf)
    print(f"\nPer-region breakdown (ms):")
    print(f"  precompute : {1000*t_precompute/n:8.3f} ms/region  "
          f"({100*t_precompute/t_total:5.1f}%)")
    print(f"  weights    : {1000*t_weights/n:8.3f} ms/region  "
          f"({100*t_weights/t_total:5.1f}%)")
    print(f"  sample     : {1000*t_sample/n:8.3f} ms/region  "
          f"({100*t_sample/t_total:5.1f}%)")
    print(f"  TOTAL      : {1000*t_total/n:8.3f} ms/region")
    print(f"\nWall clock   : {t_total:.1f}s for {n} regions")
    print(f"Extrapolation: {66649 * t_total / n / 60:.1f} min for 66,649 regions")

    # === cProfile run ===
    if args.cprofile:
        n_prof = min(args.cprofile_regions, len(rdf))
        print(f"\n=== cProfile ({n_prof} regions) ===")

        fasta2 = pysam.FastaFile(args.fasta)
        master_ss2 = np.random.SeedSequence(args.seed)
        child_seeds2 = master_ss2.spawn(n_prof)

        def profiled_loop():
            for i in range(n_prof):
                row = rdf.iloc[i]
                rng = np.random.default_rng(child_seeds2[i])
                pc = precompute_region(
                    row.contig, int(row.start), int(row.stop), "",
                    fasta=fasta2, pad=MAX_FL_HALF)
                rw = build_region_weights(
                    hex_fwd=pc.hex_fwd, hex_rc=pc.hex_rc, cum_gc=pc.cum_gc,
                    hex_tables=hex_tables, marginal_fl=marginal_fl,
                    predict_lut=predict_lut, region_len=region_len,
                    valid=pc.valid, pad=MAX_FL_HALF)
                draw_fragments_for_region(
                    hex_fwd=pc.hex_fwd, hex_rc=pc.hex_rc, cum_gc=pc.cum_gc,
                    valid=pc.valid, hex_tables=hex_tables,
                    marginal_fl=marginal_fl, predict_lut=predict_lut,
                    region_len=region_len, n_fragments=target, rng=rng,
                    region_weights=rw, pad=MAX_FL_HALF)

        prof = cProfile.Profile()
        prof.enable()
        profiled_loop()
        prof.disable()
        fasta2.close()

        stats = pstats.Stats(prof)
        stats.sort_stats("cumulative")
        print("\nTop 30 by cumulative time:")
        stats.print_stats(30)
        stats.sort_stats("tottime")
        print("\nTop 30 by total time:")
        stats.print_stats(30)


if __name__ == "__main__":
    main()
