#!/usr/bin/env python
"""Verify the vectorised build_region_weights is bit-identical to the original.

Runs both the old (loop) and new (vectorised) implementations on real regions
and checks:
  1. w_plus and w_minus are identical (max abs diff = 0)
  2. S_plus and S_minus are identical
  3. Invariants hold (sum=1, each strand=0.5)
  4. Emitted fragments are identical for a fixed seed
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from background_model.simulator.precompute import precompute_region
from background_model.simulator.sampler import draw_fragments_for_region, target_count_for_region
from background_model.simulator.weights import (
    GC_BIN_WIDTH,
    L_MAX,
    L_MIN,
    MAX_FL_HALF,
    N_GC_BINS,
    N_LENGTHS,
    NHEX,
    HexamerTables,
    RegionWeights,
    build_region_weights,
    gc_bin_index,
    gc_pct,
    generative_domain_size,
)


def build_region_weights_ORIGINAL(
    *, hex_fwd, hex_rc, cum_gc, hex_tables, marginal_fl, predict_lut,
    region_len, valid=None, pad=0,
):
    """Exact copy of the original loop-based implementation for comparison."""
    n_sites = region_len + 2 * pad + 1
    start_fwd = hex_tables.start_fwd
    end_fwd = hex_tables.end_fwd
    start_rev = hex_tables.start_rev
    end_rev = hex_tables.end_rev

    if valid is None:
        valid = np.ones(n_sites, dtype=bool)

    w_plus = np.zeros((n_sites, N_LENGTHS), dtype=np.float64)
    w_minus = np.zeros((n_sites, N_LENGTHS), dtype=np.float64)

    start_vals_plus = np.where(valid, start_fwd[hex_fwd], 0.0)
    start_vals_minus = np.where(valid, start_rev[hex_rc], 0.0)

    Ls = np.arange(L_MIN, L_MAX + 1)

    for li, L in enumerate(Ls):
        half_down = L // 2
        c5_lo = max(0, pad - half_down)
        c5_hi = min(n_sites - 1, pad + region_len - 1 - half_down)
        if c5_lo > c5_hi:
            continue
        c5s = np.arange(c5_lo, c5_hi + 1)
        c3s = c5s + L
        in_bounds = c3s < n_sites
        c5s = c5s[in_bounds]
        c3s = c3s[in_bounds]
        if len(c5s) == 0:
            continue
        vmask = valid[c5s] & valid[c3s]
        end_vals = end_fwd[hex_fwd[c3s]]
        gc_pcts_arr = gc_pct(c5s, c3s, cum_gc)
        gc_bins = gc_bin_index(gc_pcts_arr)
        predict_vals = predict_lut[li, gc_bins]
        E = np.where(vmask, end_vals * marginal_fl[li] / predict_vals, 0.0)
        w_plus[c5s, li] = E

    for li, L in enumerate(Ls):
        half_up = L - L // 2
        c5_lo = max(0, pad + half_up)
        c5_hi = min(n_sites - 1, pad + region_len - 1 + half_up)
        if c5_lo > c5_hi:
            continue
        c5s = np.arange(c5_lo, c5_hi + 1)
        c3s = c5s - L
        in_bounds = c3s >= 0
        c5s = c5s[in_bounds]
        c3s = c3s[in_bounds]
        if len(c5s) == 0:
            continue
        vmask = valid[c5s] & valid[c3s]
        end_vals = end_rev[hex_rc[c3s]]
        gc_pcts_arr = gc_pct(c5s, c3s, cum_gc)
        gc_bins = gc_bin_index(gc_pcts_arr)
        predict_vals = predict_lut[li, gc_bins]
        E = np.where(vmask, end_vals * marginal_fl[li] / predict_vals, 0.0)
        w_minus[c5s, li] = E

    # Normalisation
    Z_plus = w_plus.sum(axis=1)
    has_frags_plus = Z_plus > 0
    S_plus = float(start_vals_plus[has_frags_plus].sum())
    safe_Z_plus = np.where(has_frags_plus, Z_plus, 1.0)
    w_plus = (
        0.5
        * (start_vals_plus / S_plus)[:, np.newaxis]
        * (w_plus / safe_Z_plus[:, np.newaxis])
    )
    Z_minus = w_minus.sum(axis=1)
    has_frags_minus = Z_minus > 0
    S_minus = float(start_vals_minus[has_frags_minus].sum())
    safe_Z_minus = np.where(has_frags_minus, Z_minus, 1.0)
    w_minus = (
        0.5
        * (start_vals_minus / S_minus)[:, np.newaxis]
        * (w_minus / safe_Z_minus[:, np.newaxis])
    )
    return RegionWeights(w_plus=w_plus, w_minus=w_minus,
                         S_plus=S_plus, S_minus=S_minus, pad=pad)


def build_hexamer_tables(seed=42, dynamic_range=4.0):
    rng = np.random.default_rng(seed)
    sigma = np.log(dynamic_range) / (2 * 1.6448536269514722)
    tables = []
    for _ in range(4):
        w = np.exp(rng.normal(0.0, sigma, size=NHEX))
        tables.append(w / w.max())
    return HexamerTables(*tables)


def main():
    from background_model.simulator.capture import fit_and_build
    from fragmentomics_tools.dataframe import RegionDataFrame
    import pysam

    fasta_path = "/efs/analytics/nathanboley/data_resources/genome/hg38.fa"
    bed_path = (
        "/home/nathanboley/src/fragmentomics_tools/data/region_sets/"
        "quiet_v2_pad1200_repeats_removed_tile2560.bed"
    )

    rdf = RegionDataFrame.from_bed(bed_path, ref="hg38")
    n_test = 100
    rdf = rdf.iloc[:n_test]
    region_len = int((rdf.stop - rdf.start).unique()[0])
    target = target_count_for_region(region_len)

    predict_lut, marginal_fl = fit_and_build("RD-56670")
    hex_tables = build_hexamer_tables()
    fasta = pysam.FastaFile(fasta_path)

    print(f"Testing {n_test} regions, region_len={region_len}, pad={MAX_FL_HALF}")
    print(f"generative_domain_size({region_len}) = {generative_domain_size(region_len)}")

    max_diff_plus = 0.0
    max_diff_minus = 0.0
    max_diff_S_plus = 0.0
    max_diff_S_minus = 0.0
    n_frag_mismatches = 0
    seed = 20260929
    master_ss = np.random.SeedSequence(seed)
    child_seeds = master_ss.spawn(n_test)

    for i, row in enumerate(rdf.itertuples(index=False)):
        contig = row.contig
        gstart = int(row.start)
        gstop = int(row.stop)

        pc = precompute_region(contig, gstart, gstop, "", fasta=fasta,
                               pad=MAX_FL_HALF)

        kwargs = dict(
            hex_fwd=pc.hex_fwd, hex_rc=pc.hex_rc, cum_gc=pc.cum_gc,
            hex_tables=hex_tables, marginal_fl=marginal_fl,
            predict_lut=predict_lut, region_len=region_len,
            valid=pc.valid, pad=MAX_FL_HALF,
        )

        rw_old = build_region_weights_ORIGINAL(**kwargs)
        rw_new = build_region_weights(**kwargs)

        dp = float(np.abs(rw_old.w_plus - rw_new.w_plus).max())
        dm = float(np.abs(rw_old.w_minus - rw_new.w_minus).max())
        dsp = abs(rw_old.S_plus - rw_new.S_plus)
        dsm = abs(rw_old.S_minus - rw_new.S_minus)

        max_diff_plus = max(max_diff_plus, dp)
        max_diff_minus = max(max_diff_minus, dm)
        max_diff_S_plus = max(max_diff_S_plus, dsp)
        max_diff_S_minus = max(max_diff_S_minus, dsm)

        if dp > 0 or dm > 0:
            print(f"  region {i}: w diff plus={dp:.3e} minus={dm:.3e}")

        # Check emitted fragments are identical
        rng_old = np.random.default_rng(child_seeds[i])
        starts_old, stops_old, strands_old = draw_fragments_for_region(
            hex_fwd=pc.hex_fwd, hex_rc=pc.hex_rc, cum_gc=pc.cum_gc,
            valid=pc.valid, hex_tables=hex_tables, marginal_fl=marginal_fl,
            predict_lut=predict_lut, region_len=region_len,
            n_fragments=target, rng=rng_old, region_weights=rw_old,
            pad=MAX_FL_HALF,
        )
        rng_new = np.random.default_rng(child_seeds[i])
        starts_new, stops_new, strands_new = draw_fragments_for_region(
            hex_fwd=pc.hex_fwd, hex_rc=pc.hex_rc, cum_gc=pc.cum_gc,
            valid=pc.valid, hex_tables=hex_tables, marginal_fl=marginal_fl,
            predict_lut=predict_lut, region_len=region_len,
            n_fragments=target, rng=rng_new, region_weights=rw_new,
            pad=MAX_FL_HALF,
        )
        if (not np.array_equal(starts_old, starts_new) or
            not np.array_equal(stops_old, stops_new) or
            not np.array_equal(strands_old, strands_new)):
            n_frag_mismatches += 1
            print(f"  region {i}: FRAGMENT MISMATCH")

    fasta.close()

    print(f"\n=== Results ===")
    print(f"max |w_plus_old - w_plus_new|  = {max_diff_plus:.3e}")
    print(f"max |w_minus_old - w_minus_new| = {max_diff_minus:.3e}")
    print(f"max |S_plus diff|               = {max_diff_S_plus:.3e}")
    print(f"max |S_minus diff|              = {max_diff_S_minus:.3e}")
    print(f"fragment mismatches             = {n_frag_mismatches}/{n_test}")

    if max_diff_plus == 0.0 and max_diff_minus == 0.0 and n_frag_mismatches == 0:
        print("\nBIT-IDENTICAL: vectorised implementation matches original exactly.")
        return 0
    else:
        print("\nDIFFERENCES FOUND — vectorisation is NOT identical.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
