#!/usr/bin/env python
"""Fit per-hexamer NB dispersion parameters from real cfDNA data.

For each position in the production zarr store, we observe counts across
40 training samples.  The hexamer at each position determines a group (4096
groups).  Within each group we fit the NB2 dispersion parameter r using the
Pearson method-of-moments estimator:

    r_hat = sum(mu_p^2) / sum(var_p - mu_p)

where mu_p and var_p are the across-sample mean and variance at position p,
and the sums run over all overdispersed positions (var > mu) in the group.

Output: hexamer_dispersion.json with 4096 r values + summary statistics.

Env: /home/nathanboley/miniconda3/envs/biomarker_env/bin/python
Run: cd /home/nathanboley/src/fragmentomics_tools && PYTHONPATH=. \
     python scripts/fit_hexamer_dispersion.py [--max-tiles 100]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import zarr

# ── constants ────────────────────────────────────────────────────────────
NHEX = 4096
KMER = 6

DEFAULT_STORE = (
    "/efs/analytics/nathanboley/background_model/stores/bg_store_b67d7c95.zarr"
)
DEFAULT_OUT_DIR = "/efs/analytics/nathanboley/background_model/simulation_v3"

_BASE_LUT = np.full(256, 255, dtype=np.uint8)
_BASE_LUT[ord("A")] = 0
_BASE_LUT[ord("C")] = 1
_BASE_LUT[ord("G")] = 2
_BASE_LUT[ord("T")] = 3
_POW = (4 ** np.arange(KMER - 1, -1, -1)).astype(np.int64)

# log-space bins for approximate median (r ranges from ~0.1 to ~100k)
_LOG_R_LO, _LOG_R_HI, _N_BINS = -2.0, 12.0, 200
_LOG_R_EDGES = np.linspace(_LOG_R_LO, _LOG_R_HI, _N_BINS + 1)
_LOG_R_CENTERS = (_LOG_R_EDGES[:-1] + _LOG_R_EDGES[1:]) / 2


def _hexamer_indices(seq_bytes: np.ndarray):
    """Sliding 6-mer indices over ASCII-uint8 sequence.

    Returns (fwd_idx, valid) each of length len(seq_bytes) - 5.
    """
    codes = _BASE_LUT[seq_bytes].astype(np.int64)
    win = np.lib.stride_tricks.sliding_window_view(codes, KMER)
    valid = ~(win == 255).any(axis=1)
    safe = np.where(win == 255, 0, win)
    fwd = (safe @ _POW).astype(np.int64)
    return fwd, valid


def _approx_median_from_hist(hist: np.ndarray) -> float:
    """Approximate median from a histogram over log-r bins."""
    total = hist.sum()
    if total == 0:
        return np.nan
    cs = np.cumsum(hist)
    idx = np.searchsorted(cs, total / 2)
    idx = min(idx, _N_BINS - 1)
    return float(np.exp(_LOG_R_CENTERS[idx]))


# ── main ─────────────────────────────────────────────────────────────────

def fit(store_path: str, out_dir: str, max_tiles: int | None = None):
    t0 = time.time()

    root = zarr.open_group(store_path, mode="r")

    # ── geometry from store config ──────────────────────────────────────
    cfg = json.loads(root.attrs["config_json"])
    tile_size = cfg["tile_size"]      # 16384
    jitter = cfg["jitter"]            # 128
    rf_budget = cfg["rf_budget"]      # 2048
    n_tracks = 12

    n_total_tiles = root["tiles/seq"].shape[0]

    # ── training samples / tiles ────────────────────────────────────────
    roles = np.asarray(root["samples/role"][:])
    splits = np.asarray(root["tiles/split"][:])
    train_samples = np.nonzero(roles == 0)[0]
    train_tiles = np.nonzero(splits == 0)[0]
    if max_tiles is not None:
        train_tiles = train_tiles[:max_tiles]

    n_samples = len(train_samples)
    n_tiles = len(train_tiles)
    print(f"[fit] {n_samples} training samples, {n_tiles} training tiles", flush=True)
    print(f"[fit] tile_size={tile_size}, jitter={jitter}, rf_budget={rf_budget}",
          flush=True)

    # ── preload arrays ──────────────────────────────────────────────────
    print("[fit] preloading sparse arrays ...", flush=True)
    all_pos = np.asarray(root["counts/pos"][:])
    all_track = np.asarray(root["counts/track"][:])
    all_data = np.asarray(root["counts/data"][:])
    indptr = np.asarray(root["counts/indptr"][:])
    all_seq = np.asarray(root["tiles/seq"][:])
    print(f"[fit] preloaded ({time.time() - t0:.1f}s)", flush=True)

    # ── accumulators (Pearson MoM + mean-r + histogram) ─────────────────
    # Aggregated across tracks (one r per hexamer).
    hex_sum_mu2 = np.zeros(NHEX, dtype=np.float64)
    hex_sum_excess_var = np.zeros(NHEX, dtype=np.float64)
    hex_sum_r = np.zeros(NHEX, dtype=np.float64)
    hex_count_total = np.zeros(NHEX, dtype=np.int64)
    hex_count_od = np.zeros(NHEX, dtype=np.int64)
    hex_r_hist = np.zeros((NHEX, _N_BINS), dtype=np.int64)

    # Pre-compute indptr lookups for training (sample, tile) pairs
    # u = s_idx * n_total_tiles + t_idx
    us_matrix = train_samples[:, None] * n_total_tiles + train_tiles[None, :]
    # us_matrix shape: (n_samples, n_tiles) — row=sample, col=tile

    # seq offset: count position p → seq position p + rf_budget + jitter
    seq_offset = rf_budget + jitter

    for ti in range(n_tiles):
        t_idx = train_tiles[ti]

        # 1. Hexamer indices for center tile_size positions
        seq_bytes = all_seq[t_idx]
        hex_seq = seq_bytes[seq_offset : seq_offset + tile_size + KMER - 1]
        hex_idx, hex_valid = _hexamer_indices(hex_seq)
        hex_idx = hex_idx[:tile_size]
        hex_valid = hex_valid[:tile_size]

        # 2. Reconstruct per-position mean and variance across samples
        #    using one-pass accumulation (sum and sum-of-squares)
        sum_y = np.zeros((n_tracks, tile_size), dtype=np.float64)
        sum_y2 = np.zeros((n_tracks, tile_size), dtype=np.float64)

        for si in range(n_samples):
            u = int(us_matrix[si, ti])
            lo = int(indptr[u])
            hi = int(indptr[u + 1])

            if hi <= lo:
                # This sample has zero counts at this tile — contributes 0
                # to both sum_y and sum_y2 (already in the zeros init).
                continue

            pos = all_pos[lo:hi]
            track = all_track[lo:hi]
            data = all_data[lo:hi]

            # Center crop: positions [jitter, jitter + tile_size)
            mask = (pos >= jitter) & (pos < jitter + tile_size)
            if not mask.any():
                continue

            y = np.zeros((n_tracks, tile_size), dtype=np.float64)
            np.add.at(y, (track[mask], pos[mask] - jitter),
                       data[mask].astype(np.float64))
            sum_y += y
            sum_y2 += y * y

        # Per-position mean and sample variance (ddof=1)
        mu = sum_y / n_samples
        var = (sum_y2 - n_samples * mu * mu) / (n_samples - 1)

        # 3. Accumulate per-hexamer stats (flatten across tracks)
        # hex_idx is shared across tracks; tile to (n_tracks * tile_size) flat
        hex_flat = np.tile(hex_idx, n_tracks)         # (n_tracks * tile_size,)
        valid_flat = np.tile(hex_valid, n_tracks)
        mu_flat = mu.ravel()                           # (n_tracks * tile_size,)
        var_flat = var.ravel()

        ok = valid_flat
        np.add.at(hex_count_total, hex_flat[ok], 1)

        # Pearson estimator: accumulate over ALL mu > 0 positions (not just
        # overdispersed). Underdispersed positions contribute negative excess
        # variance, which correctly regularises the estimate. This is the
        # standard Pearson chi-squared NB dispersion estimator.
        has_counts = ok & (mu_flat > 0)
        h_c = hex_flat[has_counts]
        m_c = mu_flat[has_counts]
        v_c = var_flat[has_counts]
        np.add.at(hex_sum_mu2, h_c, m_c ** 2)
        np.add.at(hex_sum_excess_var, h_c, v_c - m_c)
        np.add.at(hex_count_od, h_c, 1)  # count of mu>0 positions

        # Per-position r stats (over overdispersed only, for reporting)
        od = has_counts & (var_flat > mu_flat)
        h_od = hex_flat[od]
        m_od = mu_flat[od]
        v_od = var_flat[od]
        r_od = m_od ** 2 / (v_od - m_od)
        np.add.at(hex_sum_r, h_od, r_od)

        # Histogram of log(r) for approximate median (overdispersed only)
        log_r = np.log(r_od)
        bin_idx = np.searchsorted(_LOG_R_EDGES, log_r) - 1
        bin_idx = np.clip(bin_idx, 0, _N_BINS - 1)
        np.add.at(hex_r_hist, (h_od, bin_idx), 1)

        if (ti + 1) % 200 == 0 or ti == 0:
            elapsed = time.time() - t0
            rate = (ti + 1) / elapsed
            eta = (n_tiles - ti - 1) / rate
            print(f"  tile {ti + 1}/{n_tiles}  ({elapsed:.1f}s, ETA {eta:.0f}s)",
                  flush=True)

    elapsed_fit = time.time() - t0
    print(f"[fit] tile loop done ({elapsed_fit:.1f}s)", flush=True)

    # ── compute final per-hexamer r ─────────────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        r_pearson = np.where(
            hex_sum_excess_var > 0,
            hex_sum_mu2 / hex_sum_excess_var,
            np.nan,
        )
        r_mean = np.where(
            hex_count_od > 0,
            hex_sum_r / hex_count_od,
            np.nan,
        )

    r_median = np.array([_approx_median_from_hist(hex_r_hist[h])
                         for h in range(NHEX)])

    # Primary r: Pearson estimator (best for NB simulation).
    # Uses sum over ALL mu>0 positions; sum_excess_var <= 0 means no
    # overdispersion detected → NaN (treated as Poisson/multinomial).
    hexamer_r = r_pearson.copy()

    # Fraction of hexamers with detected overdispersion
    finite = np.isfinite(hexamer_r)
    n_valid = int(finite.sum())
    n_nan = int((~finite).sum())
    frac_with_counts = hex_count_od.sum() / max(1, hex_count_total.sum())
    n_od_in_hist = int(hex_r_hist.sum())  # positions with per-position r

    # Per-position r stats (overdispersed positions only, for reference)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_od_per_hex = hex_r_hist.sum(axis=1)
        r_mean_od = np.where(n_od_per_hex > 0,
                             hex_sum_r / n_od_per_hex, np.nan)

    summary = {
        "n_hexamers_valid": n_valid,
        "n_hexamers_nan": n_nan,
        "frac_positions_with_counts": round(float(frac_with_counts), 4),
        "n_overdispersed_position_track_pairs": n_od_in_hist,
        "r_median": float(np.nanmedian(hexamer_r)),
        "r_mean": float(np.nanmean(hexamer_r)),
        "r_std": float(np.nanstd(hexamer_r)),
        "r_p5": float(np.nanpercentile(hexamer_r, 5)),
        "r_p25": float(np.nanpercentile(hexamer_r, 25)),
        "r_p75": float(np.nanpercentile(hexamer_r, 75)),
        "r_p95": float(np.nanpercentile(hexamer_r, 95)),
        "r_min": float(np.nanmin(hexamer_r)),
        "r_max": float(np.nanmax(hexamer_r)),
        "per_position_r_mean_od_only": float(np.nanmean(r_mean_od)),
    }

    result = {
        "hexamer_r": [None if np.isnan(v) else round(float(v), 4)
                      for v in hexamer_r],
        "summary": summary,
        "metadata": {
            "store_path": store_path,
            "config_hash": root.attrs["config_hash"],
            "n_samples": int(n_samples),
            "n_tiles": int(n_tiles),
            "tile_size": int(tile_size),
            "n_tracks": int(n_tracks),
            "method": "pearson_mom",
            "hexamer_definition": "6-mer starting at position (sliding window)",
            "date": datetime.now(timezone.utc).isoformat(),
            "runtime_s": round(time.time() - t0, 1),
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "hexamer_dispersion.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[fit] {n_valid} valid hexamers, {n_nan} NaN")
    print(f"[fit] positions with counts: {frac_with_counts:.3f}")
    print(f"[fit] r (Pearson): median={summary['r_median']:.1f}  "
          f"mean={summary['r_mean']:.1f}  "
          f"[p5={summary['r_p5']:.1f}, p95={summary['r_p95']:.1f}]")
    print(f"[fit] Saved to {out_path}  ({time.time() - t0:.1f}s total)")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--store", default=DEFAULT_STORE,
                    help="path to the production zarr store")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--max-tiles", type=int, default=None,
                    help="limit to first N training tiles (for testing)")
    args = ap.parse_args(argv)
    fit(args.store, args.out_dir, max_tiles=args.max_tiles)


if __name__ == "__main__":
    sys.exit(main())
