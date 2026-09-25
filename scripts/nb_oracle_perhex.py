#!/usr/bin/env python
"""Per-hexamer NB oracle anchor: closing open question 2 from nb_oracle.md.

Does fitting a per-hexamer dispersion r (4096 free parameters instead of 1
scalar) produce a materially tighter oracle floor?

THE HAZARD: 4096 free parameters on the val set can overfit rare hexamers,
silently biasing the "floor" below the true lower bound. Guard:
  - Fit per-hexamer r on the TRAIN split, score on VAL.
  - Report BOTH in-sample (fit-on-val) and out-of-sample (fit-on-train,
    score-on-val) values.
  - A large gap between them is evidence of overfitting, NOT evidence of a
    tighter floor.
  - Rare hexamers (< MIN_HEX_POSITIONS positions in the fitting set) fall
    back to the scalar r.

Also scores with the TRUE per-hexamer r from the ground truth (no fitting)
as a theoretical reference.

NOTE: the scalar anchor is itself fitted on val. At one parameter that is
negligible, but stated rather than implied.

Uses the FROZEN-CORE MaskedNegativeBinomialOffsetNLLLoss with the SAME config
as the scalar anchor and training runs:
  max_dispersion_ratio=2.0, clamp_margin=1.0, dispersion_window_size=1

Usage:
    cd <repo>
    PYTHONPATH=. /home/nathanboley/miniconda3/envs/biomarker_env/bin/python \
        scripts/nb_oracle_perhex.py
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.special import gammaln

# ── Paths ─────────────────────────────────────────────────────────────────
STORE = "/efs/analytics/nathanboley/background_model/simulation_v3_nb/stores/sim_store_v3nb_A.zarr"
SIM_DIR = "/efs/analytics/nathanboley/background_model/simulation_v3_nb/A"
FASTA = "/efs/analytics/nathanboley/data_resources/genome/hg38.fa"
OUT_JSON = "/efs/analytics/nathanboley/background_model/simulation_v3_nb/A/oracle_nb_perhex.json"
V2_JSON = "/efs/analytics/nathanboley/background_model/simulation_v3_nb/A/oracle_nb_v2.json"

# ── Configuration ─────────────────────────────────────────────────────────
MIN_HEX_POSITIONS = 200     # minimum POSITION count per hexamer for fitting
                            # (200 positions × 12 tracks = 2400 data points)
N_GRID = 200                # grid points for log_r search
LOG_R_LO = np.log(0.3)      # grid lower bound
LOG_R_HI = np.log(5000.0)   # grid upper bound
NHEX = 4096

# Reference values from nb_oracle_v2.py (for comparison table)
REF_SCALAR_ORACLE = 4.026351353675127
REF_UNIFORM = 4.115219691395760
REF_GAP = REF_UNIFORM - REF_SCALAR_ORACLE
REF_TRAINED_KEN = 4.060737457573413
REF_TRAINED_HYBRID = 4.067427050471306
REF_UNTRAINED_KEN = 4.117498033046722
REF_UNTRAINED_HYBRID = 4.125973814576864


# ── Hexamer index computation ─────────────────────────────────────────────

_BASE_LUT = np.full(256, 255, dtype=np.uint8)
_BASE_LUT[ord("A")] = 0
_BASE_LUT[ord("C")] = 1
_BASE_LUT[ord("G")] = 2
_BASE_LUT[ord("T")] = 3
_POW = (4 ** np.arange(5, -1, -1)).astype(np.int64)


def _hex_indices(seq_bytes):
    """Compute hexamer indices for a byte sequence. Returns (fwd, valid)."""
    codes = _BASE_LUT[seq_bytes].astype(np.int64)
    win = np.lib.stride_tricks.sliding_window_view(codes, 6)
    valid = ~(win == 255).any(axis=1)
    safe = np.where(win == 255, 0, win)
    fwd = (safe @ _POW).astype(np.int16)
    return fwd, valid


def get_hex_for_crop(contig, gstart, crop_start, crop_stop, fa):
    """Hexamer indices for cropped positions in a tile.

    Uses the same convention as sim_fragments.hexamer_indices: the hexamer at
    position p is the 6-mer whose 5' end is 3 bases before p (centered around
    the cut site). This matches w6 and hexamer_r indexing.

    Returns: (hex_idx, hex_valid) each of shape (tile_size,)
    """
    HEX_HALF = 3
    seq = fa.fetch(contig, gstart + crop_start - HEX_HALF,
                   gstart + crop_stop + HEX_HALF).upper()
    seq_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    fwd, valid = _hex_indices(seq_bytes)
    tile_size = crop_stop - crop_start
    return fwd[:tile_size], valid[:tile_size]


# ── NB NLL helpers ────────────────────────────────────────────────────────

def nb_nll_positions(y, r_eff, mu):
    """Vectorized per-position NB NLL. All inputs broadcastable numpy arrays.

    Returns -log P(y | r_eff, mu) element-wise.
    Handles mu=0 safely (contributes 0 when y=0).
    """
    safe_mu = np.maximum(mu, 1e-30)
    nll = -(gammaln(y + r_eff) - gammaln(r_eff) - gammaln(y + 1)
            + r_eff * np.log(r_eff / (r_eff + safe_mu))
            + np.where(y > 0, y * np.log(safe_mu / (r_eff + safe_mu)), 0.0))
    return np.where(mu > 1e-30, nll, 0.0)


# ── Scoring with frozen-core loss ─────────────────────────────────────────

def score_val_perhex(oracle_data, hex_per_pair, hex_r, loss_fn, scalar_r_fallback):
    """Score val pairs with per-hexamer log_dispersion via the frozen-core loss.

    Args:
        oracle_data: dict di -> (oracle_logits, uniform_logits, y_t, m_t)
        hex_per_pair: dict di -> (hex_idx, hex_valid)
        hex_r: (4096,) array of r values per hexamer
        loss_fn: MaskedNegativeBinomialOffsetNLLLoss instance
        scalar_r_fallback: r value for invalid hexamers

    Returns: mean NLL over all val pairs
    """
    losses = []
    with torch.no_grad():
        for di in sorted(oracle_data):
            oracle_logits, _, y_t, m_t = oracle_data[di]
            hex_idx, hex_valid = hex_per_pair[di]
            B, C, L = oracle_logits.shape

            # Build per-position log_r: (L,)
            r_at_pos = hex_r[hex_idx.astype(np.intp)]
            r_at_pos = np.where(hex_valid, r_at_pos, scalar_r_fallback)
            # Handle NaN in hex_r (hexamers with N bases in ground truth)
            r_at_pos = np.where(np.isfinite(r_at_pos), r_at_pos, scalar_r_fallback)
            log_r_pos = np.log(np.maximum(r_at_pos, 1e-6))

            # (1, C, L) — same r across all tracks at a given position
            ld = torch.from_numpy(log_r_pos).float()
            ld = ld.unsqueeze(0).unsqueeze(0).expand(B, C, L).contiguous()

            loss_val = loss_fn(oracle_logits, ld, y_t, m_t).item()
            losses.append(loss_val)
    return float(np.mean(losses))


# ── Per-hexamer fitting via grid search ───────────────────────────────────

def fit_perhex_grid(oracle_data, hex_per_pair, scalar_r, min_positions):
    """Fit per-hexamer r using grid search over the NB NLL.

    For each hexamer h, accumulates the total NB NLL (weighted by 1/N_track)
    over all positions in the data set where hex = h, evaluated on a grid of
    candidate r values. The minimiser on the grid is the fitted r_h.

    Hexamers with < min_positions positions fall back to scalar_r.

    Returns: (fitted_r, hex_pos_count, grid_info)
        fitted_r: (4096,) array
        hex_pos_count: (4096,) int64 — position count per hexamer
        grid_info: dict with log_r_grid and hex_nll_grid for provenance
    """
    log_r_grid = np.linspace(LOG_R_LO, LOG_R_HI, N_GRID)
    r_grid = np.exp(log_r_grid)

    hex_nll_sum = np.zeros((NHEX, N_GRID), dtype=np.float64)
    hex_pos_count = np.zeros(NHEX, dtype=np.int64)
    n_pairs_done = 0

    for di in sorted(oracle_data):
        oracle_logits, _, y_t, m_t = oracle_data[di]
        hex_idx, hex_valid = hex_per_pair[di]
        hex_idx_np = hex_idx.astype(np.intp)

        with torch.no_grad():
            logp = torch.log_softmax(oracle_logits, dim=-1)
            N = y_t.sum(dim=-1)  # (1, C)
            p = logp.exp()
            mu = N[..., None].clamp(min=1.0) * p

        logp_np = logp[0].numpy()
        N_np = N[0].numpy()
        mu_np = mu[0].numpy()
        y_np = y_t[0].numpy()
        p_np = p[0].numpy()

        # mask: True = valid
        if m_t.dim() == 3:
            mask = m_t[0, 0].numpy().astype(bool)
        else:
            mask = m_t[0].numpy().astype(bool)

        # Dispersion clamp floor: r_floor = mu / (ratio*(1-p) - 1)
        denom = np.clip(2.0 * (1.0 - p_np) - 1.0, 0.01, None)
        r_floor = mu_np / denom  # (C, L)

        C, L = y_np.shape
        combined_valid = mask & hex_valid

        for c in range(C):
            N_c = N_np[c]
            if N_c < 1:
                continue
            w = 1.0 / N_c

            vp = np.nonzero(combined_valid)[0]
            if len(vp) == 0:
                continue

            y_v = y_np[c, vp]
            mu_v = mu_np[c, vp]
            rf_v = r_floor[c, vp]
            hex_v = hex_idx_np[vp]

            # r_eff = max(r_candidate, r_floor): (N_GRID, n_valid)
            r_eff = np.maximum(r_grid[:, None], rf_v[None, :])

            # NB NLL: (N_GRID, n_valid)
            nll = nb_nll_positions(y_v[None, :], r_eff, mu_v[None, :])
            nll_weighted = nll * w

            # Accumulate per hexamer per grid point — single scatter
            # nll_weighted.T: (n_valid, N_GRID); hex_v: (n_valid,)
            # hex_nll_sum[hex_v[j], :] += nll_weighted.T[j, :]
            np.add.at(hex_nll_sum, hex_v, nll_weighted.T)

        # Count positions (once per position, not per track)
        vp_all = np.nonzero(combined_valid)[0]
        hex_pos_count += np.bincount(hex_idx_np[vp_all], minlength=NHEX)

        n_pairs_done += 1
        if n_pairs_done % 200 == 0:
            print(f"    fitting: {n_pairs_done}/{len(oracle_data)} pairs", flush=True)

    # Find optimal r per hexamer
    fitted_log_r = np.full(NHEX, np.log(scalar_r))
    n_fitted = 0
    for h in range(NHEX):
        if hex_pos_count[h] >= min_positions:
            best_gi = int(np.argmin(hex_nll_sum[h]))
            fitted_log_r[h] = log_r_grid[best_gi]
            n_fitted += 1

    fitted_r = np.exp(fitted_log_r)
    n_fallback = NHEX - n_fitted

    print(f"    fitted {n_fitted} hexamers, {n_fallback} fallback to scalar r={scalar_r:.1f}")
    print(f"    position counts: min={hex_pos_count.min()}, "
          f"median={np.median(hex_pos_count):.0f}, max={hex_pos_count.max()}")

    grid_info = {
        "log_r_grid_range": [float(LOG_R_LO), float(LOG_R_HI)],
        "n_grid": N_GRID,
        "n_fitted": n_fitted,
        "n_fallback": n_fallback,
        "min_positions_threshold": min_positions,
    }

    return fitted_r, hex_pos_count, grid_info


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    import pysam
    import zarr
    from scipy.optimize import minimize_scalar

    from background_model.config import PlumbingConfig
    from background_model.dataset import BackgroundTileDataset
    from background_model_core import MaskedNegativeBinomialOffsetNLLLoss
    from scripts.sim_fragments import GCBias2D, MAX_LEN
    from scripts.sim_oracle import compute_oracle_propensity_for_tile

    # ── Load ground truth ──────────────────────────────────────────────────
    gt = np.load(os.path.join(SIM_DIR, "ground_truth.npz"), allow_pickle=True)
    w6 = gt["w6"]
    len_vals = gt["len_vals"]
    len_p_per_sample = gt["len_p_per_sample"]
    true_hex_r = gt["hexamer_r"]  # (4096,) — the true per-hexamer r
    gcbias = GCBias2D(gt["bias_lengths"], gt["bias_gc_percents"], gt["bias_grid"])
    gc_lut = gcbias.build_lookup_table(max_len=MAX_LEN)

    n_true_finite = int(np.isfinite(true_hex_r).sum())
    n_true_nan = NHEX - n_true_finite
    print(f"[perhex] Ground truth hexamer_r: {n_true_finite} finite, {n_true_nan} NaN")
    print(f"         median={np.nanmedian(true_hex_r):.4f}, "
          f"p5={np.nanpercentile(true_hex_r, 5):.4f}, "
          f"p95={np.nanpercentile(true_hex_r, 95):.4f}")

    # ── Store metadata ─────────────────────────────────────────────────────
    root = zarr.open_group(STORE, mode="r")
    cfg = PlumbingConfig.from_json(root.attrs["config_json"])
    tile_size = cfg.tile_size
    l_target = cfg.l_target
    contigs = root["tiles/contig"][:]
    starts = root["tiles/start"][:]
    stops = root["tiles/stop"][:]

    crop_start = (l_target - tile_size) // 2
    crop_stop = crop_start + tile_size

    # ── Build val and train datasets ───────────────────────────────────────
    ds_val = BackgroundTileDataset(
        store_path=STORE, model_input_size=tile_size, split="val",
        sample_role="train", min_N=0, train_mode=False, seed=1337,
    )
    ds_train = BackgroundTileDataset(
        store_path=STORE, model_input_size=tile_size, split="train",
        sample_role="train", min_N=0, train_mode=False, seed=1337,
    )
    n_val = len(ds_val)
    n_train = len(ds_train)
    print(f"[perhex] val={n_val} pairs, train={n_train} pairs, "
          f"crop [{crop_start},{crop_stop}) ({time.time()-t0:.1f}s)", flush=True)

    # ── Phase A: Pre-compute val oracle data + hex indices ─────────────────
    print(f"\n{'='*70}")
    print("PHASE A: Pre-computing val oracle data + hexamer indices")
    print(f"{'='*70}")

    val_tile_to_pairs = {}
    for di in range(n_val):
        s_idx, t_idx = ds_val.index[di]
        val_tile_to_pairs.setdefault(t_idx, []).append((di, s_idx))

    fa = pysam.FastaFile(FASTA)
    loss_fn = MaskedNegativeBinomialOffsetNLLLoss(
        max_dispersion_ratio=2.0, clamp_margin=1.0,
    )

    val_oracle_data = {}   # di -> (oracle_logits, uniform_logits, y_t, m_t)
    val_hex_data = {}      # di -> (hex_idx, hex_valid)

    for tn, t_idx in enumerate(sorted(val_tile_to_pairs)):
        contig = str(contigs[t_idx])
        gstart, gstop = int(starts[t_idx]), int(stops[t_idx])
        pairs = val_tile_to_pairs[t_idx]
        uniq = sorted({s for _, s in pairs})
        loc = {s: i for i, s in enumerate(uniq)}

        prop_full = compute_oracle_propensity_for_tile(
            contig, gstart, gstop, l_target, w6, gcbias, len_vals,
            len_p_per_sample, np.array(uniq), fa, gc_lut=gc_lut,
        )
        prop_cropped = prop_full[:, :, crop_start:crop_stop]

        # Hexamer indices for this tile's cropped region
        hex_idx, hex_valid = get_hex_for_crop(contig, gstart, crop_start, crop_stop, fa)

        for di, s_idx in pairs:
            _, y, mask = ds_val[di]
            p = prop_cropped[loc[s_idx]]
            oracle_logits = torch.from_numpy(
                np.where(p > 0, np.log(p), -1e30)
            ).float().unsqueeze(0)
            uniform_logits = torch.zeros_like(oracle_logits)
            y_t = y.unsqueeze(0)
            m_t = mask.unsqueeze(0)
            val_oracle_data[di] = (oracle_logits, uniform_logits, y_t, m_t)
            val_hex_data[di] = (hex_idx.copy(), hex_valid.copy())

        if (tn + 1) % 50 == 0 or tn == 0:
            print(f"  val tiles: {tn+1}/{len(val_tile_to_pairs)} ({time.time()-t0:.1f}s)",
                  flush=True)

    print(f"[perhex] Cached {len(val_oracle_data)} val pairs ({time.time()-t0:.1f}s)",
          flush=True)

    # ── Phase B: Reproduce scalar anchor + score with TRUE hex_r ───────────
    print(f"\n{'='*70}")
    print("PHASE B: Scalar anchor (verification) + TRUE per-hexamer r")
    print(f"{'='*70}")

    # Scalar r: reproduce the nb_oracle_v2 result
    def eval_scalar_loss(log_r_val, use_oracle=True):
        losses = []
        with torch.no_grad():
            for di in sorted(val_oracle_data):
                oracle_logits, uniform_logits, y_t, m_t = val_oracle_data[di]
                logits = oracle_logits if use_oracle else uniform_logits
                B, C, L = logits.shape
                ld = torch.full((1, C, L), log_r_val, dtype=torch.float32)
                loss_val = loss_fn(logits, ld, y_t, m_t).item()
                losses.append(loss_val)
        return float(np.mean(losses))

    # Fit scalar r (oracle propensity) on val
    result_scalar = minimize_scalar(
        lambda lr: eval_scalar_loss(lr, use_oracle=True),
        bounds=(np.log(1.0), np.log(3000.0)),
        method="bounded",
        options={"xatol": 0.01},
    )
    scalar_r = float(np.exp(result_scalar.x))
    scalar_oracle_loss = float(result_scalar.fun)

    # Fit scalar r (uniform) on val
    result_uniform = minimize_scalar(
        lambda lr: eval_scalar_loss(lr, use_oracle=False),
        bounds=(np.log(1.0), np.log(3000.0)),
        method="bounded",
        options={"xatol": 0.01},
    )
    scalar_uniform_r = float(np.exp(result_uniform.x))
    scalar_uniform_loss = float(result_uniform.fun)
    scalar_gap = scalar_uniform_loss - scalar_oracle_loss

    print(f"  Scalar oracle NLL: {scalar_oracle_loss:.9f} (r={scalar_r:.1f})")
    print(f"  Scalar uniform NLL: {scalar_uniform_loss:.9f} (r={scalar_uniform_r:.1f})")
    print(f"  Scalar gap: {scalar_gap:.9f}")
    print(f"  Reference (v2): oracle={REF_SCALAR_ORACLE:.9f}, gap={REF_GAP:.9f}")
    assert abs(scalar_oracle_loss - REF_SCALAR_ORACLE) < 0.001, \
        f"Scalar oracle mismatch: {scalar_oracle_loss} vs {REF_SCALAR_ORACLE}"

    # Score val with TRUE per-hexamer r
    true_r_oracle_loss = score_val_perhex(
        val_oracle_data, val_hex_data, true_hex_r, loss_fn, scalar_r,
    )
    true_r_uniform_loss = score_val_perhex(
        {di: (val_oracle_data[di][1], val_oracle_data[di][1],
              val_oracle_data[di][2], val_oracle_data[di][3])
         for di in val_oracle_data},
        val_hex_data,
        np.full(NHEX, scalar_uniform_r),  # uniform doesn't benefit from per-hex r
        loss_fn, scalar_uniform_r,
    )

    print(f"\n  TRUE per-hex r oracle NLL: {true_r_oracle_loss:.9f}")
    print(f"  Delta from scalar:        {true_r_oracle_loss - scalar_oracle_loss:+.9f}")
    print(f"  (negative = tighter floor)")

    # ── Phase C: Pre-compute train oracle data + hex indices ───────────────
    print(f"\n{'='*70}")
    print("PHASE C: Pre-computing train oracle data + hexamer indices")
    print(f"{'='*70}")

    train_tile_to_pairs = {}
    for di in range(n_train):
        s_idx, t_idx = ds_train.index[di]
        train_tile_to_pairs.setdefault(t_idx, []).append((di, s_idx))

    train_oracle_data = {}
    train_hex_data = {}

    for tn, t_idx in enumerate(sorted(train_tile_to_pairs)):
        contig = str(contigs[t_idx])
        gstart, gstop = int(starts[t_idx]), int(stops[t_idx])
        pairs = train_tile_to_pairs[t_idx]
        uniq = sorted({s for _, s in pairs})
        loc = {s: i for i, s in enumerate(uniq)}

        prop_full = compute_oracle_propensity_for_tile(
            contig, gstart, gstop, l_target, w6, gcbias, len_vals,
            len_p_per_sample, np.array(uniq), fa, gc_lut=gc_lut,
        )
        prop_cropped = prop_full[:, :, crop_start:crop_stop]

        hex_idx, hex_valid = get_hex_for_crop(contig, gstart, crop_start, crop_stop, fa)

        for di, s_idx in pairs:
            _, y, mask = ds_train[di]
            p = prop_cropped[loc[s_idx]]
            oracle_logits = torch.from_numpy(
                np.where(p > 0, np.log(p), -1e30)
            ).float().unsqueeze(0)
            uniform_logits = torch.zeros_like(oracle_logits)
            y_t = y.unsqueeze(0)
            m_t = mask.unsqueeze(0)
            train_oracle_data[di] = (oracle_logits, uniform_logits, y_t, m_t)
            train_hex_data[di] = (hex_idx.copy(), hex_valid.copy())

        if (tn + 1) % 100 == 0 or tn == 0:
            print(f"  train tiles: {tn+1}/{len(train_tile_to_pairs)} ({time.time()-t0:.1f}s)",
                  flush=True)

    fa.close()
    print(f"[perhex] Cached {len(train_oracle_data)} train pairs ({time.time()-t0:.1f}s)",
          flush=True)

    # ── Phase D: Fit per-hexamer r on TRAIN (out-of-sample) ────────────────
    print(f"\n{'='*70}")
    print("PHASE D: Fitting per-hexamer r on TRAIN (out-of-sample guard)")
    print(f"{'='*70}")

    fitted_r_train, hex_count_train, grid_info_train = fit_perhex_grid(
        train_oracle_data, train_hex_data, scalar_r, MIN_HEX_POSITIONS,
    )
    print(f"  ({time.time()-t0:.1f}s)")

    # ── Phase E: Score VAL with train-fitted r (out-of-sample) ─────────────
    print(f"\n{'='*70}")
    print("PHASE E: Scoring VAL with train-fitted per-hexamer r")
    print(f"{'='*70}")

    oos_perhex_loss = score_val_perhex(
        val_oracle_data, val_hex_data, fitted_r_train, loss_fn, scalar_r,
    )
    print(f"  Out-of-sample per-hex oracle NLL: {oos_perhex_loss:.9f}")
    print(f"  Delta from scalar:                {oos_perhex_loss - scalar_oracle_loss:+.9f}")

    # ── Phase F: Fit per-hexamer r on VAL (in-sample, for comparison) ──────
    print(f"\n{'='*70}")
    print("PHASE F: Fitting per-hexamer r on VAL (in-sample, for overfitting check)")
    print(f"{'='*70}")

    fitted_r_val, hex_count_val, grid_info_val = fit_perhex_grid(
        val_oracle_data, val_hex_data, scalar_r, MIN_HEX_POSITIONS,
    )
    print(f"  ({time.time()-t0:.1f}s)")

    # ── Phase G: Score VAL with val-fitted r (in-sample) ───────────────────
    print(f"\n{'='*70}")
    print("PHASE G: Scoring VAL with val-fitted per-hexamer r (IN-SAMPLE)")
    print(f"{'='*70}")

    insample_perhex_loss = score_val_perhex(
        val_oracle_data, val_hex_data, fitted_r_val, loss_fn, scalar_r,
    )
    print(f"  In-sample per-hex oracle NLL: {insample_perhex_loss:.9f}")
    print(f"  Delta from scalar:            {insample_perhex_loss - scalar_oracle_loss:+.9f}")

    # ── Overfitting diagnostic ─────────────────────────────────────────────
    overfit_gap = insample_perhex_loss - oos_perhex_loss
    print(f"\n  Overfitting gap (in-sample - out-of-sample): {overfit_gap:+.9f}")
    if overfit_gap < -0.001:
        print(f"  *** In-sample is LOWER by {-overfit_gap:.6f} — "
              f"evidence of overfitting, not a tighter floor ***")
    elif overfit_gap < 0.0:
        # Negative IS the overfitting direction — in-sample fits better than
        # out-of-sample. What makes it benign here is the MAGNITUDE, not the
        # sign, so say that rather than implying the sign was reassuring.
        print(f"  In-sample is lower by {-overfit_gap:.6f} — the overfitting "
              f"direction, but below the 0.001 threshold, so not material")
    else:
        print(f"  Gap is non-negative — no overfitting detected")

    # ── Phase H: Comparison table + sanity gates ───────────────────────────
    print(f"\n{'='*70}")
    print("RESULTS TABLE")
    print(f"{'='*70}")

    # Use the out-of-sample result as the per-hexamer anchor
    perhex_anchor = oos_perhex_loss
    gap_oos = scalar_uniform_loss - perhex_anchor

    # CANONICAL %-bias is computed against the PUBLISHED v2 anchor (REF_*), not
    # against this script's own re-fit of the scalar. This script re-fits the
    # scalar with scipy only and lands at r=21.0 / 4.026409, whereas v2 selects
    # min-over-union and publishes r=1096 / 4.026351354 — a 5.7e-5 difference
    # that is immaterial to every verdict here (all thresholds are 1e-3) but
    # would otherwise make two artifacts in one directory report different
    # %-bias for the same checkpoint. One floor, one number.
    def pct_bias_scalar(nll):
        return 100.0 * (REF_UNIFORM - nll) / REF_GAP

    # NOT AN ANCHOR. Per-hexamer scored WORSE than scalar out-of-sample, so it
    # is not a floor; dividing by its smaller gap inflates every model. Kept
    # only as a sensitivity figure so the inflation is visible and quantified.
    def pct_bias_perhex(nll):
        return 100.0 * (scalar_uniform_loss - nll) / gap_oos

    print(f"\n  {'Anchor':<35s} {'NLL':>12s} {'Δ vs scalar':>12s}")
    print(f"  {'-'*35} {'-'*12} {'-'*12}")
    print(f"  {'Scalar oracle (1 param, fit-on-val)':<35s} "
          f"{scalar_oracle_loss:>12.6f} {'—':>12s}")
    print(f"  {'True hex_r (no fit)':<35s} "
          f"{true_r_oracle_loss:>12.6f} {true_r_oracle_loss-scalar_oracle_loss:>+12.6f}")
    print(f"  {'Per-hex fit-on-train (OOS)':<35s} "
          f"{oos_perhex_loss:>12.6f} {oos_perhex_loss-scalar_oracle_loss:>+12.6f}")
    print(f"  {'Per-hex fit-on-val (in-sample)':<35s} "
          f"{insample_perhex_loss:>12.6f} {insample_perhex_loss-scalar_oracle_loss:>+12.6f}")
    print(f"  {'Uniform (scalar r)':<35s} "
          f"{scalar_uniform_loss:>12.6f} {scalar_uniform_loss-scalar_oracle_loss:>+12.6f}")

    print(f"\n  Scalar gap (uniform - oracle):     {scalar_gap:.9f}")
    print(f"  OOS per-hex gap (uniform - oracle): {gap_oos:.9f}")

    print(f"\n  %-bias captured (with SCALAR anchor as floor):")
    print(f"  {'Model':<25s} {'NLL':>12s} {'% captured':>12s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'Trained KEN':<25s} {REF_TRAINED_KEN:>12.6f} {pct_bias_scalar(REF_TRAINED_KEN):>11.2f}%")
    print(f"  {'Trained Hybrid':<25s} {REF_TRAINED_HYBRID:>12.6f} {pct_bias_scalar(REF_TRAINED_HYBRID):>11.2f}%")

    print(f"\n  %-bias captured (with OOS PER-HEX anchor as floor):")
    print(f"  {'Model':<25s} {'NLL':>12s} {'% captured':>12s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'Trained KEN':<25s} {REF_TRAINED_KEN:>12.6f} {pct_bias_perhex(REF_TRAINED_KEN):>11.2f}%")
    print(f"  {'Trained Hybrid':<25s} {REF_TRAINED_HYBRID:>12.6f} {pct_bias_perhex(REF_TRAINED_HYBRID):>11.2f}%")

    # ── Sanity gates (§5) ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SANITY GATES (§5 against OOS per-hexamer anchor)")
    print(f"{'='*70}")

    all_pass = True

    # Gate 1: Untrained models must be ABOVE the per-hex anchor
    print("\n  [1] Untrained control (must score ABOVE per-hex oracle):")
    for name, score in [("untrained_ken", REF_UNTRAINED_KEN),
                        ("untrained_hybrid", REF_UNTRAINED_HYBRID)]:
        above = score > perhex_anchor
        status = "PASS" if above else "**FAIL**"
        if not above:
            all_pass = False
        print(f"    {status}: {name} = {score:.6f} > {perhex_anchor:.6f}? {above}")

    # Gate 2: Trained models between per-hex oracle and uniform
    print("\n  [2] Trained between per-hex oracle and uniform:")
    for name, score in [("trained_ken", REF_TRAINED_KEN),
                        ("trained_hybrid", REF_TRAINED_HYBRID)]:
        above = score > perhex_anchor
        below = score < scalar_uniform_loss
        ok = above and below
        status = "PASS" if ok else "**FAIL**"
        if not ok:
            all_pass = False
        print(f"    {status}: {name} = {score:.6f}  "
              f"({perhex_anchor:.6f} < model < {scalar_uniform_loss:.6f}? "
              f"{above} and {below})")

    print(f"\n  All gates: {'PASS' if all_pass else '**FAIL**'}")

    if not all_pass:
        print("\n  *** SANITY GATE FAILURE — stopping, do not interpret as valid ***")
        # Still write JSON for diagnostics, but flag failure

    # ── Verdict ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("VERDICT on open question 2")
    print(f"{'='*70}")

    delta_oos = oos_perhex_loss - scalar_oracle_loss
    delta_true = true_r_oracle_loss - scalar_oracle_loss
    delta_overfit = insample_perhex_loss - oos_perhex_loss

    if abs(delta_oos) < 0.001:
        verdict = "NOT_MATERIAL"
        verdict_text = (
            f"Per-hexamer r is NOT materially tighter than scalar. "
            f"The OOS delta is {delta_oos:+.6f}, less than 0.001 nats "
            f"in a gap of {scalar_gap:.6f}. The scalar anchor is adequate."
        )
    elif delta_oos < -0.001 and delta_overfit > -0.001:
        verdict = "TIGHTER_AND_REAL"
        verdict_text = (
            f"Per-hexamer r IS materially tighter (delta = {delta_oos:+.6f} nats). "
            f"The improvement holds out-of-sample (overfit gap = {delta_overfit:+.6f}). "
            f"The current %-captured numbers are overstated."
        )
    elif delta_oos < -0.001 and delta_overfit < -0.001:
        # In-sample << OOS, but OOS still tighter than scalar
        verdict = "TIGHTER_BUT_OVERFIT_RISK"
        verdict_text = (
            f"Per-hexamer r is tighter (OOS delta = {delta_oos:+.6f}), but the "
            f"in-sample fit is substantially lower (overfit gap = {delta_overfit:+.6f}). "
            f"The OOS result is credible; the in-sample result is not."
        )
    elif delta_oos > 0.001:
        verdict = "SCALAR_IS_TIGHTER"
        verdict_text = (
            f"Per-hexamer r is WORSE than scalar (delta = {delta_oos:+.6f}). "
            f"This likely means the per-hexamer fit is overfitting rare hexamers "
            f"even with the train/val guard. The scalar anchor remains the floor."
        )
    else:
        verdict = "INCONCLUSIVE"
        verdict_text = f"OOS delta = {delta_oos:+.6f}, needs further analysis."

    print(f"\n  {verdict}: {verdict_text}")

    # ── Write JSON ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("WRITING JSON")
    print(f"{'='*70}")

    payload = {
        "_what": (
            "Per-hexamer NB oracle anchor: closes open question 2 from nb_oracle.md. "
            "Compares a per-hexamer fitted r (4096 params) to the scalar anchor "
            "(1 param). Reports in-sample and out-of-sample results to detect "
            "overfitting."
        ),
        "verdict": verdict,
        "verdict_text": verdict_text,
        "scalar_anchor": {
            "oracle_nll": scalar_oracle_loss,
            "uniform_nll": scalar_uniform_loss,
            "gap": scalar_gap,
            "fitted_r": scalar_r,
            "_note": "Fitted on val (1 param — negligible in-sample bias)",
        },
        "true_hex_r_anchor": {
            "oracle_nll": true_r_oracle_loss,
            "delta_vs_scalar": true_r_oracle_loss - scalar_oracle_loss,
            "n_finite": n_true_finite,
            "n_nan_fallback_to_scalar": n_true_nan,
            "median_r": float(np.nanmedian(true_hex_r)),
            "_note": "Scored with ground-truth hexamer_r, no fitting",
        },
        "perhex_oos": {
            "oracle_nll": oos_perhex_loss,
            "delta_vs_scalar": oos_perhex_loss - scalar_oracle_loss,
            "gap_uniform_minus_oracle": gap_oos,
            "fitted_on": "train",
            "scored_on": "val",
            "n_fitted_hexamers": grid_info_train["n_fitted"],
            "n_fallback_hexamers": grid_info_train["n_fallback"],
            "min_positions_threshold": MIN_HEX_POSITIONS,
            "grid": grid_info_train,
        },
        "perhex_insample": {
            "oracle_nll": insample_perhex_loss,
            "delta_vs_scalar": insample_perhex_loss - scalar_oracle_loss,
            "fitted_on": "val",
            "scored_on": "val",
            "n_fitted_hexamers": grid_info_val["n_fitted"],
            "n_fallback_hexamers": grid_info_val["n_fallback"],
            "overfitting_gap": overfit_gap,
            "_note": (
                "In-sample (fit-on-val, score-on-val). A value substantially "
                "below the OOS result would indicate overfitting."
            ),
        },
        "pct_bias_captured": {
            "_canonical": "with_scalar_anchor",
            "with_scalar_anchor": {
                "trained_ken": round(pct_bias_scalar(REF_TRAINED_KEN), 2),
                "trained_hybrid": round(pct_bias_scalar(REF_TRAINED_HYBRID), 2),
                "gap": round(REF_GAP, 9),
                "anchor_nll": REF_SCALAR_ORACLE,
                "_note": (
                    "Computed against the PUBLISHED oracle_nb_v2.json anchor "
                    "(4.026351354, r=1096), NOT this script's own scipy-only "
                    "re-fit (4.026409, r=21). Both sit on the flat plateau and "
                    "differ by 5.7e-5, far below every threshold used here; "
                    "using the published one keeps all artifacts on one floor."
                ),
            },
            "with_oos_perhex_anchor": {
                "trained_ken": round(pct_bias_perhex(REF_TRAINED_KEN), 2),
                "trained_hybrid": round(pct_bias_perhex(REF_TRAINED_HYBRID), 2),
                "gap": round(gap_oos, 9),
                "anchor_nll": oos_perhex_loss,
                "_not_an_anchor": (
                    "DO NOT QUOTE. Per-hexamer scored WORSE than scalar "
                    "out-of-sample (+4.83e-4), so it is not a lower bound. Its "
                    "gap is smaller, which inflates %-bias for every model "
                    "(KEN reads 61.68 here vs 61.31 canonical). Retained only "
                    "to quantify that inflation."
                ),
            },
        },
        "sanity_gates": {
            "all_pass": all_pass,
            "untrained_above_oracle": {
                "ken": bool(REF_UNTRAINED_KEN > perhex_anchor),
                "hybrid": bool(REF_UNTRAINED_HYBRID > perhex_anchor),
            },
            "trained_between_oracle_and_uniform": {
                "ken": bool(perhex_anchor < REF_TRAINED_KEN < scalar_uniform_loss),
                "hybrid": bool(perhex_anchor < REF_TRAINED_HYBRID < scalar_uniform_loss),
            },
        },
        "loss_config": {
            "loss": "MaskedNegativeBinomialOffsetNLLLoss (frozen core)",
            "max_dispersion_ratio": 2.0,
            "clamp_margin": 1.0,
            "dispersion_window_size": 1,
            "_note": "Matched to scalar anchor and training runs",
        },
        "rare_hexamer_handling": {
            "method": "minimum_count_threshold",
            "threshold": MIN_HEX_POSITIONS,
            "fallback": f"scalar r = {scalar_r:.4f}",
            "justification": (
                f"Hexamers with < {MIN_HEX_POSITIONS} positions in the fitting "
                f"set have too few data points for a reliable 1-param fit. "
                f"Falling back to the scalar r avoids extrapolation. "
                f"With {MIN_HEX_POSITIONS} positions × 12 tracks = "
                f"{MIN_HEX_POSITIONS * 12} data points, the MLE is well "
                f"identified for typical hexamer-r values."
            ),
            "hex_position_counts_train": {
                "min": int(hex_count_train.min()),
                "median": int(np.median(hex_count_train)),
                "max": int(hex_count_train.max()),
                "below_threshold": int((hex_count_train < MIN_HEX_POSITIONS).sum()),
            },
        },
        "fitted_r_summary": {
            "train_fit": {
                "median": float(np.median(fitted_r_train)),
                "mean": float(np.mean(fitted_r_train)),
                "p5": float(np.percentile(fitted_r_train, 5)),
                "p95": float(np.percentile(fitted_r_train, 95)),
            },
            "val_fit": {
                "median": float(np.median(fitted_r_val)),
                "mean": float(np.mean(fitted_r_val)),
                "p5": float(np.percentile(fitted_r_val, 5)),
                "p95": float(np.percentile(fitted_r_val, 95)),
            },
            "true_r": {
                "median": float(np.nanmedian(true_hex_r)),
                "mean": float(np.nanmean(true_hex_r)),
                "p5": float(np.nanpercentile(true_hex_r, 5)),
                "p95": float(np.nanpercentile(true_hex_r, 95)),
            },
        },
        "reference_model_nlls": {
            "trained_ken": REF_TRAINED_KEN,
            "trained_hybrid": REF_TRAINED_HYBRID,
            "untrained_ken": REF_UNTRAINED_KEN,
            "untrained_hybrid": REF_UNTRAINED_HYBRID,
        },
        "store": STORE,
        "sim_dir": SIM_DIR,
        "fasta": FASTA,
        "n_val_pairs": n_val,
        "n_train_pairs": n_train,
        "tile_size": int(tile_size),
        "l_target": int(l_target),
        "crop": [int(crop_start), int(crop_stop)],
        "design_doc": "docs/pending/nb_oracle.md",
        "supersedes": None,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.executable,
        "runtime_s": round(time.time() - t0, 1),
    }

    os.makedirs(os.path.dirname(os.path.abspath(OUT_JSON)), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Wrote {OUT_JSON}")
    print(f"\n  Total runtime: {time.time()-t0:.1f}s")

    if not all_pass:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
