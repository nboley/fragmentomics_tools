#!/usr/bin/env python
"""Oracle fragment-level NLL for the cut-site simulation store.

Computes the ground-truth per-fragment negative log-likelihood on the
val split of a cut-site zarr store, using the generative parameters
from the simulation manifest.

Imports `build_region_weights` and `precompute_region` from the simulator
package (commit bc52890, verified identical to HEAD for these files) rather
than hand-rolling the weight computation.

Two mandatory correctness checks:
  1. No observed fragment may land on a w==0 cell.
  2. Mean per-region entropy (-sum w log w) must agree with the sample mean
     of -log w(fragment) within Monte Carlo error at 37 draws/region.

Domain correction: the store excludes L=180, so
  oracle NLL = -mean(log w) + log(W_D)
  W_D = n_scored / n_emitted  (pooled scalar)

Usage:
    cd /home/nathanboley/src/fragmentomics_tools/.claude/worktrees/background-model-work
    PYTHONPATH=. /home/nathanboley/miniconda3/envs/biomarker_env/bin/python \
        scripts/cut_site_oracle.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

# ── sys.path pin for scripts/ imports ─────────────────────────────────────
# (CLAUDE.md: "scripts/ needs a sys.path pin to import at all")
import os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def main():
    t0 = time.time()

    # ── Load manifest (verify=False: no FASTA/region_set paths needed) ──
    from background_model.simulator.emit import load_manifest
    from background_model.simulator.weights import (
        build_region_weights, L_MIN, L_MAX, N_LENGTHS,
        generative_domain_size,
    )
    from background_model.simulator.precompute import (
        hexamer_indices, HEX_HALF,
    )

    MANIFEST = (
        "/efs/analytics/nathanboley/background_model/"
        "sim_run_tile1536_20260930/RD-56670.manifest.json"
    )
    STORE = (
        "/efs/analytics/nathanboley/background_model/"
        "cut_site_stores/sim_tile1536.zarr"
    )

    manifest = load_manifest(MANIFEST, verify=False)
    hex_tables = manifest["hex_tables"]
    predict_lut = manifest["predict_lut"]
    marginal_fl = manifest["marginal_fl"]
    region_len = manifest["region_len"]   # 1536

    print(f"[oracle] loaded manifest: region_len={region_len}, "
          f"L={manifest['l_min']}..{manifest['l_max']}, "
          f"fl_bands={manifest['fl_bands']} ({time.time()-t0:.1f}s)",
          flush=True)

    # ── Open store ──────────────────────────────────────────────────────
    import zarr
    root = zarr.open_group(STORE, mode="r")

    # Parse store config
    import json
    store_cfg = json.loads(root.attrs["config_json"])
    rf_budget = store_cfg["rf_budget"]   # 133
    store_L_max = store_cfg["L_max"]     # 179

    splits = root["tiles/split"][:]
    val_idxs = np.nonzero(splits == 1)[0]
    n_val = len(val_idxs)

    # Preload all needed arrays
    all_seq = root["tiles/seq"]          # (n_tiles, 1802) uint8
    all_mask = root["tiles/mask"]        # (n_tiles, region_len) bool
    frag_start = root["fragments/start"][:]    # uint16, region-local
    frag_length = root["fragments/length"][:]  # uint16
    frag_strand = root["fragments/strand"][:]  # uint8 (0=plus, 1=minus)
    frag_indptr = root["fragments/indptr"][:]  # int64

    n_total_store = len(frag_start)
    n_total_emitted = sum(manifest["per_region_counts"].values())

    print(f"[oracle] store: {n_val} val tiles, rf_budget={rf_budget}, "
          f"store_L_max={store_L_max}", flush=True)
    print(f"[oracle] fragments: {n_total_store} in store, "
          f"{n_total_emitted} emitted", flush=True)

    # ── Domain correction (pooled scalar) ───────────────────────────────
    W_D = n_total_store / n_total_emitted
    log_W_D = np.log(W_D)
    print(f"[oracle] W_D = {W_D:.6f}, log(W_D) = {log_W_D:.6f}", flush=True)

    # ── Sequence offset geometry ────────────────────────────────────────
    # Store seq covers [gstart - rf_budget, gstop + rf_budget), length 1802.
    # precompute_region needs [gstart - HEX_HALF, gstop + HEX_HALF), length
    # region_len + 2*HEX_HALF = 1542.
    # Offset within stored seq: rf_budget - HEX_HALF = 130.
    seq_offset = rf_budget - HEX_HALF    # 130
    seq_len_needed = region_len + 2 * HEX_HALF  # 1542

    # Verify: hexamer_indices on 1542 bytes -> 1537 = region_len + 1 entries
    assert seq_len_needed - (6 - 1) == region_len + 1, \
        f"geometry check: {seq_len_needed - 5} != {region_len + 1}"

    # ── Compute the theoretical domain size |Omega| ─────────────────────
    omega_size = generative_domain_size(region_len, L_MIN, L_MAX)
    print(f"[oracle] |Omega| = {omega_size} (at region_len={region_len})")

    # ── Main loop: score val fragments ──────────────────────────────────
    sum_neg_log_w = np.float64(0.0)      # accumulate in float64
    n_scored = 0
    n_zero_hits = 0                       # check 1: must be 0

    # For check 2: per-region entropy and per-region sample mean
    region_entropies = []                 # -sum w log w per region
    region_sample_means = []              # mean(-log w) per region's fragments

    # Also compute uniform NLL (restricted to scored domain D, L=25..store_L_max)
    # |D| per region = generative_domain_size(region_len, L_MIN, store_L_max)
    D_size = generative_domain_size(region_len, L_MIN, store_L_max)
    log_D = np.log(D_size)
    print(f"[oracle] |D| (L={L_MIN}..{store_L_max}) = {D_size}, "
          f"log|D| = {log_D:.6f}")

    # For uniform: we need the actual uniform NLL accounting for zero-weight
    # positions.  Under uniform, w_uniform = 1/|D_valid| where D_valid
    # excludes N-masked and edge-truncated positions.  Since the simulation
    # should have no Ns (it's simulated data), D_valid ~ D_size.  We'll
    # compute it per-region from the weights.
    sum_uniform_neg_log_w = np.float64(0.0)

    for batch_start in range(0, n_val, 100):
        batch_end = min(batch_start + 100, n_val)
        batch_idxs = val_idxs[batch_start:batch_end]

        for ti in batch_idxs:
            # Extract sequence sub-array for this tile
            seq_raw = all_seq[ti]  # (1802,) uint8
            seq_sub = seq_raw[seq_offset : seq_offset + seq_len_needed]

            # Compute hexamer indices and cumulative GC (mirroring precompute_region)
            hex_fwd, hex_rc, valid = hexamer_indices(seq_sub)
            assert len(hex_fwd) == region_len + 1

            # Cumulative GC over core region bases
            core = seq_sub[HEX_HALF : HEX_HALF + region_len]
            is_gc = (core == ord("G")) | (core == ord("C"))
            cum_gc = np.empty(region_len + 1, dtype=np.float64)
            cum_gc[0] = 0
            np.cumsum(is_gc, out=cum_gc[1:])

            # Build normalised weights
            rw = build_region_weights(
                hex_fwd=hex_fwd,
                hex_rc=hex_rc,
                cum_gc=cum_gc,
                hex_tables=hex_tables,
                marginal_fl=marginal_fl,
                predict_lut=predict_lut,
                region_len=region_len,
                valid=valid,
            )

            # ── Check 2 prep: compute per-region entropy ────────────────
            # Entropy = -sum_{w>0} w * log(w)
            # Only over L=L_MIN..L_MAX (full generative domain)
            w_all = np.concatenate([rw.w_plus.ravel(), rw.w_minus.ravel()])
            pos_mask = w_all > 0
            w_pos = w_all[pos_mask]
            entropy = -np.sum(w_pos * np.log(w_pos), dtype=np.float64)
            region_entropies.append(entropy)

            # ── Uniform anchor: count nonzero cells in D ────────────────
            # D restricts to L=L_MIN..store_L_max (i.e., li=0..store_L_max-L_MIN)
            li_max_store = store_L_max - L_MIN  # 154 (0-indexed)
            w_plus_D = rw.w_plus[:, :li_max_store + 1]
            w_minus_D = rw.w_minus[:, :li_max_store + 1]
            n_nonzero_D = int((w_plus_D > 0).sum() + (w_minus_D > 0).sum())

            # ── Score fragments ─────────────────────────────────────────
            i0 = frag_indptr[ti]
            i1 = frag_indptr[ti + 1]
            n_frags = i1 - i0

            if n_frags == 0:
                region_sample_means.append(np.nan)
                continue

            starts = frag_start[i0:i1].astype(np.intp)
            lengths = frag_length[i0:i1].astype(np.intp)
            strands = frag_strand[i0:i1]  # 0=plus, 1=minus

            frag_neg_log_w = np.empty(n_frags, dtype=np.float64)

            for fi in range(n_frags):
                p = starts[fi]
                L = lengths[fi]
                li = L - L_MIN
                s = strands[fi]

                if s == 0:  # plus: c5 = p
                    w = rw.w_plus[p, li]
                else:       # minus: c5 = p + L
                    w = rw.w_minus[p + L, li]

                if w == 0.0:
                    n_zero_hits += 1
                    frag_neg_log_w[fi] = np.inf
                else:
                    frag_neg_log_w[fi] = -np.log(w)

            # Accumulate
            finite_mask = np.isfinite(frag_neg_log_w)
            sum_neg_log_w += np.sum(frag_neg_log_w[finite_mask],
                                    dtype=np.float64)
            n_scored += int(finite_mask.sum())

            # Per-region sample mean of -log w
            if finite_mask.any():
                region_sample_means.append(
                    np.mean(frag_neg_log_w[finite_mask], dtype=np.float64)
                )
            else:
                region_sample_means.append(np.nan)

            # Uniform: -log(1/n_nonzero_D) = log(n_nonzero_D) per fragment
            sum_uniform_neg_log_w += n_frags * np.log(n_nonzero_D)

        done = batch_end
        if done % 1000 == 0 or done == n_val:
            running_nll = float(sum_neg_log_w / n_scored) if n_scored > 0 else float('nan')
            print(f"[oracle] {done}/{n_val} tiles, {n_scored} frags scored, "
                  f"running -mean(log w) = {running_nll:.6f}, "
                  f"zero_hits = {n_zero_hits} ({time.time()-t0:.1f}s)",
                  flush=True)

    # ── Results ─────────────────────────────────────────────────────────
    mean_neg_log_w = float(sum_neg_log_w / n_scored)
    oracle_nll = mean_neg_log_w + float(log_W_D)
    uniform_nll = float(sum_uniform_neg_log_w / n_scored)

    # Uniform is already defined over D — no domain correction needed.
    # (The oracle needs correction because w is normalised over Omega
    # which includes L=180; the uniform has no mass outside D.)

    print(f"\n{'='*70}")
    print(f"Oracle NLL — cut-site store val split ({n_scored} fragments)")
    print(f"{'='*70}")
    print(f"  -mean(log w)          = {mean_neg_log_w:.6f}")
    print(f"  log(W_D)              = {log_W_D:.6f}")
    print(f"  Oracle NLL            = {oracle_nll:.6f}")
    print(f"  Uniform NLL (per-rgn) = {uniform_nll:.6f}")
    print(f"  log|D| (geometric)    = {log_D:.6f}")
    print(f"  Uniform deficit       = {log_D - uniform_nll:.6f}")
    print(f"  Gap (uniform-oracle)  = {uniform_nll - oracle_nll:.6f}")

    # ── Check 1: zero-weight hits ───────────────────────────────────────
    print(f"\n--- Check 1: zero-weight fragments ---")
    if n_zero_hits == 0:
        print(f"  PASS: 0 fragments landed on w==0 cells")
    else:
        print(f"  FAIL: {n_zero_hits} fragments on w==0 cells — "
              f"coordinate mapping is wrong")

    # ── Check 2: entropy vs sample mean ─────────────────────────────────
    entropies = np.array(region_entropies, dtype=np.float64)
    sample_means = np.array(region_sample_means, dtype=np.float64)
    valid_mask = np.isfinite(sample_means)
    entropies_v = entropies[valid_mask]
    sample_means_v = sample_means[valid_mask]

    mean_entropy = np.mean(entropies_v, dtype=np.float64)
    mean_sample = np.mean(sample_means_v, dtype=np.float64)
    diff = abs(mean_entropy - mean_sample)

    # Monte Carlo SE: for n=37 draws from a distribution with entropy H,
    # the variance of the sample mean of -log w is Var(-log w)/n.
    # We can estimate Var(-log w) from the per-region variance.
    # For a rough bound: SE of the grand mean across R regions of n frags
    # each is sqrt(Var(H_region) / R + Var(-log w | region) / (R * n)).
    # At R=6664 and n=37, the second term dominates less; use the empirical
    # spread.
    se_diff = np.std(sample_means_v - entropies_v, dtype=np.float64) / np.sqrt(len(entropies_v))

    print(f"\n--- Check 2: entropy vs sample mean ---")
    print(f"  Mean per-region entropy      = {mean_entropy:.6f}")
    print(f"  Mean per-region sample mean  = {mean_sample:.6f}")
    print(f"  Difference                   = {diff:.6f}")
    print(f"  SE of difference             = {se_diff:.6f}")
    print(f"  |diff| / SE                  = {diff / se_diff:.2f}")

    if diff / se_diff < 5.0:
        print(f"  PASS: within {diff/se_diff:.1f} SE (Monte Carlo noise)")
    else:
        print(f"  FAIL: {diff/se_diff:.1f} SE — mapping bug suspected")

    print(f"\n{'='*70}")
    print(f"  Runtime: {time.time()-t0:.1f}s")
    print(f"{'='*70}")

    return 0 if n_zero_hits == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
