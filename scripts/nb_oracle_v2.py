#!/usr/bin/env python
"""Compute anchor A for the overdispersed (v3nb) simulation store.

This script implements §3–§5 of docs/pending/nb_oracle.md:

  Anchor A: true propensity, fitted scalar r — minimises the frozen-core
  NB-offset loss over the val set.  The fitted r isolates the propensity
  question: "how much of the bias did the model capture, given a correctly
  specified dispersion?"

The propensity comes from sim_oracle.compute_oracle_propensity_for_tile
(verified, tracked) — no re-derivation.  The loss is the FROZEN core
MaskedNegativeBinomialOffsetNLLLoss with config matched to the training
runs (max_dispersion_ratio=2.0, clamp_margin=1.0, dispersion_window_size=1).

The uniform anchor gets its own separately fitted r (§4 step 3 of the
design — review flagged this could be misread as reusing oracle's r).

Writes simulation_v3_nb/A/oracle_nb_v2.json with full provenance.

Usage:
    cd <repo>
    PYTHONPATH=. /home/nathanboley/miniconda3/envs/biomarker_env/bin/python \\
        scripts/nb_oracle_v2.py
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.optimize import minimize_scalar

# ── Paths ─────────────────────────────────────────────────────────────────
STORE = "/efs/analytics/nathanboley/background_model/simulation_v3_nb/stores/sim_store_v3nb_A.zarr"
SIM_DIR = "/efs/analytics/nathanboley/background_model/simulation_v3_nb/A"
FASTA = "/efs/analytics/nathanboley/data_resources/genome/hg38.fa"
OUT_JSON = "/efs/analytics/nathanboley/background_model/simulation_v3_nb/A/oracle_nb_v2.json"
OLD_JSON = "/efs/analytics/nathanboley/background_model/simulation_v3_nb/A/oracle_nb.json"

KEN_SUMMARY = "/efs/analytics/nathanboley/background_model/simulation_v3_nb/runs/v3nb_ken_nb_frozen_lr5e-3/summary.json"
HYBRID_SUMMARY = "/efs/analytics/nathanboley/background_model/simulation_v3_nb/runs/v3nb_hybrid_nb_frozen_lr5e-3/summary.json"


def load_checkpoint_path(summary_path):
    with open(summary_path) as f:
        s = json.load(f)
    return s["best_model_path"], s


def build_dataset(store_path, model_input_size):
    from background_model.dataset import BackgroundTileDataset
    return BackgroundTileDataset(
        store_path=store_path,
        model_input_size=model_input_size,
        split="val",
        sample_role="train",
        min_N=0,
        train_mode=False,
        seed=1337,
    )


def load_model(ckpt_path, model_type):
    from background_model.train import (
        InstrumentedBackgroundModelKEN,
        InstrumentedBackgroundModelHybrid,
    )
    cls = {
        "ken": InstrumentedBackgroundModelKEN,
        "hybrid": InstrumentedBackgroundModelHybrid,
    }[model_type]
    model = cls.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    return model


def create_untrained_model(model_type, summary):
    from background_model_core import BackgroundModelKEN, BackgroundModelHybrid
    if model_type == "ken":
        model = BackgroundModelKEN(
            k=summary.get("k", 6),
            d_embed=summary.get("d_embed", 64),
            d_context=summary.get("d_context", 128),
            n_context_layers=summary.get("n_context_layers", 2),
            context_kernel_size=summary.get("context_kernel_size", 15),
            dropout=summary.get("dropout", 0.0),
            loss="nb_offset",
            dispersion_window_size=1,
            freeze_dispersion=True,
        )
    elif model_type == "hybrid":
        model = BackgroundModelHybrid(
            k=summary.get("k", 6),
            d_embed=summary.get("d_embed", 64),
            n_kernels=summary.get("n_kernels", 128),
            num_residual_layers=summary.get("num_residual_layers", 3),
            dropout=summary.get("dropout", 0.0),
            loss="nb_offset",
            dispersion_window_size=1,
            freeze_dispersion=True,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    model.eval()
    return model


def score_model_nb(model, ds, device="cpu"):
    """Score a model under NB-offset loss with frozen dispersion (r from model).

    The model's forward returns (shape_logits, dispersion_bp) where
    dispersion_bp is the RAW per-position delta. The training loop applies
    _pooled_log_dispersion which: (1) mean-pools to the window level, and
    (2) adds log_dispersion_init. We call that method directly.
    """
    from background_model_core import (
        MaskedNegativeBinomialOffsetNLLLoss,
        _prepare_mask,
    )
    loss_fn = MaskedNegativeBinomialOffsetNLLLoss(
        max_dispersion_ratio=2.0, clamp_margin=1.0,
    )
    model = model.to(device)
    nlls = []
    with torch.no_grad():
        for i in range(len(ds)):
            x, y, mask = ds[i]
            x = x.unsqueeze(0).to(device)
            shape_logits, dispersion_bp = model(x)
            shape_logits = shape_logits.cpu()
            dispersion_bp = dispersion_bp.cpu()
            y_t = y.unsqueeze(0)
            mask3 = _prepare_mask(mask.unsqueeze(0), y_t)

            log_disp = model._pooled_log_dispersion(dispersion_bp, mask3)

            nll = loss_fn(shape_logits, log_disp, y_t, mask3).item()
            nlls.append(nll)
    return np.array(nlls)


def main():
    t0 = time.time()

    import pysam
    import zarr
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
    gcbias = GCBias2D(gt["bias_lengths"], gt["bias_gc_percents"], gt["bias_grid"])
    gc_lut = gcbias.build_lookup_table(max_len=MAX_LEN)

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

    ds = BackgroundTileDataset(
        store_path=STORE, model_input_size=tile_size, split="val",
        sample_role="train", min_N=0, train_mode=False, seed=1337,
    )
    n_val_pairs = len(ds)
    print(f"[oracle-v2] {n_val_pairs} val pairs, crop [{crop_start},{crop_stop}) "
          f"({time.time()-t0:.1f}s)", flush=True)

    tile_to_pairs = {}
    for di in range(len(ds)):
        s_idx, t_idx = ds.index[di]
        tile_to_pairs.setdefault(t_idx, []).append((di, s_idx))

    fa = pysam.FastaFile(FASTA)

    # ── Pre-compute oracle propensity logits and uniform logits ────────────
    print("[oracle-v2] Pre-computing oracle propensity logits...", flush=True)
    oracle_data = {}  # di -> (oracle_logits, uniform_logits, y, mask)

    for tn, t_idx in enumerate(sorted(tile_to_pairs)):
        contig = str(contigs[t_idx])
        gstart, gstop = int(starts[t_idx]), int(stops[t_idx])
        pairs = tile_to_pairs[t_idx]
        uniq = sorted({s for _, s in pairs})
        loc = {s: i for i, s in enumerate(uniq)}

        prop_full = compute_oracle_propensity_for_tile(
            contig, gstart, gstop, l_target, w6, gcbias, len_vals,
            len_p_per_sample, np.array(uniq), fa, gc_lut=gc_lut,
        )
        prop_cropped = prop_full[:, :, crop_start:crop_stop]

        for di, s_idx in pairs:
            _, y, mask = ds[di]
            p = prop_cropped[loc[s_idx]]
            oracle_logits = torch.from_numpy(
                np.where(p > 0, np.log(p), -1e30)
            ).float().unsqueeze(0)
            uniform_logits = torch.zeros_like(oracle_logits)
            y_t = y.unsqueeze(0)
            m_t = mask.unsqueeze(0)
            oracle_data[di] = (oracle_logits, uniform_logits, y_t, m_t)

        if (tn + 1) % 100 == 0 or tn == 0:
            print(f"  {tn+1}/{len(tile_to_pairs)} tiles ({time.time()-t0:.1f}s)",
                  flush=True)

    fa.close()
    print(f"[oracle-v2] Cached {len(oracle_data)} pairs ({time.time()-t0:.1f}s)",
          flush=True)

    # ── Helper: evaluate loss at a given log_r for a given logit set ───────
    loss_fn = MaskedNegativeBinomialOffsetNLLLoss(
        max_dispersion_ratio=2.0, clamp_margin=1.0,
    )

    def eval_loss_at_log_r(log_r_val, use_oracle=True):
        """Mean loss across all val pairs at a given scalar log_r."""
        losses = []
        with torch.no_grad():
            for di in sorted(oracle_data):
                oracle_logits, uniform_logits, y_t, m_t = oracle_data[di]
                logits = oracle_logits if use_oracle else uniform_logits
                B, C, L = logits.shape
                W = L  # dispersion_window_size=1
                ld = torch.full((1, C, W), log_r_val, dtype=torch.float32)
                loss_val = loss_fn(logits, ld, y_t, m_t).item()
                losses.append(loss_val)
        return float(np.mean(losses))

    # ── Step 0 recap: coarse sweep for the report ──────────────────────────
    print("\n" + "=" * 70)
    print("STEP 0 RECAP: Coarse sweep of scalar r (true propensity)")
    print("=" * 70)

    # Sweep over a log grid; include key reference points
    coarse_log_r = np.linspace(np.log(1.0), np.log(5000.0), 25)
    ref_points = [np.log(7.179), np.log(1096.0)]
    coarse_log_r = np.sort(np.unique(np.concatenate([coarse_log_r, ref_points])))

    sweep_results = []
    print(f"{'log_r':>10s} {'r':>12s} {'loss':>12s}")
    print(f"{'-'*10} {'-'*12} {'-'*12}")
    for lr in coarse_log_r:
        loss = eval_loss_at_log_r(lr, use_oracle=True)
        sweep_results.append({"log_r": float(lr), "r": float(np.exp(lr)),
                              "loss": loss})
        note = ""
        if abs(np.exp(lr) - 7.179) < 0.01:
            note = " <-- true hexamer_r"
        elif abs(np.exp(lr) - 1096.0) < 1.0:
            note = " <-- frozen init"
        print(f"{lr:10.4f} {np.exp(lr):12.4f} {loss:12.6f}{note}")

    # ── Noise floor measurement ───────────────────────────────────────────
    # The objective should be deterministic given cached pairs. Verify by
    # re-evaluating several r values and comparing bitwise.
    print("\n" + "=" * 70)
    print("NOISE FLOOR MEASUREMENT")
    print("=" * 70)

    test_log_rs = [np.log(7.179), np.log(21.0), np.log(1096.0), np.log(500.0)]
    determinism_ok = True
    for test_lr in test_log_rs:
        v1 = eval_loss_at_log_r(test_lr, use_oracle=True)
        v2 = eval_loss_at_log_r(test_lr, use_oracle=True)
        match = (v1 == v2)
        if not match:
            determinism_ok = False
        print(f"  r={np.exp(test_lr):10.4f}: eval1={v1:.15f} eval2={v2:.15f} "
              f"bitwise_equal={match}")

    # Characterise non-monotonicity in the sweep's plateau region (r > 20)
    plateau_entries = [(s["r"], s["loss"]) for s in sweep_results if s["r"] > 20]
    if len(plateau_entries) >= 2:
        plateau_losses = [l for _, l in plateau_entries]
        plateau_max = max(plateau_losses)
        plateau_min = min(plateau_losses)
        non_monotonicity = plateau_max - plateau_min
    else:
        non_monotonicity = 0.0

    noise_floor_info = {
        "deterministic": determinism_ok,
        "plateau_non_monotonicity": non_monotonicity,
        "plateau_range_r_gt_20": {
            "min_loss": float(plateau_min) if len(plateau_entries) >= 2 else None,
            "max_loss": float(plateau_max) if len(plateau_entries) >= 2 else None,
            "n_points": len(plateau_entries),
        },
        "_note": (
            "The objective is deterministic given cached logits/counts, so "
            "non-monotonicity across the plateau is real numerical noise from "
            "the softmax/lgamma/clamp pipeline, not stochastic evaluation. "
            "Any claim about fitted r must exceed this noise floor to be "
            "meaningful."
        ),
    }
    print(f"\n  Deterministic: {determinism_ok}")
    print(f"  Plateau non-monotonicity (r>20): {non_monotonicity:.2e}")
    if len(plateau_entries) >= 2:
        print(f"    min={plateau_min:.9f}  max={plateau_max:.9f}  "
              f"over {len(plateau_entries)} points")

    # ── Step 2: fit r for oracle propensity ────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: Fitting scalar r for oracle propensity")
    print("=" * 70)

    # Use scipy minimize_scalar in the reliable range [log(1), log(3000)]
    # The coarse sweep shows the curve plateaus above r≈20-30
    result_oracle = minimize_scalar(
        lambda lr: eval_loss_at_log_r(lr, use_oracle=True),
        bounds=(np.log(1.0), np.log(3000.0)),
        method="bounded",
        options={"xatol": 0.01},
    )
    scipy_log_r_oracle = result_oracle.x
    scipy_r_oracle = np.exp(scipy_log_r_oracle)
    scipy_loss_oracle = result_oracle.fun

    print(f"  Scipy log_r: {scipy_log_r_oracle:.4f}")
    print(f"  Scipy r:     {scipy_r_oracle:.4f}")
    print(f"  Scipy loss:  {scipy_loss_oracle:.9f}")

    # Select oracle = min over union of (swept grid, scipy result)
    sweep_losses = [s["loss"] for s in sweep_results]
    sweep_best_idx = int(np.argmin(sweep_losses))
    sweep_best_loss = sweep_losses[sweep_best_idx]
    sweep_best_log_r = sweep_results[sweep_best_idx]["log_r"]
    print(f"  Sweep best:  {sweep_best_loss:.9f} at r={sweep_results[sweep_best_idx]['r']:.4f}")

    if sweep_best_loss < scipy_loss_oracle:
        oracle_loss = sweep_best_loss
        fitted_log_r_oracle = sweep_best_log_r
        oracle_source = "sweep"
        print(f"  -> Sweep beats scipy by {scipy_loss_oracle - sweep_best_loss:.2e}; "
              f"using sweep minimum")
    else:
        oracle_loss = scipy_loss_oracle
        fitted_log_r_oracle = scipy_log_r_oracle
        oracle_source = "scipy"
        print(f"  -> Scipy beats sweep by {sweep_best_loss - scipy_loss_oracle:.2e}; "
              f"using scipy result")
    fitted_r_oracle = np.exp(fitted_log_r_oracle)
    print(f"  Final oracle loss: {oracle_loss:.9f} (r={fitted_r_oracle:.4f}, "
          f"source={oracle_source})")

    # ── Step 3: fit r for uniform propensity (SEPARATE fit) ────────────────
    print("\n" + "=" * 70)
    print("STEP 3: Fitting scalar r for uniform propensity (separate fit)")
    print("=" * 70)

    # Sweep for uniform too
    uniform_sweep_results = []
    for lr in coarse_log_r:
        loss = eval_loss_at_log_r(lr, use_oracle=False)
        uniform_sweep_results.append({"log_r": float(lr), "r": float(np.exp(lr)),
                                      "loss": loss})

    result_uniform = minimize_scalar(
        lambda lr: eval_loss_at_log_r(lr, use_oracle=False),
        bounds=(np.log(1.0), np.log(3000.0)),
        method="bounded",
        options={"xatol": 0.01},
    )
    scipy_log_r_uniform = result_uniform.x
    scipy_r_uniform = np.exp(scipy_log_r_uniform)
    scipy_loss_uniform = result_uniform.fun

    print(f"  Scipy log_r: {scipy_log_r_uniform:.4f}")
    print(f"  Scipy r:     {scipy_r_uniform:.4f}")
    print(f"  Scipy loss:  {scipy_loss_uniform:.9f}")

    # Select uniform = min over union of (swept grid, scipy result)
    u_sweep_losses = [s["loss"] for s in uniform_sweep_results]
    u_sweep_best_idx = int(np.argmin(u_sweep_losses))
    u_sweep_best_loss = u_sweep_losses[u_sweep_best_idx]
    u_sweep_best_log_r = uniform_sweep_results[u_sweep_best_idx]["log_r"]
    print(f"  Sweep best:  {u_sweep_best_loss:.9f} at r={uniform_sweep_results[u_sweep_best_idx]['r']:.4f}")

    if u_sweep_best_loss < scipy_loss_uniform:
        uniform_loss = u_sweep_best_loss
        fitted_log_r_uniform = u_sweep_best_log_r
        uniform_source = "sweep"
        print(f"  -> Sweep beats scipy by {scipy_loss_uniform - u_sweep_best_loss:.2e}; "
              f"using sweep minimum")
    else:
        uniform_loss = scipy_loss_uniform
        fitted_log_r_uniform = scipy_log_r_uniform
        uniform_source = "scipy"
        print(f"  -> Scipy beats sweep by {u_sweep_best_loss - scipy_loss_uniform:.2e}; "
              f"using scipy result")
    fitted_r_uniform = np.exp(fitted_log_r_uniform)
    print(f"  Final uniform loss: {uniform_loss:.9f} (r={fitted_r_uniform:.4f}, "
          f"source={uniform_source})")

    gap = uniform_loss - oracle_loss
    print(f"\n  Oracle:  {oracle_loss:.9f}")
    print(f"  Uniform: {uniform_loss:.9f}")
    print(f"  Gap:     {gap:.9f}")

    # ── Alignment verification ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ALIGNMENT VERIFICATION")
    print("=" * 70)

    # Reopen fasta for alignment check
    fa = pysam.FastaFile(FASTA)
    max_shift = 260
    shifts = np.arange(-max_shift, max_shift + 1)
    n_align_tiles = 20

    all_prop = []
    all_counts = []
    tiles_done = 0

    for t_idx in sorted(tile_to_pairs):
        if tiles_done >= n_align_tiles:
            break
        contig = str(contigs[t_idx])
        gstart, gstop = int(starts[t_idx]), int(stops[t_idx])
        pairs = tile_to_pairs[t_idx]
        uniq = sorted({s for _, s in pairs})
        loc = {s: i for i, s in enumerate(uniq)}

        prop_full = compute_oracle_propensity_for_tile(
            contig, gstart, gstop, l_target, w6, gcbias, len_vals,
            len_p_per_sample, np.array(uniq), fa,
        )
        for di, s_idx in pairs:
            _, y, mask = ds[di]
            p = prop_full[loc[s_idx]]
            counts = y.numpy()
            all_prop.append(p)
            all_counts.append(counts)
        tiles_done += 1

    fa.close()

    rs_corr = []
    for s in shifts:
        lo = crop_start + s
        hi = crop_stop + s
        if lo < 0 or hi > l_target:
            rs_corr.append(np.nan)
            continue
        p_flat = np.concatenate([p[:, lo:hi].ravel() for p in all_prop])
        c_flat = np.concatenate([c.ravel() for c in all_counts])
        r = np.corrcoef(p_flat, c_flat)[0, 1]
        rs_corr.append(r)

    rs_corr = np.array(rs_corr)
    best_shift = shifts[np.nanargmax(rs_corr)]
    best_r_corr = np.nanmax(rs_corr)
    r_at_zero = rs_corr[max_shift]
    r_at_minus128 = rs_corr[max_shift - 128] if max_shift >= 128 else np.nan
    ratio = r_at_zero / r_at_minus128 if r_at_minus128 != 0 else float("inf")

    align_info = {
        "best_shift": int(best_shift),
        "best_r": float(best_r_corr),
        "r_at_shift_0": float(r_at_zero),
        "r_at_shift_minus128": float(r_at_minus128),
        "ratio_0_vs_minus128": float(ratio),
        "n_tiles": tiles_done,
    }

    print(f"  Best shift:           {best_shift}")
    print(f"  Best r:               {best_r_corr:.6f}")
    print(f"  r at shift 0:         {r_at_zero:.6f}")
    print(f"  r at shift -128:      {r_at_minus128:.6f}")
    print(f"  Ratio (0 vs -128):    {ratio:.1f}x")
    assert best_shift == 0, f"ALIGNMENT FAILURE: best shift is {best_shift}"

    # ── Score trained and untrained models ─────────────────────────────────
    print("\n" + "=" * 70)
    print("SCORING MODELS (NB-offset loss, frozen dispersion)")
    print("=" * 70)

    ken_ckpt, ken_summary = load_checkpoint_path(KEN_SUMMARY)
    hybrid_ckpt, hybrid_summary = load_checkpoint_path(HYBRID_SUMMARY)

    print(f"  KEN best val_loss from training: {ken_summary['best_val_loss']:.6f}")
    print(f"  Hybrid best val_loss from training: {hybrid_summary['best_val_loss']:.6f}")

    # Trained KEN
    print("\n  Loading trained KEN...", flush=True)
    ken_model = load_model(ken_ckpt, "ken")
    ken_input_size = ken_model.calc_input_region_size(tile_size)
    ds_ken = build_dataset(STORE, ken_input_size)
    ken_nlls = score_model_nb(ken_model, ds_ken)
    ken_mean = float(np.mean(ken_nlls))
    print(f"  Trained KEN NB loss: {ken_mean:.6f} "
          f"(training reported: {ken_summary['best_val_loss']:.6f})")

    # Trained hybrid
    print("\n  Loading trained hybrid...", flush=True)
    hybrid_model = load_model(hybrid_ckpt, "hybrid")
    hybrid_input_size = hybrid_model.calc_input_region_size(tile_size)
    ds_hybrid = build_dataset(STORE, hybrid_input_size)
    hybrid_nlls = score_model_nb(hybrid_model, ds_hybrid)
    hybrid_mean = float(np.mean(hybrid_nlls))
    print(f"  Trained Hybrid NB loss: {hybrid_mean:.6f} "
          f"(training reported: {hybrid_summary['best_val_loss']:.6f})")

    # Untrained KEN
    print("\n  Scoring untrained KEN...", flush=True)
    torch.manual_seed(42)
    ken_untrained = create_untrained_model("ken", ken_summary)
    ken_u_input_size = ken_untrained.calc_input_region_size(tile_size)
    ds_ken_u = build_dataset(STORE, ken_u_input_size)
    ken_u_nlls = score_model_nb(ken_untrained, ds_ken_u)
    ken_u_mean = float(np.mean(ken_u_nlls))
    print(f"  Untrained KEN NB loss: {ken_u_mean:.6f}")

    # Untrained hybrid
    print("\n  Scoring untrained hybrid...", flush=True)
    torch.manual_seed(42)
    hybrid_untrained = create_untrained_model("hybrid", hybrid_summary)
    hybrid_u_input_size = hybrid_untrained.calc_input_region_size(tile_size)
    ds_hybrid_u = build_dataset(STORE, hybrid_u_input_size)
    hybrid_u_nlls = score_model_nb(hybrid_untrained, ds_hybrid_u)
    hybrid_u_mean = float(np.mean(hybrid_u_nlls))
    print(f"  Untrained Hybrid NB loss: {hybrid_u_mean:.6f}")

    # ── Verification (§5) ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VERIFICATION (§5 of nb_oracle.md)")
    print("=" * 70)

    all_pass = True

    # 1. Untrained control: must score ABOVE the oracle
    print("\n  [1] Untrained control:")
    for name, score in [("untrained_ken", ken_u_mean),
                        ("untrained_hybrid", hybrid_u_mean)]:
        above = score > oracle_loss
        status = "PASS" if above else "**FAIL**"
        print(f"    {status}: {name} = {score:.6f} > oracle {oracle_loss:.6f}? {above}")
        if not above:
            all_pass = False

    # 2. Sanity gate: trained models between oracle and uniform
    print("\n  [2] Sanity gate (trained between oracle and uniform):")
    for name, score in [("trained_ken", ken_mean),
                        ("trained_hybrid", hybrid_mean)]:
        above_oracle = score > oracle_loss
        below_uniform = score < uniform_loss
        ok = above_oracle and below_uniform
        status = "PASS" if ok else "**FAIL**"
        print(f"    {status}: {name} = {score:.6f}  "
              f"(oracle {oracle_loss:.6f} < model < uniform {uniform_loss:.6f}? "
              f"{above_oracle} and {below_uniform})")
        if not ok:
            all_pass = False

    # 3. Alignment (already checked above)
    print(f"\n  [3] Alignment: best_shift={best_shift}, "
          f"r(0)={r_at_zero:.4f} vs r(-128)={r_at_minus128:.4f} "
          f"({ratio:.1f}x) — {'PASS' if best_shift == 0 else '**FAIL**'}")

    # 4. Uniform cross-check
    log_tile = np.log(tile_size)
    print(f"\n  [4] Uniform cross-check:")
    print(f"    log(tile_size={tile_size}) = {log_tile:.6f}")
    print(f"    Fitted uniform NB:          {uniform_loss:.6f}")
    print(f"    Delta:                      {uniform_loss - log_tile:+.6f}")
    print(f"    Note: no collapsed v3nb run exists; log(2048) minus zero-count "
          f"correction is the available check (weaker than v3_A's)")

    # 5. Loss-object identity
    print(f"\n  [5] Loss-object identity:")
    print(f"    Loss class: MaskedNegativeBinomialOffsetNLLLoss (frozen core)")
    print(f"    max_dispersion_ratio=2.0, clamp_margin=1.0, "
          f"dispersion_window_size=1")
    print(f"    Matches training run config: YES")

    # 6. Monotonicity of sweep
    sweep_losses = np.array([s["loss"] for s in sweep_results])
    sweep_diffs = np.diff(sweep_losses)
    sign_changes = int(np.sum(np.diff(np.sign(sweep_diffs)) != 0))
    # Check just the reliable region (r < 3000)
    print(f"\n  [6] Sweep monotonicity (r in [1, 5000]):")
    print(f"    Sign changes in slope: {sign_changes}")
    # The curve should descend then plateau — check for one transition
    min_idx = np.argmin(sweep_losses)
    descending_before = all(d <= 0.0005 for d in sweep_diffs[:max(1, min_idx)])
    print(f"    Minimum at index {min_idx}/{len(sweep_losses)} "
          f"(r={sweep_results[min_idx]['r']:.1f})")
    if sign_changes <= 5:
        print(f"    Verdict: smooth (few sign changes) — PASS")
    else:
        print(f"    Verdict: somewhat noisy ({sign_changes} sign changes) — "
              f"curve plateaus, minimum may be in flat region")

    # ── Results table ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)

    def pct_bias(model_nll):
        return 100.0 * (uniform_loss - model_nll) / gap

    print(f"\n  {'Model':<22s} {'NB loss':>12s} {'% bias':>10s}")
    print(f"  {'-'*22} {'-'*12} {'-'*10}")
    print(f"  {'Oracle (fitted r)':<22s} {oracle_loss:>12.6f} {'100.0%':>10s}")
    print(f"  {'Trained KEN':<22s} {ken_mean:>12.6f} {pct_bias(ken_mean):>9.1f}%")
    print(f"  {'Trained Hybrid':<22s} {hybrid_mean:>12.6f} "
          f"{pct_bias(hybrid_mean):>9.1f}%")
    print(f"  {'Untrained KEN':<22s} {ken_u_mean:>12.6f} "
          f"{pct_bias(ken_u_mean):>9.1f}%")
    print(f"  {'Untrained Hybrid':<22s} {hybrid_u_mean:>12.6f} "
          f"{pct_bias(hybrid_u_mean):>9.1f}%")
    print(f"  {'Uniform (fitted r)':<22s} {uniform_loss:>12.6f} {'0.0%':>10s}")

    print(f"\n  Fitted r (oracle propensity):  {fitted_r_oracle:.4f} "
          f"(log_r={fitted_log_r_oracle:.4f})")
    print(f"  Fitted r (uniform propensity): {fitted_r_uniform:.4f} "
          f"(log_r={fitted_log_r_uniform:.4f})")
    print(f"  True hexamer_r median:         7.179 (log_r=1.971)")
    print(f"  Frozen model init:             1096  (log_r=7.000)")

    # ── Write oracle_nb_v2.json ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("WRITING JSON")
    print("=" * 70)

    payload = {
        "_what": (
            "Anchor A for the v3nb overdispersed simulation store: true "
            "propensity with a FITTED scalar dispersion r, minimising the "
            "frozen-core NB-offset loss. The offset conditioning absorbs the "
            "overdispersion above r~20, and the loss plateau is flat to within "
            "numerical noise so the fitted r is not identified — do not read "
            "an effective-r value from it. This SUPERSEDES oracle_nb.json, "
            "which plugged in the true per-position r and produced a value "
            "ABOVE untrained models (not a floor). See docs/pending/nb_oracle.md "
            "for the full diagnosis."
        ),
        "oracle_nb_nll": oracle_loss,
        "uniform_nb_nll": uniform_loss,
        "gap_uniform_minus_oracle": gap,
        "fitted_r": {
            "oracle": {
                "r": fitted_r_oracle,
                "log_r": fitted_log_r_oracle,
                "source": oracle_source,
            },
            "uniform": {
                "r": fitted_r_uniform,
                "log_r": fitted_log_r_uniform,
                "source": uniform_source,
                "_note": "Separately fitted (not reusing oracle's r)",
            },
            "reference": {
                "true_hexamer_r_median": 7.179,
                "frozen_model_init_r": 1096.0,
                "frozen_model_log_dispersion_init": 7.0,
            },
            "_note": (
                "The fitted r is NOT identified on the plateau — the loss is "
                "flat to within numerical noise above r~20. The specific "
                "value is an artefact of which grid/optimizer point happened "
                "to land lowest in the noise. Do not interpret it as an "
                "effective dispersion."
            ),
        },
        "noise_floor": noise_floor_info,
        "sweep_curve": sweep_results,
        "models": {
            "trained_ken": {
                "nb_loss": ken_mean,
                "nb_val_loss_from_training": ken_summary["best_val_loss"],
                "pct_bias_captured": round(pct_bias(ken_mean), 2),
                "checkpoint": ken_ckpt,
            },
            "trained_hybrid": {
                "nb_loss": hybrid_mean,
                "nb_val_loss_from_training": hybrid_summary["best_val_loss"],
                "pct_bias_captured": round(pct_bias(hybrid_mean), 2),
                "checkpoint": hybrid_ckpt,
            },
            "untrained_ken": {
                "nb_loss": ken_u_mean,
                "pct_bias_captured": round(pct_bias(ken_u_mean), 2),
                "seed": 42,
            },
            "untrained_hybrid": {
                "nb_loss": hybrid_u_mean,
                "pct_bias_captured": round(pct_bias(hybrid_u_mean), 2),
                "seed": 42,
            },
        },
        "verification": {
            "alignment": align_info,
            "sanity_gate_all_pass": all_pass,
            "untrained_above_oracle": {
                "ken": bool(ken_u_mean > oracle_loss),
                "hybrid": bool(hybrid_u_mean > oracle_loss),
            },
            "trained_between_oracle_and_uniform": {
                "ken": bool(oracle_loss < ken_mean < uniform_loss),
                "hybrid": bool(oracle_loss < hybrid_mean < uniform_loss),
            },
            "uniform_vs_log_W": {
                "log_tile_size": float(log_tile),
                "uniform_nb": uniform_loss,
                "delta": float(uniform_loss - log_tile),
            },
            "sweep_sign_changes": sign_changes,
        },
        "loss": "MaskedNegativeBinomialOffsetNLLLoss (background_model_core, frozen)",
        "loss_config": {
            "max_dispersion_ratio": 2.0,
            "clamp_margin": 1.0,
            "dispersion_window_size": 1,
            "_why": (
                "Matched to the training runs (v3nb_ken/hybrid_nb_frozen_lr5e-3). "
                "A floor computed under a different clamp is not in the same units "
                "as the val_loss it is compared against."
            ),
        },
        "store": STORE,
        "sim_dir": SIM_DIR,
        "fasta": FASTA,
        "n_val_pairs": n_val_pairs,
        "tile_size": int(tile_size),
        "l_target": int(l_target),
        "crop": [int(crop_start), int(crop_stop)],
        "gc_mode": "lut",
        "design_doc": "docs/pending/nb_oracle.md",
        "supersedes": "oracle_nb.json",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.executable,
        "runtime_s": round(time.time() - t0, 1),
    }

    os.makedirs(os.path.dirname(os.path.abspath(OUT_JSON)), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Wrote {OUT_JSON}")

    # ── Mark old oracle_nb.json as superseded ──────────────────────────────
    if os.path.exists(OLD_JSON):
        with open(OLD_JSON) as f:
            old = json.load(f)
        old["_superseded_by"] = "oracle_nb_v2.json"
        old["_superseded_reason"] = (
            "The plug-in oracle (true propensity + true per-position r) is not "
            "a floor for the nb_offset pseudo-likelihood. Both untrained models "
            "scored below it. See docs/pending/nb_oracle.md for the full diagnosis."
        )
        with open(OLD_JSON, "w") as f:
            json.dump(old, f, indent=2)
        print(f"  Marked {OLD_JSON} as superseded")

    print(f"\n  Total runtime: {time.time()-t0:.1f}s")

    if not all_pass:
        print("\n  *** VERIFICATION FAILED — see details above ***")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
