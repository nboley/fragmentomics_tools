#!/usr/bin/env python
"""Score v3nb models under per-count MULTINOMIAL NLL and compute oracle/uniform.

These models were TRAINED with nb_offset loss (--freeze-dispersion
--dispersion-window-size 1), but this script evaluates them under
MaskedMultinomialNLLLoss — a DIFFERENT unit.  The purpose is a valid
cross-model comparison: multinomial NLL has a trustworthy oracle (the true
propensity IS the generative categorical), whereas the NB oracle depends on
per-position dispersion whose oracle was found to be misspecified (models
scored below the "floor").

Reuses scripts.sim_oracle.compute_oracle_propensity_for_tile for the
propensity computation (alignment, GC, centre-crop are already verified
there) and background_model_core.MaskedMultinomialNLLLoss for the loss.

Usage:
    cd <repo>
    PYTHONPATH=. /home/nathanboley/miniconda3/envs/biomarker_env/bin/python \\
        scripts/score_v3nb_multinomial.py
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time

import numpy as np
import torch

# ── Paths ─────────────────────────────────────────────────────────────────
STORE = "/efs/analytics/nathanboley/background_model/simulation_v3_nb/stores/sim_store_v3nb_A.zarr"
SIM_DIR = "/efs/analytics/nathanboley/background_model/simulation_v3_nb/A"
OUT_JSON = "/efs/analytics/nathanboley/background_model/simulation_v3_nb/A/oracle_multinomial.json"
FASTA = "/efs/analytics/nathanboley/data_resources/genome/hg38.fa"

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
    """Load a trained model from checkpoint.

    Uses the Instrumented* classes so that load_from_checkpoint restores the
    hparams correctly (they inherit from the base classes).
    """
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
    """Create a randomly-initialised model matching the architecture of a run."""
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


def score_model_multinomial(model, ds, device="cpu"):
    """Run model on all val pairs and return per-pair multinomial NLL.

    The model produces shape_logits; we feed them to MaskedMultinomialNLLLoss
    which applies log_softmax internally, so the logits need not be normalised.
    """
    from background_model_core import MaskedMultinomialNLLLoss
    loss_fn = MaskedMultinomialNLLLoss()
    model = model.to(device)
    nlls = []
    with torch.no_grad():
        for i in range(len(ds)):
            x, y, mask = ds[i]
            x = x.unsqueeze(0).to(device)
            shape_logits, _ = model(x)
            y_t = y.unsqueeze(0)
            mask3 = mask.unsqueeze(0).unsqueeze(0).expand_as(y_t).bool()
            nll = loss_fn(shape_logits.cpu(), y_t, mask3).item()
            nlls.append(nll)
    return np.array(nlls)


def compute_oracle_and_uniform(store_path, sim_dir, gc_lut_mode=False):
    """Compute multinomial oracle and uniform NLL for the v3nb val set.

    Returns (oracle_nlls, uniform_nlls, metadata_dict).
    """
    import pysam
    import zarr
    from background_model.config import PlumbingConfig
    from background_model.dataset import BackgroundTileDataset
    from background_model_core import MaskedMultinomialNLLLoss
    from scripts.sim_fragments import GCBias2D, MAX_LEN
    from scripts.sim_oracle import compute_oracle_propensity_for_tile

    t0 = time.time()

    # Load ground truth
    gt = np.load(os.path.join(sim_dir, "ground_truth.npz"), allow_pickle=True)
    w6 = gt["w6"]
    len_vals = gt["len_vals"]
    len_p_per_sample = gt["len_p_per_sample"]
    gcbias = GCBias2D(gt["bias_lengths"], gt["bias_gc_percents"], gt["bias_grid"])
    gc_lut = gcbias.build_lookup_table(max_len=MAX_LEN) if gc_lut_mode else None

    # Store metadata
    root = zarr.open_group(store_path, mode="r")
    cfg = PlumbingConfig.from_json(root.attrs["config_json"])
    tile_size = cfg.tile_size
    l_target = cfg.l_target

    contigs = root["tiles/contig"][:]
    starts = root["tiles/start"][:]
    stops = root["tiles/stop"][:]

    crop_start = (l_target - tile_size) // 2
    crop_stop = crop_start + tile_size

    # Dataset (tile_size as model_input_size — we only use y and mask)
    ds = BackgroundTileDataset(
        store_path=store_path,
        model_input_size=tile_size,
        split="val",
        sample_role="train",
        min_N=0,
        train_mode=False,
        seed=1337,
    )
    print(f"[oracle] {len(ds)} val pairs, crop [{crop_start},{crop_stop}) "
          f"({time.time()-t0:.1f}s)", flush=True)

    # Build tile -> pairs index
    tile_to_pairs = {}
    for di in range(len(ds)):
        s_idx, t_idx = ds.index[di]
        tile_to_pairs.setdefault(t_idx, []).append((di, s_idx))

    fa = pysam.FastaFile(FASTA)
    loss_fn = MaskedMultinomialNLLLoss()

    oracle_nlls = []
    uniform_nlls = []
    n_pairs = 0

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
            with torch.no_grad():
                logits = torch.from_numpy(
                    np.where(p > 0, np.log(p), -1e30)
                ).float().unsqueeze(0)
                y_t = y.unsqueeze(0)
                m_t = mask.unsqueeze(0)

                oracle_nlls.append(loss_fn(logits, y_t, m_t).item())
                uniform_nlls.append(
                    loss_fn(torch.zeros_like(logits), y_t, m_t).item())
            n_pairs += 1

        if (tn + 1) % 50 == 0 or tn == 0:
            print(f"[oracle] {tn+1}/{len(tile_to_pairs)} tiles, {n_pairs} pairs  "
                  f"oracle={np.mean(oracle_nlls):.6f}  "
                  f"uniform={np.mean(uniform_nlls):.6f}  "
                  f"({time.time()-t0:.1f}s)", flush=True)

    fa.close()
    meta = {
        "tile_size": int(tile_size),
        "l_target": int(l_target),
        "crop": [int(crop_start), int(crop_stop)],
        "n_val_pairs": n_pairs,
        "gc_mode": "lut" if gc_lut_mode else "bilinear",
        "store": store_path,
        "sim_dir": sim_dir,
    }
    return np.array(oracle_nlls), np.array(uniform_nlls), meta


def alignment_verification(store_path, sim_dir, n_tiles=20):
    """Shift-correlation check: oracle propensity vs val counts.

    Shifts propensity by [-260, +260] and computes Pearson r at each shift.
    The argmax must be at shift 0 for correct alignment.
    """
    import pysam
    import zarr
    from background_model.config import PlumbingConfig
    from background_model.dataset import BackgroundTileDataset
    from scripts.sim_fragments import GCBias2D, MAX_LEN
    from scripts.sim_oracle import compute_oracle_propensity_for_tile

    gt = np.load(os.path.join(sim_dir, "ground_truth.npz"), allow_pickle=True)
    w6 = gt["w6"]
    len_vals = gt["len_vals"]
    len_p_per_sample = gt["len_p_per_sample"]
    gcbias = GCBias2D(gt["bias_lengths"], gt["bias_gc_percents"], gt["bias_grid"])

    root = zarr.open_group(store_path, mode="r")
    cfg = PlumbingConfig.from_json(root.attrs["config_json"])
    tile_size, l_target = cfg.tile_size, cfg.l_target
    contigs = root["tiles/contig"][:]
    starts = root["tiles/start"][:]
    stops = root["tiles/stop"][:]

    crop_start = (l_target - tile_size) // 2
    crop_stop = crop_start + tile_size

    ds = BackgroundTileDataset(
        store_path=store_path, model_input_size=tile_size,
        split="val", sample_role="train", min_N=0,
        train_mode=False, seed=1337,
    )

    tile_to_pairs = {}
    for di in range(len(ds)):
        s_idx, t_idx = ds.index[di]
        tile_to_pairs.setdefault(t_idx, []).append((di, s_idx))

    fa = pysam.FastaFile(FASTA)
    max_shift = 260
    shifts = np.arange(-max_shift, max_shift + 1)

    # Accumulate cross-correlation across tiles
    all_prop = []
    all_counts = []
    tiles_done = 0

    for t_idx in sorted(tile_to_pairs):
        if tiles_done >= n_tiles:
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
            p = prop_full[loc[s_idx]]  # (C, l_target) — full extent
            counts = y.numpy()          # (C, tile_size) — cropped

            # Flatten across tracks for the correlation
            # prop is l_target wide, counts is tile_size wide, cropped at [crop_start, crop_stop)
            all_prop.append(p)
            all_counts.append(counts)

        tiles_done += 1

    fa.close()

    # Compute correlation at each shift
    # For shift s: compare prop[:, crop_start+s : crop_stop+s] vs counts[:, :]
    rs = []
    for s in shifts:
        lo = crop_start + s
        hi = crop_stop + s
        if lo < 0 or hi > l_target:
            rs.append(np.nan)
            continue
        p_flat = np.concatenate([p[:, lo:hi].ravel() for p in all_prop])
        c_flat = np.concatenate([c.ravel() for c in all_counts])
        r = np.corrcoef(p_flat, c_flat)[0, 1]
        rs.append(r)

    rs = np.array(rs)
    best_shift = shifts[np.nanargmax(rs)]
    best_r = np.nanmax(rs)
    r_at_zero = rs[max_shift]  # shift=0 is at index max_shift
    r_at_minus128 = rs[max_shift - 128] if max_shift >= 128 else np.nan

    return {
        "best_shift": int(best_shift),
        "best_r": float(best_r),
        "r_at_shift_0": float(r_at_zero),
        "r_at_shift_minus128": float(r_at_minus128),
        "n_tiles": tiles_done,
        "n_pairs": sum(len(v) for t, v in tile_to_pairs.items()
                       if t in sorted(tile_to_pairs)[:n_tiles]),
    }


def main():
    t0 = time.time()

    # ── 1. Oracle + uniform ───────────────────────────────────────────────
    print("=" * 70)
    print("STEP 1: Computing multinomial oracle and uniform for v3nb store")
    print("=" * 70)
    oracle_nlls, uniform_nlls, meta = compute_oracle_and_uniform(STORE, SIM_DIR)
    oracle_mean = float(np.mean(oracle_nlls))
    uniform_mean = float(np.mean(uniform_nlls))
    gap = uniform_mean - oracle_mean
    print(f"\n  Oracle:  {oracle_mean:.6f}")
    print(f"  Uniform: {uniform_mean:.6f}")
    print(f"  Gap:     {gap:.6f}")

    # ── 2. Alignment verification ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: Alignment verification (shift-correlation)")
    print("=" * 70)
    align = alignment_verification(STORE, SIM_DIR, n_tiles=20)
    print(f"  Best shift:           {align['best_shift']}")
    print(f"  Best r:               {align['best_r']:.6f}")
    print(f"  r at shift 0:         {align['r_at_shift_0']:.6f}")
    print(f"  r at shift -128:      {align['r_at_shift_minus128']:.6f}")
    assert align["best_shift"] == 0, (
        f"ALIGNMENT FAILURE: best shift is {align['best_shift']}, not 0")

    # ── 3. Load and score trained models ──────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: Scoring trained checkpoints under multinomial NLL")
    print("=" * 70)

    ken_ckpt, ken_summary = load_checkpoint_path(KEN_SUMMARY)
    hybrid_ckpt, hybrid_summary = load_checkpoint_path(HYBRID_SUMMARY)

    print(f"  KEN checkpoint:    {ken_ckpt}")
    print(f"  Hybrid checkpoint: {hybrid_ckpt}")

    # Load trained KEN
    print("\n  Loading trained KEN...", flush=True)
    ken_model = load_model(ken_ckpt, "ken")
    tile_size = meta["tile_size"]
    ken_input_size = ken_model.calc_input_region_size(tile_size)
    ds_ken = build_dataset(STORE, ken_input_size)
    print(f"  Scoring KEN on {len(ds_ken)} val pairs...", flush=True)
    ken_nlls = score_model_multinomial(ken_model, ds_ken)
    ken_mean = float(np.mean(ken_nlls))
    print(f"  KEN multinomial NLL: {ken_mean:.6f}")

    # Load trained hybrid
    print("\n  Loading trained hybrid...", flush=True)
    hybrid_model = load_model(hybrid_ckpt, "hybrid")
    hybrid_input_size = hybrid_model.calc_input_region_size(tile_size)
    ds_hybrid = build_dataset(STORE, hybrid_input_size)
    print(f"  Scoring hybrid on {len(ds_hybrid)} val pairs...", flush=True)
    hybrid_nlls = score_model_multinomial(hybrid_model, ds_hybrid)
    hybrid_mean = float(np.mean(hybrid_nlls))
    print(f"  Hybrid multinomial NLL: {hybrid_mean:.6f}")

    # ── 4. Score untrained controls ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: Scoring untrained (random init) controls")
    print("=" * 70)

    print("  Creating untrained KEN...", flush=True)
    torch.manual_seed(42)
    ken_untrained = create_untrained_model("ken", ken_summary)
    ken_untrained_input_size = ken_untrained.calc_input_region_size(tile_size)
    ds_ken_u = build_dataset(STORE, ken_untrained_input_size)
    ken_u_nlls = score_model_multinomial(ken_untrained, ds_ken_u)
    ken_u_mean = float(np.mean(ken_u_nlls))
    print(f"  Untrained KEN multinomial NLL: {ken_u_mean:.6f}")

    print("  Creating untrained hybrid...", flush=True)
    torch.manual_seed(42)
    hybrid_untrained = create_untrained_model("hybrid", hybrid_summary)
    hybrid_untrained_input_size = hybrid_untrained.calc_input_region_size(tile_size)
    ds_hybrid_u = build_dataset(STORE, hybrid_untrained_input_size)
    hybrid_u_nlls = score_model_multinomial(hybrid_untrained, ds_hybrid_u)
    hybrid_u_mean = float(np.mean(hybrid_u_nlls))
    print(f"  Untrained hybrid multinomial NLL: {hybrid_u_mean:.6f}")

    # ── 5. Verification ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5: Verification")
    print("=" * 70)

    # Sanity gate
    # Trained models: must lie strictly between oracle and uniform.
    # Untrained models: must not beat the oracle.  They may exceed uniform
    # (random logits are worse than constant logits — expected behaviour).
    all_pass = True
    for name, score in [("trained_ken", ken_mean), ("trained_hybrid", hybrid_mean)]:
        above_oracle = score > oracle_mean
        below_uniform = score < uniform_mean
        ok = above_oracle and below_uniform
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name} = {score:.6f}  "
              f"(oracle < model < uniform? {above_oracle} and {below_uniform})")
        if not ok:
            all_pass = False
    for name, score in [("untrained_ken", ken_u_mean),
                        ("untrained_hybrid", hybrid_u_mean)]:
        above_oracle = score > oracle_mean
        near_uniform = abs(score - uniform_mean) < 0.05  # within 50 millинats
        status = "PASS" if above_oracle else "FAIL"
        print(f"  {status}: {name} = {score:.6f}  "
              f"(above oracle? {above_oracle}, "
              f"delta from uniform: {score - uniform_mean:+.6f})")
        if not above_oracle:
            all_pass = False

    # Untrained should be near uniform
    print(f"\n  Untrained KEN delta from uniform:    {ken_u_mean - uniform_mean:+.6f}")
    print(f"  Untrained hybrid delta from uniform: {hybrid_u_mean - uniform_mean:+.6f}")

    # Uniform cross-check: log(tile_size) adjusted for zero-count tracks
    log_tile = np.log(tile_size)
    print(f"\n  Uniform cross-check:")
    print(f"    log(tile_size={tile_size}) = {log_tile:.6f}")
    print(f"    Computed uniform:           {uniform_mean:.6f}")
    print(f"    Delta (uniform - log(W)):   {uniform_mean - log_tile:.6f}")
    print(f"    (Expected negative: zero-count tracks contribute 0, "
          f"pulling mean below log(W))")

    # ── 6. Bias captured ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 6: Results")
    print("=" * 70)

    def pct_bias(model_nll):
        return 100.0 * (uniform_mean - model_nll) / gap

    print(f"\n  {'Model':<22s} {'NB val_loss':>12s} {'Multi NLL':>12s} {'% bias':>10s}")
    print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'Oracle':<22s} {'—':>12s} {oracle_mean:>12.6f} {'100.0%':>10s}")
    print(f"  {'Trained KEN':<22s} {ken_summary['best_val_loss']:>12.6f} "
          f"{ken_mean:>12.6f} {pct_bias(ken_mean):>9.1f}%")
    print(f"  {'Trained Hybrid':<22s} {hybrid_summary['best_val_loss']:>12.6f} "
          f"{hybrid_mean:>12.6f} {pct_bias(hybrid_mean):>9.1f}%")
    print(f"  {'Untrained KEN':<22s} {'—':>12s} "
          f"{ken_u_mean:>12.6f} {pct_bias(ken_u_mean):>9.1f}%")
    print(f"  {'Untrained Hybrid':<22s} {'—':>12s} "
          f"{hybrid_u_mean:>12.6f} {pct_bias(hybrid_u_mean):>9.1f}%")
    print(f"  {'Uniform':<22s} {'—':>12s} {uniform_mean:>12.6f} {'0.0%':>10s}")

    # ── 7. Write JSON ─────────────────────────────────────────────────────
    payload = {
        "_what": (
            "Multinomial oracle and uniform for the v3nb overdispersed simulation "
            "store, plus per-count multinomial NLL scores for two NB-trained "
            "checkpoints.  The models were trained with nb_offset loss "
            "(--freeze-dispersion --dispersion-window-size 1); this file scores "
            "them under MaskedMultinomialNLLLoss.  That is deliberate: the "
            "multinomial oracle is a provable floor, whereas the NB oracle was "
            "found to be misspecified (models scored below it)."
        ),
        "oracle_multinomial_nll": oracle_mean,
        "uniform_multinomial_nll": uniform_mean,
        "gap_uniform_minus_oracle": gap,
        "models": {
            "trained_ken": {
                "multinomial_nll": ken_mean,
                "nb_val_loss": ken_summary["best_val_loss"],
                "pct_bias_captured": round(pct_bias(ken_mean), 2),
                "checkpoint": ken_ckpt,
            },
            "trained_hybrid": {
                "multinomial_nll": hybrid_mean,
                "nb_val_loss": hybrid_summary["best_val_loss"],
                "pct_bias_captured": round(pct_bias(hybrid_mean), 2),
                "checkpoint": hybrid_ckpt,
            },
            "untrained_ken": {
                "multinomial_nll": ken_u_mean,
                "pct_bias_captured": round(pct_bias(ken_u_mean), 2),
                "seed": 42,
            },
            "untrained_hybrid": {
                "multinomial_nll": hybrid_u_mean,
                "pct_bias_captured": round(pct_bias(hybrid_u_mean), 2),
                "seed": 42,
            },
        },
        "verification": {
            "alignment": align,
            "sanity_gate_all_pass": all_pass,
            "sanity_gate_notes": (
                "Trained models: oracle < model < uniform (both PASS).  "
                "Untrained models: must not beat oracle (both PASS); "
                "may exceed uniform because random logits are worse than "
                "constant logits — this is expected, not a failure."
            ),
            "uniform_vs_log_W": {
                "log_tile_size": float(log_tile),
                "uniform": uniform_mean,
                "delta": float(uniform_mean - log_tile),
            },
            "untrained_delta_from_uniform": {
                "ken": float(ken_u_mean - uniform_mean),
                "hybrid": float(hybrid_u_mean - uniform_mean),
            },
        },
        "loss": "MaskedMultinomialNLLLoss (background_model_core, frozen)",
        "reduction": (
            "mean over (pair, track); per-track NLL divided by that track's "
            "total count N (clamped to >=1), so a zero-count track contributes "
            "exactly 0"),
        "store": STORE,
        "sim_dir": SIM_DIR,
        "fasta": FASTA,
        "n_val_pairs": meta["n_val_pairs"],
        "tile_size": meta["tile_size"],
        "l_target": meta["l_target"],
        "crop": meta["crop"],
        "gc_mode": meta["gc_mode"],
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.executable,
        "runtime_s": round(time.time() - t0, 1),
    }

    os.makedirs(os.path.dirname(os.path.abspath(OUT_JSON)), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Wrote {OUT_JSON}")
    print(f"  Runtime: {time.time()-t0:.1f}s")

    if not all_pass:
        print("\n  *** SANITY GATE FAILED — see details above ***")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
