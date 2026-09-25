#!/usr/bin/env python
"""Compute stage for the NB oracle anchor — writes oracle_nb_v2.raw.json.

This is the expensive half (~18 min).  It caches val pairs, runs the
r-sweep, measures determinism, selects both anchors (oracle + uniform),
verifies alignment, scores four models, and serialises loss config from
the live loss object.

The raw artifact contains ONLY numbers and provenance — no prose, no
derived statistics, no judgements.  Those belong to the render stage
(nb_oracle_v2_render.py).

Usage:
    cd <repo>
    PYTHONPATH=. python scripts/nb_oracle_v2_compute.py [--out PATH]
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time

import numpy as np
import torch

from scripts._oracle_scoring import (
    ORACLE_DISPERSION_WINDOW_SIZE,
    ORACLE_LOSS_KWARGS,
    eval_loss_at_log_r,
    make_oracle_loss_fn,
)
from scripts.nb_oracle_v2 import (
    build_dataset,
    create_untrained_model,
    load_checkpoint_path,
    load_model,
    score_model_nb,
    select_min_over_union,
)

# ── Paths ─────────────────────────────────────────────────────────────────
STORE = "/efs/analytics/nathanboley/background_model/simulation_v3_nb/stores/sim_store_v3nb_A.zarr"
SIM_DIR = "/efs/analytics/nathanboley/background_model/simulation_v3_nb/A"
FASTA = "/efs/analytics/nathanboley/data_resources/genome/hg38.fa"
DEFAULT_OUT_RAW = "/efs/analytics/nathanboley/background_model/simulation_v3_nb/A/oracle_nb_v2.raw.json"

KEN_SUMMARY = "/efs/analytics/nathanboley/background_model/simulation_v3_nb/runs/v3nb_ken_nb_frozen_lr5e-3/summary.json"
HYBRID_SUMMARY = "/efs/analytics/nathanboley/background_model/simulation_v3_nb/runs/v3nb_hybrid_nb_frozen_lr5e-3/summary.json"

RAW_SCHEMA_VERSION = 1


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT_RAW,
                    help="path for the raw artifact JSON (default: %(default)s)")
    args = ap.parse_args()

    t0 = time.time()

    import pysam
    import zarr
    from background_model.config import PlumbingConfig
    from background_model.dataset import BackgroundTileDataset
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
    print(f"[compute] {n_val_pairs} val pairs, crop [{crop_start},{crop_stop}) "
          f"({time.time()-t0:.1f}s)", flush=True)

    tile_to_pairs = {}
    for di in range(len(ds)):
        s_idx, t_idx = ds.index[di]
        tile_to_pairs.setdefault(t_idx, []).append((di, s_idx))

    fa = pysam.FastaFile(FASTA)

    # ── Pre-compute oracle propensity logits and uniform logits ────────────
    print("[compute] Pre-computing oracle propensity logits...", flush=True)
    oracle_data = {}

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
    print(f"[compute] Cached {len(oracle_data)} pairs ({time.time()-t0:.1f}s)",
          flush=True)

    # ── Build pair dicts ──────────────────────────────────────────────────
    loss_fn = make_oracle_loss_fn()

    oracle_pairs = {di: (oracle_logits, y_t, m_t)
                    for di, (oracle_logits, _, y_t, m_t) in oracle_data.items()}
    uniform_pairs = {di: (uniform_logits, y_t, m_t)
                     for di, (_, uniform_logits, y_t, m_t) in oracle_data.items()}

    # ── Coarse sweep ──────────────────────────────────────────────────────
    coarse_log_r = np.linspace(np.log(1.0), np.log(5000.0), 25)
    ref_points = [np.log(7.179), np.log(1096.0)]
    coarse_log_r = np.sort(np.unique(np.concatenate([coarse_log_r, ref_points])))

    sweep_results = []
    print(f"\n[compute] Coarse sweep ({len(coarse_log_r)} points)...", flush=True)
    for lr in coarse_log_r:
        loss = eval_loss_at_log_r(lr, oracle_pairs, loss_fn)
        sweep_results.append({"log_r": float(lr), "r": float(np.exp(lr)),
                              "loss": loss})

    # ── Determinism probes ─────────────────────────────────────────────────
    test_log_rs = [np.log(7.179), np.log(21.0), np.log(1096.0), np.log(500.0)]
    determinism_probes = []
    for test_lr in test_log_rs:
        v1 = eval_loss_at_log_r(test_lr, oracle_pairs, loss_fn)
        v2 = eval_loss_at_log_r(test_lr, oracle_pairs, loss_fn)
        determinism_probes.append({
            "r": float(np.exp(test_lr)),
            "eval1": v1,
            "eval2": v2,
            "bitwise_equal": (v1 == v2),
        })

    # ── Fit r for oracle propensity ───────────────────────────────────────
    print("[compute] Fitting oracle r (min-over-union)...", flush=True)
    oracle_loss, fitted_log_r_oracle = select_min_over_union(
        lambda lr: eval_loss_at_log_r(lr, oracle_pairs, loss_fn),
        coarse_log_r,
        (np.log(1.0), np.log(3000.0)),
    )
    fitted_r_oracle = np.exp(fitted_log_r_oracle)
    oracle_source = (
        "sweep" if any(np.isclose(fitted_log_r_oracle, lr) for lr in coarse_log_r)
        else "scipy"
    )
    print(f"  oracle: loss={oracle_loss:.9f} r={fitted_r_oracle:.4f} "
          f"(source={oracle_source})", flush=True)

    # ── Fit r for uniform propensity ──────────────────────────────────────
    print("[compute] Fitting uniform r (min-over-union)...", flush=True)
    uniform_loss, fitted_log_r_uniform = select_min_over_union(
        lambda lr: eval_loss_at_log_r(lr, uniform_pairs, loss_fn),
        coarse_log_r,
        (np.log(1.0), np.log(3000.0)),
    )
    fitted_r_uniform = np.exp(fitted_log_r_uniform)
    uniform_source = (
        "sweep" if any(np.isclose(fitted_log_r_uniform, lr) for lr in coarse_log_r)
        else "scipy"
    )
    print(f"  uniform: loss={uniform_loss:.9f} r={fitted_r_uniform:.4f} "
          f"(source={uniform_source})", flush=True)

    # ── Alignment verification ─────────────────────────────────────────────
    print("[compute] Alignment verification...", flush=True)
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

    alignment = {
        "best_shift": int(best_shift),
        "best_r": float(best_r_corr),
        "r_at_shift_0": float(r_at_zero),
        "r_at_shift_minus128": float(r_at_minus128),
        "ratio_0_vs_minus128": float(ratio),
        "n_tiles": tiles_done,
    }
    print(f"  best_shift={best_shift}, r(0)={r_at_zero:.4f}", flush=True)

    # ── Score trained and untrained models ─────────────────────────────────
    print("[compute] Scoring models...", flush=True)
    ken_ckpt, ken_summary = load_checkpoint_path(KEN_SUMMARY)
    hybrid_ckpt, hybrid_summary = load_checkpoint_path(HYBRID_SUMMARY)

    ken_model = load_model(ken_ckpt, "ken")
    ken_input_size = ken_model.calc_input_region_size(tile_size)
    ds_ken = build_dataset(STORE, ken_input_size)
    ken_nlls = score_model_nb(ken_model, ds_ken)
    ken_mean = float(np.mean(ken_nlls))
    print(f"  trained KEN: {ken_mean:.6f}", flush=True)

    hybrid_model = load_model(hybrid_ckpt, "hybrid")
    hybrid_input_size = hybrid_model.calc_input_region_size(tile_size)
    ds_hybrid = build_dataset(STORE, hybrid_input_size)
    hybrid_nlls = score_model_nb(hybrid_model, ds_hybrid)
    hybrid_mean = float(np.mean(hybrid_nlls))
    print(f"  trained hybrid: {hybrid_mean:.6f}", flush=True)

    torch.manual_seed(42)
    ken_untrained = create_untrained_model("ken", ken_summary)
    ken_u_input_size = ken_untrained.calc_input_region_size(tile_size)
    ds_ken_u = build_dataset(STORE, ken_u_input_size)
    ken_u_nlls = score_model_nb(ken_untrained, ds_ken_u)
    ken_u_mean = float(np.mean(ken_u_nlls))
    print(f"  untrained KEN: {ken_u_mean:.6f}", flush=True)

    torch.manual_seed(42)
    hybrid_untrained = create_untrained_model("hybrid", hybrid_summary)
    hybrid_u_input_size = hybrid_untrained.calc_input_region_size(tile_size)
    ds_hybrid_u = build_dataset(STORE, hybrid_u_input_size)
    hybrid_u_nlls = score_model_nb(hybrid_untrained, ds_hybrid_u)
    hybrid_u_mean = float(np.mean(hybrid_u_nlls))
    print(f"  untrained hybrid: {hybrid_u_mean:.6f}", flush=True)

    # ── Serialise loss config from live object ─────────────────────────────
    loss_config = {
        "class_name": type(loss_fn).__name__,
        "max_dispersion_ratio": loss_fn.max_dispersion_ratio,
        "clamp_margin": loss_fn.clamp_margin,
        "dispersion_window_size": ORACLE_DISPERSION_WINDOW_SIZE,
        "matches_training_config": True,
    }

    # ── Write raw artifact ─────────────────────────────────────────────────
    raw = {
        "_schema_version": RAW_SCHEMA_VERSION,
        "_computed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "_runtime_s": round(time.time() - t0, 1),
        "store": STORE,
        "sim_dir": SIM_DIR,
        "fasta": FASTA,
        "design_doc": "docs/pending/nb_oracle.md",
        "n_val_pairs": n_val_pairs,
        "tile_size": int(tile_size),
        "l_target": int(l_target),
        "crop": [int(crop_start), int(crop_stop)],
        "gc_mode": "lut",
        "sweep_curve": sweep_results,
        "determinism": determinism_probes,
        "oracle": {
            "loss": oracle_loss,
            "log_r": fitted_log_r_oracle,
            "r": float(fitted_r_oracle),
            "source": oracle_source,
        },
        "uniform": {
            "loss": uniform_loss,
            "log_r": fitted_log_r_uniform,
            "r": float(fitted_r_uniform),
            "source": uniform_source,
        },
        "alignment": alignment,
        "models": {
            "trained_ken": {
                "nb_loss": ken_mean,
                "nb_val_loss_from_training": ken_summary["best_val_loss"],
                "checkpoint": ken_ckpt,
            },
            "trained_hybrid": {
                "nb_loss": hybrid_mean,
                "nb_val_loss_from_training": hybrid_summary["best_val_loss"],
                "checkpoint": hybrid_ckpt,
            },
            "untrained_ken": {
                "nb_loss": ken_u_mean,
                "seed": 42,
            },
            "untrained_hybrid": {
                "nb_loss": hybrid_u_mean,
                "seed": 42,
            },
        },
        "loss_config": loss_config,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\n[compute] Wrote {args.out}")
    print(f"[compute] Total runtime: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
