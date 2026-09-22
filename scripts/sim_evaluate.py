#!/usr/bin/env python
"""Evaluate trained models against simulation ground truth (plan §7).

Metrics:
  1. Corrected flatness — the product metric: after applying per-fragment
     weights, the corrected profile should be flat (uniform across positions).
     Residual structure IS the error.
  2. Shape recovery — KL divergence between predicted probs and true
     per-position endpoint propensity (computed exactly from ground-truth w6).
  3. Clamp sweep — which clamp bounds minimise corrected-flatness error.

Usage:
    PYTHONPATH=. python scripts/sim_evaluate.py \
        --sim-dir /efs/.../simulation/B \
        --store /efs/.../simulation/stores/sim_store_B.zarr \
        --runs-root /efs/.../simulation/runs \
        --out /efs/.../simulation/eval_B

Env: /home/nathanboley/miniconda3/envs/biomarker_env/bin/python
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

FASTA = "/efs/analytics/nathanboley/data_resources/genome/hg38.fa"
LOSSES = ["multinomial", "dirichlet_multinomial", "nb_offset"]

# ── Track layout (must match preprocess/store) ────────────────────────────
STRANDS = ("+", "-")
FL_BANDS = ((40, 65), (120, 175))
COVERAGE_TYPES = ("first", "last", "midpoint")
TRACK_INDEX = {}
_idx = 0
for _s in STRANDS:
    for _fl in FL_BANDS:
        for _c in COVERAGE_TYPES:
            TRACK_INDEX[(_s, _fl, _c)] = _idx
            _idx += 1
N_TRACKS = _idx


def find_best_checkpoint(runs_root, run_name):
    """Find the best checkpoint (lowest val_loss) in a run directory."""
    run_dir = os.path.join(runs_root, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        # Try finding the run directory by pattern
        for d in sorted(os.listdir(runs_root)):
            if run_name in d:
                ckpt_dir = os.path.join(runs_root, d, "checkpoints")
                run_dir = os.path.join(runs_root, d)
                break
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"No checkpoints found for {run_name} in {runs_root}")
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files in {ckpt_dir}")
    # Best checkpoint by name (Lightning names them by val_loss)
    best = sorted(ckpts)[0]  # lowest val_loss alphabetically
    return os.path.join(ckpt_dir, best), run_dir


def load_model(ckpt_path, loss):
    """Load a trained BackgroundModel from checkpoint."""
    from background_model.train import InstrumentedBackgroundModel
    model = InstrumentedBackgroundModel.load_from_checkpoint(
        ckpt_path, loss=loss, learning_rate=1e-4
    )
    model.eval()
    return model


def predict_on_store(model, store_path, split="val", sample_role="train",
                     device="cpu"):
    """Run model on all tiles in a split, returning predicted probs per tile.

    Returns dict: tile_idx -> (probs, mask) where probs is (C, tile_size).
    """
    from background_model.dataset import BackgroundTileDataset
    from background_model.config import PlumbingConfig
    import zarr

    root = zarr.open_group(store_path, mode="r")
    cfg = PlumbingConfig.from_json(root.attrs["config_json"])
    tile_size = cfg.tile_size
    model_input_size = model.calc_input_region_size(tile_size)

    ds = BackgroundTileDataset(
        store_path=store_path,
        model_input_size=model_input_size,
        split=split,
        sample_role=sample_role,
        min_N=0,
        train_mode=False,
        seed=1337,
    )

    results = {}
    model = model.to(device)
    with torch.no_grad():
        for i in range(len(ds)):
            x, y, m = ds[i]
            s_idx, t_idx = ds.index[i]
            # predict_profile is a method on BackgroundModel; takes (4, L_in)
            # numpy array + optional (L_out,) mask, returns dict with 'probs'
            x_np = x.numpy()
            mask_np = m.numpy() if hasattr(m, 'numpy') else np.array(m, dtype=bool)
            out = model.predict_profile(x_np, mask=mask_np)
            probs_np = out["probs"]  # (C, tile_size)

            if t_idx not in results:
                results[t_idx] = {"probs": probs_np, "mask": mask_np,
                                  "obs": {}, "N": {}}
            results[t_idx]["obs"][s_idx] = y.numpy()
            N_per_track = (y.numpy() * mask_np[None, :]).sum(axis=1)
            results[t_idx]["N"][s_idx] = N_per_track

    return results


def compute_true_propensity(sim_dir, store_path):
    """Compute the true per-position endpoint propensity from ground truth w6.

    For each tile and each track, compute the expected density from the known
    hexamer weights. This is the gold standard against which predicted probs
    are compared.

    Returns dict: tile_idx -> (C, tile_size) true propensity array.
    """
    import pysam
    import zarr
    from background_model.config import PlumbingConfig
    from scripts.sim_fragments import (
        hexamer_indices, GCBias2D, HEX_HALF, NHEX, RC_PERM,
        GC_BIAS_JSON,
    )

    gt = np.load(os.path.join(sim_dir, "ground_truth.npz"), allow_pickle=True)
    w6 = gt["w6"]
    len_vals = gt["len_vals"]
    if "len_p" in gt:
        len_p = gt["len_p"]
    else:
        # Per-sample FL distributions: use mean for oracle propensity
        len_p = gt["len_p_per_sample"].mean(axis=0)
    gcbias = GCBias2D(gt["bias_lengths"], gt["bias_gc_percents"], gt["bias_grid"])

    root = zarr.open_group(store_path, mode="r")
    cfg = PlumbingConfig.from_json(root.attrs["config_json"])
    tile_size = cfg.tile_size

    contigs = root["tiles/contig"][:]
    starts = root["tiles/start"][:]
    stops = root["tiles/stop"][:]
    splits = root["tiles/split"][:]

    fa = pysam.FastaFile(FASTA)
    true_prop = {}

    for t_idx in range(len(contigs)):
        if splits[t_idx] != 1:  # val split only
            continue
        contig = str(contigs[t_idx])
        gstart = int(starts[t_idx])
        gstop = int(stops[t_idx])
        region_len = gstop - gstart

        # Fetch sequence with hexamer margin
        seq = fa.fetch(contig, gstart - HEX_HALF, gstop + HEX_HALF).upper()
        seq_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        fwd_cut, rc_cut, valid = hexamer_indices(seq_bytes)

        # Cumulative GC for the region
        core = seq_bytes[HEX_HALF:HEX_HALF + region_len]
        is_gc = (core == ord("G")) | (core == ord("C"))
        cum_gc = np.concatenate([[0], np.cumsum(is_gc)]).astype(np.int64)

        prop = np.zeros((N_TRACKS, tile_size), dtype=np.float64)

        # For each fragment length, compute per-position propensity
        for li, L in enumerate(len_vals):
            if len_p[li] <= 0:
                continue
            for fl_idx, (fl_lo, fl_hi) in enumerate(FL_BANDS):
                if L < fl_lo or L >= fl_hi:
                    continue

                # For each start position p, fragment [p, p+L)
                max_p = region_len - L
                if max_p < 0:
                    continue
                ps = np.arange(0, max_p + 1)
                qs = ps + L

                # Check validity at both cut sites
                v = valid[ps] & valid[qs]
                ps_v = ps[v]
                qs_v = qs[v]

                if len(ps_v) == 0:
                    continue

                # Acceptance weight for each fragment placement
                lw = w6[fwd_cut[ps_v]]
                rw = w6[rc_cut[qs_v]]
                gc_pct = 100.0 * (cum_gc[qs_v] - cum_gc[ps_v]) / L
                gb = gcbias(L, gc_pct)
                weight = lw * rw * gb * len_p[li]

                # Distribute weight to endpoint tracks
                for strand_idx, strand in enumerate(STRANDS):
                    # "first" = 5' end, "last" = 3' end, "midpoint"
                    if strand == "+":
                        first_pos = ps_v
                        last_pos = qs_v - 1
                    else:
                        first_pos = qs_v - 1
                        last_pos = ps_v
                    mid_pos = (ps_v + qs_v) // 2

                    for ci, (cov_type, endpoints) in enumerate(
                        [("first", first_pos), ("last", last_pos),
                         ("midpoint", mid_pos)]
                    ):
                        track = TRACK_INDEX[(strand, (fl_lo, fl_hi), cov_type)]
                        # Clip to tile bounds
                        in_bounds = (endpoints >= 0) & (endpoints < tile_size)
                        if in_bounds.any():
                            np.add.at(prop[track], endpoints[in_bounds],
                                      weight[in_bounds] * 0.5)  # 50% per strand

        # Normalize each track to a probability distribution
        for t in range(N_TRACKS):
            s = prop[t].sum()
            if s > 0:
                prop[t] /= s

        true_prop[t_idx] = prop

    fa.close()
    return true_prop


def evaluate_shape_recovery(predictions, true_prop):
    """Compare predicted shape to true propensity.

    Returns per-tile and aggregate metrics.
    """
    kl_divs = []
    pearson_rs = []
    from scipy.stats import pearsonr

    for t_idx in sorted(set(predictions.keys()) & set(true_prop.keys())):
        pred = predictions[t_idx]["probs"]
        true = true_prop[t_idx]
        mask = predictions[t_idx]["mask"]

        for track in range(N_TRACKS):
            p = pred[track][mask]
            t = true[track][mask]
            if p.sum() <= 0 or t.sum() <= 0:
                continue
            # Renormalize
            p = p / p.sum()
            t = t / t.sum()
            # KL(true || pred)
            safe = (t > 0) & (p > 0)
            if safe.sum() < 10:
                continue
            kl = float(np.sum(t[safe] * np.log(t[safe] / p[safe])))
            kl_divs.append(kl)
            r, _ = pearsonr(t[safe], p[safe])
            pearson_rs.append(float(r))

    return {
        "n_comparisons": len(kl_divs),
        "kl_divergence_mean": float(np.mean(kl_divs)) if kl_divs else None,
        "kl_divergence_median": float(np.median(kl_divs)) if kl_divs else None,
        "pearson_r_mean": float(np.mean(pearson_rs)) if pearson_rs else None,
        "pearson_r_median": float(np.median(pearson_rs)) if pearson_rs else None,
    }


def evaluate_corrected_flatness(predictions):
    """Measure residual structure in corrected profiles.

    For each (sample, tile), compute corrected = observed / (N * probs).
    A perfect model produces corrected ≡ 1 everywhere.
    Coefficient of variation of corrected is the error metric.
    """
    cvs = []
    for t_idx, data in predictions.items():
        probs = data["probs"]
        mask = data["mask"]
        for s_idx, obs in data["obs"].items():
            N = data["N"][s_idx]
            for track in range(N_TRACKS):
                if N[track] <= 0:
                    continue
                expected = N[track] * probs[track]
                valid = mask & (expected > 1e-10)
                if valid.sum() < 20:
                    continue
                corrected = obs[track][valid] / expected[valid]
                cv = float(np.std(corrected) / np.mean(corrected))
                cvs.append(cv)

    return {
        "n_comparisons": len(cvs),
        "corrected_cv_mean": float(np.mean(cvs)) if cvs else None,
        "corrected_cv_median": float(np.median(cvs)) if cvs else None,
        "corrected_cv_p90": float(np.percentile(cvs, 90)) if cvs else None,
    }


def clamp_sweep(predictions, bounds_list=None):
    """Sweep clamp bounds and measure corrected flatness at each."""
    if bounds_list is None:
        bounds_list = [
            (None, None),     # identity
            (0.1, 10.0),
            (0.2, 5.0),
            (0.5, 2.0),
            (0.3, 3.0),
        ]

    results = []
    for lo, hi in bounds_list:
        cvs = []
        for t_idx, data in predictions.items():
            probs = data["probs"]
            mask = data["mask"]
            for s_idx, obs in data["obs"].items():
                N = data["N"][s_idx]
                for track in range(N_TRACKS):
                    if N[track] <= 0:
                        continue
                    expected = N[track] * probs[track]
                    valid = mask & (expected > 1e-10)
                    if valid.sum() < 20:
                        continue
                    weights = 1.0 / (probs[track][valid] * valid.sum())
                    if lo is not None or hi is not None:
                        weights = np.clip(weights, lo, hi)
                    corrected = obs[track][valid] * weights
                    cv = float(np.std(corrected) / np.mean(corrected)) if np.mean(corrected) > 0 else np.nan
                    cvs.append(cv)

        results.append({
            "clamp_lo": lo,
            "clamp_hi": hi,
            "label": f"[{lo}, {hi}]" if lo is not None else "identity",
            "cv_mean": float(np.mean(cvs)) if cvs else None,
            "cv_median": float(np.median(cvs)) if cvs else None,
        })
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-dir", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--losses", default="all",
                    help="comma-separated losses or 'all'")
    ap.add_argument("--run-prefix", default="sim_v2_B",
                    help="run name prefix (run name = {prefix}_{loss})")
    ap.add_argument("--run-names", default=None,
                    help="explicit loss:run_name pairs, comma-separated "
                         "(e.g. 'multinomial:my_run_1,nb_offset:my_run_2'). "
                         "Overrides --run-prefix.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    losses = LOSSES if args.losses == "all" else args.losses.split(",")
    t0 = time.time()

    # Compute true propensity (shared across models)
    print("[eval] computing true propensity from ground truth...", flush=True)
    true_prop = compute_true_propensity(args.sim_dir, args.store)
    print(f"[eval] true propensity for {len(true_prop)} val tiles "
          f"({time.time()-t0:.1f}s)", flush=True)

    # Build loss -> run_name mapping
    run_name_map = {}
    if args.run_names:
        for pair in args.run_names.split(","):
            l, rn = pair.split(":")
            run_name_map[l.strip()] = rn.strip()

    all_results = {}
    for loss in losses:
        print(f"\n[eval] === {loss} ===", flush=True)
        run_name = run_name_map.get(loss, f"{args.run_prefix}_{loss}")
        try:
            ckpt_path, run_dir = find_best_checkpoint(args.runs_root, run_name)
        except FileNotFoundError as e:
            print(f"[eval] SKIP {loss}: {e}", flush=True)
            continue
        print(f"[eval] checkpoint: {ckpt_path}", flush=True)

        model = load_model(ckpt_path, loss)
        print(f"[eval] model loaded ({time.time()-t0:.1f}s)", flush=True)

        predictions = predict_on_store(model, args.store, split="val",
                                       device=args.device)
        print(f"[eval] predictions on {len(predictions)} val tiles "
              f"({time.time()-t0:.1f}s)", flush=True)

        shape = evaluate_shape_recovery(predictions, true_prop)
        flatness = evaluate_corrected_flatness(predictions)
        clamp = clamp_sweep(predictions)

        result = {
            "loss": loss,
            "checkpoint": ckpt_path,
            "shape_recovery": shape,
            "corrected_flatness": flatness,
            "clamp_sweep": clamp,
        }
        all_results[loss] = result

        print(f"[eval] {loss}: shape r={shape.get('pearson_r_mean', 'N/A'):.4f}  "
              f"KL={shape.get('kl_divergence_mean', 'N/A'):.6f}  "
              f"flatness CV={flatness.get('corrected_cv_mean', 'N/A'):.4f}",
              flush=True)

    # Save results
    out_file = os.path.join(args.out, "eval_results.json")
    json.dump(all_results, open(out_file, "w"), indent=2, default=str)
    print(f"\n[eval] wrote {out_file} ({time.time()-t0:.1f}s)", flush=True)

    # Print comparison table
    print("\n=== Model Comparison ===")
    print(f"{'Loss':<25} {'Shape r':>10} {'KL div':>10} {'Flat CV':>10}")
    print("-" * 60)
    for loss in losses:
        if loss not in all_results:
            continue
        r = all_results[loss]
        sr = r["shape_recovery"]
        cf = r["corrected_flatness"]
        print(f"{loss:<25} "
              f"{sr.get('pearson_r_mean', 0):.4f}     "
              f"{sr.get('kl_divergence_mean', 0):.6f}   "
              f"{cf.get('corrected_cv_mean', 0):.4f}")


if __name__ == "__main__":
    sys.exit(main())
