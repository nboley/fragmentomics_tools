#!/usr/bin/env python
"""Step 0: sweep scalar r with propensity at truth to test §2.3 of nb_oracle.md.

Holds the propensity at the true generating surface (via
sim_oracle.compute_oracle_propensity_for_tile) and sweeps a single scalar
log_r over a log grid. For each r value, computes the mean NB-offset loss
over all 1600 val pairs using the FROZEN core loss
(MaskedNegativeBinomialOffsetNLLLoss), configured from
``scripts._oracle_scoring.ORACLE_LOSS_KWARGS`` and
``ORACLE_DISPERSION_WINDOW_SIZE``. The values are deliberately NOT restated
here: they had been transcribed into this docstring and four other places,
which is the duplication this refactor exists to remove.

Three outcomes (per §4 of the design):
  - r far above 7.18 (>500), minimum below untrained models → premise confirmed
  - r intermediate (20–500) → plateau begins; r not identified if loss is flat
    to within numerical noise (see §9.1 of the design doc)
  - r at or near 7.18 → premise FALSE, STOP

Reference values:
  - claimed-bad oracle:    4.115647  (oracle_nb.json, r = true hexamer_r ≈ 7.18)
  - uniform:               4.204456
  - untrained KEN val_loss:  4.093257
  - untrained hybrid val_loss: 4.108722
  - true hexamer_r median: 7.179
  - frozen models use log_dispersion_init = 7.0 → r ≈ 1096

Usage:
    cd <repo>
    PYTHONPATH=. /home/nathanboley/miniconda3/envs/biomarker_env/bin/python \
        scripts/nb_oracle_sweep.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

# ── Paths ─────────────────────────────────────────────────────────────────
STORE = "/efs/analytics/nathanboley/background_model/simulation_v3_nb/stores/sim_store_v3nb_A.zarr"
SIM_DIR = "/efs/analytics/nathanboley/background_model/simulation_v3_nb/A"
FASTA = "/efs/analytics/nathanboley/data_resources/genome/hg38.fa"


def main():
    t0 = time.time()

    # ── Load ground truth ──────────────────────────────────────────────────
    import pysam
    import zarr
    from background_model.config import PlumbingConfig
    from background_model.dataset import BackgroundTileDataset
    from scripts._oracle_scoring import eval_loss_at_log_r, make_oracle_loss_fn
    from scripts.sim_fragments import GCBias2D, MAX_LEN
    from scripts.sim_oracle import compute_oracle_propensity_for_tile

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
    print(f"[sweep] {len(ds)} val pairs, crop [{crop_start},{crop_stop}) "
          f"({time.time()-t0:.1f}s)", flush=True)

    tile_to_pairs = {}
    for di in range(len(ds)):
        s_idx, t_idx = ds.index[di]
        tile_to_pairs.setdefault(t_idx, []).append((di, s_idx))

    fa = pysam.FastaFile(FASTA)

    # ── Pre-compute oracle propensity logits for all val pairs ─────────────
    # Store them so we only compute propensity once, then sweep r values.
    print("[sweep] Pre-computing oracle propensity logits...", flush=True)
    pair_logits = {}  # di -> (logits_tensor, y_tensor, mask_tensor)

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
            logits = torch.from_numpy(
                np.where(p > 0, np.log(p), -1e30)
            ).float().unsqueeze(0)
            y_t = y.unsqueeze(0)
            m_t = mask.unsqueeze(0)
            pair_logits[di] = (logits, y_t, m_t)

        if (tn + 1) % 100 == 0 or tn == 0:
            print(f"  {tn+1}/{len(tile_to_pairs)} tiles ({time.time()-t0:.1f}s)",
                  flush=True)

    fa.close()
    n_pairs = len(pair_logits)
    print(f"[sweep] Cached {n_pairs} pairs ({time.time()-t0:.1f}s)", flush=True)

    # ── Sweep scalar r ─────────────────────────────────────────────────────
    # Log grid from r=1 to r=1e5, 30 points (includes region around 7.18,
    # region around 1096, and above)
    log_r_values = np.linspace(np.log(1.0), np.log(1e5), 30)
    # Also include exact reference points
    exact_points = [np.log(7.179), np.log(1096.0)]
    log_r_values = np.sort(np.unique(np.concatenate([log_r_values, exact_points])))

    loss_fn = make_oracle_loss_fn()

    print(f"\n[sweep] Sweeping {len(log_r_values)} log_r values...", flush=True)
    print(f"{'log_r':>10s} {'r':>12s} {'mean_loss':>12s}")
    print(f"{'-'*10} {'-'*12} {'-'*12}")

    results = []
    for log_r_val in log_r_values:
        r_val = np.exp(log_r_val)
        mean_loss = eval_loss_at_log_r(float(log_r_val), pair_logits, loss_fn)
        results.append((float(log_r_val), float(r_val), mean_loss))
        print(f"{log_r_val:10.4f} {r_val:12.4f} {mean_loss:12.6f}")

    # ── Find minimum ───────────────────────────────────────────────────────
    losses_arr = np.array([r[2] for r in results])
    log_r_arr = np.array([r[0] for r in results])
    r_arr = np.array([r[1] for r in results])

    best_idx = np.argmin(losses_arr)
    best_log_r = log_r_arr[best_idx]
    best_r = r_arr[best_idx]
    best_loss = losses_arr[best_idx]

    print(f"\n{'='*60}")
    print(f"STEP 0 RESULTS")
    print(f"{'='*60}")
    print(f"  Best log_r:      {best_log_r:.4f}")
    print(f"  Best r:          {best_r:.4f}")
    print(f"  Best loss:       {best_loss:.6f}")
    print(f"")
    print(f"  Reference points:")
    print(f"    true hexamer_r median (7.179):  log_r={np.log(7.179):.4f}")
    print(f"    frozen model init (1096):       log_r={np.log(1096):.4f}")
    print(f"    claimed-bad oracle:             4.115647")
    print(f"    untrained KEN val_loss:         4.093257")
    print(f"    untrained hybrid val_loss:      4.108722")
    print(f"    uniform:                        4.204456")
    print(f"")

    # Assess monotonicity / smoothness
    diffs = np.diff(losses_arr)
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    print(f"  Curve smoothness: {sign_changes} sign changes in slope "
          f"(0 = monotone after min, low = smooth)")

    # Verdict
    if best_r > 500:
        verdict = "CONFIRMED"
        detail = (f"r={best_r:.1f} >> 7.18, and best_loss={best_loss:.6f} — "
                  f"compare untrained KEN 4.093257, hybrid 4.108722")
        if best_loss < 4.093257:
            detail += " → minimum BELOW both untrained models. §2.3 confirmed."
        else:
            detail += " → minimum NOT below untrained models. INVESTIGATE."
    elif best_r > 20:
        verdict = "PLATEAU"
        detail = (f"r={best_r:.1f} is on the plateau (>20). The offset conditioning "
                  f"absorbs the overdispersion; r is not identified. "
                  f"Proceed with anchor A.")
    else:
        verdict = "REFUTED"
        detail = (f"r={best_r:.1f} is near true r ≈ 7.18. §2.3 is FALSE. "
                  f"STOP — do not proceed to step 1.")

    print(f"\n  VERDICT: {verdict}")
    print(f"  {detail}")
    print(f"\n  Runtime: {time.time()-t0:.1f}s")

    # Print the full curve for reporting
    print(f"\n{'='*60}")
    print(f"FULL SWEPT CURVE")
    print(f"{'='*60}")
    print(f"{'log_r':>10s} {'r':>12s} {'loss':>12s} {'note':>20s}")
    print(f"{'-'*10} {'-'*12} {'-'*12} {'-'*20}")
    for log_r_val, r_val, loss in results:
        note = ""
        if abs(r_val - 7.179) < 0.01:
            note = "<-- true hexamer_r"
        elif abs(r_val - 1096.0) < 1.0:
            note = "<-- frozen init"
        elif abs(log_r_val - best_log_r) < 0.001:
            note = "<-- MINIMUM"
        print(f"{log_r_val:10.4f} {r_val:12.4f} {loss:12.6f} {note:>20s}")

    return 0 if verdict != "REFUTED" else 1


if __name__ == "__main__":
    sys.exit(main())
