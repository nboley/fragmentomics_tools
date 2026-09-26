#!/usr/bin/env python
"""Oracle NLL for the v3 simulation: the theoretical best any model can achieve.

Computes the true per-position endpoint propensity from ground-truth w6 +
GC bias + per-sample FL distributions, then evaluates multinomial NLL of the
actual validation counts against this propensity using MaskedMultinomialNLLLoss.

The KEY correctness requirement is coordinate alignment:
  - Store tiles span l_target = tile_size + 2*jitter = 2304 positions.
  - The BackgroundTileDataset center-crops to tile_size=2048 (positions [128, 2176)
    within the l_target extent) for val (jitter=0).
  - The oracle propensity MUST be computed over the FULL l_target extent and
    then center-cropped identically, so that propensity[i] matches counts[i].

The previous compute_true_propensity had a 128-position offset: it accumulated
into a (C, tile_size=2048) array clipping to [0, 2048), but the counts come from
positions [128, 2176). This script fixes that.

Usage:
    cd /home/nathanboley/src/fragmentomics_tools
    PYTHONPATH=. /home/nathanboley/miniconda3/envs/biomarker_env/bin/python \
        scripts/sim_oracle.py \
        --sim-dir /efs/analytics/nathanboley/background_model/simulation_v3/A \
        --store /efs/analytics/nathanboley/background_model/simulation_v3/stores/sim_store_v3_A.zarr
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

FASTA = "/efs/analytics/nathanboley/data_resources/genome/hg38.fa"

from background_model.tracks import (
    COVERAGE_TYPES,
    FL_BANDS,
    N_TRACKS,
    STRANDS,
    TRACK_INDEX,
)


def compute_oracle_propensity_for_tile(
    contig, gstart, gstop, l_target, w6, gcbias, len_vals,
    len_p_per_sample, sample_idxs, fa, gc_lut=None,
):
    """Compute per-sample propensity over the full l_target extent for one tile.

    For Regime A (shared w6), the only per-sample variation is len_p.
    Strategy: for each fragment length L, compute a shared per-position delta
    (hexamer * GC weights) via one np.add.at, then scale by each sample's
    len_p[L] — avoids n_samples inner scatter calls.

    Args:
        sample_idxs: array of sample indices to compute for.
        len_p_per_sample: (n_samples_total, n_lengths) FL distributions.
        gc_lut: optional padded GC lookup table from
            ``GCBias2D.build_lookup_table``.  scripts/sim_fragments.py DREW the
            fragments through this table (integer-rounded GC%), so passing it
            reproduces the generative weights exactly; passing None uses the
            exact bilinear surface instead (a slightly different propensity).

    Returns: prop  shape (len(sample_idxs), C, l_target), unnormalized.
    """
    from scripts.sim_fragments import hexamer_indices, HEX_HALF, GCBias2D

    region_len = gstop - gstart  # == l_target
    assert region_len == l_target

    # Fetch sequence with hexamer margin
    seq = fa.fetch(contig, gstart - HEX_HALF, gstop + HEX_HALF).upper()
    seq_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    fwd_cut, rc_cut, valid = hexamer_indices(seq_bytes)

    # Cumulative GC over the region
    core = seq_bytes[HEX_HALF:HEX_HALF + region_len]
    is_gc = (core == ord("G")) | (core == ord("C"))
    cum_gc = np.concatenate([[0], np.cumsum(is_gc)]).astype(np.int64)

    # Pre-compute w6 weights at every cut position
    lw_all = w6[fwd_cut]    # (region_len+1,)
    rw_all = w6[rc_cut]     # (region_len+1,)

    # Collect per-length deltas: for each length L, build a (C, l_target)
    # contribution from fragment placements (shared across samples), then
    # scale by per-sample len_p[L] and accumulate.
    n_out = len(sample_idxs)
    prop = np.zeros((n_out, N_TRACKS, l_target), dtype=np.float64)
    # len_p subset for our sample indices: (n_out, n_lengths)
    lp_sub = len_p_per_sample[sample_idxs]

    for li, L_val in enumerate(len_vals):
        L = int(L_val)
        # Skip if no sample has weight at this length
        lp_col = lp_sub[:, li]  # (n_out,)
        if lp_col.max() < 1e-12:
            continue

        # Which FL band(s)?
        fl_match = [(lo, hi) for lo, hi in FL_BANDS if lo <= L < hi]
        if not fl_match:
            continue

        max_p = region_len - L
        if max_p < 0:
            continue

        ok = valid[:max_p + 1] & valid[L:L + max_p + 1]
        ps = np.nonzero(ok)[0]
        if len(ps) == 0:
            continue
        qs = ps + L

        gc_pct = 100.0 * (cum_gc[qs] - cum_gc[ps]) / L
        if gc_lut is not None:
            gc_w = gc_lut[L, np.rint(gc_pct).astype(np.intp) + GCBias2D._GC_PAD]
        else:
            gc_w = gcbias(L, gc_pct)
        frag_w = lw_all[ps] * rw_all[qs] * gc_w  # shared

        mid = (ps + qs) // 2

        # Build per-track delta for this length (shared across samples)
        # IMPORTANT: "first" = starts_0 (left end) and "last" = stops_0-1
        # (right end) for ALL strands — the RFA's first/last_covered_base
        # are strand-INDEPENDENT.  Strand assignment is 50/50, so + and -
        # tracks get identical endpoint distributions.
        first_pos = ps       # starts_0 = left endpoint
        last_pos = qs - 1    # stops_0 - 1 = right endpoint

        delta = np.zeros((N_TRACKS, l_target), dtype=np.float64)
        for fl_lo, fl_hi in fl_match:
            for strand in STRANDS:
                for cov_type, endpoints in [
                    ("first", first_pos), ("last", last_pos),
                    ("midpoint", mid),
                ]:
                    track = TRACK_INDEX[(strand, (fl_lo, fl_hi), cov_type)]
                    in_bounds = (endpoints >= 0) & (endpoints < l_target)
                    if not in_bounds.any():
                        continue
                    np.add.at(delta[track], endpoints[in_bounds],
                              frag_w[in_bounds] * 0.5)

        # Scale by per-sample len_p and accumulate
        # prop[s] += lp_col[s] * delta  for each sample s
        for si in range(n_out):
            if lp_col[si] >= 1e-12:
                prop[si] += lp_col[si] * delta

    return prop


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-dir", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None,
                    help="write results + provenance to this JSON path")
    ap.add_argument("--gc-lut", action="store_true",
                    help="use the simulator's integer-GC lookup table for the "
                         "GC weight (reproduces the generative weights exactly) "
                         "instead of the exact bilinear surface")
    args = ap.parse_args()

    t0 = time.time()

    # ── Load ground truth ──────────────────────────────────────────────────
    gt = np.load(os.path.join(args.sim_dir, "ground_truth.npz"), allow_pickle=True)
    w6 = gt["w6"]
    len_vals = gt["len_vals"]
    len_p_per_sample = gt["len_p_per_sample"]  # (n_samples_total, n_lengths)

    from scripts.sim_fragments import GCBias2D, MAX_LEN
    gcbias = GCBias2D(gt["bias_lengths"], gt["bias_gc_percents"], gt["bias_grid"])
    gc_lut = gcbias.build_lookup_table(max_len=MAX_LEN) if args.gc_lut else None

    print(f"[oracle] loaded ground truth ({time.time()-t0:.1f}s), "
          f"gc_mode={'lut' if args.gc_lut else 'bilinear'}", flush=True)

    # ── Load store metadata ────────────────────────────────────────────────
    import zarr
    from background_model.config import PlumbingConfig

    root = zarr.open_group(args.store, mode="r")
    cfg = PlumbingConfig.from_json(root.attrs["config_json"])
    tile_size = cfg.tile_size
    l_target = cfg.l_target
    jitter = cfg.jitter

    contigs = root["tiles/contig"][:]
    starts = root["tiles/start"][:]
    stops = root["tiles/stop"][:]
    splits = root["tiles/split"][:]
    roles = root["samples/role"][:]

    # Train sample indices (role == 0)
    train_sample_idxs = np.nonzero(roles == 0)[0]
    n_samples_train = len(train_sample_idxs)

    # Val tile indices (split == 1)
    val_tile_idxs = np.nonzero(splits == 1)[0]
    n_val_tiles = len(val_tile_idxs)

    print(f"[oracle] store: tile_size={tile_size}, l_target={l_target}, jitter={jitter}")
    print(f"[oracle] {n_samples_train} train samples, {n_val_tiles} val tiles")
    print(f"[oracle] total val pairs: {n_samples_train * n_val_tiles}")

    # Center-crop offset (jitter=0 in val mode)
    crop_start = (l_target - tile_size) // 2  # = jitter = 128
    crop_stop = crop_start + tile_size
    print(f"[oracle] center-crop: [{crop_start}, {crop_stop}) within l_target={l_target}")

    # ── Create dataset to get correctly-cropped counts and masks ───────────
    from background_model.dataset import BackgroundTileDataset

    # We need model_input_size to create the dataset. Use a reasonable value.
    # The model_input_size must satisfy: l_seq >= model_input_size + 2*jitter
    # l_seq = tile_size + 2*(jitter + rf_budget) = 2048 + 2*(128 + 2048) = 6400
    # For val (jitter=0 in the dataset), we just need model_input_size <= l_seq
    # The actual value doesn't matter for y and mask — only x depends on it.
    # Use tile_size as model_input_size (we won't use x).
    model_input_size = tile_size

    ds = BackgroundTileDataset(
        store_path=args.store,
        model_input_size=model_input_size,
        split="val",
        sample_role="train",
        min_N=0,
        train_mode=False,
        seed=1337,
    )

    print(f"[oracle] dataset: {len(ds)} val pairs ({time.time()-t0:.1f}s)", flush=True)

    # Build lookup: tile_idx -> list of (dataset_index, sample_idx)
    tile_to_pairs = {}
    for di in range(len(ds)):
        s_idx, t_idx = ds.index[di]
        if t_idx not in tile_to_pairs:
            tile_to_pairs[t_idx] = []
        tile_to_pairs[t_idx].append((di, s_idx))

    # ── Compute oracle NLL ─────────────────────────────────────────────────
    import pysam
    from background_model_core import MaskedMultinomialNLLLoss

    fa = pysam.FastaFile(FASTA)
    loss_fn = MaskedMultinomialNLLLoss()

    oracle_nlls = []
    uniform_nlls = []
    n_pairs = 0

    for tile_num, t_idx in enumerate(sorted(tile_to_pairs.keys())):
        contig = str(contigs[t_idx])
        gstart = int(starts[t_idx])
        gstop = int(stops[t_idx])

        pairs = tile_to_pairs[t_idx]
        # Which train samples appear for this tile?
        sample_idxs_in_pairs = [s_idx for _, s_idx in pairs]

        # Which sample indices appear for this tile?
        unique_sidxs = sorted(set(s_idx for _, s_idx in pairs))
        sidx_to_local = {s: i for i, s in enumerate(unique_sidxs)}

        # Compute propensity for those samples at this tile
        prop_full = compute_oracle_propensity_for_tile(
            contig, gstart, gstop, l_target, w6, gcbias, len_vals,
            len_p_per_sample, np.array(unique_sidxs), fa, gc_lut=gc_lut,
        )
        # prop_full: (n_unique_samples, C, l_target)

        # Center-crop propensity to match dataset output
        prop_cropped = prop_full[:, :, crop_start:crop_stop]
        # prop_cropped: (n_unique_samples, C, tile_size)

        # Evaluate NLL for each (sample, tile) pair
        for di, s_idx in pairs:
            _, y, mask = ds[di]
            # y: (C, tile_size), mask: (tile_size,)

            prop_s = prop_cropped[sidx_to_local[s_idx]]  # (C, tile_size)

            # Oracle NLL: use log(propensity) as logits
            # The loss applies log_softmax which normalizes, so unnormalized is fine.
            # But we need to handle zeros: positions with zero propensity should
            # get -inf logits (correctly excluded by the loss).
            with torch.no_grad():
                logits = torch.from_numpy(
                    np.where(prop_s > 0, np.log(prop_s), -1e30)
                ).float().unsqueeze(0)  # (1, C, tile_size)
                y_t = y.unsqueeze(0)    # (1, C, tile_size)
                m_t = mask.unsqueeze(0) # (1, tile_size)

                oracle_nll = loss_fn(logits, y_t, m_t).item()

                # Uniform baseline: all logits = 0 (uniform over unmasked positions)
                uniform_logits = torch.zeros_like(logits)
                uniform_nll = loss_fn(uniform_logits, y_t, m_t).item()

            oracle_nlls.append(oracle_nll)
            uniform_nlls.append(uniform_nll)
            n_pairs += 1

        if (tile_num + 1) % 50 == 0 or tile_num == 0:
            print(f"[oracle] processed {tile_num+1}/{len(tile_to_pairs)} tiles, "
                  f"{n_pairs} pairs  "
                  f"oracle={np.mean(oracle_nlls):.4f}  "
                  f"uniform={np.mean(uniform_nlls):.4f}  "
                  f"({time.time()-t0:.1f}s)", flush=True)

    fa.close()

    # ── Report ─────────────────────────────────────────────────────────────
    oracle_mean = float(np.mean(oracle_nlls))
    uniform_mean = float(np.mean(uniform_nlls))
    gap = uniform_mean - oracle_mean

    print(f"\n{'='*60}")
    print(f"Oracle NLL evaluation ({n_pairs} val pairs)")
    print(f"{'='*60}")
    print(f"  Oracle NLL:   {oracle_mean:.6f}")
    print(f"  Uniform NLL:  {uniform_mean:.6f}")
    print(f"  Gap (uniform - oracle): {gap:.6f}")
    print(f"  Oracle < uniform? {oracle_mean < uniform_mean}")
    print(f"{'='*60}")
    print(f"  Runtime: {time.time()-t0:.1f}s")

    if args.out:
        import datetime
        import json
        import subprocess

        try:
            sha = subprocess.run(
                ["git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            sha = "unknown"

        payload = {
            "oracle_nll": oracle_mean,
            "uniform_nll": uniform_mean,
            "gap_uniform_minus_oracle": gap,
            "loss": "MaskedMultinomialNLLLoss (background_model_core, frozen)",
            "reduction": "mean over (pair, track); per-track NLL divided by that "
                         "track's total count N (clamped to >=1), so a "
                         "zero-count track contributes exactly 0",
            "gc_weight_mode": "lut" if args.gc_lut else "bilinear",
            "store": args.store,
            "sim_dir": args.sim_dir,
            "fasta": FASTA,
            "store_config_hash": root.attrs["config_hash"],
            "store_split_version": int(root.attrs["split_version"]),
            "split": "tiles/split == 1 (val) x samples/role == 0 (train)",
            "dataset_kwargs": {
                "split": "val", "sample_role": "train", "min_N": 0,
                "train_mode": False, "seed": 1337,
                "model_input_size": int(model_input_size),
            },
            "n_val_pairs": int(n_pairs),
            "n_val_tiles": int(n_val_tiles),
            "n_train_samples": int(n_samples_train),
            "tile_size": int(tile_size),
            "l_target": int(l_target),
            "jitter": int(jitter),
            "crop": [int(crop_start), int(crop_stop)],
            "git_sha": sha,
            "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "python": sys.executable,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[oracle] wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
