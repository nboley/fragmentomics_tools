#!/usr/bin/env python
"""Evaluate 12-track background models on the per-fragment NLL metric.

Loads a trained 12-track model checkpoint and a cut-site fragment store,
then for each val region derives per-fragment log-probabilities from
the model's endpoint shape predictions and reports val_mult_nll — the
same metric the cut-site fragment model optimises.

Fragment log-probability from a 12-track model
-----------------------------------------------
The 12-track model predicts per-position probabilities for each of 12
tracks (2 strands × 2 FL bands × 3 coverage types).  For a fragment
(centre m, length L, strand s):

    score(m, L, s) = p_first[first_pos | s, band(L)]
                   × p_last[last_pos  | s, band(L)]
                   × p_marginal(L)

    p(m, L, s) = score(m, L, s) / Z

where:
  - p_first / p_last come from the masked softmax of the shape_head's
    "first" and "last" coverage-type tracks for (strand s, FL band of L)
  - p_marginal(L) is the empirical length distribution (global over val)
  - Z normalises over D = {(m, L, s) : both endpoints in [0, P),
    L ∈ [L_min, L_max], s ∈ {0, 1}}

Strand/endpoint mapping:
  +strand: first (5') at start=p,      last (3') at p+L-1
  −strand: first (5') at p+L-1,        last (3') at p

Usage::

    python scripts/eval_fragment_nll.py \\
        --checkpoint /efs/.../best.ckpt \\
        --model ken \\
        --store /efs/.../quiet_tile1536.zarr
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import zarr

from background_model.constants import L_MAX, L_MIN
from background_model.tracks import FL_BANDS, TRACK_INDEX
from background_model_core import (
    BackgroundModel,
    BackgroundModelHybrid,
    BackgroundModelKEN,
    calculate_start_and_stop_from_jitter,
    jitter_matrix,
)
from background_model.cut_site_store import CutSiteStoreConfig
from fragmentomics_tools.region import one_hot_encode_sequences


MODEL_CLASSES = {
    "cnn": BackgroundModel,
    "ken": BackgroundModelKEN,
    "hybrid": BackgroundModelHybrid,
}

_SPLIT_CODES = {"train": 0, "val": 1, "heldout_inactive": 2, "positive_control": 3}
_ROLE_CODES = {"train": 0, "heldout": 1, "dropped_low_depth": 2}

# L_MIN / L_MAX are the shared inclusive bounds from background_model.constants
# (owner decision 185; this script used the half-open max(hi) - 1 = 179 before).
# L_MAX = 180 is admitted by the cut-site store and model but lies in NO
# half-open FL band, so a 12-track model has no track to score it with.  How
# such a fragment should be scored is an open owner decision; until it is made,
# _eval_region refuses any admitted fragment that falls in no band rather than
# score it with uninitialised memory.


def _build_endpoint_map():
    """Return dict: (strand_idx, band_idx) -> (first_track, last_track)."""
    strands = ("+", "-")
    mapping = {}
    for si, s in enumerate(strands):
        for bi, fl in enumerate(FL_BANDS):
            mapping[(si, bi)] = (
                TRACK_INDEX[(s, fl, "first")],
                TRACK_INDEX[(s, fl, "last")],
            )
    return mapping


ENDPOINT_MAP = _build_endpoint_map()


def _compute_global_p_marginal(index, n_tiles, mem_frag_indptr, mem_frag_length):
    """Compute global empirical p(L) from all indexed fragments."""
    counts = np.zeros(L_MAX - L_MIN + 1, dtype=np.float64)
    for s, t in index:
        u = s * n_tiles + t
        flo = int(mem_frag_indptr[u])
        fhi = int(mem_frag_indptr[u + 1])
        lengths = mem_frag_length[flo:fhi].astype(np.int64)
        valid = (lengths >= L_MIN) & (lengths <= L_MAX)
        if valid.any():
            np.add.at(counts, lengths[valid] - L_MIN, 1)
    total = counts.sum()
    if total == 0:
        return np.ones_like(counts) / len(counts)
    return counts / total


def _compute_log_Z(probs, p_marginal, P):
    """log Z = log sum_{m,L,s} p_first[f] × p_last[l] × p_marginal(L).

    Vectorised over m for each (s, band, L).
    """
    Z = 0.0
    for s_idx in (0, 1):
        for b_idx, (lo, hi) in enumerate(FL_BANDS):
            first_t, last_t = ENDPOINT_MAP[(s_idx, b_idx)]
            p_first = probs[first_t]  # (P,)
            p_last = probs[last_t]

            for L in range(max(L_MIN, lo), min(L_MAX + 1, hi)):
                pm = p_marginal[L - L_MIN]
                if pm <= 0:
                    continue
                max_start = P - L  # inclusive upper bound on start p
                if max_start < 0:
                    continue

                if s_idx == 0:  # +: first at p, last at p+L-1
                    Z += pm * np.dot(
                        p_first[: max_start + 1],
                        p_last[L - 1 : L + max_start],
                    )
                else:  # −: first at p+L-1, last at p
                    Z += pm * np.dot(
                        p_first[L - 1 : L + max_start],
                        p_last[: max_start + 1],
                    )

    return np.log(max(Z, 1e-300))


def _eval_region(
    probs_np,
    mask_np,
    frags_start,
    frags_length,
    frags_strand,
    tile_size,
    l_target,
    p_marginal,
):
    """Compute fragment NLL for one (sample, region) pair.

    Returns (per_region_nll, n_valid) — the per-fragment mean NLL for
    this region, and the number of valid fragments used.
    """
    P = tile_size

    # --- Fragment coordinates in crop frame ---
    crop_start, _ = calculate_start_and_stop_from_jitter(l_target, tile_size, 0)
    frag_p = frags_start.astype(np.int64) - crop_start
    frag_L = frags_length.astype(np.int64)
    frag_s = frags_strand.astype(np.int64)
    frag_end = frag_p + frag_L - 1

    # Keep fragments whose centre is in [0, P) AND both endpoints in [0, P)
    frag_m = frag_p + frag_L // 2
    keep = (
        (frag_m >= 0)
        & (frag_m < P)
        & (frag_L >= L_MIN)
        & (frag_L <= L_MAX)
        & (frag_p >= 0)
        & (frag_end < P)
    )
    frag_p = frag_p[keep]
    frag_L = frag_L[keep]
    frag_s = frag_s[keep]
    frag_end = frag_end[keep]

    n_frag = len(frag_p)
    if n_frag == 0:
        return None, 0

    # Log-probs from the 12-track model (clamp for safety)
    log_probs_np = np.log(np.maximum(probs_np, 1e-30))  # (C, P)

    # log Z for this region
    log_Z = _compute_log_Z(probs_np, p_marginal, P)

    # Band index per fragment
    band_idx = np.full(n_frag, -1, dtype=np.int64)
    for bi, (lo, hi) in enumerate(FL_BANDS):
        band_idx[(frag_L >= lo) & (frag_L < hi)] = bi

    # log_frag_p is np.empty below and only band members are written, so an
    # admitted fragment in no band would contribute uninitialised memory.
    unbanded = band_idx < 0
    if unbanded.any():
        raise ValueError(
            f"{int(unbanded.sum())} admitted fragment(s) fall in no FL band "
            f"(lengths {sorted(set(frag_L[unbanded].tolist()))}; bands "
            f"{FL_BANDS} are half-open, L_MAX={L_MAX} is inclusive).  Scoring "
            f"them needs an owner decision; see the note at L_MIN / L_MAX."
        )

    # Vectorised per-(strand, band) group
    log_frag_p = np.empty(n_frag, dtype=np.float64)
    for s_idx in (0, 1):
        for b_idx in range(len(FL_BANDS)):
            sel = (frag_s == s_idx) & (band_idx == b_idx)
            if not sel.any():
                continue

            first_t, last_t = ENDPOINT_MAP[(s_idx, b_idx)]
            p_sel = frag_p[sel]
            end_sel = frag_end[sel]
            L_sel = frag_L[sel]

            if s_idx == 0:  # +: first (5') at p, last (3') at end
                f_pos = p_sel
                l_pos = end_sel
            else:  # −: first (5') at end, last (3') at p
                f_pos = end_sel
                l_pos = p_sel

            L_idx = L_sel - L_MIN
            log_p_L = np.log(np.maximum(p_marginal[L_idx], 1e-30))

            log_frag_p[sel] = (
                log_probs_np[first_t, f_pos]
                + log_probs_np[last_t, l_pos]
                + log_p_L
                - log_Z
            )

    per_region_nll = -log_frag_p.sum() / n_frag
    return float(per_region_nll), n_frag


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="path to trained 12-track model checkpoint (.ckpt)",
    )
    p.add_argument(
        "--model",
        choices=list(MODEL_CLASSES),
        required=True,
        help="model architecture (cnn, ken, hybrid)",
    )
    p.add_argument(
        "--store",
        required=True,
        help="path to cut-site fragment store (.zarr)",
    )
    p.add_argument(
        "--split",
        default="val",
        choices=sorted(_SPLIT_CODES),
        help="which split to evaluate (default: val)",
    )
    p.add_argument(
        "--sample-role",
        default="train",
        choices=sorted(_ROLE_CODES),
    )
    p.add_argument(
        "--min-N",
        type=int,
        default=10,
        help="min fragments per region to include",
    )
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    # --- Load model ---
    print(f"Loading {args.model} model from {args.checkpoint}")
    cls = MODEL_CLASSES[args.model]
    model = cls.load_from_checkpoint(args.checkpoint, map_location=args.device)
    model.eval()
    model.to(args.device)

    # --- Open store and read geometry ---
    root = zarr.open_group(args.store, mode="r")
    config = CutSiteStoreConfig.from_json(root.attrs["config_json"])
    tile_size = config.tile_size
    l_target = config.l_target
    l_seq = config.l_seq

    model_input_size = model.calc_input_region_size(tile_size)
    print(f"Tile size: {tile_size}, Model input: {model_input_size}, "
          f"Store l_seq: {l_seq}")

    if model_input_size > l_seq:
        print(
            f"ERROR: model needs {model_input_size} bp input but store "
            f"sequences are only {l_seq} bp"
        )
        sys.exit(1)
    # jitter_matrix requires matched parity
    if model_input_size % 2 != l_seq % 2:
        print(
            f"ERROR: parity mismatch — model_input_size={model_input_size} "
            f"vs l_seq={l_seq}"
        )
        sys.exit(1)

    # --- Build (sample, tile) index ---
    split_arr = root["tiles/split"][:]
    role_arr = root["samples/role"][:]
    frag_indptr = root["fragments/indptr"][:]
    n_tiles = split_arr.shape[0]

    split_code = _SPLIT_CODES[args.split]
    role_code = _ROLE_CODES[args.sample_role]
    sample_ok = np.nonzero(role_arr == role_code)[0]
    tile_ok = np.nonzero(split_arr == split_code)[0]

    index = []
    for s in sample_ok:
        for t in tile_ok:
            u = int(s) * n_tiles + int(t)
            n_frags = int(frag_indptr[u + 1] - frag_indptr[u])
            if n_frags >= args.min_N:
                index.append((int(s), int(t)))

    print(
        f"Evaluating {len(index)} (sample, region) pairs "
        f"({len(sample_ok)} samples × {len(tile_ok)} tiles, "
        f"min_N={args.min_N})"
    )
    if not index:
        print("No valid pairs found.")
        sys.exit(0)

    # --- Preload arrays ---
    mem_frag_indptr = np.asarray(frag_indptr)
    mem_frag_start = np.asarray(root["fragments/start"][:])
    mem_frag_length = np.asarray(root["fragments/length"][:])
    mem_frag_strand = np.asarray(root["fragments/strand"][:])
    mem_mask = np.asarray(root["tiles/mask"][:])
    mem_seq = np.asarray(root["tiles/seq"][:])

    # --- Global empirical length distribution ---
    print("Computing global p(L) from val fragments ...")
    p_marginal = _compute_global_p_marginal(
        index, n_tiles, mem_frag_indptr, mem_frag_length
    )

    # --- Evaluate ---
    per_region_nlls = []
    total_frags = 0

    for i, (s, t) in enumerate(index):
        u = s * n_tiles + t

        # Fragments
        flo = int(mem_frag_indptr[u])
        fhi = int(mem_frag_indptr[u + 1])
        frags_start = mem_frag_start[flo:fhi]
        frags_length = mem_frag_length[flo:fhi]
        frags_strand = mem_frag_strand[flo:fhi]

        # Sequence → one-hot for the 12-track model
        seq_full = mem_seq[t]  # (l_seq,)
        x_tokens = np.ascontiguousarray(
            jitter_matrix(seq_full, 0, model_input_size), dtype=np.uint8
        )
        onehot = one_hot_encode_sequences([x_tokens.tobytes()])[0]
        x_np = np.ascontiguousarray(onehot.T, dtype=np.float32)  # (4, input_size)

        # Mask
        mask_full = mem_mask[t]  # (l_target,)
        mask_np = np.asarray(jitter_matrix(mask_full, 0, tile_size), dtype=bool)

        # Model forward pass
        x = torch.from_numpy(x_np).unsqueeze(0).to(args.device)
        with torch.no_grad():
            shape_logits, _ = model(x)  # (1, C, P)

        # Masked softmax → probabilities
        sl = shape_logits[0]  # (C, P)
        mask_t = torch.from_numpy(mask_np).to(args.device)
        sl = sl.masked_fill(~mask_t.unsqueeze(0), float("-inf"))
        probs_np = torch.softmax(sl, dim=-1).cpu().numpy().astype(np.float64)

        # Fragment NLL
        nll, n_valid = _eval_region(
            probs_np,
            mask_np,
            frags_start,
            frags_length,
            frags_strand,
            tile_size,
            l_target,
            p_marginal,
        )

        if nll is not None:
            per_region_nlls.append(nll)
            total_frags += n_valid

        if (i + 1) % 200 == 0 or i == len(index) - 1:
            if per_region_nlls:
                running = np.mean(per_region_nlls)
                print(
                    f"  [{i + 1}/{len(index)}] val_mult_nll = {running:.6f} "
                    f"({total_frags} frags, {len(per_region_nlls)} regions)"
                )

    if not per_region_nlls:
        print("No valid fragments found.")
        sys.exit(0)

    mean_nll = np.mean(per_region_nlls)
    print(f"\n{'=' * 60}")
    print(f"Model:   {args.model} from {os.path.basename(args.checkpoint)}")
    print(f"Store:   {args.store}")
    print(f"Split:   {args.split}")
    print(f"Regions: {len(per_region_nlls)}")
    print(f"Frags:   {total_frags}")
    print(f"val_mult_nll = {mean_nll:.6f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
