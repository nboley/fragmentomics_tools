#!/usr/bin/env python
"""Stage 1: Compute empirical hexamer cut-site tables and evaluate baseline NLL.

Three outputs:
1. Empirical hex tables (4 × 4096): start_fwd, stop_fwd, start_rev, stop_rev
2. Marginal fragment length distribution p(L)
3. Baseline fragment NLL from the factored model:
       log p₀(m, L, s) = log p_start + log p_stop + log p(L) - log Z

Also trains a linear Conv1d(4, 1, k=6) to verify that gradient-based
optimisation recovers the same hex table values.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
import torch.nn as nn
import zarr

from background_model.cut_site_model import CUT_OFFSET_IN_KMER
from background_model.cut_site_store import CutSiteStoreConfig, validate_length_range
from background_model_core import calculate_start_and_stop_from_jitter, jitter_matrix
from fragmentomics_tools.region import one_hot_encode_sequences

_SPLIT_CODES = {"train": 0, "val": 1, "heldout_inactive": 2}


_ASCII_TO_BASE = np.full(256, 255, dtype=np.uint8)
_ASCII_TO_BASE[ord('A')] = 0
_ASCII_TO_BASE[ord('C')] = 1
_ASCII_TO_BASE[ord('G')] = 2
_ASCII_TO_BASE[ord('T')] = 3
# Also handle lowercase
_ASCII_TO_BASE[ord('a')] = 0
_ASCII_TO_BASE[ord('c')] = 1
_ASCII_TO_BASE[ord('g')] = 2
_ASCII_TO_BASE[ord('t')] = 3
# Handle 0-3 encoding directly
_ASCII_TO_BASE[0] = 0
_ASCII_TO_BASE[1] = 1
_ASCII_TO_BASE[2] = 2
_ASCII_TO_BASE[3] = 3


def _seq_to_hexamer_indices(seq_uint8, k=6):
    """Convert a (L,) uint8 sequence to (L-k+1,) hexamer indices in [0, 4^k).

    Handles both ASCII (A=65,C=67,G=71,T=84) and numeric (0-3) encodings.
    Returns -1 for hexamers containing N or other non-ACGT bases.
    """
    bases = _ASCII_TO_BASE[seq_uint8]  # 255 for unknown
    powers = 4 ** np.arange(k, dtype=np.int64)
    n = len(seq_uint8) - k + 1
    indices = np.zeros(n, dtype=np.int64)
    has_n = np.zeros(n, dtype=bool)
    for i in range(k):
        b = bases[i:i + n]
        has_n |= (b == 255)
        indices += b.astype(np.int64) * powers[i]
    indices[has_n] = -1
    return indices


def hex_window_for_tile(seq_full, tile_size, k=6):
    """k-mer index per tile position, registered on the cut junction.

    Returns ``(tile_size,)`` where entry ``p`` is the k-mer STARTING
    ``CUT_OFFSET_IN_KMER`` bases before tile position ``p``.

    The registration is done by explicit slicing rather than by choosing a crop
    width and letting ``jitter_matrix`` centre it.  That indirection is what
    broke before: ``jitter_matrix`` crops from the full stored sequence (1802bp,
    not ``l_target``) and CEILS on odd differences, so the realised offset did
    not match what ``calculate_start_and_stop_from_jitter`` predicted and nobody
    noticed the k-mer had moved.
    """
    tile_start = (len(seq_full) - tile_size) // 2
    lo = tile_start - CUT_OFFSET_IN_KMER
    if lo < 0 or lo + tile_size + k - 1 > len(seq_full):
        raise ValueError(
            f"sequence of length {len(seq_full)} cannot supply k-mers for a "
            f"{tile_size}bp tile at offset {CUT_OFFSET_IN_KMER} with k={k}"
        )
    return _seq_to_hexamer_indices(seq_full[lo:lo + tile_size + k - 1], k)


def compute_hex_tables(root, config, split="train", L_min=25, L_max=180, k=6):
    """Compute empirical hexamer cut-site frequency tables from training data.

    Returns:
        hex_tables: dict with keys 'start_fwd', 'stop_fwd', 'start_rev', 'stop_rev',
                    each a (4^k,) float64 array of probabilities (sums to 1 within each table)
        p_L: (n_L,) float64 marginal length distribution
        stats: dict with counts and diagnostics
    """
    n_hex = 4 ** k
    tile_size = config.tile_size
    l_target = config.l_target
    jitter = config.jitter

    # Hex occurrence counts and cut counts
    hex_occurrences = np.zeros(n_hex, dtype=np.float64)
    start_fwd_counts = np.zeros(n_hex, dtype=np.float64)
    stop_fwd_counts = np.zeros(n_hex, dtype=np.float64)
    start_rev_counts = np.zeros(n_hex, dtype=np.float64)
    stop_rev_counts = np.zeros(n_hex, dtype=np.float64)

    # Length distribution
    n_L = L_max - L_min + 1
    length_counts = np.zeros(n_L, dtype=np.float64)

    # Load arrays
    split_arr = root["tiles/split"][:]
    seq_all = root["tiles/seq"][:]
    mask_all = root["tiles/mask"][:]
    frag_indptr = root["fragments/indptr"][:]
    frag_start = root["fragments/start"][:]
    frag_length = root["fragments/length"][:]
    frag_strand = root["fragments/strand"][:]

    split_code = _SPLIT_CODES[split]
    tile_ok = np.nonzero(split_arr == split_code)[0]

    # Crop geometry (j=0 for centre crop)
    crop_start, _ = calculate_start_and_stop_from_jitter(l_target, tile_size, 0)

    n_tiles = seq_all.shape[0]
    n_samples = frag_indptr.shape[0] // n_tiles - 1  # indptr has S*T + 1 entries
    # Actually: indptr length = n_samples * n_tiles + 1
    n_samples = (len(frag_indptr) - 1) // n_tiles

    total_frags = 0
    total_tiles = 0

    for t in tile_ok:
        # Get sequence in crop frame
        seq_full = seq_all[t]
        mask_full = mask_all[t]

        mask_crop = jitter_matrix(mask_full, 0, tile_size)

        # valid_hex[p] = k-mer starting CUT_OFFSET_IN_KMER before tile pos p
        valid_hex = hex_window_for_tile(seq_full, tile_size, k)
        for p in range(tile_size):
            if mask_crop[p] and valid_hex[p] >= 0:
                hex_occurrences[valid_hex[p]] += 1

        # Process fragments for all samples
        for s in range(n_samples):
            u = s * n_tiles + t
            flo = int(frag_indptr[u])
            fhi = int(frag_indptr[u + 1])
            if flo == fhi:
                continue

            starts = frag_start[flo:fhi].astype(np.int64)
            lengths = frag_length[flo:fhi].astype(np.int64)
            strands = frag_strand[flo:fhi].astype(np.int64)

            # Convert to crop coordinates
            starts_crop = starts - crop_start

            for fi in range(len(starts)):
                st = starts_crop[fi]
                L = lengths[fi]
                s_val = strands[fi]
                end = st + L - 1

                # Filter: centre in [0, tile_size), valid length
                centre = st + L // 2
                if centre < 0 or centre >= tile_size:
                    continue
                if L < L_min or L > L_max:
                    continue

                # Length distribution
                length_counts[L - L_min] += 1
                total_frags += 1

                # Cut-site hex indices
                # start position hexamer (position of the 5' cut)
                # stop position hexamer (position of the 3' cut)
                if 0 <= st < tile_size and 0 <= end < tile_size:
                    hex_start = valid_hex[st] if st < len(valid_hex) else -1
                    hex_stop = valid_hex[end] if end < len(valid_hex) else -1

                    if s_val == 0:  # + strand: 5' at start, 3' at end
                        if hex_start >= 0:
                            start_fwd_counts[hex_start] += 1
                        if hex_stop >= 0:
                            stop_fwd_counts[hex_stop] += 1
                    else:  # - strand: 5' at end, 3' at start
                        if hex_stop >= 0:
                            start_rev_counts[hex_stop] += 1
                        if hex_start >= 0:
                            stop_rev_counts[hex_start] += 1

        total_tiles += 1
        if (total_tiles) % 5000 == 0:
            print(f"  Processed {total_tiles}/{len(tile_ok)} tiles, {total_frags:,} frags so far")

    # Normalize to probabilities
    # p(cut | hexamer) = cut_count / hex_occurrences, then normalize
    # Use Laplace smoothing to avoid zeros
    alpha = 1.0
    def _normalize(counts):
        # Rate: cuts per hexamer occurrence
        rate = (counts + alpha) / (hex_occurrences + alpha * n_hex)
        # Normalize to probability distribution
        return rate / rate.sum()

    hex_tables = {
        "start_fwd": _normalize(start_fwd_counts),
        "stop_fwd": _normalize(stop_fwd_counts),
        "start_rev": _normalize(start_rev_counts),
        "stop_rev": _normalize(stop_rev_counts),
    }

    # Marginal p(L)
    p_L = length_counts / max(length_counts.sum(), 1)

    stats = {
        "total_frags": total_frags,
        "total_tiles": total_tiles,
        "n_hex": n_hex,
        "k": k,
        "nonzero_hexamers": int((hex_occurrences > 0).sum()),
    }

    return hex_tables, p_L, stats


def train_conv_baseline(root, config, hex_tables, split="train",
                        L_min=25, L_max=180, k=6, N1=128,
                        epochs=5, lr=1e-3, batch_size=256, max_tiles=5000):
    """Train Conv1d(4, N1, k=6) + LeakyReLU + Conv1d(N1, 4, k=1) to predict cut-site density.

    Matches the actual model's L1 architecture (N1=128 filters + nonlinearity).
    The 4 output channels are (start_fwd, stop_fwd, start_rev, stop_rev).
    Compare learned cut-site predictions to the analytical hex table values.
    """
    tile_size = config.tile_size
    l_target = config.l_target

    split_arr = root["tiles/split"][:]
    seq_all = root["tiles/seq"][:]
    mask_all = root["tiles/mask"][:]
    frag_indptr = root["fragments/indptr"][:]
    frag_start_arr = root["fragments/start"][:]
    frag_length_arr = root["fragments/length"][:]
    frag_strand_arr = root["fragments/strand"][:]

    split_code = _SPLIT_CODES[split]
    tile_ok = np.nonzero(split_arr == split_code)[0]
    if max_tiles and len(tile_ok) > max_tiles:
        rng = np.random.default_rng(42)
        tile_ok = rng.choice(tile_ok, max_tiles, replace=False)

    crop_start, _ = calculate_start_and_stop_from_jitter(l_target, tile_size, 0)
    n_tiles = seq_all.shape[0]
    n_samples = (len(frag_indptr) - 1) // n_tiles

    # Build training data: one-hot sequence + 4-channel cut-site density targets
    print(f"Building training data from {len(tile_ok)} tiles...")

    X_list = []
    Y_list = []  # (N, 4, tile_size): start_fwd, stop_fwd, start_rev, stop_rev

    for t in tile_ok:
        seq_full = seq_all[t]
        seq_crop = jitter_matrix(seq_full, 0, tile_size + k - 1)
        tokens = np.ascontiguousarray(seq_crop, dtype=np.uint8)
        onehot = one_hot_encode_sequences([tokens.tobytes()])[0]
        x = onehot.T.astype(np.float32)  # (4, tile_size + k - 1)
        X_list.append(x)

        targets = np.zeros((4, tile_size), dtype=np.float32)

        for s in range(n_samples):
            u = s * n_tiles + t
            flo = int(frag_indptr[u])
            fhi = int(frag_indptr[u + 1])
            if flo == fhi:
                continue

            starts = frag_start_arr[flo:fhi].astype(np.int64) - crop_start
            lengths = frag_length_arr[flo:fhi].astype(np.int64)
            strands = frag_strand_arr[flo:fhi].astype(np.int64)

            for fi in range(len(starts)):
                st = starts[fi]
                L = lengths[fi]
                s_val = strands[fi]
                end = st + L - 1
                if L < L_min or L > L_max:
                    continue
                if s_val == 0:  # + strand: 5' at start, 3' at end
                    if 0 <= st < tile_size:
                        targets[0, st] += 1   # start_fwd
                    if 0 <= end < tile_size:
                        targets[1, end] += 1  # stop_fwd
                else:  # - strand: 5' at end, 3' at start
                    if 0 <= end < tile_size:
                        targets[2, end] += 1  # start_rev
                    if 0 <= st < tile_size:
                        targets[3, st] += 1   # stop_rev

        # Normalize each channel to a density
        for c in range(4):
            s = targets[c].sum()
            if s > 0:
                targets[c] /= s
        Y_list.append(targets)

    X = np.stack(X_list)   # (N, 4, tile_size + k - 1)
    Y = np.stack(Y_list)   # (N, 4, tile_size)

    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)

    # Model: matches the actual L1 architecture
    conv1 = nn.Conv1d(4, N1, kernel_size=k, bias=True)
    act = nn.LeakyReLU(0.1)
    head = nn.Conv1d(N1, 4, kernel_size=1, bias=True)  # 4 outputs

    nn.init.kaiming_normal_(conv1.weight, nonlinearity='leaky_relu')
    nn.init.zeros_(conv1.bias)
    nn.init.kaiming_normal_(head.weight, nonlinearity='linear')
    nn.init.zeros_(head.bias)

    params = list(conv1.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    N = X_t.shape[0]
    print(f"Training Conv1d(4, {N1}, k={k}) + LeakyReLU + Conv1d({N1}, 4, 1)")
    print(f"  {N} tiles, {epochs} epochs, bs={batch_size}")
    n_params = sum(p.numel() for p in params)
    print(f"  Parameters: {n_params:,}")

    for epoch in range(epochs):
        perm = torch.randperm(N)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_t[idx]       # (B, 4, tile_size + k - 1)
            yb = Y_t[idx]       # (B, 4, tile_size)

            # Forward
            h = act(conv1(xb))  # (B, N1, tile_size)
            pred = head(h)      # (B, 4, tile_size)

            # Per-channel multinomial loss: -sum(target * log_softmax(pred))
            log_p = torch.log_softmax(pred, dim=-1)  # softmax over positions
            loss = -(yb * log_p).sum(dim=-1).mean()   # mean over (batch, channels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch}: loss = {avg_loss:.6f}")

    # Evaluate: compare conv predictions to analytical hex-table predictions
    # on the training tiles (can't decompose into hex tables due to nonlinearity,
    # but can compare per-position output)
    conv1.eval()
    head.eval()

    print("\n  Evaluating conv vs analytical hex tables on training tiles...")
    table_names = ["start_fwd", "stop_fwd", "start_rev", "stop_rev"]
    n_eval = min(500, N)
    with torch.no_grad():
        pred_all = head(act(conv1(X_t[:n_eval])))  # (n_eval, 4, tile_size)
        pred_probs = torch.softmax(pred_all, dim=-1).numpy()

    # Build analytical predictions from hex tables
    from scipy.stats import pearsonr
    for c, tname in enumerate(table_names):
        analytical_table = hex_tables[tname]
        all_corrs = []
        for i in range(n_eval):
            hex_idx = hex_window_for_tile(seq_all[tile_ok[i]], tile_size, k)
            valid = hex_idx >= 0
            if valid.sum() < 10:
                continue
            analytical_pred = analytical_table[hex_idx[valid]]
            analytical_pred = analytical_pred / analytical_pred.sum()
            conv_pred = pred_probs[i, c, :][valid]
            conv_pred = conv_pred / conv_pred.sum()
            r, _ = pearsonr(analytical_pred, conv_pred)
            all_corrs.append(r)
        mean_r = np.mean(all_corrs) if all_corrs else 0.0
        print(f"    {tname}: mean Pearson r = {mean_r:.4f} ({len(all_corrs)} tiles)")

    # Also report the final train loss vs the analytical hex-table loss
    # (the analytical loss is the theoretical minimum for this factorization)
    with torch.no_grad():
        pred_full = head(act(conv1(X_t)))
        log_p = torch.log_softmax(pred_full, dim=-1)
        conv_loss = -(Y_t * log_p).sum(dim=-1).mean().item()
    print(f"\n  Final conv train loss: {conv_loss:.6f}")

    return {"conv1": conv1, "head": head}

    return learned_start, learned_stop


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", required=True, help="cut-site fragment store (.zarr)")
    p.add_argument("--k", type=int, default=6, help="hexamer size (default: 6)")
    p.add_argument("--L-min", type=int, default=25)
    p.add_argument("--L-max", type=int, default=180)
    p.add_argument("--train-conv", action="store_true",
                   help="train a linear Conv1d and compare to analytical solution")
    p.add_argument("--conv-epochs", type=int, default=5)
    p.add_argument("--conv-max-tiles", type=int, default=5000)
    p.add_argument("--output", default=None, help="save hex tables + p_L to .npz")
    args = p.parse_args()

    root = zarr.open_group(args.store, mode="r")
    config = CutSiteStoreConfig.from_json(root.attrs["config_json"])

    print("=" * 60)
    print("Stage 1: Computing empirical hex tables")
    print(f"  Store: {args.store}")
    print(f"  k={args.k}, L=[{args.L_min}, {args.L_max}]")
    print("=" * 60)

    hex_tables, p_L, stats = compute_hex_tables(
        root, config, split="train", L_min=args.L_min, L_max=args.L_max, k=args.k
    )

    print(f"\nHex table stats:")
    print(f"  Total training fragments: {stats['total_frags']:,}")
    print(f"  Total training tiles: {stats['total_tiles']:,}")
    print(f"  Non-zero hexamers: {stats['nonzero_hexamers']}/{stats['n_hex']}")
    print(f"  p(L) range: [{p_L.min():.6f}, {p_L.max():.6f}]")

    # Top hexamers
    for table_name in ["start_fwd", "stop_fwd", "start_rev", "stop_rev"]:
        table = hex_tables[table_name]
        top_idx = np.argsort(table)[-5:][::-1]
        ratio = table.max() / table.min()
        print(f"  {table_name}: max/min ratio = {ratio:.1f}")

    # This script deliberately does NOT report a baseline NLL. It used to, via
    # evaluate_baseline_nll, which was a second implementation of the fragment
    # rule and had silently drifted from CutSiteFragmentDataset on four axes
    # (filter, min_N, normalizer, estimator). The number it printed was ~0.8
    # nats easier than the figure the model is actually scored against, which
    # made a working Stage 2 look like it could not beat its own prior.
    print(f"\n{'=' * 60}")
    print("Hex tables written. To measure the baseline NLL, run:")
    print("  train_cut_site.py --baseline-only --hex-tables <this .npz>")
    print("which zeroes the correction and reports the figure through the")
    print("model's own validation path, so the fragment population, estimator")
    print("and normalizer match the model by construction.")
    print("=" * 60)

    # Optional: train conv and compare
    if args.train_conv:
        print(f"\n{'=' * 60}")
        print("Training Conv1d(4, N1, k=6) + LeakyReLU + Conv1d(N1, 4, 1)")
        print("=" * 60)

        learned_tables = train_conv_baseline(
            root, config, hex_tables,
            split="train", L_min=args.L_min, L_max=args.L_max, k=args.k,
            N1=128, epochs=args.conv_epochs, max_tiles=args.conv_max_tiles,
        )

        # The nonlinear model can't be decomposed into hex tables directly,
        # but we can evaluate its per-position predictions against the
        # hex-table predictions on held-out sequences. The train_conv_baseline
        # function already reports the training loss which shows how well
        # the conv architecture fits the cut-site density.

    # Save
    if args.output:
        np.savez(
            args.output,
            start_fwd=hex_tables["start_fwd"],
            stop_fwd=hex_tables["stop_fwd"],
            start_rev=hex_tables["start_rev"],
            stop_rev=hex_tables["stop_rev"],
            p_L=p_L,
            k=args.k,
            L_min=args.L_min,
            L_max=args.L_max,
        )
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
