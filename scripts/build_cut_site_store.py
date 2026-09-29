#!/usr/bin/env python
"""Build a cut-site fragment store from a fragment h5 + region BED.

Usage:
    python scripts/build_cut_site_store.py \
        --regions regions.bed \
        --reference GRCh38.p12.genome.fa.gz \
        --sample sample.fragments.h5 \
        --output store.zarr \
        --tile-size 2048 --jitter 256 \
        --L-min 25 --L-max 179 \
        --val-frac 0.1 --test-frac 0.1

The output store is consumed by CutSiteFragmentDataset for training the
CutSiteFragmentModel.  The interface is identical for sim and real data.
"""

import argparse
import datetime
import hashlib
import json
import logging
import multiprocessing as mp
import os
import sys

import numpy as np
import zarr

# Bootstrap: background_model is not an installed package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py

from background_model.cut_site_store import (
    CutSiteStoreConfig,
    recompute_endpoints_from_fragments,
)
from fragmentomics_tools.region import Region, one_hot_encode_sequences

log = logging.getLogger(__name__)

_ZARR_MAJOR = int(zarr.__version__.split(".")[0])


def _create_array(group, name, **kwargs):
    if _ZARR_MAJOR >= 3:
        return group.create_array(name, **kwargs)
    return group.create_dataset(name, **kwargs)


def _load_regions(bed_path, ref_path, tile_size, jitter, rf_budget):
    """Load regions from BED, expand, extract sequences."""
    l_target = tile_size + 2 * jitter
    l_seq = tile_size + 2 * (jitter + rf_budget)

    regions = []
    with open(bed_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            chrom, start, stop = parts[0], int(parts[1]), int(parts[2])
            centre = (start + stop) // 2
            regions.append(Region(chrom, centre - l_target // 2,
                                  centre + l_target // 2, strand=None))

    # Extract sequences over l_seq extent (wider than l_target for conv margin)
    import pysam
    ref = pysam.FastaFile(ref_path)
    sequences = []
    masks = []
    valid_regions = []
    for r in regions:
        seq_start = r.start - (l_seq - l_target) // 2
        seq_stop = seq_start + l_seq
        if seq_start < 0:
            continue
        try:
            seq = ref.fetch(r.chrom, seq_start, seq_stop).upper()
        except (ValueError, KeyError):
            continue
        if len(seq) != l_seq:
            continue
        sequences.append(seq.encode())
        # Mask: all valid (blacklist handled upstream in region set)
        masks.append(np.ones(l_target, dtype=bool))
        valid_regions.append(r)
    ref.close()
    return valid_regions, sequences, masks


_BIN_SIZE = 5000  # fragments_h5 index bin size

# Per-worker h5 handle cache (one open per process, not per region)
_worker_h5 = {}


def _extract_worker(args_tuple):
    """Worker for multiprocessing.Pool.map.  Opens h5 once per process."""
    h5_path, chrom, reg_start, reg_stop, L_min, L_max = args_tuple

    # Lazy per-process h5 open
    if h5_path not in _worker_h5:
        _worker_h5[h5_path] = h5py.File(h5_path, 'r')
    f = _worker_h5[h5_path]

    if chrom not in f['data']:
        return np.empty(0, np.uint16), np.empty(0, np.uint16), np.empty(0, np.uint8)

    # Use the h5 index to find the slice of fragments near this region
    idx = f['index'][chrom]
    d = f['data'][chrom]
    start_bin = max(0, reg_start // _BIN_SIZE - 1)
    end_bin = min(len(idx) - 1, reg_stop // _BIN_SIZE + 1)
    lo = int(idx[start_bin])
    hi = int(idx[end_bin])
    if lo >= hi:
        return np.empty(0, np.uint16), np.empty(0, np.uint16), np.empty(0, np.uint8)

    # Read the candidate slice
    starts_abs = d['starts'][lo:hi].astype(np.int64)
    lengths = d['lengths'][lo:hi].astype(np.int64)
    raw_strand = d['strand'][lo:hi]

    # Filter: overlaps region
    overlap = (starts_abs + lengths > reg_start) & (starts_abs < reg_stop)
    starts_abs = starts_abs[overlap]
    lengths = lengths[overlap]
    raw_strand = raw_strand[overlap]

    if len(starts_abs) == 0:
        return np.empty(0, np.uint16), np.empty(0, np.uint16), np.empty(0, np.uint8)

    # Region-local coordinates
    starts = starts_abs - reg_start

    # Encode strand: '+' or b'+' → 0, else 1
    strands = np.where(
        (raw_strand == b'+') | (raw_strand == '+'), 0, 1
    ).astype(np.uint8)

    # Filter: in-band length, centre in [0, l_target), start >= 0
    l_target = reg_stop - reg_start
    keep = (lengths >= L_min) & (lengths <= L_max)
    centres = starts + lengths // 2
    keep &= (centres >= 0) & (centres < l_target)
    keep &= (starts >= 0)

    return (
        starts[keep].astype(np.uint16),
        lengths[keep].astype(np.uint16),
        strands[keep],
    )


def build_store(args):
    cfg = CutSiteStoreConfig(
        tile_size=args.tile_size,
        jitter=args.jitter,
        rf_budget=args.rf_budget,
        L_min=args.L_min,
        L_max=args.L_max,
        seed=args.split_seed,
        n_train_samples=1,
    )

    log.info("Loading regions from %s", args.regions)
    regions, sequences, masks = _load_regions(
        args.regions, args.reference,
        cfg.tile_size, cfg.jitter, cfg.rf_budget,
    )
    T = len(regions)
    log.info("Loaded %d valid regions", T)
    if T == 0:
        raise ValueError("No valid regions found")

    # -- Extract fragments (parallel) -------------------------------------
    n_workers = args.workers
    log.info("Extracting fragments from %s (%d regions, %d workers)",
             args.sample, T, n_workers)

    worker_args = [
        (args.sample, r.chrom, r.start, r.stop, cfg.L_min, cfg.L_max)
        for r in regions
    ]

    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            results = pool.map(_extract_worker, worker_args)
    else:
        results = [_extract_worker(a) for a in worker_args]

    # -- Accumulate fragments + compute endpoints (sequential, fast) ------
    frag_indptr = [0]
    all_frag_start, all_frag_length, all_frag_strand = [], [], []
    all_ep_pos, all_ep_track, all_ep_data = [], [], []
    ep_indptr = [0]
    S = 1
    N_totals = np.zeros((S, T, 2), dtype="uint32")
    margin = (cfg.l_target - cfg.tile_size) // 2

    for t, (fs, fl, fst) in enumerate(results):
        all_frag_start.append(fs)
        all_frag_length.append(fl)
        all_frag_strand.append(fst)
        frag_indptr.append(frag_indptr[-1] + len(fs))

        ep = recompute_endpoints_from_fragments(fs, fl, cfg.l_target)
        for track_idx in range(2):
            nz = np.nonzero(ep[track_idx])[0]
            if len(nz):
                all_ep_pos.extend(nz.tolist())
                all_ep_track.extend([track_idx] * len(nz))
                all_ep_data.extend(ep[track_idx, nz].astype(int).tolist())
            N_totals[0, t, track_idx] = int(
                ep[track_idx, margin:margin + cfg.tile_size].sum()
            )
        ep_indptr.append(len(all_ep_pos))

    total_frags = frag_indptr[-1]
    log.info("Extracted %d fragments from %d regions", total_frags, T)

    # -- Assign splits -----------------------------------------------------
    rng = np.random.default_rng(cfg.seed)
    split_arr = np.zeros(T, dtype=np.uint8)
    perm = rng.permutation(T)
    n_val = int(T * args.val_frac)
    n_test = int(T * args.test_frac)
    split_arr[perm[:n_val]] = 1
    split_arr[perm[n_val:n_val + n_test]] = 2
    log.info("Splits: train=%d, val=%d, test=%d",
             (split_arr == 0).sum(), (split_arr == 1).sum(), (split_arr == 2).sum())

    # -- Write store -------------------------------------------------------
    open_kwargs = {"mode": "w"}
    if _ZARR_MAJOR >= 3:
        open_kwargs["zarr_format"] = 2
    root = zarr.open_group(args.output, **open_kwargs)

    root.attrs.update({
        "config_json": cfg.to_json(),
        "config_hash": cfg.config_hash(),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "split_version": 0,
    })

    tiles = root.create_group("tiles")
    _create_array(tiles, "contig", data=np.array([r.chrom for r in regions], dtype="<U32"))
    _create_array(tiles, "start", data=np.array([r.start for r in regions], dtype="int64"))
    _create_array(tiles, "stop", data=np.array([r.stop for r in regions], dtype="int64"))
    _create_array(tiles, "strand", data=np.array(["."] * T, dtype="<U1"))
    _create_array(tiles, "region_id", data=np.array([f"r{i}" for i in range(T)], dtype="<U64"))
    _create_array(tiles, "split", data=split_arr)
    _create_array(tiles, "seq", data=np.array([list(s) for s in sequences], dtype="uint8"))
    _create_array(tiles, "mask", data=np.array(masks, dtype=bool))

    samples = root.create_group("samples")
    _create_array(samples, "library", data=np.array([os.path.basename(args.sample)], dtype="<U256"))
    _create_array(samples, "role", data=np.zeros(S, dtype="uint8"))
    _create_array(samples, "total_fragments", data=np.array([total_frags], dtype="uint64"))

    frags = root.create_group("fragments")
    _create_array(frags, "indptr", data=np.array(frag_indptr, dtype="int64"))
    all_fs = np.concatenate(all_frag_start) if all_frag_start else np.empty(0, "uint16")
    all_fl = np.concatenate(all_frag_length) if all_frag_length else np.empty(0, "uint16")
    all_fst = np.concatenate(all_frag_strand) if all_frag_strand else np.empty(0, "uint8")
    _create_array(frags, "start", data=all_fs)
    _create_array(frags, "length", data=all_fl)
    _create_array(frags, "strand", data=all_fst)

    endpoints = root.create_group("endpoints")
    _create_array(endpoints, "indptr", data=np.array(ep_indptr, dtype="int64"))
    _create_array(endpoints, "pos", data=np.array(all_ep_pos, dtype="uint16") if all_ep_pos else np.empty(0, "uint16"))
    _create_array(endpoints, "track", data=np.array(all_ep_track, dtype="uint8") if all_ep_track else np.empty(0, "uint8"))
    _create_array(endpoints, "data", data=np.array(all_ep_data, dtype="uint16") if all_ep_data else np.empty(0, "uint16"))

    totals = root.create_group("totals")
    _create_array(totals, "N", data=N_totals)

    log.info("Store written to %s (%d tiles, %d fragments)", args.output, T, total_frags)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--regions", required=True, help="BED file with region centres")
    p.add_argument("--reference", required=True, help="Reference genome FASTA")
    p.add_argument("--sample", required=True, help="Fragment h5 file")
    p.add_argument("--output", required=True, help="Output zarr store path")
    p.add_argument("--tile-size", type=int, default=2048)
    p.add_argument("--jitter", type=int, default=256)
    p.add_argument("--rf-budget", type=int, default=133)
    p.add_argument("--L-min", type=int, default=25)
    p.add_argument("--L-max", type=int, default=179)
    p.add_argument("--workers", type=int, default=1, help="Parallel workers for h5 queries")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build_store(args)


if __name__ == "__main__":
    main()
