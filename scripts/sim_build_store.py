#!/usr/bin/env python
"""Build a zarr store from simulation output for background model training.

Reads sim_fragments.py output (.npz files) and constructs a zarr store
compatible with BackgroundTileDataset, using the real fragment_array library
to build coverage counts (plan §6 step 5: "using the existing preprocess path").

Geometry: TILE=2048, JITTER=0 (no augmentation needed for simulation),
RF_BUDGET=2048 (architecture-dependent receptive field).

Usage:
    PYTHONPATH=. python scripts/sim_build_store.py \
        --sim-dir /efs/.../simulation/B \
        --out /efs/.../simulation/stores/sim_store_B.zarr

Env: /home/nathanboley/miniconda3/envs/biomarker_env/bin/python
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import os
import sys
import time

import numpy as np
import pysam
import zarr

# ── geometry for the simulation store ─────────────────────────────────────
SIM_TILE = 2_048
SIM_JITTER = 0
SIM_RF_BUDGET = 2_048
SIM_L_TARGET = SIM_TILE + 2 * SIM_JITTER           # 2048
SIM_L_SEQ = SIM_TILE + 2 * (SIM_JITTER + SIM_RF_BUDGET)  # 6144

FASTA = "/efs/analytics/nathanboley/data_resources/genome/hg38.fa"

# Track layout (must match background_model/preprocess.py)
STRANDS = ("+", "-")
FL_BANDS = ((40, 65), (120, 175))
COVERAGE_TYPES = ("first", "last", "midpoint")
N_TRACKS = len(STRANDS) * len(FL_BANDS) * len(COVERAGE_TYPES)  # 12

TRACK_INDEX = {}
_idx = 0
for _s in STRANDS:
    for _fl in FL_BANDS:
        for _c in COVERAGE_TYPES:
            TRACK_INDEX[(_s, _fl, _c)] = _idx
            _idx += 1

# Split/role codes (must match dataset.py)
SPLIT_CODES = {"train": 0, "val": 1, "heldout_inactive": 2}
ROLE_CODES = {"train": 0, "heldout": 1}


def load_sim_output(sim_dir):
    """Load region table + all sample fragment arrays from sim output."""
    rt = np.load(os.path.join(sim_dir, "region_table.npz"), allow_pickle=True)
    regions = [
        {"contig": str(c), "gstart": int(a), "gstop": int(b)}
        for c, a, b in zip(rt["contig"], rt["gstart"], rt["gstop"])
    ]
    region_len = int(rt["region_len"])

    sample_files = sorted(
        f for f in os.listdir(sim_dir)
        if f.startswith("sample_") and f.endswith(".npz")
    )
    samples = []
    for sf in sample_files:
        d = np.load(os.path.join(sim_dir, sf))
        samples.append({
            "region_idx": d["region_idx"],
            "start": d["start"],
            "stop": d["stop"],
            "strand": d["strand"],
        })
    return regions, region_len, samples


def fetch_sequence(fa, contig, gstart, gstop, l_seq, region_len):
    """Fetch padded sequence for a tile (tile center + RF_BUDGET margin).

    Returns l_seq bytes. Pads with 'N' at contig edges.
    """
    margin = (l_seq - region_len) // 2
    fetch_start = gstart - margin
    fetch_stop = gstop + margin
    assert fetch_stop - fetch_start == l_seq

    # Clamp to contig boundaries, pad with N
    contig_len = fa.get_reference_length(contig)
    left_pad = max(0, -fetch_start)
    right_pad = max(0, fetch_stop - contig_len)
    clamped_start = max(0, fetch_start)
    clamped_stop = min(contig_len, fetch_stop)

    seq = fa.fetch(contig, clamped_start, clamped_stop).upper()
    seq = "N" * left_pad + seq + "N" * right_pad
    assert len(seq) == l_seq, (len(seq), l_seq)
    return np.frombuffer(seq.encode("ascii"), dtype=np.uint8)


def build_sparse_counts_for_sample_tile(sample_data, tile_idx, region_len):
    """Build sparse coverage counts for one (sample, tile) using the library.

    Returns (pos_arr, track_arr, data_arr) as flat arrays, plus N (C,).
    """
    from fragmentomics_tools.fragment_array.fragment_array import RegionFragmentArray
    from fragmentomics_tools.region import Region

    mask = sample_data["region_idx"] == tile_idx
    if not mask.any():
        return (np.empty(0, np.uint16), np.empty(0, np.uint8),
                np.empty(0, np.uint16), np.zeros(N_TRACKS, np.uint32))

    starts = sample_data["start"][mask].astype(np.int64)
    stops = sample_data["stop"][mask].astype(np.int64)
    strands = sample_data["strand"][mask]

    # Construct a dummy region — only the length matters for coverage counts
    region = Region(chrom="meta", start=0, stop=region_len, strand=".")
    max_frag_len = int(stops.max() - starts.min()) + 1 if len(starts) > 0 else 500

    rfa = RegionFragmentArray(
        starts_0=starts, stops_0=stops,
        region=region, max_frag_len=max(max_frag_len, 500),
        validate_data=True,
        fragment_strands=strands,
    )

    sparse_counts = rfa.build_coverage_counts(
        fl_bands=list(FL_BANDS), split_strand=True, return_sparse=True
    )

    all_pos, all_track, all_data = [], [], []
    for key, vec in sparse_counts.items():
        strand, fl_band, cov_type = key
        track_idx = TRACK_INDEX[(strand, fl_band, cov_type)]
        if len(vec.coords) > 0:
            # Clip to L_TARGET range (with JITTER=0, L_TARGET == region_len)
            valid = vec.coords < SIM_L_TARGET
            if valid.any():
                all_pos.append(vec.coords[valid].astype(np.uint16))
                all_track.append(np.full(int(valid.sum()), track_idx, dtype=np.uint8))
                all_data.append(vec.data[valid].astype(np.uint16))

    if all_pos:
        pos = np.concatenate(all_pos)
        track = np.concatenate(all_track)
        data = np.concatenate(all_data)
    else:
        pos = np.empty(0, np.uint16)
        track = np.empty(0, np.uint8)
        data = np.empty(0, np.uint16)

    # Compute N: per-track totals over the center tile (all positions valid)
    y_dense = np.zeros((N_TRACKS, SIM_L_TARGET), dtype=np.float32)
    if len(pos) > 0:
        np.add.at(y_dense, (track, pos), data.astype(np.float32))
    # With JITTER=0, center tile == full tile, mask == all True
    N = y_dense.sum(axis=1).astype(np.uint32)

    return pos, track, data, N


def _process_sample(sample_data, n_tiles, region_len):
    """Process all tiles for one sample. Top-level function for multiprocessing."""
    sparse_list = []
    N_arr = np.zeros((n_tiles, N_TRACKS), dtype=np.uint32)
    for t_idx in range(n_tiles):
        pos, track, data, N = build_sparse_counts_for_sample_tile(
            sample_data, t_idx, region_len
        )
        sparse_list.append((pos, track, data))
        N_arr[t_idx] = N
    return sparse_list, N_arr


def build_store(sim_dir, out_path, n_train_samples=16, seed=1337, workers=1):
    """Build a zarr store from simulation output."""
    t0 = time.time()
    regions, region_len, samples = load_sim_output(sim_dir)
    n_tiles = len(regions)
    n_samples = len(samples)
    assert region_len == SIM_TILE, (region_len, SIM_TILE)

    print(f"[store] {n_tiles} tiles, {n_samples} samples, "
          f"region_len={region_len}, workers={workers}", flush=True)

    # ── Assign splits (regions) and roles (samples) ──────────────────────
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_tiles)
    n_train_tiles = int(n_tiles * 0.8)
    n_val_tiles = int(n_tiles * 0.1)
    split_arr = np.full(n_tiles, SPLIT_CODES["heldout_inactive"], dtype=np.uint8)
    split_arr[perm[:n_train_tiles]] = SPLIT_CODES["train"]
    split_arr[perm[n_train_tiles:n_train_tiles + n_val_tiles]] = SPLIT_CODES["val"]

    role_arr = np.full(n_samples, ROLE_CODES["heldout"], dtype=np.uint8)
    role_arr[:n_train_samples] = ROLE_CODES["train"]

    # ── Build config for store attrs ─────────────────────────────────────
    from background_model.config import PlumbingConfig
    config = PlumbingConfig(
        sample_sheet="",
        region_beds={},
        blacklist_bed="",
        fasta="",
        tile_size=SIM_TILE,
        jitter=SIM_JITTER,
        rf_budget=SIM_RF_BUDGET,
        fl_bands=FL_BANDS,
        seed=seed,
        n_train_samples=n_train_samples,
        n_heldout_samples=n_samples - n_train_samples,
    )

    # ── First pass: accumulate all sparse counts to compute nnz ──────────
    print("[store] building sparse counts...", flush=True)
    all_sparse = []  # list of (pos, track, data) per flat index s*T+t
    all_N = np.zeros((n_samples, n_tiles, N_TRACKS), dtype=np.uint32)
    total_nnz = 0

    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for s_idx, sample in enumerate(samples):
                fut = pool.submit(_process_sample, sample, n_tiles, region_len)
                futures[fut] = s_idx
            for fut in concurrent.futures.as_completed(futures):
                s_idx = futures[fut]
                sparse_list, N_arr = fut.result()
                all_N[s_idx] = N_arr
                for t_idx, (pos, track, data) in enumerate(sparse_list):
                    all_sparse.append((s_idx, t_idx, pos, track, data))
                    total_nnz += len(pos)
                print(f"[store]   sample {s_idx+1}/{n_samples} "
                      f"({time.time()-t0:.1f}s)", flush=True)
        # Sort by (s_idx, t_idx) to match the sequential ordering
        all_sparse.sort(key=lambda x: (x[0], x[1]))
        all_sparse = [(pos, track, data) for _, _, pos, track, data in all_sparse]
    else:
        for s_idx, sample in enumerate(samples):
            for t_idx in range(n_tiles):
                pos, track, data, N = build_sparse_counts_for_sample_tile(
                    sample, t_idx, region_len
                )
                all_sparse.append((pos, track, data))
                all_N[s_idx, t_idx] = N
                total_nnz += len(pos)
            if (s_idx + 1) % 5 == 0:
                print(f"[store]   sample {s_idx+1}/{n_samples} "
                      f"({time.time()-t0:.1f}s)", flush=True)

    print(f"[store] total nnz={total_nnz:,} ({time.time()-t0:.1f}s)", flush=True)

    # ── Create zarr store ────────────────────────────────────────────────
    open_kwargs = {"mode": "w"}
    if int(zarr.__version__.split(".")[0]) >= 3:
        open_kwargs["zarr_format"] = 2
    root = zarr.open_group(out_path, **open_kwargs)

    root.attrs.update({
        "config_json": config.full_config_json(),
        "config_hash": config.config_hash(with_content_hashes=False),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "zarr_format_note": f"zarr v2, zarr-python {zarr.__version__}",
        "split_version": 1,
        "sim_dir": sim_dir,
        "sim_study": True,
    })

    S, T = n_samples, n_tiles

    # /tiles/
    tiles_grp = root.create_group("tiles")
    z2_create = (lambda g, n, **kw: g.create_dataset(n, **kw)
                 if int(zarr.__version__.split(".")[0]) < 3
                 else lambda g, n, **kw: g.create_array(n, **kw))

    def _ca(grp, name, **kwargs):
        if int(zarr.__version__.split(".")[0]) >= 3:
            return grp.create_array(name, **kwargs)
        return grp.create_dataset(name, **kwargs)

    _ca(tiles_grp, "contig", shape=(T,), dtype="<U32", chunks=(T,))
    _ca(tiles_grp, "start", shape=(T,), dtype="int64", chunks=(T,))
    _ca(tiles_grp, "stop", shape=(T,), dtype="int64", chunks=(T,))
    _ca(tiles_grp, "strand", shape=(T,), dtype="<U1", chunks=(T,))
    _ca(tiles_grp, "region_id", shape=(T,), dtype="<U64", chunks=(T,))
    _ca(tiles_grp, "split", shape=(T,), dtype="uint8", chunks=(T,))
    _ca(tiles_grp, "seq", shape=(T, SIM_L_SEQ), dtype="uint8",
        chunks=(min(64, T), SIM_L_SEQ))
    _ca(tiles_grp, "mask", shape=(T, SIM_L_TARGET), dtype="bool",
        chunks=(min(64, T), SIM_L_TARGET))

    # /samples/
    samples_grp = root.create_group("samples")
    _ca(samples_grp, "library", shape=(S,), dtype="<U64", chunks=(S,))
    _ca(samples_grp, "seqrun", shape=(S,), dtype="<U64", chunks=(S,))
    _ca(samples_grp, "endo_category", shape=(S,), dtype="<U64", chunks=(S,))
    _ca(samples_grp, "h5_path", shape=(S,), dtype="<U256", chunks=(S,))
    _ca(samples_grp, "role", shape=(S,), dtype="uint8", chunks=(S,))
    _ca(samples_grp, "total_fragments", shape=(S,), dtype="uint64", chunks=(S,))

    # /counts/ (CSR)
    counts_grp = root.create_group("counts")
    _ca(counts_grp, "indptr", shape=(S * T + 1,), dtype="int64",
        chunks=(min(1 << 20, S * T + 1),))
    _ca(counts_grp, "pos", shape=(total_nnz,), dtype="uint16",
        chunks=(min(1 << 20, max(1, total_nnz)),))
    _ca(counts_grp, "track", shape=(total_nnz,), dtype="uint8",
        chunks=(min(1 << 20, max(1, total_nnz)),))
    _ca(counts_grp, "data", shape=(total_nnz,), dtype="uint16",
        chunks=(min(1 << 20, max(1, total_nnz)),))

    # /totals/
    totals_grp = root.create_group("totals")
    _ca(totals_grp, "N", shape=(S, T, N_TRACKS), dtype="uint32",
        chunks=(S, T, N_TRACKS))

    # ── Fill tile metadata + sequences (BATCHED to avoid EFS thrashing) ──
    print("[store] fetching sequences (batched)...", flush=True)
    fa = pysam.FastaFile(FASTA)
    contigs = np.array([r["contig"] for r in regions], dtype="<U32")
    starts = np.array([r["gstart"] for r in regions], dtype=np.int64)
    stops = np.array([r["gstop"] for r in regions], dtype=np.int64)
    strands_arr = np.full(T, ".", dtype="<U1")
    region_ids = np.array(
        [f"{r['contig']}:{r['gstart']}-{r['gstop']}" for r in regions], dtype="<U64"
    )
    seq_bulk = np.zeros((T, SIM_L_SEQ), dtype=np.uint8)
    for t_idx, r in enumerate(regions):
        seq_bulk[t_idx] = fetch_sequence(
            fa, r["contig"], r["gstart"], r["gstop"], SIM_L_SEQ, region_len
        )
    fa.close()
    mask_bulk = np.ones((T, SIM_L_TARGET), dtype=bool)

    # Single bulk writes (one I/O per array, not per tile)
    tiles_grp["contig"][:] = contigs
    tiles_grp["start"][:] = starts
    tiles_grp["stop"][:] = stops
    tiles_grp["strand"][:] = strands_arr
    tiles_grp["region_id"][:] = region_ids
    tiles_grp["split"][:] = split_arr
    tiles_grp["seq"][:] = seq_bulk
    tiles_grp["mask"][:] = mask_bulk
    print(f"[store] sequences done ({time.time()-t0:.1f}s)", flush=True)

    # ── Fill sample metadata (batched) ────────────────────────────────────
    libraries = np.array([f"sim_sample_{i:03d}" for i in range(S)], dtype="<U64")
    samples_grp["library"][:] = libraries
    samples_grp["seqrun"][:] = np.full(S, "sim", dtype="<U64")
    samples_grp["endo_category"][:] = np.full(S, "simulated", dtype="<U64")
    samples_grp["h5_path"][:] = np.full(S, "", dtype="<U256")
    samples_grp["role"][:] = role_arr
    total_frags = np.array([len(s["region_idx"]) for s in samples], dtype=np.uint64)
    samples_grp["total_fragments"][:] = total_frags

    # ── Fill CSR counts ──────────────────────────────────────────────────
    print("[store] writing CSR counts...", flush=True)
    indptr = np.zeros(S * T + 1, dtype=np.int64)
    all_pos_flat = []
    all_track_flat = []
    all_data_flat = []
    cursor = 0
    for u, (pos, track, data) in enumerate(all_sparse):
        n = len(pos)
        indptr[u + 1] = indptr[u] + n
        if n > 0:
            all_pos_flat.append(pos)
            all_track_flat.append(track)
            all_data_flat.append(data)

    counts_grp["indptr"][:] = indptr
    if total_nnz > 0:
        counts_grp["pos"][:] = np.concatenate(all_pos_flat)
        counts_grp["track"][:] = np.concatenate(all_track_flat)
        counts_grp["data"][:] = np.concatenate(all_data_flat)

    # ── Fill N totals ────────────────────────────────────────────────────
    totals_grp["N"][:] = all_N

    elapsed = time.time() - t0
    print(f"[store] wrote {out_path}  "
          f"({S} samples x {T} tiles, nnz={total_nnz:,}) "
          f"in {elapsed:.1f}s", flush=True)

    # Summary
    train_pairs = int(((role_arr == ROLE_CODES["train"])[:, None] &
                       (split_arr == SPLIT_CODES["train"])[None, :]).sum())
    min_N_50 = int((all_N.min(axis=2) >= 50).sum())
    print(f"[store] train pairs (before min_N filter): {train_pairs}")
    print(f"[store] pairs with min_N >= 50: {min_N_50}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-dir", required=True,
                    help="Path to simulation output (e.g. .../simulation/B)")
    ap.add_argument("--out", required=True,
                    help="Output zarr store path")
    ap.add_argument("--n-train-samples", type=int, default=16,
                    help="Number of train samples (rest are heldout)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of parallel workers (default 1, sequential)")
    args = ap.parse_args()
    build_store(args.sim_dir, args.out, args.n_train_samples, args.seed,
                workers=args.workers)


if __name__ == "__main__":
    sys.exit(main())
