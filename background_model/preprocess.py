"""Preprocess pipeline: Phase A (per-sample shards) + Phase B (zarr assembly).

Phase A is embarrassingly parallel (ProcessPoolExecutor); Phase B is serial.
See docs/pending/data_plumbing_design.md for the full design.
"""

import argparse
import logging
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

from background_model.config import (
    C,
    L_SEQ,
    L_TARGET,
    TILE,
    PlumbingConfig,
    _file_md5,
)
from background_model.store import (
    compute_N_for_tile,
    create_store,
    densify_counts,
    increment_split_version,
    open_store,
    record_phase_b_params,
)

log = logging.getLogger(__name__)

# ── Track index (maps build_coverage_counts keys to canonical order) ─────

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


# ── Contig geometry ──────────────────────────────────────────────────────

def _contig_length(ref: str, contig: str):
    """Return the contig length for `ref`, or None if unavailable.

    Used to clamp margined regions/masks at contig ends so positions past the
    contig boundary contribute zero counts and are masked invalid.
    """
    try:
        from fragmentomics_tools.contig import CONTIG_LENGTHS
        return CONTIG_LENGTHS[ref][contig]
    except (KeyError, TypeError, ImportError):
        return None


# ── Tiling ───────────────────────────────────────────────────────────────

def build_tiles(region_beds: dict, tile_size: int, jitter: int, rf_budget: int, ref: str):
    """Generate tiles from region BED files.

    Returns a list of dicts with keys:
        contig, start, stop, region_id, split_name, tile_idx
    where (start, stop) are the CENTER tile coordinates (not margined).
    """
    from fragmentomics_tools.dataframe import RegionDataFrame

    tiles = []
    for split_name, bed_path in sorted(region_beds.items()):
        rdf = RegionDataFrame.from_bed(bed_path, ref=ref)
        for row in rdf.itertuples():
            region_start = int(row.start)
            region_stop = int(row.stop)
            contig = str(row.contig)
            region_id = f"{contig}:{region_start}-{region_stop}"

            # Tile the region
            pos = region_start
            while pos + tile_size <= region_stop:
                tiles.append({
                    "contig": contig,
                    "start": pos,
                    "stop": pos + tile_size,
                    "region_id": region_id,
                    "split_name": split_name,
                })
                pos += tile_size

    return tiles


# ── Phase A — per-sample shard worker ────────────────────────────────────

def _worker_process_sample(
    sample_row: dict,
    tiles: list,
    config: PlumbingConfig,
    shard_dir: str,
    ref: str,
):
    """Process one sample: read fragment h5, build sparse counts, save shard.

    Writes shards/<library>.npz (+ .done marker).
    On failure writes <library>.error.
    """
    library = sample_row["library"]
    h5_path = sample_row["h5_path"]
    done_marker = os.path.join(shard_dir, f"{library}.done")
    error_marker = os.path.join(shard_dir, f"{library}.error")

    if os.path.exists(done_marker):
        log.info(f"Skipping {library} (already done)")
        return library, True

    try:
        return _worker_inner(library, h5_path, tiles, config, shard_dir, ref)
    except Exception as e:
        tb = traceback.format_exc()
        with open(error_marker, "w") as f:
            f.write(f"{e}\n{tb}")
        log.error(f"Sample {library} failed: {e}")
        return library, False


def _worker_inner(library, h5_path, tiles, config, shard_dir, ref):
    from fragments_h5 import FragmentsH5
    from fragmentomics_tools.dataframe import RegionDataFrame
    from fragmentomics_tools.fragment_array.fragment_array import RegionFragmentArray
    from fragmentomics_tools.region import Region

    h5 = FragmentsH5(h5_path, cache_pointers=False)

    # Total fragments from the global fragment-length histogram (condition #6)
    total_fragments = int(h5.fragment_length_counts.sum())

    # Load blacklist once per worker
    blacklist_rdf = None
    if config.blacklist_bed:
        blacklist_rdf = RegionDataFrame.from_bed(config.blacklist_bed, ref=ref)

    all_pos = []
    all_track = []
    all_data = []
    tile_offsets = []  # (tile_idx, nnz_start, nnz_end)

    l_target = config.l_target

    for tile_idx, tile in enumerate(tiles):
        # Margined region for counts: tile ± JITTER (= L_TARGET extent)
        # CRITICAL: strand='.' ALWAYS (condition #1 — no strand flip)
        margin = config.jitter
        # The L_TARGET-relative count frame is anchored at `count_start` (tile
        # start minus the jitter margin), which can fall before the contig start
        # (or the region can run past the contig end).  Clamp the fetched region
        # to [0, contig_len) so Region(start>=0) holds, then shift shard coords by
        # `left_pad` so they stay L_TARGET-relative with the tile centered.  This
        # mirrors the Phase B sequence padding convention below.  Positions
        # outside the contig simply carry no fragments (zero counts).
        count_start = tile["start"] - margin
        count_stop = tile["stop"] + margin
        clamped_start = max(0, count_start)
        clamped_stop = count_stop
        contig_len = _contig_length(ref, tile["contig"])
        if contig_len is not None:
            clamped_stop = min(clamped_stop, contig_len)
        left_pad = clamped_start - count_start  # = max(0, -count_start)

        margined_region = Region(
            chrom=tile["contig"],
            start=clamped_start,
            stop=clamped_stop,
            strand=".",
        )

        rfa = RegionFragmentArray.from_fragments_h5(
            h5,
            margined_region,
            min_mapq=config.min_mapq,
            max_frag_len=config.max_frag_len,
        )

        if config.dedup:
            rfa = rfa.drop_duplicate_fragments()

        # Blacklist masking (fragments with endpoints on blacklisted positions)
        if blacklist_rdf is not None and len(blacklist_rdf) > 0:
            # Find blacklist regions overlapping this tile
            bl_regions = _get_overlapping_blacklist_regions(
                blacklist_rdf, tile["contig"],
                tile["start"] - margin, tile["stop"] + margin,
            )
            if bl_regions:
                rfa = rfa.mask_overlapping_fragments(
                    bl_regions, expansion=config.blacklist_expansion
                )

        # Build sparse coverage counts (condition #5: half-open [lo, hi) semantics)
        sparse_counts = rfa.build_coverage_counts(
            fl_bands=list(config.fl_bands),
            split_strand=True,
            return_sparse=True,
        )

        # Convert SparseIntVector triples to (pos, track, data) arrays
        tile_pos = []
        tile_track = []
        tile_data = []
        for key, vec in sparse_counts.items():
            strand, fl_band, cov_type = key
            track_idx = TRACK_INDEX[(strand, fl_band, cov_type)]
            if len(vec.coords) > 0:
                # Shift region-relative coords into the L_TARGET frame (accounts
                # for left-clamping at the contig start).
                tile_pos.append((vec.coords + left_pad).astype(np.uint16))
                tile_track.append(np.full(len(vec.coords), track_idx, dtype=np.uint8))
                tile_data.append(vec.data.astype(np.uint16))

        if tile_pos:
            tp = np.concatenate(tile_pos)
            tt = np.concatenate(tile_track)
            td = np.concatenate(tile_data)
            # Sort by (track, pos) within tile as per design
            sort_idx = np.lexsort((tp, tt))
            tp = tp[sort_idx]
            tt = tt[sort_idx]
            td = td[sort_idx]
        else:
            tp = np.empty(0, dtype=np.uint16)
            tt = np.empty(0, dtype=np.uint8)
            td = np.empty(0, dtype=np.uint16)

        nnz_start = sum(len(p) for p in all_pos) if all_pos else 0
        all_pos.append(tp)
        all_track.append(tt)
        all_data.append(td)
        tile_offsets.append((tile_idx, nnz_start, nnz_start + len(tp)))

    h5.close()

    # Save shard
    if all_pos:
        pos_arr = np.concatenate(all_pos)
        track_arr = np.concatenate(all_track)
        data_arr = np.concatenate(all_data)
    else:
        pos_arr = np.empty(0, dtype=np.uint16)
        track_arr = np.empty(0, dtype=np.uint8)
        data_arr = np.empty(0, dtype=np.uint16)

    tile_offsets_arr = np.array(tile_offsets, dtype=np.int64)

    shard_path = os.path.join(shard_dir, f"{library}.npz")
    np.savez(
        shard_path,
        pos=pos_arr,
        track=track_arr,
        data=data_arr,
        tile_offsets=tile_offsets_arr,
        total_fragments=np.array(total_fragments, dtype=np.uint64),
    )

    # Write done marker
    done_marker = os.path.join(shard_dir, f"{library}.done")
    Path(done_marker).touch()

    log.info(f"Sample {library}: {len(pos_arr)} nnz, total_fragments={total_fragments}")
    return library, True


def _get_overlapping_blacklist_regions(blacklist_rdf, contig, start, stop):
    """Return a list of Region objects from the blacklist that overlap [start, stop)."""
    from fragmentomics_tools.region import Region

    regions = []
    for row in blacklist_rdf.itertuples():
        if str(row.contig) != contig:
            continue
        bl_start = int(row.start)
        bl_stop = int(row.stop)
        if bl_start < stop and bl_stop > start:
            regions.append(Region(chrom=contig, start=bl_start, stop=bl_stop, strand="."))
    return regions


# ── Phase A orchestrator ─────────────────────────────────────────────────

def draw_samples(sheet: pd.DataFrame, config: PlumbingConfig):
    """Draw train + heldout samples from the sheet (pre-Phase A).

    Returns a DataFrame with an added 'role' column:
        0 = train, 1 = heldout
    """
    rng = np.random.default_rng(config.seed)
    n_total = config.n_train_samples + config.n_heldout_samples
    if n_total > len(sheet):
        raise ValueError(
            f"Need {n_total} samples but sheet has only {len(sheet)}. "
            f"Increase the pool or decrease n_train_samples/n_heldout_samples."
        )
    chosen_idx = rng.choice(len(sheet), size=n_total, replace=False)
    chosen_idx.sort()
    drawn = sheet.iloc[chosen_idx].copy().reset_index(drop=True)
    roles = np.zeros(n_total, dtype=np.uint8)
    roles[config.n_train_samples:] = 1  # heldout
    drawn["role"] = roles
    return drawn


def run_phase_a(
    config: PlumbingConfig,
    drawn_sheet: pd.DataFrame,
    tiles: list,
    shard_dir: str,
    ref: str,
    n_workers: int = 8,
):
    """Run Phase A: parallel per-sample shard generation."""
    os.makedirs(shard_dir, exist_ok=True)

    # Check for .error files
    errors = list(Path(shard_dir).glob("*.error"))
    if errors:
        raise RuntimeError(
            f"Phase A cannot proceed: {len(errors)} error file(s) exist in {shard_dir}. "
            f"Fix or remove: {[e.name for e in errors]}"
        )

    sample_rows = drawn_sheet.to_dict("records")

    if n_workers <= 1:
        results = []
        for row in sample_rows:
            results.append(
                _worker_process_sample(row, tiles, config, shard_dir, ref)
            )
    else:
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _worker_process_sample, row, tiles, config, shard_dir, ref
                ): row["library"]
                for row in sample_rows
            }
            for future in as_completed(futures):
                results.append(future.result())

    failed = [lib for lib, ok in results if not ok]
    if failed:
        raise RuntimeError(f"Phase A failed for: {failed}")

    log.info(f"Phase A complete: {len(results)} samples processed")
    return results


# ── Phase B — store assembly ─────────────────────────────────────────────

def assign_region_splits(
    tiles: list,
    config: PlumbingConfig,
):
    """Assign splits to regions, then propagate to tiles.

    Split values: 0=train, 1=val, 2=heldout_inactive, 3=positive_control.
    Regions from a 'positive_control' BED get split=3 unconditionally.
    Other regions are split at REGION granularity using config.region_fracs.
    """
    rng = np.random.default_rng(config.seed)

    # Collect unique regions and their split_name
    region_info = {}
    for tile in tiles:
        rid = tile["region_id"]
        if rid not in region_info:
            region_info[rid] = tile["split_name"]

    # Assign splits per region
    region_splits = {}
    non_pc_regions = [
        rid for rid, sn in region_info.items() if sn != "positive_control"
    ]

    # Shuffle for random assignment
    rng.shuffle(non_pc_regions)

    n = len(non_pc_regions)
    train_frac, val_frac, heldout_frac = config.region_fracs
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    # Rest goes to heldout_inactive

    for i, rid in enumerate(non_pc_regions):
        if i < n_train:
            region_splits[rid] = 0  # train
        elif i < n_train + n_val:
            region_splits[rid] = 1  # val
        else:
            region_splits[rid] = 2  # heldout_inactive

    # Positive control regions
    for rid, sn in region_info.items():
        if sn == "positive_control":
            region_splits[rid] = 3

    # Propagate to tiles
    tile_splits = np.array(
        [region_splits[tile["region_id"]] for tile in tiles], dtype=np.uint8
    )
    return tile_splits


def run_phase_b(
    config: PlumbingConfig,
    drawn_sheet: pd.DataFrame,
    tiles: list,
    shard_dir: str,
    output_dir: str,
    ref: str,
):
    """Phase B: assemble shards into a zarr store.

    Builds under <output>.building/, then atomic-renames to final name.
    """
    store_name = config.store_name()
    building_path = os.path.join(output_dir, store_name.replace(".zarr", ".building"))
    final_path = os.path.join(output_dir, store_name)

    S = len(drawn_sheet)
    T = len(tiles)

    log.info(f"Phase B: assembling {S} samples × {T} tiles")

    # ── Load all shards to compute total nnz ─────────────────────────────
    shards = {}
    total_fragments_map = {}
    total_nnz = 0
    for _, row in drawn_sheet.iterrows():
        library = row["library"]
        shard_path = os.path.join(shard_dir, f"{library}.npz")
        shard = np.load(shard_path)
        shards[library] = shard
        total_fragments_map[library] = int(shard["total_fragments"])
        total_nnz += len(shard["pos"])

    # ── Create store ─────────────────────────────────────────────────────
    if os.path.exists(building_path):
        shutil.rmtree(building_path)
    root = create_store(building_path, config, T, S, total_nnz)

    # ── Write /tiles/ metadata ───────────────────────────────────────────
    tile_contigs = np.array([t["contig"] for t in tiles])
    tile_starts = np.array([t["start"] for t in tiles], dtype=np.int64)
    tile_stops = np.array([t["stop"] for t in tiles], dtype=np.int64)
    tile_strands = np.array(["." for _ in tiles])  # always strandless
    tile_region_ids = np.array([t["region_id"] for t in tiles])

    # Assign region splits
    tile_splits = assign_region_splits(tiles, config)

    root["tiles/contig"][:] = tile_contigs
    root["tiles/start"][:] = tile_starts
    root["tiles/stop"][:] = tile_stops
    root["tiles/strand"][:] = tile_strands
    root["tiles/region_id"][:] = tile_region_ids
    root["tiles/split"][:] = tile_splits

    # ── Write sequence data ──────────────────────────────────────────────
    fasta = pysam.FastaFile(config.fasta)
    l_seq = config.l_seq
    seq_margin = config.jitter + config.rf_budget

    for t_idx, tile in enumerate(tiles):
        # Sequence extent: tile center ± (jitter + rf_budget) = L_SEQ
        seq_start = tile["start"] - seq_margin
        seq_stop = tile["stop"] + seq_margin
        seq_str = fasta.fetch(tile["contig"], max(0, seq_start), seq_stop)

        # Pad if near chromosome boundary
        left_pad = max(0, -seq_start)
        right_pad = l_seq - len(seq_str) - left_pad
        if left_pad > 0 or right_pad > 0:
            seq_str = "N" * left_pad + seq_str + "N" * max(0, right_pad)

        seq_bytes = np.frombuffer(seq_str.upper().encode("ascii"), dtype=np.uint8)
        if len(seq_bytes) != l_seq:
            seq_bytes = np.pad(
                seq_bytes, (0, l_seq - len(seq_bytes)),
                constant_values=ord("N"),
            )[:l_seq]
        root["tiles/seq"][t_idx] = seq_bytes

    fasta.close()

    # ── Write blacklist mask ─────────────────────────────────────────────
    l_target = config.l_target
    mask_margin = config.jitter
    # SYMMETRIC blacklist expansion (approved 2026-08-27): the position mask
    # marks invalid EVERY position within blacklist_expansion bp of any
    # blacklist region — the SAME expansion Phase A applies to fragment masking
    # (mask_overlapping_fragments(expansion=...)).  This excludes fragment-
    # dropped zones from both the likelihood support and N.
    expansion = config.blacklist_expansion

    if config.blacklist_bed:
        from fragmentomics_tools.dataframe import RegionDataFrame
        blacklist_rdf = RegionDataFrame.from_bed(config.blacklist_bed, ref=ref)
    else:
        blacklist_rdf = None

    for t_idx, tile in enumerate(tiles):
        mask = np.ones(l_target, dtype=bool)
        count_start = tile["start"] - mask_margin  # L_TARGET frame origin (genomic)

        # Positions outside the contig are invalid (they carry zero counts).
        left_invalid = max(0, -count_start)
        if left_invalid > 0:
            mask[:left_invalid] = False
        contig_len = _contig_length(ref, tile["contig"])
        if contig_len is not None:
            right_valid = contig_len - count_start  # first out-of-contig position
            if right_valid < l_target:
                mask[max(0, right_valid):] = False

        if blacklist_rdf is not None:
            # Widen the overlap query by `expansion` so a blacklist region just
            # outside the L_TARGET frame whose expanded zone reaches into it is
            # still caught.
            bl_regions = _get_overlapping_blacklist_regions(
                blacklist_rdf, tile["contig"],
                count_start - expansion, tile["stop"] + mask_margin + expansion,
            )
            for bl_reg in bl_regions:
                # Expand each blacklist region by `expansion` bp on both sides,
                # then convert to local (L_TARGET-frame) coordinates.
                local_start = max(0, bl_reg.start - expansion - count_start)
                local_stop = min(l_target, bl_reg.stop + expansion - count_start)
                if local_start < local_stop:
                    mask[local_start:local_stop] = False
        root["tiles/mask"][t_idx] = mask

    # ── Write /samples/ metadata ─────────────────────────────────────────
    libraries = list(drawn_sheet["library"])
    root["samples/library"][:] = np.array(libraries)
    root["samples/seqrun"][:] = np.array(list(drawn_sheet["seqrun"]))
    root["samples/endo_category"][:] = np.array(list(drawn_sheet["endo_category"]))
    root["samples/h5_path"][:] = np.array(list(drawn_sheet["h5_path"]))
    root["samples/total_fragments"][:] = np.array(
        [total_fragments_map[lib] for lib in libraries], dtype=np.uint64
    )

    # Depth filter: downgrade low-depth samples to role=2 (condition from §6)
    roles = np.array(list(drawn_sheet["role"]), dtype=np.uint8)
    for s_idx, lib in enumerate(libraries):
        tf = total_fragments_map[lib]
        if tf < config.min_total_fragments:
            roles[s_idx] = 2  # dropped_low_depth
    root["samples/role"][:] = roles

    # ── Write /counts/ CSR ───────────────────────────────────────────────
    indptr = np.zeros(S * T + 1, dtype=np.int64)
    all_pos = []
    all_track = []
    all_data = []
    cursor = 0

    for s_idx, lib in enumerate(libraries):
        shard = shards[lib]
        shard_pos = shard["pos"]
        shard_track = shard["track"]
        shard_data = shard["data"]
        tile_offsets = shard["tile_offsets"]  # (tile_idx, nnz_start, nnz_end)

        # Index tile offsets by tile index for O(1) lookup (avoids O(T^2) scan)
        offsets_by_tile = {
            int(row[0]): (int(row[1]), int(row[2])) for row in tile_offsets
        }

        for t_idx in range(T):
            u = s_idx * T + t_idx
            # Find this tile's data in the shard
            nnz_start, nnz_end = offsets_by_tile.get(t_idx, (0, 0))

            n_entries = nnz_end - nnz_start
            indptr[u + 1] = indptr[u] + n_entries
            if n_entries > 0:
                all_pos.append(shard_pos[nnz_start:nnz_end])
                all_track.append(shard_track[nnz_start:nnz_end])
                all_data.append(shard_data[nnz_start:nnz_end])

    if all_pos:
        final_pos = np.concatenate(all_pos)
        final_track = np.concatenate(all_track)
        final_data = np.concatenate(all_data)
    else:
        final_pos = np.empty(0, dtype=np.uint16)
        final_track = np.empty(0, dtype=np.uint8)
        final_data = np.empty(0, dtype=np.uint16)

    actual_nnz = len(final_pos)

    # Resize CSR arrays if nnz estimate was off
    if actual_nnz != total_nnz:
        from background_model.store import _create_array
        _create_array(root["counts"], "pos", shape=(actual_nnz,), dtype="uint16", chunks=(1 << 20,), overwrite=True)
        _create_array(root["counts"], "track", shape=(actual_nnz,), dtype="uint8", chunks=(1 << 20,), overwrite=True)
        _create_array(root["counts"], "data", shape=(actual_nnz,), dtype="uint16", chunks=(1 << 20,), overwrite=True)

    root["counts/indptr"][:] = indptr
    if actual_nnz > 0:
        root["counts/pos"][:] = final_pos
        root["counts/track"][:] = final_track
        root["counts/data"][:] = final_data

    # ── Compute /totals/N ────────────────────────────────────────────────
    N = np.zeros((S, T, C), dtype=np.uint32)
    for s_idx in range(S):
        for t_idx in range(T):
            pos_arr, track_arr, data_arr = _csr_slice_from_indptr(
                indptr, final_pos, final_track, final_data, s_idx, t_idx, T
            )
            y_dense = densify_counts(pos_arr, track_arr, data_arr, C, l_target)
            mask = np.asarray(root["tiles/mask"][t_idx])
            N[s_idx, t_idx] = compute_N_for_tile(
                y_dense, mask, config.tile_size, l_target
            )
    root["totals/N"][:] = N

    # ── Attrs: Phase B params + split_version ────────────────────────────
    split_version = increment_split_version(root)
    record_phase_b_params(root, config)

    # ── Write sidecar config JSON ────────────────────────────────────────
    sidecar_path = os.path.join(
        output_dir,
        store_name.replace(".zarr", ".config.json"),
    )
    with open(sidecar_path, "w") as f:
        f.write(config.full_config_json())

    # ── Atomic rename ────────────────────────────────────────────────────
    if os.path.exists(final_path):
        shutil.rmtree(final_path)
    os.rename(building_path, final_path)

    log.info(f"Phase B complete: {final_path} (split_version={split_version})")
    return final_path


def _csr_slice_from_indptr(indptr, all_pos, all_track, all_data, s_idx, t_idx, T):
    """Get CSR triples from in-memory arrays (used during Phase B assembly)."""
    u = s_idx * T + t_idx
    lo = int(indptr[u])
    hi = int(indptr[u + 1])
    if lo == hi:
        return np.empty(0, "uint16"), np.empty(0, "uint8"), np.empty(0, "uint16")
    return all_pos[lo:hi], all_track[lo:hi], all_data[lo:hi]


# ── Full pipeline ────────────────────────────────────────────────────────

def run_preprocess(
    config: PlumbingConfig,
    output_dir: str,
    ref: str = "hg38",
    n_workers: int = 8,
):
    """Run the full preprocessing pipeline (Phase A + Phase B)."""
    sheet = pd.read_csv(config.sample_sheet, sep="\t")
    drawn_sheet = draw_samples(sheet, config)

    tiles = build_tiles(
        config.region_beds, config.tile_size, config.jitter, config.rf_budget, ref
    )

    shard_dir = os.path.join(output_dir, f"shards_{config.config_hash8()}")
    run_phase_a(config, drawn_sheet, tiles, shard_dir, ref, n_workers)

    store_path = run_phase_b(config, drawn_sheet, tiles, shard_dir, output_dir, ref)
    return store_path


# ── CLI ──────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="Background model preprocessing")
    parser.add_argument("--config", required=True, help="Config JSON path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--ref", default="hg38", help="Reference genome name")
    parser.add_argument("--workers", type=int, default=8, help="Number of Phase A workers")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        config = PlumbingConfig.from_json(f.read())

    logging.basicConfig(level=logging.INFO)
    run_preprocess(config, args.output_dir, args.ref, args.workers)


if __name__ == "__main__":
    main()
