"""Integration test for the cut-site store builder.

Uses the golden chr6 fragment h5 with real fragments.  Bypasses the reference
genome by injecting synthetic sequence (the builder's sequence extraction is
tested separately; this test exercises the fragment pipeline).
"""

import os
import tempfile

import numpy as np
import pytest
import zarr

from background_model.cut_site_model import CutSiteFragmentModel, _compute_geometry
from background_model.cut_site_store import (
    CutSiteStoreConfig,
    recompute_endpoints_from_fragments,
    validate_length_range,
)
from background_model.cut_site_dataset import CutSiteFragmentDataset, cut_site_collate_fn
from fragmentomics_tools.fragment_array import RegionFragmentArray
from fragmentomics_tools.region import Region


GOLDEN_H5 = os.path.join(os.path.dirname(__file__), "data", "golden.small.chr6.frag.h5")


def _build_test_store(tmp_path, tile_size=2048, jitter=256, L_min=25, L_max=179):
    """Build a cut-site store from the golden h5 with synthetic sequence."""
    rf_budget = 133
    cfg = CutSiteStoreConfig(
        tile_size=tile_size, jitter=jitter, rf_budget=rf_budget,
        L_min=L_min, L_max=L_max, seed=42, n_train_samples=1,
    )

    # Regions in chr6 that have fragments in the golden h5
    region_starts = list(range(51510000, 51520000, cfg.l_target))
    regions = [
        Region("chr6", s, s + cfg.l_target, strand=None) for s in region_starts
    ]

    # Extract fragments, keep only regions with data
    rng = np.random.default_rng(42)
    valid_regions = []
    frag_data = []  # list of (start, length, strand) per region

    for r in regions:
        rfa = RegionFragmentArray.from_fragments_h5(GOLDEN_H5, r)
        if rfa.n_fragments == 0:
            continue
        starts = (rfa.starts - r.start).astype(np.int64)
        lengths = rfa.lengths.astype(np.int64)
        strands = np.array(
            [0 if s in ("+", b"+") else 1 for s in rfa.fragment_strands], dtype=np.uint8
        )
        keep = (lengths >= L_min) & (lengths <= L_max)
        centres = starts + lengths // 2
        keep &= (centres >= 0) & (centres < cfg.l_target)
        keep &= (starts >= 0)  # uint16 can't hold negatives
        if keep.sum() == 0:
            continue
        valid_regions.append(r)
        frag_data.append((
            starts[keep].astype(np.uint16),
            lengths[keep].astype(np.uint16),
            strands[keep],
        ))

    T = len(valid_regions)
    if T == 0:
        pytest.skip("No fragments in golden h5 for these regions")
    S = 1

    # Build the store
    store_path = str(tmp_path / "golden_test.zarr")
    root = zarr.open_group(store_path, mode="w")
    root.attrs.update({
        "config_json": cfg.to_json(),
        "config_hash": cfg.config_hash(),
        "created_utc": "2026-01-01T00:00:00Z",
        "split_version": 0,
    })

    # /tiles/ — synthetic sequence (random ACGT), real mask
    tiles = root.create_group("tiles")
    tiles.create_dataset("contig", data=np.array([r.chrom for r in valid_regions], dtype="<U32"))
    tiles.create_dataset("start", data=np.array([r.start for r in valid_regions], dtype="int64"))
    tiles.create_dataset("stop", data=np.array([r.stop for r in valid_regions], dtype="int64"))
    tiles.create_dataset("strand", data=np.array(["."] * T, dtype="<U1"))
    tiles.create_dataset("region_id", data=np.array([f"r{i}" for i in range(T)], dtype="<U64"))
    split_arr = np.zeros(T, dtype="uint8")
    if T > 1:
        split_arr[T - 1] = 1  # last region is val
    tiles.create_dataset("split", data=split_arr)
    # Synthetic sequence: random ACGT bytes
    seq = rng.choice(list(b"ACGT"), size=(T, cfg.l_seq)).astype(np.uint8)
    tiles.create_dataset("seq", data=seq)
    tiles.create_dataset("mask", data=np.ones((T, cfg.l_target), dtype=bool))

    # /samples/
    samples = root.create_group("samples")
    samples.create_dataset("library", data=np.array(["golden_chr6"], dtype="<U256"))
    samples.create_dataset("role", data=np.zeros(S, dtype="uint8"))
    total_frags = sum(len(fd[0]) for fd in frag_data)
    samples.create_dataset("total_fragments", data=np.array([total_frags], dtype="uint64"))

    # /fragments/ + /endpoints/
    frag_indptr = [0]
    all_fs, all_fl, all_fst = [], [], []
    all_ep_pos, all_ep_track, all_ep_data = [], [], []
    ep_indptr = [0]
    N_totals = np.zeros((S, T, 2), dtype="uint32")

    for t, (fs, fl, fst) in enumerate(frag_data):
        all_fs.append(fs)
        all_fl.append(fl)
        all_fst.append(fst)
        frag_indptr.append(frag_indptr[-1] + len(fs))

        ep = recompute_endpoints_from_fragments(fs, fl, cfg.l_target)
        for track_idx in range(2):
            nz = np.nonzero(ep[track_idx])[0]
            for p in nz:
                all_ep_pos.append(p)
                all_ep_track.append(track_idx)
                all_ep_data.append(int(ep[track_idx, p]))
            margin = (cfg.l_target - tile_size) // 2
            N_totals[0, t, track_idx] = int(
                ep[track_idx, margin:margin + tile_size].sum()
            )
        ep_indptr.append(len(all_ep_pos))

    frags = root.create_group("fragments")
    frags.create_dataset("indptr", data=np.array(frag_indptr, dtype="int64"))
    frags.create_dataset("start", data=np.concatenate(all_fs) if all_fs else np.empty(0, "uint16"))
    frags.create_dataset("length", data=np.concatenate(all_fl) if all_fl else np.empty(0, "uint16"))
    frags.create_dataset("strand", data=np.concatenate(all_fst) if all_fst else np.empty(0, "uint8"))

    endpoints = root.create_group("endpoints")
    endpoints.create_dataset("indptr", data=np.array(ep_indptr, dtype="int64"))
    endpoints.create_dataset("pos", data=np.array(all_ep_pos, dtype="uint16") if all_ep_pos else np.empty(0, "uint16"))
    endpoints.create_dataset("track", data=np.array(all_ep_track, dtype="uint8") if all_ep_track else np.empty(0, "uint8"))
    endpoints.create_dataset("data", data=np.array(all_ep_data, dtype="uint16") if all_ep_data else np.empty(0, "uint16"))

    totals = root.create_group("totals")
    totals.create_dataset("N", data=N_totals)

    return store_path, cfg, T, total_frags


class TestBuilderIntegration:
    """End-to-end: golden h5 → store → dataset → model forward."""

    def test_store_roundtrip(self, tmp_path):
        store_path, cfg, T, total_frags = _build_test_store(tmp_path)
        root = zarr.open_group(store_path, mode="r")

        assert root["tiles/contig"].shape[0] == T
        assert root["fragments/indptr"].shape[0] == T + 1
        assert root["endpoints/indptr"].shape[0] == T + 1
        assert root["totals/N"].shape == (1, T, 2)

        # Config round-trip
        stored_cfg = CutSiteStoreConfig.from_json(root.attrs["config_json"])
        assert stored_cfg == cfg

        # Length range guard
        validate_length_range(root, cfg.L_min, cfg.L_max)

    def test_endpoint_consistency(self, tmp_path):
        """Stored endpoints match recomputation from fragments."""
        store_path, cfg, T, _ = _build_test_store(tmp_path)
        root = zarr.open_group(store_path, mode="r")
        frag_indptr = root["fragments/indptr"][:]
        ep_indptr = root["endpoints/indptr"][:]

        for t in range(T):
            flo, fhi = int(frag_indptr[t]), int(frag_indptr[t + 1])
            fs = np.asarray(root["fragments/start"][flo:fhi])
            fl = np.asarray(root["fragments/length"][flo:fhi])
            recomputed = recompute_endpoints_from_fragments(fs, fl, cfg.l_target)

            elo, ehi = int(ep_indptr[t]), int(ep_indptr[t + 1])
            from background_model.cut_site_store import densify_endpoints
            stored = densify_endpoints(
                np.asarray(root["endpoints/pos"][elo:ehi]),
                np.asarray(root["endpoints/track"][elo:ehi]),
                np.asarray(root["endpoints/data"][elo:ehi]),
                cfg.l_target,
            )
            np.testing.assert_array_equal(stored, recomputed)

    def test_dataset_loads_from_store(self, tmp_path):
        store_path, cfg, T, _ = _build_test_store(tmp_path)
        model = CutSiteFragmentModel(
            L_min=cfg.L_min, L_max=cfg.L_max,
            tile_size=cfg.tile_size, N1=8, K=8, N2=4, H=4, L_chunk=32,
        )
        ds = CutSiteFragmentDataset(
            store_path=store_path,
            model_input_size=model.model_input_size,
            model_L_min=cfg.L_min, model_L_max=cfg.L_max,
            split="train", sample_role="train",
            min_N=0, train_mode=False,
        )
        assert len(ds) > 0
        item = ds[0]
        assert item["frag_N"].item() > 0
        assert item["x"].shape[0] == 4

    def test_model_forward_on_real_fragments(self, tmp_path):
        """Model forward pass on a batch built from real fragment data."""
        store_path, cfg, T, _ = _build_test_store(tmp_path)
        model = CutSiteFragmentModel(
            L_min=cfg.L_min, L_max=cfg.L_max,
            tile_size=cfg.tile_size, N1=8, K=8, N2=4, H=4, L_chunk=32,
        )
        ds = CutSiteFragmentDataset(
            store_path=store_path,
            model_input_size=model.model_input_size,
            model_L_min=cfg.L_min, model_L_max=cfg.L_max,
            split="train", sample_role="train",
            min_N=0, train_mode=False,
        )
        batch = cut_site_collate_fn([ds[0]])
        import torch
        n_L, _, _, _ = _compute_geometry(cfg.L_min, cfg.L_max)
        logits, ep_shape, disp = model(batch["x"])
        assert logits.shape == (1, cfg.tile_size, n_L, 2)
        assert torch.isfinite(logits).all()

        # Verify fragments can index into the logit grid
        frag_m = batch["frag_m"][0]
        frag_L_idx = batch["frag_L_idx"][0]
        frag_s = batch["frag_s"][0]
        assert (frag_m >= 0).all() and (frag_m < cfg.tile_size).all()
        assert (frag_L_idx >= 0).all() and (frag_L_idx < n_L).all()
        assert (frag_s >= 0).all() and (frag_s <= 1).all()

    def test_fragments_are_real(self, tmp_path):
        """Sanity: the stored fragments actually came from the h5."""
        store_path, cfg, T, total_frags = _build_test_store(tmp_path)
        assert total_frags > 0, "No fragments extracted from golden h5"
        root = zarr.open_group(store_path, mode="r")
        lengths = np.asarray(root["fragments/length"][:])
        assert (lengths >= cfg.L_min).all()
        assert (lengths <= cfg.L_max).all()
        strands = np.asarray(root["fragments/strand"][:])
        assert set(np.unique(strands)).issubset({0, 1})


class TestExtractWorkerVsLibrary:
    """Verify _extract_worker produces the same fragments as the library path."""

    def test_worker_matches_RegionFragmentArray(self):
        """_extract_worker output must match RegionFragmentArray for the same region."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from scripts.build_cut_site_store import _extract_worker
        from fragmentomics_tools.fragment_array import RegionFragmentArray

        L_min, L_max = 25, 179
        # Region that has fragments in the golden h5
        chrom, reg_start, reg_stop = "chr6", 51512560, 51512560 + 2560

        # Builder path
        worker_starts, worker_lengths, worker_strands = _extract_worker(
            (GOLDEN_H5, chrom, reg_start, reg_stop, L_min, L_max, True)
        )

        # Library path — same filters as the builder
        r = Region(chrom, reg_start, reg_stop, strand=None)
        rfa = RegionFragmentArray.from_fragments_h5(GOLDEN_H5, r)
        if rfa.n_fragments > 0:
            rfa = rfa.drop_duplicate_fragments()
            lib_starts = (rfa.starts - reg_start).astype(np.int64)
            lib_lengths = rfa.lengths.astype(np.int64)
            keep = (lib_lengths >= L_min) & (lib_lengths <= L_max)
            centres = lib_starts + lib_lengths // 2
            keep &= (centres >= 0) & (centres < 2560)
            keep &= (lib_starts >= 0)
            lib_starts = lib_starts[keep]
        else:
            lib_starts = np.array([], dtype=np.int64)

        # Compare counts — they should match (same h5, same region, same filters)
        assert len(worker_starts) == len(lib_starts), (
            f"Count mismatch: worker={len(worker_starts)}, library={len(lib_starts)}"
        )
        if len(worker_starts) > 0:
            # Sort both by start for comparison
            w_order = np.argsort(worker_starts)
            assert np.array_equal(
                np.sort(worker_starts), np.sort(lib_starts.astype(np.uint16))
            ), "Start positions differ between worker and library paths"
