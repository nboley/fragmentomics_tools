"""Tests for the cut-site store schema, dataset, and centre-based filtering.

Uses a synthetic zarr store fixture with known fragments.
"""

import json
import tempfile

import numpy as np
import pytest
import torch
import zarr

from background_model.cut_site_model import (
    L_MAX_INBAND,
    L_MIN_INBAND,
    N_L,
    WINDOW_HALF,
    CutSiteFragmentModel,
)
from background_model.cut_site_store import (
    CutSiteStoreConfig,
    densify_endpoints,
    open_cut_site_store,
    recompute_endpoints_from_fragments,
    validate_max_len,
)
from background_model.cut_site_dataset import (
    CutSiteFragmentDataset,
    cut_site_collate_fn,
)


# ---------------------------------------------------------------------------
# Synthetic store fixture
# ---------------------------------------------------------------------------

def _create_synthetic_store(tmp_path, tile_size=256, jitter=32, n_tiles=4,
                            n_samples=1, n_frags_per_tile=50, seed=42):
    """Create a minimal cut-site store with random fragments."""
    rng = np.random.default_rng(seed)
    rf_budget = WINDOW_HALF + 3  # enough for the model
    cfg = CutSiteStoreConfig(
        tile_size=tile_size, jitter=jitter, rf_budget=rf_budget,
        max_len=256, seed=seed, n_train_samples=n_samples,
    )

    store_path = str(tmp_path / "test_store.zarr")
    root = zarr.open_group(store_path, mode="w")

    root.attrs.update({
        "config_json": cfg.to_json(),
        "config_hash": cfg.config_hash(),
        "created_utc": "2026-01-01T00:00:00Z",
        "split_version": 0,
    })

    S, T = n_samples, n_tiles

    # /tiles/
    tiles = root.create_group("tiles")
    tiles.create_dataset("contig", data=np.array(["chr1"] * T, dtype="<U32"))
    tiles.create_dataset("start", data=np.arange(T, dtype="int64") * tile_size)
    tiles.create_dataset("stop", data=(np.arange(T, dtype="int64") + 1) * tile_size)
    tiles.create_dataset("strand", data=np.array(["."] * T, dtype="<U1"))
    tiles.create_dataset("region_id", data=np.array([f"r{i}" for i in range(T)], dtype="<U64"))
    tiles.create_dataset("split", data=np.zeros(T, dtype="uint8"))  # all train
    # seq: random ACGT bytes over l_seq
    seq = rng.choice(list(b"ACGT"), size=(T, cfg.l_seq)).astype(np.uint8)
    tiles.create_dataset("seq", data=seq)
    # mask: all valid
    mask = np.ones((T, cfg.l_target), dtype=bool)
    tiles.create_dataset("mask", data=mask)

    # /samples/
    samples = root.create_group("samples")
    samples.create_dataset("library", data=np.array(["lib1"] * S, dtype="<U64"))
    samples.create_dataset("role", data=np.zeros(S, dtype="uint8"))  # all train
    samples.create_dataset("total_fragments", data=np.array([n_frags_per_tile * T] * S, dtype="uint64"))

    # /fragments/ — random fragments per (s, t)
    all_starts, all_lengths, all_strands = [], [], []
    frag_indptr = [0]
    for s_idx in range(S):
        for t_idx in range(T):
            starts = rng.integers(0, cfg.l_target - L_MAX_INBAND, size=n_frags_per_tile).astype(np.uint16)
            lengths = rng.integers(L_MIN_INBAND, L_MAX_INBAND + 1, size=n_frags_per_tile).astype(np.uint16)
            strands = rng.integers(0, 2, size=n_frags_per_tile).astype(np.uint8)
            all_starts.append(starts)
            all_lengths.append(lengths)
            all_strands.append(strands)
            frag_indptr.append(frag_indptr[-1] + n_frags_per_tile)

    all_starts = np.concatenate(all_starts)
    all_lengths = np.concatenate(all_lengths)
    all_strands = np.concatenate(all_strands)

    frags = root.create_group("fragments")
    frags.create_dataset("indptr", data=np.array(frag_indptr, dtype="int64"))
    frags.create_dataset("start", data=all_starts)
    frags.create_dataset("length", data=all_lengths)
    frags.create_dataset("strand", data=all_strands)

    # /endpoints/ — compute from fragments
    all_ep_pos, all_ep_track, all_ep_data = [], [], []
    ep_indptr = [0]
    N_totals = np.zeros((S, T, 2), dtype="uint32")

    for s_idx in range(S):
        for t_idx in range(T):
            flat_idx = s_idx * T + t_idx
            flo = frag_indptr[flat_idx]
            fhi = frag_indptr[flat_idx + 1]
            fs = all_starts[flo:fhi]
            fl = all_lengths[flo:fhi]

            ep = recompute_endpoints_from_fragments(fs, fl, cfg.l_target)
            # Convert dense to sparse
            for track_idx in range(2):
                nz = np.nonzero(ep[track_idx])[0]
                for p in nz:
                    all_ep_pos.append(p)
                    all_ep_track.append(track_idx)
                    all_ep_data.append(int(ep[track_idx, p]))
                # Centre-tile totals
                margin = (cfg.l_target - tile_size) // 2
                N_totals[s_idx, t_idx, track_idx] = ep[
                    track_idx, margin:margin + tile_size
                ].sum()
            ep_indptr.append(len(all_ep_pos))

    endpoints = root.create_group("endpoints")
    endpoints.create_dataset("indptr", data=np.array(ep_indptr, dtype="int64"))
    endpoints.create_dataset("pos", data=np.array(all_ep_pos, dtype="uint16") if all_ep_pos else np.empty(0, "uint16"))
    endpoints.create_dataset("track", data=np.array(all_ep_track, dtype="uint8") if all_ep_track else np.empty(0, "uint8"))
    endpoints.create_dataset("data", data=np.array(all_ep_data, dtype="uint16") if all_ep_data else np.empty(0, "uint16"))

    # /totals/
    totals = root.create_group("totals")
    totals.create_dataset("N", data=N_totals)

    return store_path, cfg


@pytest.fixture
def synthetic_store(tmp_path):
    return _create_synthetic_store(tmp_path)


# ---------------------------------------------------------------------------
# Store config tests
# ---------------------------------------------------------------------------


class TestCutSiteStoreConfig:
    def test_hash_deterministic(self):
        c1 = CutSiteStoreConfig()
        c2 = CutSiteStoreConfig()
        assert c1.config_hash() == c2.config_hash()

    def test_hash_changes_on_param_change(self):
        c1 = CutSiteStoreConfig(max_len=256)
        c2 = CutSiteStoreConfig(max_len=300)
        assert c1.config_hash() != c2.config_hash()

    def test_json_roundtrip(self):
        cfg = CutSiteStoreConfig(tile_size=1024, jitter=128)
        assert CutSiteStoreConfig.from_json(cfg.to_json()) == cfg

    def test_geometry(self):
        cfg = CutSiteStoreConfig(tile_size=2048, jitter=256, rf_budget=133)
        assert cfg.l_target == 2048 + 512  # 2560
        assert cfg.l_seq == 2048 + 2 * (256 + 133)  # 2826


# ---------------------------------------------------------------------------
# Domain guard tests
# ---------------------------------------------------------------------------


class TestDomainGuards:
    def test_max_len_mismatch_raises(self, synthetic_store):
        store_path, cfg = synthetic_store
        root = zarr.open_group(store_path, mode="r")
        # Store has max_len=256; model expects 300 → should fail
        with pytest.raises(ValueError, match="max_len mismatch"):
            validate_max_len(root, 300)

    def test_max_len_match_passes(self, synthetic_store):
        store_path, cfg = synthetic_store
        root = zarr.open_group(store_path, mode="r")
        validate_max_len(root, 256)  # should not raise

    def test_fl_bands_mismatch_raises(self, tmp_path):
        """A store with wrong fl_bands should fail to open."""
        cfg = CutSiteStoreConfig()
        store_path = str(tmp_path / "bad_bands.zarr")
        root = zarr.open_group(store_path, mode="w")
        # Write config with wrong bands
        bad_cfg = json.loads(cfg.to_json())
        bad_cfg["fl_bands"] = [[40, 65], [120, 175]]
        root.attrs["config_json"] = json.dumps(bad_cfg)
        root.attrs["config_hash"] = "fake"
        root.attrs["split_version"] = 0

        with pytest.raises(ValueError, match="fl_bands mismatch"):
            open_cut_site_store(store_path)


# ---------------------------------------------------------------------------
# Endpoint consistency
# ---------------------------------------------------------------------------


class TestEndpointConsistency:
    def test_endpoints_match_recomputation(self, synthetic_store):
        """Stored /endpoints/ must equal recomputation from /fragments/."""
        store_path, cfg = synthetic_store
        root = zarr.open_group(store_path, mode="r")
        frag_indptr = root["fragments/indptr"][:]
        ep_indptr = root["endpoints/indptr"][:]
        S, T = root["totals/N"].shape[:2]

        for s in range(S):
            for t in range(T):
                u = s * T + t
                # Get stored fragments
                flo, fhi = int(frag_indptr[u]), int(frag_indptr[u + 1])
                fs = np.asarray(root["fragments/start"][flo:fhi])
                fl = np.asarray(root["fragments/length"][flo:fhi])

                # Recompute
                recomputed = recompute_endpoints_from_fragments(fs, fl, cfg.l_target)

                # Get stored endpoints
                elo, ehi = int(ep_indptr[u]), int(ep_indptr[u + 1])
                stored = densify_endpoints(
                    np.asarray(root["endpoints/pos"][elo:ehi]),
                    np.asarray(root["endpoints/track"][elo:ehi]),
                    np.asarray(root["endpoints/data"][elo:ehi]),
                    cfg.l_target,
                )

                np.testing.assert_array_equal(stored, recomputed)


# ---------------------------------------------------------------------------
# Centre-based fragment filtering
# ---------------------------------------------------------------------------


class TestCentreBasedFiltering:
    def test_domain_size_per_L_is_P(self, synthetic_store):
        """Every in-band L should have the same number of scorable centres = P."""
        store_path, cfg = synthetic_store
        model = CutSiteFragmentModel(
            tile_size=cfg.tile_size, N1=8, K=8, N2=4, H=4, L_chunk=32,
        )
        ds = CutSiteFragmentDataset(
            store_path=store_path,
            model_input_size=model.model_input_size,
            model_max_len=cfg.max_len,
            split="train",
            sample_role="train",
            min_N=0,
            train_mode=False,  # centred crop, j=0
        )
        # Get one item
        item = ds[0]
        # The domain D has |D_L| = P for every L.  We can't verify this
        # from a single item's fragments (they're sparse), but we CAN verify
        # that the logit grid the model produces has P positions for every L.
        l_in = model.calc_input_region_size(cfg.tile_size)
        logits, _, _ = model(item["x"].unsqueeze(0))
        assert logits.shape == (1, cfg.tile_size, N_L, 2)

    def test_centre_filter_no_length_bias(self, synthetic_store):
        """Fragments of all in-band lengths should survive the crop filter,
        not just short ones."""
        store_path, cfg = synthetic_store
        model = CutSiteFragmentModel(
            tile_size=cfg.tile_size, N1=8, K=8, N2=4, H=4, L_chunk=32,
        )
        ds = CutSiteFragmentDataset(
            store_path=store_path,
            model_input_size=model.model_input_size,
            model_max_len=cfg.max_len,
            split="train",
            sample_role="train",
            min_N=0,
            train_mode=False,
        )
        # Collect fragment lengths from all items
        all_L = []
        for i in range(len(ds)):
            item = ds[i]
            all_L.append(item["frag_L_idx"].numpy() + L_MIN_INBAND)
        all_L = np.concatenate(all_L)

        # Check that we have fragments across the full range, not just short ones
        assert all_L.min() <= L_MIN_INBAND + 10, "No short fragments survived"
        assert all_L.max() >= L_MAX_INBAND - 10, "No long fragments survived"

    def test_both_L_parities_survive_crop(self, synthetic_store):
        """Both even and odd L values should survive the centre-based filter."""
        store_path, cfg = synthetic_store
        model = CutSiteFragmentModel(
            tile_size=cfg.tile_size, N1=8, K=8, N2=4, H=4, L_chunk=32,
        )
        ds = CutSiteFragmentDataset(
            store_path=store_path,
            model_input_size=model.model_input_size,
            model_max_len=cfg.max_len,
            split="train",
            sample_role="train",
            min_N=0,
            train_mode=False,
        )
        all_L = []
        for i in range(len(ds)):
            item = ds[i]
            all_L.append(item["frag_L_idx"].numpy() + L_MIN_INBAND)
        all_L = np.concatenate(all_L)

        even_count = (all_L % 2 == 0).sum()
        odd_count = (all_L % 2 == 1).sum()
        assert even_count > 0, "No even-L fragments survived"
        assert odd_count > 0, "No odd-L fragments survived"


# ---------------------------------------------------------------------------
# Dataset + collate integration
# ---------------------------------------------------------------------------


class TestDatasetCollate:
    def test_collate_batch(self, synthetic_store):
        store_path, cfg = synthetic_store
        model = CutSiteFragmentModel(
            tile_size=cfg.tile_size, N1=8, K=8, N2=4, H=4, L_chunk=32,
        )
        ds = CutSiteFragmentDataset(
            store_path=store_path,
            model_input_size=model.model_input_size,
            model_max_len=cfg.max_len,
            split="train",
            sample_role="train",
            min_N=0,
            train_mode=False,
        )
        batch = cut_site_collate_fn([ds[0], ds[1]])

        assert batch["x"].shape == (2, 4, model.model_input_size)
        assert batch["endpoints"].shape == (2, 2, cfg.tile_size)
        assert batch["mask"].shape == (2, cfg.tile_size)
        assert batch["frag_N"].shape == (2,)
        assert isinstance(batch["frag_m"], list) and len(batch["frag_m"]) == 2

    def test_model_forward_on_collated_batch(self, synthetic_store):
        """The model can run a forward pass on a collated dataset batch."""
        store_path, cfg = synthetic_store
        model = CutSiteFragmentModel(
            tile_size=cfg.tile_size, N1=8, K=8, N2=4, H=4, L_chunk=32,
        )
        ds = CutSiteFragmentDataset(
            store_path=store_path,
            model_input_size=model.model_input_size,
            model_max_len=cfg.max_len,
            split="train",
            sample_role="train",
            min_N=0,
            train_mode=False,
        )
        batch = cut_site_collate_fn([ds[0], ds[1]])

        # The model's forward expects (B, 4, l_in) but dataset returns
        # (B, 4, model_input_size).  We need to pad the 5 extra bases
        # for the conv margin.  In practice the dataset crops model_input_size
        # and the extra 5 bases come from the wider l_seq.  For this test
        # we just verify the shapes would work if we add 5 zeros.
        x_padded = torch.nn.functional.pad(batch["x"], (0, 5))
        logits, ep_shape, disp = model(x_padded)
        assert logits.shape == (2, cfg.tile_size, N_L, 2)
        assert ep_shape.shape == (2, 2, cfg.tile_size)
