"""Pin the four admission boundaries of ``admit_fragments`` / ``crop_origin``.

A hand-mutation exercise flipped each of these and all cut-site test files
stayed green:
    (p_crop < tile_size)        -> (p_crop <= tile_size)
    (p_crop >= 0)                -> (p_crop >= 1)
    (frag_length <= L_max)       -> (frag_length < L_max)
    (frag_length >= L_min)       -> (frag_length > L_min)
This file pins the rule at the unit level (``admit_fragments``/``crop_origin``
directly) and at the loader level (``CutSiteFragmentDataset``), so each
boundary has a test that goes red under its mutation.
"""

import numpy as np
import pytest
import zarr

from background_model.band_model.store import _create_array
from background_model.constants import L_MAX, L_MIN
from background_model.cut_site_model.dataset import (
    CutSiteFragmentDataset,
    admit_fragments,
    crop_origin,
)
from background_model.cut_site_model.model import CutSiteFragmentModel, WINDOW_HALF
from background_model.cut_site_model.store import (
    CutSiteStoreConfig,
    recompute_endpoints_from_fragments,
)
from background_model_core import calculate_start_and_stop_from_jitter, jitter_matrix
from fragmentomics_tools.region import one_hot_encode_sequences


# ---------------------------------------------------------------------------
# Synthetic store with exactly-placed fragments (one sample, one tile)
# ---------------------------------------------------------------------------


def _build_store_with_fragments(tmp_path, starts, lengths, strands=None,
                                 tile_size=256, jitter=32,
                                 L_min=L_MIN, L_max=L_MAX, seed=0):
    """Build a 1-sample, 1-tile cut-site store holding exactly the given
    fragment rows, bypassing ``build_store.py`` so admission boundaries can
    be placed precisely.  Modelled on
    ``tests/test_cut_site_store_dataset.py::_create_synthetic_store``.
    """
    rng = np.random.default_rng(seed)
    starts = np.asarray(starts, dtype=np.uint16)
    lengths = np.asarray(lengths, dtype=np.uint16)
    n = len(starts)
    strands = (np.zeros(n, dtype=np.uint8) if strands is None
               else np.asarray(strands, dtype=np.uint8))

    rf_budget = WINDOW_HALF + 5
    cfg = CutSiteStoreConfig(tile_size=tile_size, jitter=jitter,
                              rf_budget=rf_budget, L_min=L_min, L_max=L_max,
                              seed=seed, n_train_samples=1)

    store_path = str(tmp_path / "exact_store.zarr")
    root = zarr.open_group(store_path, mode="w")
    root.attrs.update({
        "config_json": cfg.to_json(),
        "config_hash": cfg.config_hash(),
        "created_utc": "2026-01-01T00:00:00Z",
        "split_version": 0,
    })

    T = 1
    tiles = root.create_group("tiles")
    _create_array(tiles, "contig", data=np.array(["chr1"], dtype="<U32"))
    _create_array(tiles, "start", data=np.array([0], dtype="int64"))
    _create_array(tiles, "stop", data=np.array([tile_size], dtype="int64"))
    _create_array(tiles, "strand", data=np.array(["."], dtype="<U1"))
    _create_array(tiles, "region_id", data=np.array(["r0"], dtype="<U64"))
    _create_array(tiles, "split", data=np.zeros(T, dtype="uint8"))  # train
    seq = rng.choice(list(b"ACGT"), size=(T, cfg.l_seq)).astype(np.uint8)
    _create_array(tiles, "seq", data=seq)
    _create_array(tiles, "mask", data=np.ones((T, cfg.l_target), dtype=bool))

    samples = root.create_group("samples")
    _create_array(samples, "library", data=np.array(["lib1"], dtype="<U64"))
    _create_array(samples, "total_fragments", data=np.array([n], dtype="uint64"))

    frags = root.create_group("fragments")
    _create_array(frags, "indptr", data=np.array([0, n], dtype="int64"))
    _create_array(frags, "start", data=starts)
    _create_array(frags, "length", data=lengths)
    _create_array(frags, "strand", data=strands)

    ep = recompute_endpoints_from_fragments(starts, lengths, cfg.l_target)
    ep_pos, ep_track, ep_data = [], [], []
    for track_idx in range(2):
        for p in np.nonzero(ep[track_idx])[0]:
            ep_pos.append(p)
            ep_track.append(track_idx)
            ep_data.append(int(ep[track_idx, p]))

    endpoints = root.create_group("endpoints")
    _create_array(endpoints, "indptr", data=np.array([0, len(ep_pos)], dtype="int64"))
    _create_array(endpoints, "pos",
                   data=np.array(ep_pos, dtype="uint16") if ep_pos else np.empty(0, "uint16"))
    _create_array(endpoints, "track",
                   data=np.array(ep_track, dtype="uint8") if ep_track else np.empty(0, "uint8"))
    _create_array(endpoints, "data",
                   data=np.array(ep_data, dtype="uint16") if ep_data else np.empty(0, "uint16"))

    margin = (cfg.l_target - tile_size) // 2
    N_totals = np.zeros((1, 1, 2), dtype="uint32")
    for track_idx in range(2):
        N_totals[0, 0, track_idx] = ep[track_idx, margin:margin + tile_size].sum()
    totals = root.create_group("totals")
    _create_array(totals, "N", data=N_totals)

    return store_path, cfg


def _recover_j(ds, i, x):
    """Read the jitter an item was cropped at off its one-hot sequence.
    Modelled on ``tests/test_cut_site_store_dataset.py::_recover_j``."""
    seq = np.asarray(zarr.open_group(ds.store_path, mode="r")["tiles/seq"][ds.index[i][1]])
    hits = []
    for j in range(-ds.jitter, ds.jitter + 1):
        tok = np.ascontiguousarray(
            jitter_matrix(seq, j, ds.model_input_size), dtype=np.uint8)
        onehot = one_hot_encode_sequences([tok.tobytes()])[0].T
        if np.array_equal(onehot.astype(np.float32), x.numpy()):
            hits.append(j)
    assert len(hits) == 1, f"jitter not identifiable: {hits}"
    return hits[0]


# ---------------------------------------------------------------------------
# Unit tests: admit_fragments boundaries
# ---------------------------------------------------------------------------


class TestAdmitFragmentsStartBoundary:
    def test_start_boundary_half_open(self):
        """keep = [F, T, T, F] at crop_start-1, crop_start,
        crop_start+tile_size-1, crop_start+tile_size: the start window is
        half-open [0, tile_size).  Uint16 starts, cast to int64 before the
        subtraction, so crop_start-1 does not wrap."""
        crop_start, tile_size = 100, 256
        starts = np.array(
            [crop_start - 1, crop_start, crop_start + tile_size - 1,
             crop_start + tile_size],
            dtype=np.uint16)
        lengths = np.full(4, 100, dtype=np.uint16)
        p_crop, keep = admit_fragments(
            starts, lengths, crop_start, tile_size, L_MIN, L_MAX)
        np.testing.assert_array_equal(keep, [False, True, True, False])
        np.testing.assert_array_equal(p_crop, starts.astype(np.int64) - crop_start)


class TestAdmitFragmentsLengthBoundary:
    @pytest.mark.parametrize("l_min,l_max", [(L_MIN, L_MAX), (5, 10)])
    def test_length_boundary_closed_interval(self, l_min, l_max):
        """keep = [F, T, T, F] at L_min-1, L_min, L_max, L_max+1: both ends of
        the length range are inclusive (L_max is NOT half-open), checked
        against both the real constants and an unrelated range."""
        crop_start, tile_size = 0, 1000
        starts = np.full(4, crop_start, dtype=np.uint16)  # fixed in-crop start
        lengths = np.array([l_min - 1, l_min, l_max, l_max + 1], dtype=np.uint16)
        _, keep = admit_fragments(starts, lengths, crop_start, tile_size, l_min, l_max)
        np.testing.assert_array_equal(keep, [False, True, True, False])


class TestAdmitFragmentsEndPastCrop:
    def test_end_past_crop_edge_still_admitted(self):
        """Only the start is tested: a fragment starting at the crop's last
        valid position with length L_max (end far past tile_size in crop
        coords) is still admitted."""
        crop_start, tile_size = 100, 256
        start = crop_start + tile_size - 1  # last valid start
        starts = np.array([start], dtype=np.uint16)
        lengths = np.array([L_MAX], dtype=np.uint16)
        p_crop, keep = admit_fragments(
            starts, lengths, crop_start, tile_size, L_MIN, L_MAX)
        assert keep[0]
        end_in_crop = p_crop[0] + L_MAX - 1
        assert end_in_crop >= tile_size


class TestAdmitFragmentsPCrop:
    def test_p_crop_dtype_and_values_including_negative(self):
        """p_crop is returned for every row, admitted or not, as int64, equal
        to start - crop_start, including negative values."""
        crop_start, tile_size = 50, 200
        starts = np.array([0, 10, 49, 50, 100, 300], dtype=np.uint16)
        lengths = np.full(len(starts), 100, dtype=np.uint16)
        p_crop, keep = admit_fragments(
            starts, lengths, crop_start, tile_size, L_MIN, L_MAX)
        assert p_crop.dtype == np.int64
        np.testing.assert_array_equal(p_crop, starts.astype(np.int64) - crop_start)
        assert (p_crop[:3] < 0).all()  # starts before crop_start
        assert len(keep) == len(starts) == len(p_crop)  # one row per input, always


# ---------------------------------------------------------------------------
# Unit test: crop_origin vs the shared jitter helper
# ---------------------------------------------------------------------------


class TestCropOrigin:
    def test_matches_jitter_helper_and_centred_at_zero(self):
        """crop_origin's crop_start equals
        calculate_start_and_stop_from_jitter's first return value over a
        range of j, and j=0 is the centred crop."""
        l_target, tile_size = 2560, 2048
        assert crop_origin(l_target, tile_size, 0) == (l_target - tile_size) // 2
        for j in range(-256, 257, 17):
            expected_start, _ = calculate_start_and_stop_from_jitter(l_target, tile_size, j)
            assert crop_origin(l_target, tile_size, j) == expected_start


# ---------------------------------------------------------------------------
# Loader-level tests
# ---------------------------------------------------------------------------


class TestLoaderEvalModeAdmission:
    @pytest.mark.parametrize("preload", [True, False])
    def test_exact_edges_admitted_rows_and_frag_m(self, tmp_path, preload):
        """Fragments at the four start edges and at L_min-1/L_min/L_max/
        L_max+1; assert exactly which stored rows the loader admits
        (frag_store_idx) and their frag_m, for both preload paths."""
        tile_size, jitter = 256, 32
        # l_target = 256 + 2*32 = 320; centred crop_start = (320-256)//2 = 32
        starts = np.array([31, 32, 287, 288, 32, 32, 32, 32], dtype=np.uint16)
        lengths = np.array(
            [100, 100, 100, 100, L_MIN - 1, L_MIN, L_MAX, L_MAX + 1], dtype=np.uint16)
        store_path, cfg = _build_store_with_fragments(
            tmp_path, starts, lengths, tile_size=tile_size, jitter=jitter, seed=2)
        assert cfg.l_target == 320

        model = CutSiteFragmentModel(
            tile_size=tile_size, N1=4, K=4, N2=2, H=2, L_chunk=32)
        ds = CutSiteFragmentDataset(
            store_path=store_path, model_input_size=model.model_input_size,
            model_L_min=cfg.L_min, model_L_max=cfg.L_max,
            split="train", min_N=0, train_mode=False, preload=preload)
        assert len(ds) == 1
        item = ds[0]

        # row0 excluded (start=31, p_crop=-1); row3 excluded (start=288, p_crop=256);
        # row4 excluded (length=L_min-1); row7 excluded (length=L_max+1).
        np.testing.assert_array_equal(item["frag_store_idx"].numpy(), [1, 2, 5, 6])
        np.testing.assert_array_equal(item["frag_m"].numpy(), [0, 255, 0, 0])
        np.testing.assert_array_equal(
            (item["frag_L_idx"] + cfg.L_min).numpy(), [100, 100, L_MIN, L_MAX])


class TestLoaderTrainModeMatchesAdmitFragments:
    def test_train_mode_admission_uses_jittered_origin(self, tmp_path):
        """The admitted set at train_mode=True equals admit_fragments applied
        at crop_origin(l_target, tile_size, j) for the j actually drawn --
        the loader uses the jittered origin, not the centred one."""
        tile_size, jitter = 256, 32
        rng = np.random.default_rng(123)
        n = 300
        starts = rng.integers(0, tile_size + 2 * jitter, size=n).astype(np.uint16)
        lengths = rng.integers(L_MIN, L_MAX + 1, size=n).astype(np.uint16)
        store_path, cfg = _build_store_with_fragments(
            tmp_path, starts, lengths, tile_size=tile_size, jitter=jitter, seed=1)

        model = CutSiteFragmentModel(
            tile_size=tile_size, N1=4, K=4, N2=2, H=2, L_chunk=32)
        ds = CutSiteFragmentDataset(
            store_path=store_path, model_input_size=model.model_input_size,
            model_L_min=cfg.L_min, model_L_max=cfg.L_max,
            split="train", min_N=0, train_mode=True, seed=7)

        js_seen = set()
        for _ in range(20):
            item = ds[0]
            j = _recover_j(ds, 0, item["x"])
            js_seen.add(j)
            crop_start = crop_origin(cfg.l_target, tile_size, j)
            expected_p_crop, expected_keep = admit_fragments(
                starts, lengths, crop_start, tile_size, cfg.L_min, cfg.L_max)
            np.testing.assert_array_equal(
                item["frag_m"].numpy(), expected_p_crop[expected_keep])
            np.testing.assert_array_equal(
                item["frag_store_idx"].numpy(), np.nonzero(expected_keep)[0])

        assert js_seen - {0}, "never drew a non-centred jitter; test did not exercise it"
