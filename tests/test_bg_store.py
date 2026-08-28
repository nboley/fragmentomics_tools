"""Tests for background_model.store — layout, CSR access, N computation."""

import os
import shutil
import tempfile

import numpy as np
import pytest
import zarr

from background_model.config import C, L_TARGET, TILE, PlumbingConfig
from background_model.store import (
    compute_N_for_tile,
    create_store,
    csr_slice,
    densify_counts,
    increment_split_version,
    open_store,
    record_phase_b_params,
)


@pytest.fixture
def store_fixture():
    """Create a temporary store with some test data."""
    d = tempfile.mkdtemp()

    # Dummy files for config
    sheet = os.path.join(d, "sheet.tsv")
    with open(sheet, "w") as f:
        f.write("library\th5_path\tseqrun\tendo_category\n")
    bed = os.path.join(d, "train.bed")
    with open(bed, "w") as f:
        f.write("chr1\t0\t100000\n")
    fasta = os.path.join(d, "test.fa")
    with open(fasta, "w") as f:
        f.write(">chr1\nACGT\n")
    fai = os.path.join(d, "test.fa.fai")
    with open(fai, "w") as f:
        f.write("chr1\t4\t6\t4\t5\n")

    cfg = PlumbingConfig(
        sample_sheet=sheet,
        region_beds={"train_pool": bed},
        fasta=fasta,
    )

    store_path = os.path.join(d, "test.zarr")
    root = create_store(store_path, cfg, n_tiles=3, n_samples=2, nnz=50)

    yield {
        "dir": d,
        "store_path": store_path,
        "root": root,
        "config": cfg,
        "n_tiles": 3,
        "n_samples": 2,
    }
    shutil.rmtree(d)


class TestStoreCreation:
    def test_store_has_required_groups(self, store_fixture):
        root = store_fixture["root"]
        assert "tiles" in root
        assert "samples" in root
        assert "counts" in root
        assert "totals" in root

    def test_store_attrs(self, store_fixture):
        root = store_fixture["root"]
        assert "config_json" in root.attrs
        assert "config_hash" in root.attrs
        assert "created_utc" in root.attrs
        assert "split_version" in root.attrs
        assert root.attrs["split_version"] == 0

    def test_tile_arrays_shape(self, store_fixture):
        root = store_fixture["root"]
        T = store_fixture["n_tiles"]
        cfg = store_fixture["config"]
        assert root["tiles/contig"].shape == (T,)
        assert root["tiles/start"].shape == (T,)
        assert root["tiles/stop"].shape == (T,)
        assert root["tiles/strand"].shape == (T,)
        assert root["tiles/region_id"].shape == (T,)
        assert root["tiles/split"].shape == (T,)
        assert root["tiles/seq"].shape == (T, cfg.l_seq)
        assert root["tiles/mask"].shape == (T, cfg.l_target)

    def test_sample_arrays_shape(self, store_fixture):
        root = store_fixture["root"]
        S = store_fixture["n_samples"]
        assert root["samples/library"].shape == (S,)
        assert root["samples/role"].shape == (S,)
        assert root["samples/total_fragments"].shape == (S,)

    def test_counts_arrays_shape(self, store_fixture):
        root = store_fixture["root"]
        S, T = store_fixture["n_samples"], store_fixture["n_tiles"]
        assert root["counts/indptr"].shape == (S * T + 1,)

    def test_totals_shape(self, store_fixture):
        root = store_fixture["root"]
        S, T = store_fixture["n_samples"], store_fixture["n_tiles"]
        assert root["totals/N"].shape == (S, T, C)


class TestCSRRoundtrip:
    def test_densify_accumulates_triples(self):
        """densify_counts is densify-ONLY: it scatter-adds (pos, track, data)
        triples into a dense array. With duplicate (track, pos) entries it must
        ACCUMULATE (np.add.at semantics), matching a manual construction. This is
        not a roundtrip (the sparse side is the input); see the dedicated
        roundtrip test below for densify(sparsify(dense)) == dense.
        """
        rng = np.random.default_rng(42)
        n_entries = 500
        tracks = rng.integers(0, C, size=n_entries).astype(np.uint8)
        positions = rng.integers(0, L_TARGET, size=n_entries).astype(np.uint16)
        values = rng.integers(1, 100, size=n_entries).astype(np.uint16)

        # Build dense from triples (duplicates present -> accumulation exercised)
        y_dense = densify_counts(positions, tracks, values, C, L_TARGET)

        # Verify it matches manual accumulation
        y_expected = np.zeros((C, L_TARGET), dtype=np.float32)
        for p, t, v in zip(positions, tracks, values):
            y_expected[t, p] += v
        np.testing.assert_array_equal(y_dense, y_expected)

    def test_dense_to_csr_to_dense_roundtrip(self):
        """True roundtrip: build CSR triples from a dense array the way Phase A
        does (per nonzero (track, pos) emit one uint16 triple, lexsort by
        (track, pos)), densify back, and require exact equality with the
        original dense array.
        """
        rng = np.random.default_rng(7)
        # Sparse-ish dense array with distinct (track, pos) nonzeros.
        dense = np.zeros((C, L_TARGET), dtype=np.float32)
        n = 400
        tracks = rng.integers(0, C, size=n)
        positions = rng.integers(0, L_TARGET, size=n)
        values = rng.integers(1, 500, size=n)
        # Deduplicate (track, pos) so each cell has a single defined value
        # (a genuine dense array, not an accumulation of collisions).
        seen = {}
        for t, p, v in zip(tracks, positions, values):
            seen[(int(t), int(p))] = int(v)
        for (t, p), v in seen.items():
            dense[t, p] = v

        # Build CSR triples from the dense array (mirrors preprocess: uint16
        # pos/data, uint8 track, lexsort by (track, pos)).
        nz_tracks, nz_pos = np.nonzero(dense)
        nz_data = dense[nz_tracks, nz_pos]
        pos = nz_pos.astype(np.uint16)
        track = nz_tracks.astype(np.uint8)
        data = nz_data.astype(np.uint16)
        order = np.lexsort((pos, track))
        pos, track, data = pos[order], track[order], data[order]

        got = densify_counts(pos, track, data, C, L_TARGET)
        np.testing.assert_array_equal(got, dense)

    def test_empty_triples(self):
        pos = np.empty(0, dtype=np.uint16)
        track = np.empty(0, dtype=np.uint8)
        data = np.empty(0, dtype=np.uint16)
        y = densify_counts(pos, track, data)
        assert y.shape == (C, L_TARGET)
        assert y.sum() == 0


class TestComputeN:
    def test_center_only(self):
        """N counts only the center tile, not the margins."""
        l_target = L_TARGET
        tile_size = TILE
        margin = (l_target - tile_size) // 2  # 128

        y = np.zeros((C, l_target), dtype=np.float32)
        mask = np.ones(l_target, dtype=bool)

        # Put counts in margin only
        y[0, :margin] = 10.0
        N = compute_N_for_tile(y, mask, tile_size, l_target)
        assert N[0] == 0

        # Put counts in center
        y[0, margin] = 5.0
        N = compute_N_for_tile(y, mask, tile_size, l_target)
        assert N[0] == 5

    def test_mask_exclusion(self):
        """Masked positions excluded from N."""
        margin = (L_TARGET - TILE) // 2
        y = np.zeros((C, L_TARGET), dtype=np.float32)
        mask = np.ones(L_TARGET, dtype=bool)

        y[0, margin] = 10.0
        y[0, margin + 1] = 20.0
        mask[margin + 1] = False  # mask out second position

        N = compute_N_for_tile(y, mask, TILE, L_TARGET)
        assert N[0] == 10  # only unmasked position counted


class TestSplitVersion:
    def test_increment(self, store_fixture):
        root = store_fixture["root"]
        assert root.attrs["split_version"] == 0
        v1 = increment_split_version(root)
        assert v1 == 1
        assert root.attrs["split_version"] == 1
        v2 = increment_split_version(root)
        assert v2 == 2

    def test_phase_b_params_recorded(self, store_fixture):
        root = store_fixture["root"]
        cfg = store_fixture["config"]
        record_phase_b_params(root, cfg)
        assert root.attrs["applied_region_fracs"] == list(cfg.region_fracs)
        assert root.attrs["applied_min_total_fragments"] == cfg.min_total_fragments


class TestCSRResize:
    def test_resize_counts_arrays(self, store_fixture):
        """The defensive CSR resize path (overwrite=True) must work across
        zarr 2/3 and yield writable arrays of the new shape."""
        from background_model.store import _create_array

        root = store_fixture["root"]
        counts = root["counts"]
        # Original nnz was 50 (from the fixture)
        assert root["counts/pos"].shape == (50,)

        new_nnz = 7
        _create_array(counts, "pos", shape=(new_nnz,), dtype="uint16", chunks=(1 << 20,), overwrite=True)
        _create_array(counts, "track", shape=(new_nnz,), dtype="uint8", chunks=(1 << 20,), overwrite=True)
        _create_array(counts, "data", shape=(new_nnz,), dtype="uint16", chunks=(1 << 20,), overwrite=True)

        assert root["counts/pos"].shape == (new_nnz,)
        assert root["counts/track"].shape == (new_nnz,)
        assert root["counts/data"].shape == (new_nnz,)

        # arrays are writable at the new shape
        vals = np.arange(new_nnz, dtype=np.uint16)
        root["counts/pos"][:] = vals
        np.testing.assert_array_equal(np.asarray(root["counts/pos"][:]), vals)


class TestOpenStore:
    def test_open_with_matching_config(self, store_fixture):
        root = open_store(store_fixture["store_path"], store_fixture["config"])
        assert "config_hash" in root.attrs

    def test_open_with_drift_raises(self, store_fixture):
        # Create a config with different content
        d = store_fixture["dir"]
        sheet2 = os.path.join(d, "sheet2.tsv")
        with open(sheet2, "w") as f:
            f.write("library\th5_path\tseqrun\tendo_category\ndifferent_content\n")

        cfg2 = PlumbingConfig(
            sample_sheet=sheet2,
            region_beds=store_fixture["config"].region_beds,
            fasta=store_fixture["config"].fasta,
        )
        with pytest.raises(ValueError, match="Config drift"):
            open_store(store_fixture["store_path"], cfg2)
