"""End-to-end tests for the background model preprocessing pipeline.

Tests the full preprocess flow using synthetic data where possible,
and mock/skip where real h5 fixtures are unavailable.
"""

import json
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from background_model.config import C, L_TARGET, TILE, PlumbingConfig
from background_model.preprocess import (
    TRACK_INDEX,
    assign_region_splits,
    build_tiles,
    draw_samples,
)
from background_model.store import (
    compute_N_for_tile,
    create_store,
    densify_counts,
    increment_split_version,
)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


def _make_sample_sheet(d, n_samples=10):
    """Create a test sample sheet TSV."""
    sheet_path = os.path.join(d, "sample_sheet.tsv")
    rows = []
    for i in range(n_samples):
        rows.append({
            "library": f"LIB-{i:03d}",
            "h5_path": f"/path/to/lib{i}.frag.h5",
            "seqrun": f"SR-{i:03d}",
            "endo_category": "Asymptomatic",
        })
    df = pd.DataFrame(rows)
    df.to_csv(sheet_path, sep="\t", index=False)
    return sheet_path


def _make_region_bed(d, name="regions.bed", regions=None):
    """Create a test region BED file."""
    if regions is None:
        regions = [
            ("chr1", 0, 100000),
            ("chr1", 200000, 300000),
            ("chr2", 0, 50000),
        ]
    bed_path = os.path.join(d, name)
    with open(bed_path, "w") as f:
        for contig, start, stop in regions:
            f.write(f"{contig}\t{start}\t{stop}\n")
    return bed_path


def _make_config_files(d, n_samples=10, tile_size=TILE):
    """Create all config files and return a PlumbingConfig."""
    sheet = _make_sample_sheet(d, n_samples)
    bed_train = _make_region_bed(d, "train.bed")
    bed_pc = _make_region_bed(d, "pc.bed", [("chr1", 500000, 516384)])
    blacklist = _make_region_bed(d, "blacklist.bed", [("chr1", 50000, 50100)])
    fasta = os.path.join(d, "test.fa")
    with open(fasta, "w") as f:
        f.write(">chr1\n" + "ACGT" * 200000 + "\n")
        f.write(">chr2\n" + "TGCA" * 100000 + "\n")
    fai = os.path.join(d, "test.fa.fai")
    with open(fai, "w") as f:
        f.write("chr1\t800000\t6\t80\t81\n")
        f.write("chr2\t400000\t810006\t80\t81\n")

    cfg = PlumbingConfig(
        sample_sheet=sheet,
        region_beds={"train_pool": bed_train, "positive_control": bed_pc},
        blacklist_bed=blacklist,
        fasta=fasta,
        tile_size=tile_size,
        n_train_samples=min(7, n_samples - 1),
        n_heldout_samples=min(3, n_samples - min(7, n_samples - 1)),
    )
    return cfg


class TestDrawSamples:
    def test_draw_correct_counts(self, tmp_dir):
        cfg = _make_config_files(tmp_dir, n_samples=20)
        sheet = pd.read_csv(cfg.sample_sheet, sep="\t")
        drawn = draw_samples(sheet, cfg)

        assert len(drawn) == cfg.n_train_samples + cfg.n_heldout_samples
        assert (drawn["role"] == 0).sum() == cfg.n_train_samples
        assert (drawn["role"] == 1).sum() == cfg.n_heldout_samples

    def test_draw_deterministic(self, tmp_dir):
        cfg = _make_config_files(tmp_dir, n_samples=20)
        sheet = pd.read_csv(cfg.sample_sheet, sep="\t")

        drawn1 = draw_samples(sheet, cfg)
        drawn2 = draw_samples(sheet, cfg)
        pd.testing.assert_frame_equal(drawn1, drawn2)

    def test_draw_different_seed(self, tmp_dir):
        d = tmp_dir
        sheet_path = _make_sample_sheet(d, 20)
        sheet = pd.read_csv(sheet_path, sep="\t")

        cfg1 = PlumbingConfig(sample_sheet=sheet_path, seed=1, n_train_samples=7, n_heldout_samples=3)
        cfg2 = PlumbingConfig(sample_sheet=sheet_path, seed=2, n_train_samples=7, n_heldout_samples=3)

        drawn1 = draw_samples(sheet, cfg1)
        drawn2 = draw_samples(sheet, cfg2)
        # Different seeds should (almost certainly) draw different samples
        assert not drawn1["library"].equals(drawn2["library"])

    def test_draw_insufficient_samples_raises(self, tmp_dir):
        cfg = _make_config_files(tmp_dir, n_samples=5)
        # Override to need more samples than available
        cfg_bad = PlumbingConfig(
            sample_sheet=cfg.sample_sheet,
            n_train_samples=40,
            n_heldout_samples=10,
        )
        sheet = pd.read_csv(cfg_bad.sample_sheet, sep="\t")
        with pytest.raises(ValueError, match="Need 50 samples"):
            draw_samples(sheet, cfg_bad)


class TestBuildTiles:
    def test_tile_count(self, tmp_dir):
        """Tiles should cover each region with non-overlapping tiles."""
        bed = _make_region_bed(tmp_dir, regions=[("chr1", 0, 49152)])  # 3 tiles of 16384
        tiles = build_tiles({"train": bed}, tile_size=16384, jitter=128, rf_budget=2048, ref="hg38")
        assert len(tiles) == 3

    def test_tile_coordinates(self, tmp_dir):
        bed = _make_region_bed(tmp_dir, regions=[("chr1", 0, 32768)])  # 2 tiles
        tiles = build_tiles({"train": bed}, tile_size=16384, jitter=128, rf_budget=2048, ref="hg38")
        assert tiles[0]["start"] == 0
        assert tiles[0]["stop"] == 16384
        assert tiles[1]["start"] == 16384
        assert tiles[1]["stop"] == 32768

    def test_partial_tile_dropped(self, tmp_dir):
        """Region not divisible by tile_size should drop the partial tail."""
        bed = _make_region_bed(tmp_dir, regions=[("chr1", 0, 20000)])
        tiles = build_tiles({"train": bed}, tile_size=16384, jitter=128, rf_budget=2048, ref="hg38")
        assert len(tiles) == 1  # only one full tile fits

    def test_split_name_preserved(self, tmp_dir):
        bed1 = _make_region_bed(tmp_dir, "train.bed", [("chr1", 0, 16384)])
        bed2 = _make_region_bed(tmp_dir, "pc.bed", [("chr2", 0, 16384)])
        tiles = build_tiles(
            {"train_pool": bed1, "positive_control": bed2},
            tile_size=16384, jitter=128, rf_budget=2048, ref="hg38",
        )
        split_names = {t["split_name"] for t in tiles}
        assert split_names == {"train_pool", "positive_control"}


class TestRegionSplits:
    def test_positive_control_always_split_3(self, tmp_dir):
        """Positive-control regions always get split=3."""
        bed_train = _make_region_bed(tmp_dir, "train.bed", [("chr1", 0, 32768)])
        bed_pc = _make_region_bed(tmp_dir, "pc.bed", [("chr2", 0, 16384)])
        tiles = build_tiles(
            {"train_pool": bed_train, "positive_control": bed_pc},
            tile_size=16384, jitter=128, rf_budget=2048, ref="hg38",
        )
        cfg = PlumbingConfig(sample_sheet="dummy", region_fracs=(0.5, 0.25, 0.25))
        splits = assign_region_splits(tiles, cfg)

        # Find the positive_control tile
        for i, tile in enumerate(tiles):
            if tile["split_name"] == "positive_control":
                assert splits[i] == 3, f"PC tile {i} should have split=3"

    def test_split_deterministic(self, tmp_dir):
        bed = _make_region_bed(tmp_dir, regions=[("chr1", i * 16384, (i + 1) * 16384) for i in range(20)])
        tiles = build_tiles({"train": bed}, tile_size=16384, jitter=128, rf_budget=2048, ref="hg38")
        cfg = PlumbingConfig(sample_sheet="dummy", seed=42)

        splits1 = assign_region_splits(tiles, cfg)
        splits2 = assign_region_splits(tiles, cfg)
        np.testing.assert_array_equal(splits1, splits2)

    def test_split_proportions_approximate(self, tmp_dir):
        """Region fracs should be approximately respected."""
        regions = [("chr1", i * 16384, (i + 1) * 16384) for i in range(100)]
        bed = _make_region_bed(tmp_dir, regions=regions)
        tiles = build_tiles({"train": bed}, tile_size=16384, jitter=128, rf_budget=2048, ref="hg38")
        cfg = PlumbingConfig(
            sample_sheet="dummy",
            region_fracs=(0.8, 0.1, 0.1),
        )
        splits = assign_region_splits(tiles, cfg)

        train_frac = (splits == 0).mean()
        val_frac = (splits == 1).mean()
        heldout_frac = (splits == 2).mean()

        assert 0.6 < train_frac < 0.95, f"train_frac={train_frac}"
        assert 0.01 < val_frac < 0.25, f"val_frac={val_frac}"
        assert 0.01 < heldout_frac < 0.25, f"heldout_frac={heldout_frac}"

    def test_tiles_inherit_region_split(self, tmp_dir):
        """Adjacent tiles from the same region must have the same split."""
        regions = [("chr1", 0, 49152)]  # 3 tiles, all same region
        bed = _make_region_bed(tmp_dir, regions=regions)
        tiles = build_tiles({"train": bed}, tile_size=16384, jitter=128, rf_budget=2048, ref="hg38")
        cfg = PlumbingConfig(sample_sheet="dummy")
        splits = assign_region_splits(tiles, cfg)

        # All 3 tiles come from the same region → same split
        assert splits[0] == splits[1] == splits[2]


class TestTrackIndex:
    def test_track_index_covers_all_tracks(self):
        assert len(TRACK_INDEX) == C

    def test_track_index_values_unique(self):
        values = list(TRACK_INDEX.values())
        assert len(set(values)) == len(values)

    def test_track_index_range(self):
        assert min(TRACK_INDEX.values()) == 0
        assert max(TRACK_INDEX.values()) == C - 1


class TestCSRDensifyRoundtrip:
    def test_roundtrip_with_known_data(self):
        """Sparsify → densify must reproduce the original."""
        rng = np.random.default_rng(99)
        y = np.zeros((C, L_TARGET), dtype=np.float32)
        # Place known counts
        for _ in range(200):
            c = rng.integers(0, C)
            p = rng.integers(0, L_TARGET)
            y[c, p] += rng.integers(1, 10)

        # Extract sparse triples
        nz = np.nonzero(y)
        track = nz[0].astype(np.uint8)
        pos = nz[1].astype(np.uint16)
        data = y[nz].astype(np.uint16)

        y2 = densify_counts(pos, track, data, C, L_TARGET)
        np.testing.assert_array_equal(y, y2)


class TestSplitVersion:
    def test_increment_on_phase_b(self, tmp_dir):
        """split_version must increment on every Phase B write (condition #2)."""
        # Simulate two Phase B runs
        cfg = _make_config_files(tmp_dir)
        store_path = os.path.join(tmp_dir, "test_sv.zarr")
        root = create_store(store_path, cfg, n_tiles=2, n_samples=2, nnz=10)

        assert root.attrs["split_version"] == 0
        v1 = increment_split_version(root)
        assert v1 == 1
        v2 = increment_split_version(root)
        assert v2 == 2

    def test_applied_params_in_attrs(self, tmp_dir):
        """Phase B must record applied region_fracs/min_total_fragments in attrs (condition #2)."""
        from background_model.store import record_phase_b_params

        cfg = _make_config_files(tmp_dir)
        store_path = os.path.join(tmp_dir, "test_attrs.zarr")
        root = create_store(store_path, cfg, n_tiles=2, n_samples=2, nnz=10)

        record_phase_b_params(root, cfg)
        assert root.attrs["applied_region_fracs"] == list(cfg.region_fracs)
        assert root.attrs["applied_min_total_fragments"] == cfg.min_total_fragments


class TestDepthFilter:
    def test_low_depth_downgrade(self, tmp_dir):
        """Samples below min_total_fragments get role=2 (dropped_low_depth)."""
        cfg = _make_config_files(tmp_dir, n_samples=20)
        sheet = pd.read_csv(cfg.sample_sheet, sep="\t")
        drawn = draw_samples(sheet, cfg)

        # Simulate: first sample has low depth
        total_fragments_map = {}
        for lib in drawn["library"]:
            total_fragments_map[lib] = 30_000_000  # above threshold
        low_lib = drawn["library"].iloc[0]
        total_fragments_map[low_lib] = 1_000_000  # below threshold

        # Apply depth filter (same logic as Phase B)
        roles = np.array(list(drawn["role"]), dtype=np.uint8)
        for i, lib in enumerate(drawn["library"]):
            if total_fragments_map[lib] < cfg.min_total_fragments:
                roles[i] = 2

        assert roles[0] == 2  # first sample dropped
        assert (roles == 2).sum() == 1  # only one dropped


class TestSampleSheetBuilder:
    def test_build_sheet(self, tmp_dir):
        from background_model.sample_sheet import build_sample_sheet

        # Create mock manifest
        manifest_path = os.path.join(tmp_dir, "manifest.tsv")
        manifest_df = pd.DataFrame({
            "key": ["key1", "key2", "key3"],
            "path": ["/path/1.h5", "/path/2.h5", "/path/3.h5"],
            "notes": [
                json.dumps({"library": "LIB-001", "seqrun": "SR-1"}),
                json.dumps({"library": "LIB-002", "seqrun": "SR-2"}),
                json.dumps({"library": "LIB-003", "seqrun": "SR-3"}),
            ],
        })
        manifest_df.to_csv(manifest_path, sep="\t", index=False)

        # Create mock clinical
        clinical_path = os.path.join(tmp_dir, "clinical.csv")
        clinical_df = pd.DataFrame({
            "library": ["LIB-001", "LIB-002", "LIB-003"],
            "ENDO_CATEGORY": ["Asymptomatic", "Active", "Remission"],
        })
        clinical_df.to_csv(clinical_path, index=False)

        sheet = build_sample_sheet(manifest_path, clinical_path)
        # Should filter to quiescent: Asymptomatic + Remission
        assert len(sheet) == 2
        assert set(sheet["endo_category"]) == {"Asymptomatic", "Remission"}
        assert "library" in sheet.columns
        assert "h5_path" in sheet.columns
