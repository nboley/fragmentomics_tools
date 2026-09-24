"""End-to-end tests for the background model preprocessing pipeline.

Tests the full preprocess flow using synthetic data where possible,
and mock/skip where real h5 fixtures are unavailable.
"""

import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from background_model.config import C, TILE, PlumbingConfig
from background_model.preprocess import (
    TRACK_INDEX,
    _contig_length,
    _worker_process_sample,
    assign_region_splits,
    build_tiles,
    draw_samples,
    run_phase_a,
    run_phase_b,
)
from background_model.store import (
    compute_N_for_tile,
    csr_slice,
    densify_counts,
    open_store,
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

    def test_track_order_matches_core_canonical(self):
        """Cross-file invariant: all track constants come from the canonical
        background_model.tracks module.  A silent order mismatch between
        TRACK_INDEX and the model's DEFAULT_OUTPUT_TRACKS would corrupt every
        stored count, so verify alignment here."""
        core = pytest.importorskip("background_model_core")
        from background_model import tracks as bg_tracks
        from background_model import config as bg_config

        # config.py re-exports from tracks (identity, not just equality)
        assert bg_config._FL_BANDS_DEFAULT is bg_tracks.FL_BANDS

        # core re-exports from tracks
        assert core.STRANDS is bg_tracks.STRANDS
        assert core.FL_BANDS is bg_tracks.FL_BANDS
        assert core.COVERAGE_TYPES is bg_tracks.COVERAGE_TYPES

        # TRACK_INDEX maps each key to the exact position of
        # its track name in core's DEFAULT_OUTPUT_TRACKS
        assert len(TRACK_INDEX) == len(core.DEFAULT_OUTPUT_TRACKS)
        for (strand, fl_band, cov), idx in TRACK_INDEX.items():
            name = core.index_key_to_track_name(strand, fl_band, cov)
            assert core.DEFAULT_OUTPUT_TRACKS[idx] == name, (
                f"track order drift: {name} at TRACK_INDEX {idx} but "
                f"core position {core.DEFAULT_OUTPUT_TRACKS.index(name)}"
            )


class TestDepthFilter:
    def test_low_depth_downgrade_through_phase_b(self, tmp_dir, monkeypatch):
        """Real Phase A -> Phase B: a sample whose total_fragments falls below
        min_total_fragments is downgraded to role=2 by PRODUCTION code (the
        depth filter in preprocess.run_phase_b), other roles are untouched, S is
        unchanged (no replacement/backfill), and applied_min_total_fragments is
        recorded in store attrs.

        The old version of this test reimplemented the filter loop locally and
        asserted against that copy; it has been replaced so the assertions read
        only what run_phase_b actually wrote to the store.  Helpers from the
        synthetic A->B harness (defined below) are reused.
        """
        cfg = _make_synth_config(tmp_dir, n_samples=2, n_train=1, n_heldout=1)
        sheet = pd.read_csv(cfg.sample_sheet, sep="\t")
        drawn = draw_samples(sheet, cfg)
        S = len(drawn)
        assert S == 2
        # role 0 (train) is at index 0, role 1 (heldout) at index 1.
        assert drawn["role"].iloc[0] == 0 and drawn["role"].iloc[1] == 1
        low_lib = drawn["library"].iloc[0]
        low_h5 = drawn["h5_path"].iloc[0]
        low_depth = cfg.min_total_fragments // 20  # well below threshold, > 0

        # Drive Phase A with a per-h5 depth map so the train sample reads back a
        # sub-threshold fragment_length_counts sum; the heldout sample stays at
        # the default (above-threshold) depth.
        _patch_phase_a_depths(monkeypatch, _synthetic_fragments, {low_h5: low_depth})

        tiles = build_tiles(
            cfg.region_beds, cfg.tile_size, cfg.jitter, cfg.rf_budget, "hg38"
        )
        shard_dir = os.path.join(tmp_dir, "shards")
        run_phase_a(cfg, drawn, tiles, shard_dir, "hg38", n_workers=1)
        out = os.path.join(tmp_dir, "out")
        os.makedirs(out)
        store_path = run_phase_b(cfg, drawn, tiles, shard_dir, out, "hg38")

        root = open_store(store_path)
        roles = np.asarray(root["samples/role"])
        libs = [str(x) for x in np.asarray(root["samples/library"])]
        tot = np.asarray(root["samples/total_fragments"])

        # low-depth train sample downgraded; heldout untouched.
        assert roles[0] == 2, "low-depth sample must be downgraded to role=2"
        assert roles[1] == 1, "other roles must be unchanged"
        assert (roles == 2).sum() == 1
        # S unchanged: no replacement/backfill introduced a new sample.
        assert len(libs) == S == 2
        assert set(libs) == set(drawn["library"])
        assert libs[0] == low_lib
        # raw depth recorded (the low sample is downgraded, not replaced).
        assert int(tot[0]) == low_depth
        # applied threshold recorded in attrs.
        assert root.attrs["applied_min_total_fragments"] == cfg.min_total_fragments


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

    def test_column_set_active_dropped_join_by_library(self, tmp_dir):
        """Exact emitted column set is {library, h5_path, seqrun, endo_category};
        'Active' (non-quiescent) rows are dropped; and the clinical endo join is
        BY LIBRARY (not positional) — proven with mismatched row order and an
        extra clinical-only library."""
        from background_model.sample_sheet import build_sample_sheet

        manifest_path = os.path.join(tmp_dir, "manifest.tsv")
        manifest_df = pd.DataFrame({
            "key": ["k1", "k2", "k3"],
            "path": ["/data/LIB-001.h5", "/data/LIB-002.h5", "/data/LIB-003.h5"],
            "notes": [
                json.dumps({"library": "LIB-001", "seqrun": "SR-1"}),
                json.dumps({"library": "LIB-002", "seqrun": "SR-2"}),
                json.dumps({"library": "LIB-003", "seqrun": "SR-3"}),
            ],
        })
        manifest_df.to_csv(manifest_path, sep="\t", index=False)

        clinical_path = os.path.join(tmp_dir, "clinical.csv")
        # Different row order + an extra library absent from the manifest, so a
        # positional (rather than by-library) join would attach the wrong endo.
        clinical_df = pd.DataFrame({
            "library": ["LIB-003", "LIB-999", "LIB-001", "LIB-002"],
            "ENDO_CATEGORY": ["Remission", "Remission", "Asymptomatic", "Active"],
        })
        clinical_df.to_csv(clinical_path, index=False)

        sheet = build_sample_sheet(manifest_path, clinical_path)

        # exact emitted column set
        assert set(sheet.columns) == {
            "library", "h5_path", "seqrun", "endo_category"
        }
        # 'Active' row (LIB-002) dropped; clinical-only LIB-999 absent (inner join)
        assert set(sheet["library"]) == {"LIB-001", "LIB-003"}
        assert "LIB-002" not in set(sheet["library"])
        assert "LIB-999" not in set(sheet["library"])
        # join is BY LIBRARY: each library keeps its own endo + manifest fields
        by_lib = sheet.set_index("library")
        assert by_lib.loc["LIB-001", "endo_category"] == "Asymptomatic"
        assert by_lib.loc["LIB-001", "h5_path"] == "/data/LIB-001.h5"
        assert by_lib.loc["LIB-001", "seqrun"] == "SR-1"
        assert by_lib.loc["LIB-003", "endo_category"] == "Remission"
        assert by_lib.loc["LIB-003", "h5_path"] == "/data/LIB-003.h5"


# ── Synthetic Phase A → Phase B integration (bypasses fragments_h5) ───────

SMALL_TILE = 256  # keep L_TARGET / L_SEQ small for fast synthetic stores


class _FakeFragmentsH5:
    """Stand-in for fragments_h5.FragmentsH5 used to drive Phase A synthetically."""

    TOTAL_FRAGMENTS = 25_000_000  # above default min_total_fragments (20M)

    def __init__(self, h5_path, cache_pointers=False):
        self.h5_path = h5_path
        self.fragment_length_counts = np.array([self.TOTAL_FRAGMENTS], dtype=np.int64)

    def close(self):
        pass


def _synthetic_fragments(h5, region, max_frag_len):
    """Deterministic synthetic fragments for (h5, region).

    Length 150 → falls in fl_band (120, 175) only. Deterministic via an md5
    seed so the Phase A worker and the test's independent recount agree.
    """
    import hashlib

    key = f"{h5.h5_path}:{int(region.start)}".encode()
    seed = int(hashlib.md5(key).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    length = int(region.length)
    n = 4
    hi = max(1, length - 160)
    starts = rng.integers(0, hi, size=n).astype(np.int64)
    stops = (starts + 150).astype(np.int64)
    strands = rng.choice(np.array(["+", "-"]), size=n)
    return starts, stops, strands


def _make_fake_from_h5(frag_fn):
    def _fake(h5, region, min_mapq=None, max_frag_len=None):
        from fragmentomics_tools.fragment_array.fragment_array import (
            RegionFragmentArray,
        )

        starts, stops, strands = frag_fn(h5, region, max_frag_len)
        return RegionFragmentArray(
            starts_0=starts,
            stops_0=stops,
            region=region,
            max_frag_len=max_frag_len or 175,
            fragment_strands=strands,
            validate_data=False,
        )

    return _fake


def _patch_phase_a(monkeypatch, frag_fn):
    """Monkeypatch fragments_h5 + from_fragments_h5 so Phase A runs synthetically."""
    from fragmentomics_tools.fragment_array.fragment_array import RegionFragmentArray

    fake_mod = types.ModuleType("fragments_h5")
    fake_mod.FragmentsH5 = _FakeFragmentsH5
    monkeypatch.setitem(sys.modules, "fragments_h5", fake_mod)
    monkeypatch.setattr(
        RegionFragmentArray,
        "from_fragments_h5",
        staticmethod(_make_fake_from_h5(frag_fn)),
    )
    return RegionFragmentArray


def _patch_phase_a_depths(monkeypatch, frag_fn, depth_map):
    """Like _patch_phase_a but the fake FragmentsH5 reports a per-h5_path total
    fragment count (from `depth_map`, default 25M) so the Phase-B depth filter
    can be exercised through production code."""
    from fragmentomics_tools.fragment_array.fragment_array import RegionFragmentArray

    class _DepthFakeFragmentsH5:
        def __init__(self, h5_path, cache_pointers=False):
            self.h5_path = h5_path
            depth = int(depth_map.get(h5_path, _FakeFragmentsH5.TOTAL_FRAGMENTS))
            self.fragment_length_counts = np.array([depth], dtype=np.int64)

        def close(self):
            pass

    fake_mod = types.ModuleType("fragments_h5")
    fake_mod.FragmentsH5 = _DepthFakeFragmentsH5
    monkeypatch.setitem(sys.modules, "fragments_h5", fake_mod)
    monkeypatch.setattr(
        RegionFragmentArray,
        "from_fragments_h5",
        staticmethod(_make_fake_from_h5(frag_fn)),
    )
    return RegionFragmentArray


def _make_synth_config(
    tmp_dir,
    *,
    contig="chr1",
    n_samples=2,
    n_train=1,
    n_heldout=1,
    region=(3072, 3072 + 2 * SMALL_TILE),
    blacklist=None,
    blacklist_expansion=None,
):
    """Build a small synthetic PlumbingConfig with a real (indexed) FASTA."""
    import pysam

    sheet_path = os.path.join(tmp_dir, "sheet.tsv")
    rows = [
        {
            "library": f"LIB-{i:03d}",
            "h5_path": f"/synthetic/LIB-{i:03d}.h5",
            "seqrun": f"SR-{i:03d}",
            "endo_category": "Asymptomatic",
        }
        for i in range(n_samples)
    ]
    pd.DataFrame(rows).to_csv(sheet_path, sep="\t", index=False)

    bed = os.path.join(tmp_dir, "train.bed")
    with open(bed, "w") as f:
        f.write(f"{contig}\t{region[0]}\t{region[1]}\n")

    # FASTA must cover the sequence extent (tile ± (jitter + rf_budget)).
    seq_stop = region[1] + 4096
    fasta = os.path.join(tmp_dir, "genome.fa")
    with open(fasta, "w") as f:
        f.write(f">{contig}\n" + ("ACGT" * ((seq_stop // 4) + 1)) + "\n")
    pysam.faidx(fasta)

    blk = ""
    if blacklist is not None:
        blk = os.path.join(tmp_dir, "blacklist.bed")
        with open(blk, "w") as f:
            for bc, bs, bp in blacklist:
                f.write(f"{bc}\t{bs}\t{bp}\n")

    extra = {}
    if blacklist_expansion is not None:
        extra["blacklist_expansion"] = blacklist_expansion
    return PlumbingConfig(
        sample_sheet=sheet_path,
        region_beds={"train_pool": bed},
        blacklist_bed=blk,
        fasta=fasta,
        tile_size=SMALL_TILE,
        n_train_samples=n_train,
        n_heldout_samples=n_heldout,
        **extra,
    )


def _write_empty_shards(shard_dir, drawn, tiles):
    """Write Phase A shards with no counts (used to exercise Phase B alone)."""
    os.makedirs(shard_dir, exist_ok=True)
    tile_offsets = np.array([[i, 0, 0] for i in range(len(tiles))], dtype=np.int64)
    for lib in drawn["library"]:
        np.savez(
            os.path.join(shard_dir, f"{lib}.npz"),
            pos=np.empty(0, np.uint16),
            track=np.empty(0, np.uint8),
            data=np.empty(0, np.uint16),
            tile_offsets=tile_offsets,
            total_fragments=np.array(25_000_000, np.uint64),
        )


def _expected_dense(RegionFragmentArray, cfg, h5_path, tile, ref="hg38"):
    """Independently recount one (sample, tile) into an (C, L_TARGET) array."""
    from fragmentomics_tools.region import Region

    margin = cfg.jitter
    count_start = tile["start"] - margin
    clamped_start = max(0, count_start)
    clamped_stop = tile["stop"] + margin
    cl = _contig_length(ref, tile["contig"])
    if cl is not None:
        clamped_stop = min(clamped_stop, cl)
    left_pad = clamped_start - count_start

    region = Region(
        chrom=tile["contig"], start=clamped_start, stop=clamped_stop, strand="."
    )
    h5 = _FakeFragmentsH5(h5_path)
    rfa = RegionFragmentArray.from_fragments_h5(
        h5, region, min_mapq=cfg.min_mapq, max_frag_len=cfg.max_frag_len
    )
    if cfg.dedup:
        rfa = rfa.drop_duplicate_fragments()
    sparse = rfa.build_coverage_counts(
        fl_bands=list(cfg.fl_bands), split_strand=True, return_sparse=True
    )

    l_target = cfg.l_target
    dense = np.zeros((C, l_target), dtype=np.float32)
    for key, vec in sparse.items():
        ti = TRACK_INDEX[key]
        if len(vec.coords) > 0:
            np.add.at(
                dense,
                (ti, np.asarray(vec.coords) + left_pad),
                np.asarray(vec.data, dtype=np.float32),
            )
    return dense, left_pad


class TestNegativeStartTile:
    def test_worker_handles_contig_start(self, tmp_dir, monkeypatch):
        """A tile at contig position 0 must not crash Phase A; counts land at
        L_TARGET-relative positions and out-of-contig positions are masked."""
        cfg = _make_synth_config(tmp_dir, region=(0, 2 * SMALL_TILE))

        def frag_fn(h5, region, max_frag_len):
            # single '+' fragment: region-relative [50, 200), length 150
            return (
                np.array([50], np.int64),
                np.array([200], np.int64),
                np.array(["+"]),
            )

        _patch_phase_a(monkeypatch, frag_fn)

        sheet = pd.read_csv(cfg.sample_sheet, sep="\t")
        drawn = draw_samples(sheet, cfg)
        tiles = build_tiles(
            cfg.region_beds, cfg.tile_size, cfg.jitter, cfg.rf_budget, "hg38"
        )
        shard_dir = os.path.join(tmp_dir, "shards")

        # Tile 0 starts at contig 0 → count_start = -jitter (would crash pre-fix)
        run_phase_a(cfg, drawn, tiles, shard_dir, "hg38", n_workers=1)

        left_pad = cfg.jitter
        lib0 = drawn["library"].iloc[0]
        shard = np.load(os.path.join(shard_dir, f"{lib0}.npz"))
        off = {int(r[0]): (int(r[1]), int(r[2])) for r in shard["tile_offsets"]}
        lo, hi = off[0]
        pos0 = shard["pos"][lo:hi]
        track0 = shard["track"][lo:hi]

        first_t = TRACK_INDEX[("+", (120, 175), "first")]
        last_t = TRACK_INDEX[("+", (120, 175), "last")]
        mid_t = TRACK_INDEX[("+", (120, 175), "midpoint")]

        # first covered base 50, last 199, midpoint 125 → +left_pad in L_TARGET frame
        assert list(pos0[track0 == first_t]) == [50 + left_pad]
        assert list(pos0[track0 == last_t]) == [199 + left_pad]
        assert list(pos0[track0 == mid_t]) == [125 + left_pad]
        # nothing lands in the out-of-contig left margin
        assert (pos0 >= left_pad).all()

        # Phase B mask: out-of-contig positions invalid, rest valid (no blacklist)
        out = os.path.join(tmp_dir, "out")
        os.makedirs(out)
        run_phase_b(cfg, drawn, tiles, shard_dir, out, "hg38")
        root = open_store(os.path.join(out, cfg.store_name()))
        mask0 = np.asarray(root["tiles/mask"][0])
        assert not mask0[:left_pad].any()
        assert mask0[left_pad:].all()


class TestPhaseAtoBIntegration:
    def test_two_samples_two_tiles(self, tmp_dir, monkeypatch):
        cfg = _make_synth_config(
            tmp_dir, n_samples=2, n_train=1, n_heldout=1,
            region=(3072, 3072 + 2 * SMALL_TILE),
        )
        RFA = _patch_phase_a(monkeypatch, _synthetic_fragments)

        sheet = pd.read_csv(cfg.sample_sheet, sep="\t")
        drawn = draw_samples(sheet, cfg)
        tiles = build_tiles(
            cfg.region_beds, cfg.tile_size, cfg.jitter, cfg.rf_budget, "hg38"
        )
        S, T = len(drawn), len(tiles)
        assert (S, T) == (2, 2)

        shard_dir = os.path.join(tmp_dir, "shards")
        run_phase_a(cfg, drawn, tiles, shard_dir, "hg38", n_workers=1)
        out = os.path.join(tmp_dir, "out")
        os.makedirs(out)
        store_path = run_phase_b(cfg, drawn, tiles, shard_dir, out, "hg38")

        root = open_store(store_path)
        l_target = cfg.l_target
        libraries = list(drawn["library"])
        for s_idx, lib in enumerate(libraries):
            h5_path = drawn["h5_path"].iloc[s_idx]
            for t_idx, tile in enumerate(tiles):
                dense, _ = _expected_dense(RFA, cfg, h5_path, tile)
                pos, track, data = csr_slice(root, s_idx, t_idx, T)
                got = densify_counts(pos, track, data, C, l_target)
                np.testing.assert_array_equal(
                    got, dense, err_msg=f"CSR mismatch sample={lib} tile={t_idx}"
                )
                mask = np.asarray(root["tiles/mask"][t_idx])
                exp_N = compute_N_for_tile(dense, mask, cfg.tile_size, l_target)
                np.testing.assert_array_equal(
                    np.asarray(root["totals/N"][s_idx, t_idx]), exp_N
                )

        # split arrays present and valid
        splits = np.asarray(root["tiles/split"])
        assert splits.shape == (T,)
        assert set(np.unique(splits)).issubset({0, 1, 2, 3})

        # split_version bumped exactly once
        assert root.attrs["split_version"] == 1

        # attrs recorded
        assert root.attrs["config_hash"] == cfg.config_hash()
        assert root.attrs["applied_region_fracs"] == list(cfg.region_fracs)
        assert root.attrs["applied_min_total_fragments"] == cfg.min_total_fragments


def _expected_blacklist_mask(l_target, count_start, bstart, bstop, expansion):
    """Independent expected mask: invalid within `expansion` bp of [bstart, bstop)."""
    expected = np.ones(l_target, dtype=bool)
    local_start = max(0, bstart - expansion - count_start)
    local_stop = min(l_target, bstop + expansion - count_start)
    if local_start < local_stop:
        expected[local_start:local_stop] = False
    return expected


class TestBlacklistMaskConstruction:
    def test_mask_symmetric_expansion(self, tmp_dir):
        """Scope B (approved 2026-08-27): the position mask marks invalid every
        position within cfg.blacklist_expansion bp of a blacklist region — the
        SAME expansion Phase A applies to fragment masking."""
        P = 3072
        region = (P, P + 2 * SMALL_TILE)
        bstart, bstop = P + 50, P + 80
        cfg = _make_synth_config(
            tmp_dir, region=region, blacklist=[("chr1", bstart, bstop)]
        )
        assert cfg.blacklist_expansion == 120  # default under test

        sheet = pd.read_csv(cfg.sample_sheet, sep="\t")
        drawn = draw_samples(sheet, cfg)
        tiles = build_tiles(
            cfg.region_beds, cfg.tile_size, cfg.jitter, cfg.rf_budget, "hg38"
        )
        shard_dir = os.path.join(tmp_dir, "shards")
        _write_empty_shards(shard_dir, drawn, tiles)
        out = os.path.join(tmp_dir, "out")
        os.makedirs(out)
        run_phase_b(cfg, drawn, tiles, shard_dir, out, "hg38")

        root = open_store(os.path.join(out, cfg.store_name()))
        l_target = cfg.l_target
        exp = cfg.blacklist_expansion

        # Tile 0: expanded blacklist lands mid-frame.
        count_start0 = tiles[0]["start"] - cfg.jitter
        expected0 = _expected_blacklist_mask(
            l_target, count_start0, bstart, bstop, exp
        )
        np.testing.assert_array_equal(np.asarray(root["tiles/mask"][0]), expected0)
        # sanity: expansion widens the invalid zone beyond the core region
        core_lo = bstart - count_start0
        assert not expected0[core_lo - 1], "expansion must invalidate bp before the core region"

        # Tile 1: the EXPANDED zone reaches into tile 1's frame even though the
        # core blacklist region does not (this is the behavior change).
        count_start1 = tiles[1]["start"] - cfg.jitter
        expected1 = _expected_blacklist_mask(
            l_target, count_start1, bstart, bstop, exp
        )
        np.testing.assert_array_equal(np.asarray(root["tiles/mask"][1]), expected1)
        assert not expected1.all(), "expanded blacklist must reach tile 1's frame"


class TestResumeSemantics:
    def test_done_marker_skips_worker(self, tmp_dir, monkeypatch):
        import background_model.preprocess as ppmod

        cfg = _make_synth_config(tmp_dir)
        sheet = pd.read_csv(cfg.sample_sheet, sep="\t")
        drawn = draw_samples(sheet, cfg)
        tiles = build_tiles(
            cfg.region_beds, cfg.tile_size, cfg.jitter, cfg.rf_budget, "hg38"
        )
        shard_dir = os.path.join(tmp_dir, "shards")
        os.makedirs(shard_dir)

        row = drawn.to_dict("records")[0]
        lib = row["library"]
        Path(os.path.join(shard_dir, f"{lib}.done")).touch()

        def _boom(*a, **k):
            raise AssertionError("_worker_inner must be skipped when .done exists")

        monkeypatch.setattr(ppmod, "_worker_inner", _boom)
        result = _worker_process_sample(row, tiles, cfg, shard_dir, "hg38")
        assert result == (lib, True)

    def test_error_file_refuses(self, tmp_dir):
        cfg = _make_synth_config(tmp_dir)
        sheet = pd.read_csv(cfg.sample_sheet, sep="\t")
        drawn = draw_samples(sheet, cfg)
        tiles = build_tiles(
            cfg.region_beds, cfg.tile_size, cfg.jitter, cfg.rf_budget, "hg38"
        )
        shard_dir = os.path.join(tmp_dir, "shards")
        os.makedirs(shard_dir)
        Path(os.path.join(shard_dir, "LIB-XXX.error")).touch()

        with pytest.raises(RuntimeError, match="error file"):
            run_phase_a(cfg, drawn, tiles, shard_dir, "hg38", n_workers=1)


class TestWorkerErrorHandling:
    def test_exception_writes_error_file_and_surfaces(self, tmp_dir, monkeypatch):
        """When per-tile counting raises, the Phase A worker
        (_worker_process_sample) catches it, writes shards/<library>.error
        containing the exception message followed by the formatted traceback,
        and returns (library, False); run_phase_a then surfaces the failure by
        raising RuntimeError('Phase A failed for: ...').  This asserts the ACTUAL
        error-handling contract in preprocess.py, not a guess.
        """
        cfg = _make_synth_config(tmp_dir, n_samples=2, n_train=1, n_heldout=1)

        def _boom_frag(h5, region, max_frag_len):
            raise RuntimeError("synthetic count failure XYZ")

        _patch_phase_a(monkeypatch, _boom_frag)

        sheet = pd.read_csv(cfg.sample_sheet, sep="\t")
        drawn = draw_samples(sheet, cfg)
        tiles = build_tiles(
            cfg.region_beds, cfg.tile_size, cfg.jitter, cfg.rf_budget, "hg38"
        )
        shard_dir = os.path.join(tmp_dir, "shards")

        with pytest.raises(RuntimeError, match="Phase A failed"):
            run_phase_a(cfg, drawn, tiles, shard_dir, "hg38", n_workers=1)

        lib0 = drawn["library"].iloc[0]
        err_path = os.path.join(shard_dir, f"{lib0}.error")
        assert os.path.exists(err_path), "worker must write <library>.error on exception"
        text = Path(err_path).read_text()
        # message line then traceback (worker writes f"{e}\n{traceback}")
        assert "synthetic count failure XYZ" in text
        assert "Traceback (most recent call last)" in text
        # a failed sample must NOT be marked done.
        assert not os.path.exists(os.path.join(shard_dir, f"{lib0}.done"))


class TestBlacklistSpanningFragmentModelE2E:
    """Real preprocess (Phase A synthetic-count / Phase B real) -> Dataset ->
    BackgroundModel._step on all three losses, with a blacklist positioned so a
    blacklist-SPANNING fragment survives Phase A's mask_overlapping_fragments
    (which drops only FULLY-CONTAINED fragments, fragment_array.py:1402) and
    deposits its midpoint endpoint inside the Phase-B-masked expanded zone.

    Without the Dataset's mask-zeroing (dataset.py `y *= m`), that stray count at
    a masked position violates the frozen model's precondition (targets zero at
    masked positions, background_model_core._prepare_mask) and crashes the step
    with an AssertionError. This test therefore fails without fix 1 (verified by
    temporary revert) and passes with it.
    """

    @pytest.mark.parametrize(
        "loss", ["multinomial", "dirichlet_multinomial", "nb_offset"]
    )
    def test_spanning_fragment_masked_endpoint_step_finite(
        self, tmp_dir, monkeypatch, loss
    ):
        pytest.importorskip("torch")
        pytest.importorskip("background_model_core")
        import torch
        from background_model_core import BackgroundModel, _prepare_mask

        from background_model.dataset import BackgroundTileDataset

        # ── geometry (SMALL_TILE=256, jitter=128 -> L_TARGET=512) ──────────
        # Single tile [3072, 3328); L_TARGET frame origin count_start = 2944.
        # Small blacklist_expansion=8 keeps the masked zone narrow so the tile
        # still has valid positions carrying counts (real code paths otherwise).
        # Local (L_TARGET-frame) coords == region-relative counts (left_pad=0):
        #   blacklist genomic [3144,3148) -> local [200,204); +/-8 -> masked
        #   [192,212). The '+' spanning fragment [120,270) has first=120,
        #   last=269 (valid) and midpoint=195 (INSIDE the masked zone) yet is not
        #   fully contained in [192,212) (start 120 < 192) so Phase A keeps it.
        P = 3072
        cfg = _make_synth_config(
            tmp_dir,
            n_samples=2,
            n_train=1,
            n_heldout=1,
            region=(P, P + SMALL_TILE),          # single tile
            blacklist=[("chr1", 3144, 3148)],
            blacklist_expansion=8,
        )
        assert cfg.blacklist_expansion == 8

        def frag_fn(h5, region, max_frag_len):
            # region-relative (== L_TARGET-frame) fragments, all length 150 ->
            # fl_band (120,175), strand '+'. First is the blacklist-spanning one.
            starts = np.array([120, 220, 228, 236], dtype=np.int64)
            stops = starts + 150
            strands = np.array(["+", "+", "+", "+"])
            return starts, stops, strands

        _patch_phase_a(monkeypatch, frag_fn)

        sheet = pd.read_csv(cfg.sample_sheet, sep="\t")
        drawn = draw_samples(sheet, cfg)
        tiles = build_tiles(
            cfg.region_beds, cfg.tile_size, cfg.jitter, cfg.rf_budget, "hg38"
        )
        assert len(tiles) == 1
        shard_dir = os.path.join(tmp_dir, "shards")
        run_phase_a(cfg, drawn, tiles, shard_dir, "hg38", n_workers=1)
        out = os.path.join(tmp_dir, "out")
        os.makedirs(out)
        store_path = run_phase_b(cfg, drawn, tiles, shard_dir, out, "hg38")

        # Sanity: the RAW store keeps the (unzeroed) midpoint count at a masked
        # position — i.e. the precondition-violating count really is present, so
        # the Dataset (not the store) is what must zero it.
        root = open_store(store_path)
        mask0 = np.asarray(root["tiles/mask"][0])
        assert not mask0[195], "expected local position 195 to be masked"
        dense0 = densify_counts(
            *csr_slice(root, 0, 0, 1), C, cfg.l_target
        )
        mid_t = TRACK_INDEX[("+", (120, 175), "midpoint")]
        assert dense0[mid_t, 195] == 1, "raw store must retain the masked-pos count"

        # ── Dataset -> model _step on the chosen loss (CPU, deterministic) ──
        model = BackgroundModel(
            n_kernels=8, kernel_size=4, num_residual_layers=1, loss=loss,
            dispersion_window_size=SMALL_TILE,
        )
        mis = model.calc_input_region_size(cfg.tile_size)  # even, << L_SEQ
        ds = BackgroundTileDataset(
            store_path, model_input_size=mis, split="train",
            sample_role="train", min_N=0, train_mode=False,
        )
        assert len(ds) == 1
        loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=0)
        x, y, mask = next(iter(loader))

        # With fix 1 the Dataset has zeroed y at masked positions, so
        # _prepare_mask's assertion holds; without it, this raises AssertionError.
        mask3 = _prepare_mask(mask, y)
        shape_logits, dispersion_bp = model(x)
        if loss == "multinomial":
            l = model.loss_fn(shape_logits, y, mask3)
        else:
            log_disp = model._pooled_log_dispersion(dispersion_bp, mask3)
            l = model.loss_fn(shape_logits, log_disp, y, mask3)
        assert torch.isfinite(l), f"{loss} loss not finite"
