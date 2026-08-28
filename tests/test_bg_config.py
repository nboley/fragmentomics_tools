"""Tests for background_model.config — PlumbingConfig + hashing."""

import json
import os
import tempfile

import pytest

from background_model.config import (
    C,
    L_SEQ,
    L_TARGET,
    TILE,
    PlumbingConfig,
    _HASH_EXCLUDED_FIELDS,
    verify_config_drift,
)


@pytest.fixture
def tmp_files():
    """Create temporary files for config testing."""
    d = tempfile.mkdtemp()
    sheet = os.path.join(d, "sheet.tsv")
    with open(sheet, "w") as f:
        f.write("library\th5_path\tseqrun\tendo_category\n")
        f.write("lib1\t/path/to/lib1.h5\tSR-1\tAsymptomatic\n")

    bed_train = os.path.join(d, "train.bed")
    with open(bed_train, "w") as f:
        f.write("chr1\t0\t100000\n")

    bed_pc = os.path.join(d, "positive_control.bed")
    with open(bed_pc, "w") as f:
        f.write("chr1\t200000\t210000\n")

    blacklist = os.path.join(d, "blacklist.bed")
    with open(blacklist, "w") as f:
        f.write("chr1\t5000\t5100\n")

    fasta = os.path.join(d, "test.fa")
    with open(fasta, "w") as f:
        f.write(">chr1\nACGT\n")
    fai = os.path.join(d, "test.fa.fai")
    with open(fai, "w") as f:
        f.write("chr1\t4\t6\t4\t5\n")

    yield {
        "dir": d,
        "sheet": sheet,
        "bed_train": bed_train,
        "bed_pc": bed_pc,
        "blacklist": blacklist,
        "fasta": fasta,
    }

    import shutil
    shutil.rmtree(d)


def _make_config(files, **overrides):
    kw = dict(
        sample_sheet=files["sheet"],
        region_beds={"train_pool": files["bed_train"]},
        blacklist_bed=files["blacklist"],
        fasta=files["fasta"],
    )
    kw.update(overrides)
    return PlumbingConfig(**kw)


class TestGeometryConstants:
    def test_tile_size(self):
        assert TILE == 16_384

    def test_l_target(self):
        assert L_TARGET == 16_640  # TILE + 2 * 128

    def test_l_seq(self):
        assert L_SEQ == 20_736  # TILE + 2 * (128 + 2048)

    def test_c_tracks(self):
        assert C == 12  # 2 strands * 2 fl_bands * 3 coverage types

    def test_all_even_parity(self):
        """Design §0: all four lengths must be even."""
        assert TILE % 2 == 0
        assert L_TARGET % 2 == 0
        assert L_SEQ % 2 == 0

    def test_derived_properties(self):
        cfg = PlumbingConfig(sample_sheet="dummy")
        assert cfg.l_target == L_TARGET
        assert cfg.l_seq == L_SEQ


class TestConfigHash:
    def test_hash_stability(self, tmp_files):
        """Same config produces same hash."""
        cfg1 = _make_config(tmp_files)
        cfg2 = _make_config(tmp_files)
        assert cfg1.config_hash() == cfg2.config_hash()

    def test_hash_path_independence(self, tmp_files):
        """File content determines hash, not path.

        Two configs pointing at identical-content files at different paths
        must produce the same hash.
        """
        import shutil
        d2 = tempfile.mkdtemp()
        sheet2 = os.path.join(d2, "sheet2.tsv")
        shutil.copy(tmp_files["sheet"], sheet2)
        bed2 = os.path.join(d2, "train2.bed")
        shutil.copy(tmp_files["bed_train"], bed2)
        bl2 = os.path.join(d2, "bl2.bed")
        shutil.copy(tmp_files["blacklist"], bl2)
        fa2 = os.path.join(d2, "test2.fa")
        shutil.copy(tmp_files["fasta"], fa2)
        fai2 = os.path.join(d2, "test2.fa.fai")
        shutil.copy(tmp_files["fasta"] + ".fai", fai2)

        cfg1 = _make_config(tmp_files)
        cfg2 = PlumbingConfig(
            sample_sheet=sheet2,
            region_beds={"train_pool": bed2},
            blacklist_bed=bl2,
            fasta=fa2,
        )
        assert cfg1.config_hash() == cfg2.config_hash()
        shutil.rmtree(d2)

    def test_hash_excludes_min_N(self, tmp_files):
        """min_N is excluded from hash (condition #4)."""
        cfg1 = _make_config(tmp_files, min_N=50)
        cfg2 = _make_config(tmp_files, min_N=100)
        assert cfg1.config_hash() == cfg2.config_hash()

    def test_hash_excludes_region_fracs(self, tmp_files):
        """region_fracs is excluded from hash (condition #4)."""
        cfg1 = _make_config(tmp_files, region_fracs=(0.8, 0.1, 0.1))
        cfg2 = _make_config(tmp_files, region_fracs=(0.6, 0.2, 0.2))
        assert cfg1.config_hash() == cfg2.config_hash()

    def test_hash_excludes_min_total_fragments(self, tmp_files):
        """min_total_fragments is excluded from hash (condition #4)."""
        cfg1 = _make_config(tmp_files, min_total_fragments=10_000_000)
        cfg2 = _make_config(tmp_files, min_total_fragments=30_000_000)
        assert cfg1.config_hash() == cfg2.config_hash()

    def test_hash_includes_geometry(self, tmp_files):
        """Geometry changes must change the hash (condition #4)."""
        cfg1 = _make_config(tmp_files, tile_size=16384)
        cfg2 = _make_config(tmp_files, tile_size=8192)
        assert cfg1.config_hash() != cfg2.config_hash()

    def test_hash_includes_fl_bands(self, tmp_files):
        """fl_bands changes must change the hash (condition #4)."""
        cfg1 = _make_config(tmp_files, fl_bands=((40, 65), (120, 175)))
        cfg2 = _make_config(tmp_files, fl_bands=((40, 65),))
        assert cfg1.config_hash() != cfg2.config_hash()

    def test_hash_includes_filters(self, tmp_files):
        """Filter changes must change the hash (condition #4)."""
        cfg1 = _make_config(tmp_files, min_mapq=10)
        cfg2 = _make_config(tmp_files, min_mapq=20)
        assert cfg1.config_hash() != cfg2.config_hash()

    def test_hash_includes_seed(self, tmp_files):
        """Seed changes must change the hash (condition #4)."""
        cfg1 = _make_config(tmp_files, seed=1337)
        cfg2 = _make_config(tmp_files, seed=42)
        assert cfg1.config_hash() != cfg2.config_hash()

    def test_hash_includes_sample_draw_params(self, tmp_files):
        """Sample draw params change must change hash (condition #4)."""
        cfg1 = _make_config(tmp_files, n_train_samples=40)
        cfg2 = _make_config(tmp_files, n_train_samples=30)
        assert cfg1.config_hash() != cfg2.config_hash()

    def test_hash_includes_blacklist_content(self, tmp_files):
        """Blacklist BED CONTENT change must change the hash (content-addressed,
        not path-addressed)."""
        bl2 = os.path.join(tmp_files["dir"], "blacklist2.bed")
        with open(bl2, "w") as f:
            f.write("chr1\t9000\t9500\n")  # differs from fixture blacklist
        cfg1 = _make_config(tmp_files)
        cfg2 = _make_config(tmp_files, blacklist_bed=bl2)
        assert cfg1.config_hash() != cfg2.config_hash()

    def test_hash_includes_region_bed_content(self, tmp_files):
        """Region BED CONTENT change must change the hash."""
        bed2 = os.path.join(tmp_files["dir"], "train2.bed")
        with open(bed2, "w") as f:
            f.write("chr1\t0\t200000\n")  # differs from fixture train.bed span
        cfg1 = _make_config(tmp_files)
        cfg2 = _make_config(tmp_files, region_beds={"train_pool": bed2})
        assert cfg1.config_hash() != cfg2.config_hash()

    def test_hash_includes_fasta_fai_content(self, tmp_files):
        """FASTA .fai CONTENT change must change the hash (the .fai is hashed as
        a proxy for FASTA content identity)."""
        fa2 = os.path.join(tmp_files["dir"], "test2.fa")
        with open(fa2, "w") as f:
            f.write(">chr1\nACGTACGT\n")
        with open(fa2 + ".fai", "w") as f:
            f.write("chr1\t8\t6\t8\t9\n")  # differs from fixture .fai
        cfg1 = _make_config(tmp_files)
        cfg2 = _make_config(tmp_files, fasta=fa2)
        assert cfg1.config_hash() != cfg2.config_hash()

    def test_hash_includes_blacklist_expansion(self, tmp_files):
        """blacklist_expansion value change must change the hash (it affects the
        counts Phase A produces)."""
        cfg1 = _make_config(tmp_files, blacklist_expansion=120)
        cfg2 = _make_config(tmp_files, blacklist_expansion=60)
        assert cfg1.config_hash() != cfg2.config_hash()

    def test_hash_includes_dedup(self, tmp_files):
        """dedup flag change must change the hash (it affects the counts)."""
        cfg1 = _make_config(tmp_files, dedup=True)
        cfg2 = _make_config(tmp_files, dedup=False)
        assert cfg1.config_hash() != cfg2.config_hash()


class TestConfigSerialization:
    def test_roundtrip(self, tmp_files):
        cfg = _make_config(tmp_files)
        json_str = cfg.full_config_json()
        cfg2 = PlumbingConfig.from_json(json_str)
        assert cfg == cfg2

    def test_full_config_includes_all_fields(self, tmp_files):
        cfg = _make_config(tmp_files)
        d = json.loads(cfg.full_config_json())
        assert "min_N" in d
        assert "region_fracs" in d
        assert "min_total_fragments" in d

    def test_canonical_json_excludes_phase_b(self, tmp_files):
        cfg = _make_config(tmp_files)
        cj = cfg.canonical_json(with_content_hashes=False)
        d = json.loads(cj)
        for excluded in _HASH_EXCLUDED_FIELDS:
            assert excluded not in d


class TestMaxFragLen:
    def test_default_max_frag_len(self):
        cfg = PlumbingConfig(sample_sheet="dummy")
        assert cfg.max_frag_len == 175

    def test_half_open_semantics(self):
        """fl_bands (40,65) means [40, 65) — max_frag_len=175 is exclusive (condition #5)."""
        cfg = PlumbingConfig(sample_sheet="dummy", fl_bands=((40, 65), (120, 175)))
        # max_frag_len = max of hi values = 175, used as exclusive bound
        assert cfg.max_frag_len == 175


class TestDriftRule:
    def test_drift_detected(self, tmp_files):
        """Config drift raises ValueError."""
        import zarr
        d = tempfile.mkdtemp()
        store_path = os.path.join(d, "test.zarr")
        root = zarr.open_group(store_path, mode="w")
        root.attrs["config_hash"] = "deadbeef" * 8

        cfg = _make_config(tmp_files)
        with pytest.raises(ValueError, match="Config drift"):
            verify_config_drift(store_path, cfg)

        import shutil
        shutil.rmtree(d)

    def test_no_drift(self, tmp_files):
        """Matching config passes."""
        import zarr
        d = tempfile.mkdtemp()
        store_path = os.path.join(d, "test.zarr")
        cfg = _make_config(tmp_files)
        root = zarr.open_group(store_path, mode="w")
        root.attrs["config_hash"] = cfg.config_hash()

        verify_config_drift(store_path, cfg)  # should not raise

        import shutil
        shutil.rmtree(d)
