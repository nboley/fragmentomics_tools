"""Inference-path tests (design §6): equivalence-to-Dataset, seam-free
stitching, coordinate mapping, and the ``locate`` property test.

T1 is the single most important test — it proves the store-free inference input
(``build_window_onehot`` / ``build_window_mask``) is BYTE-IDENTICAL to the
val-mode ``BackgroundTileDataset`` input for every tile.  It is written and run
FIRST (the design mandates it gates everything).
"""

import os
import shutil
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("background_model_core")

import pysam

from background_model.config import PlumbingConfig
from background_model.dataset import BackgroundTileDataset
from background_model.inference import (
    WindowGeometry,
    build_window_mask,
    build_window_onehot,
    iter_windows,
    locate,
    predict_region_profiles,
)
from background_model.preprocess import run_preprocess
from background_model.store import open_store
from background_model_core import BackgroundModel

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")
_FASTA = os.path.join(_DATA, "GRCh38.p12.genome.chr6_99110000_99130000.fa.gz")
_H5 = os.path.join(_DATA, "golden.small.chr6.frag.h5")

_FIXTURES = os.path.exists(_FASTA) and os.path.exists(_H5)
pytestmark = pytest.mark.skipif(
    not _FIXTURES, reason=f"golden fixtures missing ({_FASTA}, {_H5})"
)

# Small tile so L_SEQ / model_input are cheap; region well inside the populated
# window chr6:99110000-99130000 with room for the seq margin (jitter+rf_budget).
_TILE = 256
_CONTIG = "chr6"
_REGION = (99_114_000, 99_114_000 + 3 * _TILE)   # 3 tiles


def _small_model(loss="multinomial", seed=0):
    torch.manual_seed(seed)
    return BackgroundModel(
        n_kernels=8, kernel_size=4, num_residual_layers=1, loss=loss,
    )


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


def _make_config(tmp_dir, blacklist=None):
    """Build a PlumbingConfig over the real golden h5 + chr6 window FASTA."""
    sheet = os.path.join(tmp_dir, "sheet.tsv")
    with open(sheet, "w") as f:
        f.write("library\th5_path\tseqrun\tendo_category\n")
        for i in range(2):
            f.write(f"LIB-{i:03d}\t{_H5}\tSR-{i}\tAsymptomatic\n")

    bed = os.path.join(tmp_dir, "train.bed")
    with open(bed, "w") as f:
        f.write(f"{_CONTIG}\t{_REGION[0]}\t{_REGION[1]}\n")

    blk = ""
    if blacklist is not None:
        blk = os.path.join(tmp_dir, "blacklist.bed")
        with open(blk, "w") as f:
            for bc, bs, bp in blacklist:
                f.write(f"{bc}\t{bs}\t{bp}\n")

    return PlumbingConfig(
        sample_sheet=sheet,
        region_beds={"train_pool": bed},
        blacklist_bed=blk,
        fasta=_FASTA,
        tile_size=_TILE,
        n_train_samples=1,
        n_heldout_samples=1,
        min_total_fragments=0,   # keep the sample at role=train
        min_N=0,                 # admit every tile regardless of counts
    )


def _build_store(tmp_dir, **kw):
    cfg = _make_config(tmp_dir, **kw)
    out = os.path.join(tmp_dir, "out")
    os.makedirs(out)
    store_path = run_preprocess(cfg, out, ref="hg38", n_workers=1)
    return cfg, store_path


# ── T1: equivalence to the val-mode Dataset input (gates everything) ──────

class TestEquivalenceToDataset:
    def test_onehot_and_mask_byte_exact_no_blacklist(self, tmp_dir):
        cfg, store_path = _build_store(tmp_dir)
        model = _small_model()
        geom = WindowGeometry.from_model(model, cfg.tile_size)
        mis = geom.model_input_size

        ds = BackgroundTileDataset(
            store_path, model_input_size=mis, split="train",
            sample_role="train", min_N=0, train_mode=False,
        )
        assert len(ds) >= 1

        root = open_store(store_path)
        starts = np.asarray(root["tiles/start"])
        stops = np.asarray(root["tiles/stop"])
        contigs = [str(c) for c in np.asarray(root["tiles/contig"])]
        jitter = cfg.jitter

        for i in range(len(ds)):
            _, t = ds.index[i]
            x_ds = ds[i][0].numpy()
            x_inf = build_window_onehot(
                pysam.FastaFile(_FASTA), contigs[t], int(starts[t]),
                int(stops[t]), geom,
            )
            np.testing.assert_array_equal(
                x_inf, x_ds, err_msg=f"onehot mismatch tile={t}"
            )
            # mask: build_window_mask == center-crop of the store L_TARGET mask.
            store_mask = np.asarray(root["tiles/mask"][t])
            center = store_mask[jitter:jitter + cfg.tile_size]
            m_inf = build_window_mask(
                contigs[t], int(starts[t]), int(stops[t]), geom,
                blacklist_rdf=None, contig_len=170_805_979,
            )
            np.testing.assert_array_equal(
                m_inf, center, err_msg=f"mask mismatch tile={t}"
            )

    def test_mask_byte_exact_with_blacklist(self, tmp_dir):
        # blacklist landing inside tile 0's window drives the expansion path.
        bstart, bstop = _REGION[0] + 100, _REGION[0] + 140
        cfg, store_path = _build_store(
            tmp_dir, blacklist=[(_CONTIG, bstart, bstop)]
        )
        from fragmentomics_tools.dataframe import RegionDataFrame

        model = _small_model()
        geom = WindowGeometry.from_model(model, cfg.tile_size)
        root = open_store(store_path)
        starts = np.asarray(root["tiles/start"])
        stops = np.asarray(root["tiles/stop"])
        contigs = [str(c) for c in np.asarray(root["tiles/contig"])]
        jitter = cfg.jitter
        bl_rdf = RegionDataFrame.from_bed(cfg.blacklist_bed, ref="hg38")

        any_masked = False
        for t in range(len(starts)):
            store_mask = np.asarray(root["tiles/mask"][t])
            center = store_mask[jitter:jitter + cfg.tile_size]
            m_inf = build_window_mask(
                contigs[t], int(starts[t]), int(stops[t]), geom,
                blacklist_rdf=bl_rdf, blacklist_expansion=cfg.blacklist_expansion,
                contig_len=170_805_979,
            )
            np.testing.assert_array_equal(
                m_inf, center, err_msg=f"blacklist mask mismatch tile={t}"
            )
            any_masked = any_masked or (not center.all())
        assert any_masked, "expected the blacklist to invalidate some positions"


# ── T2: seam-free stitching (of the SHAPE) ───────────────────────────────

class TestSeamFree:
    def test_two_windows_equal_two_standalone(self, tmp_dir):
        model = _small_model(seed=3)
        geom = WindowGeometry.from_model(model, _TILE)
        fasta = pysam.FastaFile(_FASTA)
        a = _REGION[0]

        rp = predict_region_profiles(
            model, fasta, _CONTIG, a, a + 2 * _TILE,
            contig_len=170_805_979, tile_size=_TILE,
        )

        # two independent single-window predict_profile calls.
        parts = []
        for w0 in (a, a + _TILE):
            oh = build_window_onehot(fasta, _CONTIG, w0, w0 + _TILE, geom)
            mk = build_window_mask(
                _CONTIG, w0, w0 + _TILE, geom, contig_len=170_805_979
            )
            parts.append(np.asarray(model.predict_profile(oh, mask=mk)["probs"]))
        standalone = np.concatenate(parts, axis=1)

        np.testing.assert_allclose(rp.probs, standalone, atol=0, rtol=0)


# ── T3: coordinate mapping ───────────────────────────────────────────────

class TestCoordinateMapping:
    def test_coord0_bounds_and_trim(self, tmp_dir):
        model = _small_model()
        fasta = pysam.FastaFile(_FASTA)
        a = _REGION[0]

        # exact multiple of tile: bounds tile [a, a+2T) with no gaps/overlaps.
        rp = predict_region_profiles(
            model, fasta, _CONTIG, a, a + 2 * _TILE,
            contig_len=170_805_979, tile_size=_TILE,
        )
        assert rp.coord0 == a
        assert rp.probs.shape[1] == 2 * _TILE
        bounds = rp.window_bounds
        assert bounds[0][:2] == (a, a + _TILE)
        assert bounds[1][:2] == (a + _TILE, a + 2 * _TILE)
        for (ws, we, lo, hi) in bounds:
            assert lo == 0 and hi == _TILE

        # sub-grid stop: final window trimmed so probs len == stop - start.
        stop = a + _TILE + 100
        rp2 = predict_region_profiles(
            model, fasta, _CONTIG, a, stop,
            contig_len=170_805_979, tile_size=_TILE,
        )
        assert rp2.probs.shape[1] == stop - a
        assert rp2.window_bounds[-1] == (a + _TILE, a + 2 * _TILE, 0, 100)


# ── T10 (part): locate property test (round-trip / off-grid / straddle) ──

class TestLocate:
    def test_roundtrip_random(self):
        rng = np.random.default_rng(0)
        start, stop, tile = 1000, 1000 + 5 * 256, 256
        for _ in range(500):
            g = int(rng.integers(start, stop))
            win, j = locate(g, start, stop, tile)
            assert win >= 0 and 0 <= j < tile
            assert (start + win * tile) + j == g

    def test_off_grid_rejected(self):
        start, stop, tile = 1000, 1000 + 3 * 256, 256
        assert locate(start - 1, start, stop, tile) == (-1, -1)
        assert locate(stop, start, stop, tile) == (-1, -1)
        assert locate(stop + 50, start, stop, tile) == (-1, -1)

    def test_off_grid_start_anchor(self):
        # region.start NOT a tile multiple: anchor is region.start.
        start, stop, tile = 99_114_037, 99_114_037 + 3 * 256, 256
        win, j = locate(start, start, stop, tile)
        assert (win, j) == (0, 0)
        win, j = locate(start + 256, start, stop, tile)
        assert (win, j) == (1, 0)

    def test_boundary_straddle_distinct_windows(self):
        start, stop, tile = 1000, 1000 + 3 * 256, 256
        # positions on either side of the window-0/1 boundary
        assert locate(1000 + 255, start, stop, tile) == (0, 255)
        assert locate(1000 + 256, start, stop, tile) == (1, 0)

    def test_subgrid_stop_rejects_beyond(self):
        start, stop, tile = 1000, 1000 + 256 + 100, 256  # final window trimmed
        assert locate(start + 256 + 99, start, stop, tile) == (1, 99)
        assert locate(start + 256 + 100, start, stop, tile) == (-1, -1)


# ── geometry failure modes (design §7) ───────────────────────────────────

class TestGeometryFailureModes:
    def test_from_model_asserts_even(self):
        class _OddModel:
            def calc_input_region_size(self, n):
                return n + 1  # odd difference
        with pytest.raises(AssertionError, match="even"):
            WindowGeometry.from_model(_OddModel(), 256)

    def test_onehot_length_matches_model_input(self, tmp_dir):
        model = _small_model()
        geom = WindowGeometry.from_model(model, _TILE)
        fasta = pysam.FastaFile(_FASTA)
        x = build_window_onehot(fasta, _CONTIG, _REGION[0], _REGION[0] + _TILE, geom)
        assert x.shape == (4, geom.model_input_size)

    def test_iter_windows_covers_and_anchors(self):
        wins = list(iter_windows(100, 100 + 2 * 256 + 10, 256))
        assert wins[0] == (100, 356)
        assert wins[-1][0] == 100 + 2 * 256      # final window runs full length
        assert wins[-1][1] > 100 + 2 * 256 + 10  # overruns stop (caller trims)
