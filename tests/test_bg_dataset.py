"""Tests for BackgroundTileDataset (design §5, §7).

The headline test is the CROP-ALIGNMENT PROPERTY TEST (the design's #1 stated
risk): a synthetic store whose track-0 count at target position p encodes the
base identity at the SAME genomic position must, after any (jitter, RC) crop,
still line up with the one-hot sequence the model actually consumes. Plus an
RC-involution check at the Dataset level.
"""

import os
import shutil
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("background_model_core")

from background_model.config import C, PlumbingConfig
from background_model.dataset import _COMPLEMENT_LUT, BackgroundTileDataset
from background_model.store import create_store

# tiny geometry for fast synthetic stores
_TILE = 256
_JIT = 8
_RF = 16
_MODEL_IN = 280  # even; <= l_seq - 2*jitter = 304 - 16 = 288
_MARGIN_X = (_MODEL_IN - _TILE) // 2  # 12
_BASES = np.array([ord("A"), ord("C"), ord("G"), ord("T")], dtype=np.uint8)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


def _build_synthetic_store(store_path, seed=0, n_val_high=100, invalid_prefix=20):
    """One-sample, one-tile store.

    seq_full is random ACGT over L_SEQ. Track-0 counts encode base identity at
    the matching genomic position: y_full[0, p] = code(seq_full[p + RF]) with
    code A/C/G/T -> 1/2/3/4. Mask marks the first `invalid_prefix` target
    positions invalid (a distinctive pattern to verify mask-crop alignment). N
    is set high so the min_N filter admits the tile.
    """
    cfg = PlumbingConfig(sample_sheet="", tile_size=_TILE, jitter=_JIT, rf_budget=_RF)
    l_target = cfg.l_target  # 272
    l_seq = cfg.l_seq        # 304
    nnz = l_target

    rng = np.random.default_rng(seed)
    seq_codes = rng.integers(0, 4, size=l_seq)         # 0..3
    seq_full = _BASES[seq_codes].astype(np.uint8)
    tgt_codes = (seq_codes[_RF:_RF + l_target] + 1).astype(np.uint16)  # 1..4

    root = create_store(store_path, cfg, n_tiles=1, n_samples=1, nnz=nnz)
    root["counts/indptr"][:] = np.array([0, nnz], dtype=np.int64)
    root["counts/pos"][:] = np.arange(l_target, dtype=np.uint16)
    root["counts/track"][:] = np.zeros(l_target, dtype=np.uint8)
    root["counts/data"][:] = tgt_codes

    mask_full = np.ones(l_target, dtype=bool)
    mask_full[:invalid_prefix] = False
    root["tiles/mask"][0] = mask_full
    root["tiles/seq"][0] = seq_full
    root["tiles/split"][:] = np.array([0], dtype=np.uint8)     # train
    root["tiles/contig"][:] = np.array(["chr1"])
    root["tiles/start"][:] = np.array([1000], dtype=np.int64)
    root["tiles/stop"][:] = np.array([1000 + _TILE], dtype=np.int64)
    root["tiles/strand"][:] = np.array(["."])
    root["tiles/region_id"][:] = np.array(["chr1:1000-1256"])

    root["samples/role"][:] = np.array([0], dtype=np.uint8)    # train
    root["samples/library"][:] = np.array(["LIB0"])
    root["samples/seqrun"][:] = np.array(["SR0"])
    root["samples/endo_category"][:] = np.array(["Asymptomatic"])
    root["samples/h5_path"][:] = np.array(["/x.h5"])
    root["samples/total_fragments"][:] = np.array([1], dtype=np.uint64)

    root["totals/N"][:] = np.full((1, 1, C), n_val_high, dtype=np.uint32)

    y_full = np.zeros((C, l_target), dtype=np.float32)
    y_full[0, :] = tgt_codes.astype(np.float32)
    return cfg, seq_full, mask_full, y_full


def _make_ds(store_path, **kw):
    kw.setdefault("model_input_size", _MODEL_IN)
    kw.setdefault("split", "train")
    kw.setdefault("sample_role", "train")
    kw.setdefault("min_N", 0)
    return BackgroundTileDataset(store_path, **kw)


# ── geometry / init contract ─────────────────────────────────────────────

class TestInitContract:
    def test_reproducibility_attrs(self, tmp_dir):
        sp = os.path.join(tmp_dir, "s.zarr")
        cfg, *_ = _build_synthetic_store(sp)
        ds = _make_ds(sp)
        assert ds.config_hash == cfg.config_hash()
        assert ds.split_version == 0  # create_store initializes to 0

    def test_odd_length_fails_loudly(self, tmp_dir):
        sp = os.path.join(tmp_dir, "s.zarr")
        _build_synthetic_store(sp)
        with pytest.raises(AssertionError, match="even"):
            _make_ds(sp, model_input_size=_MODEL_IN + 1)  # odd

    def test_rf_budget_too_small_fails(self, tmp_dir):
        sp = os.path.join(tmp_dir, "s.zarr")
        _build_synthetic_store(sp)
        # model_input_size 300 -> needs l_seq >= 300 + 16 = 316 > 304
        with pytest.raises(AssertionError, match="receptive field|l_seq"):
            _make_ds(sp, model_input_size=300)

    def test_bad_split_role_raise(self, tmp_dir):
        sp = os.path.join(tmp_dir, "s.zarr")
        _build_synthetic_store(sp)
        with pytest.raises(ValueError):
            _make_ds(sp, split="nope")
        with pytest.raises(ValueError):
            _make_ds(sp, sample_role="nope")

    def test_lazy_handle_not_opened_in_init(self, tmp_dir):
        """Worker safety: no zarr handle is created in __init__ (would be shared
        across a fork)."""
        sp = os.path.join(tmp_dir, "s.zarr")
        _build_synthetic_store(sp)
        ds = _make_ds(sp)
        assert ds._root is None and ds._root_pid is None
        _ = ds[0]
        assert ds._root is not None and ds._root_pid == os.getpid()


class TestIndex:
    def test_min_N_filter(self, tmp_dir):
        sp = os.path.join(tmp_dir, "s.zarr")
        _build_synthetic_store(sp, n_val_high=100)
        assert len(_make_ds(sp, min_N=50)) == 1
        assert len(_make_ds(sp, min_N=101)) == 0  # ALL tracks must pass

    def test_split_role_selection(self, tmp_dir):
        sp = os.path.join(tmp_dir, "s.zarr")
        _build_synthetic_store(sp)
        # tile split=0 (train), sample role=0 (train)
        assert len(_make_ds(sp, split="train", sample_role="train")) == 1
        assert len(_make_ds(sp, split="val", sample_role="train")) == 0
        assert len(_make_ds(sp, split="train", sample_role="heldout")) == 0

    def test_dropped_low_depth_sample_excluded(self, tmp_dir):
        """A role=2 (dropped_low_depth) sample's (sample, tile) pairs must be
        absent from the Dataset index for BOTH sample_role='train' and
        sample_role='heldout' — the depth-filtered sample is never trainable nor
        evaluatable, regardless of tile split or N."""
        sp = os.path.join(tmp_dir, "s.zarr")
        cfg = PlumbingConfig(
            sample_sheet="", tile_size=_TILE, jitter=_JIT, rf_budget=_RF
        )
        S, T = 3, 1
        root = create_store(sp, cfg, n_tiles=T, n_samples=S, nnz=0)
        root["counts/indptr"][:] = np.zeros(S * T + 1, dtype=np.int64)
        root["tiles/split"][:] = np.array([0], dtype=np.uint8)     # train tile
        root["tiles/contig"][:] = np.array(["chr1"])
        root["tiles/start"][:] = np.array([1000], dtype=np.int64)
        root["tiles/stop"][:] = np.array([1000 + _TILE], dtype=np.int64)
        root["tiles/strand"][:] = np.array(["."])
        root["tiles/region_id"][:] = np.array(["chr1:1000-1256"])
        # roles: 0=train, 1=heldout, 2=dropped_low_depth
        root["samples/role"][:] = np.array([0, 1, 2], dtype=np.uint8)
        root["samples/library"][:] = np.array(["LIB0", "LIB1", "LIB2"])
        root["samples/seqrun"][:] = np.array(["SR0", "SR1", "SR2"])
        root["samples/endo_category"][:] = np.array(["Asymptomatic"] * 3)
        root["samples/h5_path"][:] = np.array(["/a", "/b", "/c"])
        root["samples/total_fragments"][:] = np.array([1, 1, 1], dtype=np.uint64)
        # N high so the min_N filter would admit the dropped sample if role
        # exclusion were broken.
        root["totals/N"][:] = np.full((S, T, C), 100, dtype=np.uint32)

        DROPPED = 2
        ds_train = _make_ds(sp, split="train", sample_role="train")
        ds_heldout = _make_ds(sp, split="train", sample_role="heldout")

        assert all(s != DROPPED for s, _ in ds_train.index)
        assert all(s != DROPPED for s, _ in ds_heldout.index)
        assert (DROPPED, 0) not in ds_train.index
        assert (DROPPED, 0) not in ds_heldout.index
        # sanity: the eligible samples ARE present in their respective queries
        assert ds_train.index == [(0, 0)]
        assert ds_heldout.index == [(1, 0)]


# ── the crop-alignment property test (design risk #1) ────────────────────

class TestCropAlignment:
    @pytest.mark.parametrize("j", [-8, -3, 0, 1, 5, 8])
    def test_sequence_target_mask_share_center_no_rc(self, tmp_dir, j):
        sp = os.path.join(tmp_dir, "s.zarr")
        cfg, seq_full, mask_full, y_full = _build_synthetic_store(sp)
        ds = _make_ds(sp)
        x, y, m = ds._transform(y_full, mask_full, seq_full, j, do_rc=False)
        x = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
        y = np.asarray(y)
        m = np.asarray(m)
        assert x.shape == (4, _MODEL_IN)
        assert y.shape == (C, _TILE)
        assert m.shape == (_TILE,)

        # mask crop shares the same center (resize_start = JITTER = 8)
        expected_mask = mask_full[_JIT + j: _JIT + j + _TILE]
        np.testing.assert_array_equal(m, expected_mask)
        # targets are ZEROED at masked positions (Dataset-boundary policy that
        # satisfies the frozen model's _prepare_mask precondition).
        assert not y[:, ~m].any(), "targets must be zero at masked positions"
        # base identity recovered from the one-hot at position (k + margin_x)
        # must equal the count-encoded base at target position k (valid only).
        base_from_x = np.argmax(x[:, _MARGIN_X:_MARGIN_X + _TILE], axis=0) + 1
        np.testing.assert_array_equal(
            y[0][m].astype(int), base_from_x[m],
            err_msg=f"sequence/target misaligned at j={j}",
        )

    @pytest.mark.parametrize("j", [-8, 0, 5, 8])
    def test_rc_involution(self, tmp_dir, j):
        """Applying the RC augmentation is a clean involution across x, y, m."""
        sp = os.path.join(tmp_dir, "s.zarr")
        cfg, seq_full, mask_full, y_full = _build_synthetic_store(sp)
        ds = _make_ds(sp)
        x0, y0, m0 = (np.asarray(a) for a in ds._transform(y_full, mask_full, seq_full, j, False))
        x1, y1, m1 = (np.asarray(a) for a in ds._transform(y_full, mask_full, seq_full, j, True))

        # RC must actually change the data (not a no-op)
        assert not np.array_equal(x0, x1)

        # one-hot RC = complement (reverse base axis) + reverse position
        np.testing.assert_array_equal(x0, x1[::-1, ::-1])
        # target RC = channel permutation + reverse position (perm is involution)
        np.testing.assert_array_equal(y0, y1[ds.rc_perm][:, ::-1])
        # mask RC = reverse position
        np.testing.assert_array_equal(m0, m1[::-1])

    def test_rc_sequence_matches_target_after_rc(self, tmp_dir):
        """Under RC the target still encodes the base the model sees, now on the
        RC'd strand: y_rc's channel-unpermuted track-0 base equals the RC one-hot
        base at the aligned position."""
        sp = os.path.join(tmp_dir, "s.zarr")
        cfg, seq_full, mask_full, y_full = _build_synthetic_store(sp)
        ds = _make_ds(sp)
        j = 3
        x, y, m = (np.asarray(a) for a in ds._transform(y_full, mask_full, seq_full, j, True))
        # invert the channel permutation to recover the position-reversed track 0
        inv = np.empty_like(ds.rc_perm)
        inv[ds.rc_perm] = np.arange(len(ds.rc_perm))
        y_track0_reversed = y[inv][0]  # == complement-encoded base at pos k
        base_from_x = np.argmax(x[:, _MARGIN_X:_MARGIN_X + _TILE], axis=0) + 1
        # complement code: A(1)<->T(4), C(2)<->G(3) i.e. code -> 5 - code.
        # Compare at valid positions only (targets are zeroed where masked).
        np.testing.assert_array_equal(
            y_track0_reversed[m].astype(int), (5 - base_from_x)[m],
        )


# ── store-level __getitem__ / DataLoader ─────────────────────────────────

class TestGetItem:
    def test_val_mode_deterministic(self, tmp_dir):
        sp = os.path.join(tmp_dir, "s.zarr")
        _build_synthetic_store(sp)
        ds = _make_ds(sp, train_mode=False)
        x1, y1, m1 = ds[0]
        x2, y2, m2 = ds[0]
        np.testing.assert_array_equal(x1.numpy(), x2.numpy())  # j=0, no RC
        assert x1.dtype == torch.float32
        assert y1.dtype == torch.float32
        assert m1.dtype == torch.bool
        assert tuple(x1.shape) == (4, _MODEL_IN)
        assert tuple(y1.shape) == (C, _TILE)
        assert tuple(m1.shape) == (_TILE,)

    def test_dataloader_collates(self, tmp_dir):
        sp = os.path.join(tmp_dir, "s.zarr")
        _build_synthetic_store(sp)
        ds = _make_ds(sp, train_mode=True, seed=7)
        loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=0)
        x, y, m = next(iter(loader))
        assert tuple(x.shape) == (1, 4, _MODEL_IN)
        assert tuple(y.shape) == (1, C, _TILE)
        assert tuple(m.shape) == (1, _TILE)
        assert not x.is_cuda and not y.is_cuda and not m.is_cuda


class TestModelE2E:
    """Dataset -> BackgroundModel._step forward runs finite on each loss (CPU)."""

    @pytest.mark.parametrize(
        "loss", ["multinomial", "dirichlet_multinomial", "nb_offset"]
    )
    def test_forward_finite(self, tmp_dir, loss):
        from background_model_core import BackgroundModel, _prepare_mask

        sp = os.path.join(tmp_dir, "s.zarr")
        # all-valid mask: _prepare_mask requires targets zero at masked positions
        cfg, *_ = _build_synthetic_store(sp, invalid_prefix=0)

        # small model so calc_input_region_size(256) fits the store's L_SEQ=304
        model = BackgroundModel(
            n_kernels=8, kernel_size=4, num_residual_layers=1, loss=loss,
            dispersion_window_size=_TILE,
        )
        mis = model.calc_input_region_size(cfg.tile_size)  # 268, even
        ds = _make_ds(sp, model_input_size=mis, train_mode=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=0)
        x, y, mask = next(iter(loader))

        # replicate _step WITHOUT self.log (no Trainer attached)
        mask3 = _prepare_mask(mask, y)
        shape_logits, dispersion_bp = model(x)
        if loss == "multinomial":
            l = model.loss_fn(shape_logits, y, mask3)
        else:
            log_disp = model._pooled_log_dispersion(dispersion_bp, mask3)
            l = model.loss_fn(shape_logits, log_disp, y, mask3)
        assert torch.isfinite(l), f"{loss} loss not finite"


def test_complement_lut():
    assert _COMPLEMENT_LUT[ord("A")] == ord("T")
    assert _COMPLEMENT_LUT[ord("T")] == ord("A")
    assert _COMPLEMENT_LUT[ord("C")] == ord("G")
    assert _COMPLEMENT_LUT[ord("G")] == ord("C")
    assert _COMPLEMENT_LUT[ord("N")] == ord("N")  # self
    assert _COMPLEMENT_LUT[ord("a")] == ord("t")
