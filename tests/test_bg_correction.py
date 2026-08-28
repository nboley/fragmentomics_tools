"""Correction-output tests (design §6): the two locked interfaces
``apply_fragment_weights`` (a) and ``expected_profile`` (b).

Numbering follows the design doc §6 (T4/T6/T7/T9/T11 + failure modes §7).
T1/T2/T3 and the ``locate`` round-trip half of T10 live in
``test_bg_inference.py``; the minus-strand REFUSAL half of T6/T10(v) lives here
because it exercises ``apply_fragment_weights`` directly.

Everything runs on an UNTRAINED or FORCED model — Phase 2 machinery is
statistical-quality-agnostic (design §6 preamble).
"""

import os
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("background_model_core")

import pysam

from background_model.correction import (
    ExpectedProfile,
    WeightClampConfig,
    apply_fragment_weights,
    expected_profile,
)
from background_model.inference import predict_region_profiles
from background_model.preprocess import TRACK_INDEX
from background_model_core import BackgroundModel
from fragmentomics_tools.fragment_array.fragment_array import RegionFragmentArray
from fragmentomics_tools.region import Region

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")
_FASTA = os.path.join(_DATA, "GRCh38.p12.genome.chr6_99110000_99130000.fa.gz")
_GOLDEN_H5 = os.path.join(_DATA, "golden.small.chr6.frag.h5")

pytestmark = pytest.mark.skipif(
    not os.path.exists(_FASTA), reason=f"golden fasta missing ({_FASTA})"
)

_CONTIG = "chr6"
_CONTIG_LEN = 170_805_979
_TILE = 256
_START = 99_114_000
_STOP = _START + 2 * _TILE          # exactly 2 full windows


def _fasta():
    return pysam.FastaFile(_FASTA)


def _random_model(loss="multinomial", seed=0):
    torch.manual_seed(seed)
    return BackgroundModel(
        n_kernels=8, kernel_size=4, num_residual_layers=1, loss=loss,
    )


def _uniform_model(seed=0):
    """Model with the shape head zeroed → constant logits → masked softmax is
    uniform ``1/L_valid`` over valid positions (design T4)."""
    m = _random_model(seed=seed)
    with torch.no_grad():
        m.shape_head.weight.zero_()
        m.shape_head.bias.zero_()
    return m


def _region(strand=".", start=_START, stop=_STOP):
    return Region(chrom=_CONTIG, start=start, stop=stop, strand=strand)


def _rfa(starts_0, stops_0, strands, *, region=None, is_flipped=False):
    if region is None:
        region = _region()
    return RegionFragmentArray(
        starts_0=np.asarray(starts_0, dtype=np.int64),
        stops_0=np.asarray(stops_0, dtype=np.int64),
        region=region,
        max_frag_len=511,
        fragment_strands=np.asarray(strands),
        is_flipped=is_flipped,
    )


def _bl_rdf(intervals):
    """Build a RegionDataFrame blacklist from genomic (start, stop) intervals."""
    from fragmentomics_tools.dataframe import RegionDataFrame

    fd, path = tempfile.mkstemp(suffix=".bed")
    with os.fdopen(fd, "w") as f:
        for s, e in intervals:
            f.write(f"{_CONTIG}\t{int(s)}\t{int(e)}\n")
    try:
        return RegionDataFrame.from_bed(path, ref="hg38")
    finally:
        os.unlink(path)


# ── emitted-slice-N lock (design §4 / §5.1 reconciliation) ────────────────

class TestEmittedSliceN:
    def test_subgrid_final_window_N_over_emitted_slice_only(self):
        # A sub-grid stop trims the final window to 100 emitted columns.  The
        # reconciled semantics (revision-log r1): N for that window sums ONLY the
        # emitted columns the caller supplied — full-window N is UNREALIZABLE
        # (observed_counts spans only [start, stop)).  Locking this makes a future
        # "sum a full TILE window" fix fail loudly.  contig_len=stop puts the
        # contig edge at `stop`, so the emitted slice IS the window's valid
        # support and the flatness identity sum(expected)==N holds over it.
        m = _uniform_model()
        stop = _START + _TILE + 100          # final window emits only 100 cols
        L = _TILE + 100
        rng = np.random.default_rng(3)
        obs = rng.integers(1, 6, size=(12, L)).astype(np.float64)
        ep = expected_profile(
            m, _fasta(), _CONTIG, _START, stop, obs,
            contig_len=stop, tile_size=_TILE,
        )
        assert ep.N.shape == (2, 12)
        # N[1] sums ONLY the emitted 100 columns [_TILE, _TILE+100) — a full
        # window (256 cols) is unrealizable: obs has no data past `stop`.
        emitted = obs[:, _TILE:_TILE + 100]
        np.testing.assert_allclose(ep.N[1], emitted.sum(axis=1))
        assert emitted.shape[1] == 100 and emitted.shape[1] < _TILE
        # flatness over the window's valid emitted columns == N[1].
        win1_cols = np.arange(_TILE, _TILE + 100)
        valid1 = win1_cols[ep.mask[win1_cols]]
        assert valid1.size == 100          # contig edge at stop ⇒ all emitted valid
        np.testing.assert_allclose(
            ep.expected[:, valid1].sum(axis=1), ep.N[1], atol=1e-6
        )


# ── T4: uniform ⇒ weights 1, expected N/L_valid, NaN at masked ────────────

class TestUniformAnalytic:
    def test_weights_all_one_at_valid_in_band(self):
        m = _uniform_model()
        # +len50 (band0) @100, -len130 (band1) @300 — all endpoints on-grid,
        # unmasked, in-band.
        rfa = _rfa([100, 300], [150, 430], ["+", "-"])
        new = apply_fragment_weights(
            rfa, m, _fasta(), clamp=WeightClampConfig.identity(),
            drop_uncorrectable=False, contig_len=_CONTIG_LEN, tile_size=_TILE,
        )
        for attr in ("first_covered_base_weights",
                     "last_covered_base_weights", "weights"):
            w = getattr(new, attr)
            np.testing.assert_allclose(w, np.ones_like(w), atol=1e-6)

    def test_expected_is_N_over_L_and_nan_at_masked(self):
        m = _uniform_model()
        rng = np.random.default_rng(1)
        obs = rng.integers(0, 5, size=(12, 2 * _TILE)).astype(np.float64)
        # mask a single position (col 100) via expansion-0 blacklist.
        bl = _bl_rdf([(_START + 100, _START + 101)])
        ep = expected_profile(
            m, _fasta(), _CONTIG, _START, _STOP, obs,
            blacklist_rdf=bl, blacklist_expansion=0,
            contig_len=_CONTIG_LEN, tile_size=_TILE,
        )
        assert isinstance(ep, ExpectedProfile)
        assert ep.N.shape == (2, 12)
        assert not ep.mask[100]
        # masked column → NaN everywhere.
        assert np.all(np.isnan(ep.expected[:, 100]))
        # N excludes the masked column; L_valid = TILE-1 in window 0.
        L0 = int(ep.mask[:_TILE].sum())
        assert L0 == _TILE - 1
        exp_obs_N0 = (obs[:, :_TILE] * ep.mask[None, :_TILE]).sum(axis=1)
        np.testing.assert_allclose(ep.N[0], exp_obs_N0)
        # expected == N_w/L_valid at valid positions (uniform shape).
        valid0 = np.nonzero(ep.mask[:_TILE])[0]
        for c in range(12):
            np.testing.assert_allclose(
                ep.expected[c, valid0], ep.N[0, c] / L0, atol=1e-6
            )
        # flatness: sum over each window's valid positions == N_w.
        for w in range(2):
            cols = np.nonzero(ep.mask[w * _TILE:(w + 1) * _TILE])[0] + w * _TILE
            np.testing.assert_allclose(
                ep.expected[:, cols].sum(axis=1), ep.N[w], atol=1e-6
            )


# ── T6: S1-lock (strand×band = AND), band membership, minus-strand refusal ─

def _reference_weight(rp, strand, band, cov, off):
    trk = TRACK_INDEX[(strand, band, cov)]
    win = off // _TILE
    l_valid = int(rp.mask[win * _TILE:(win + 1) * _TILE].sum())
    return 1.0 / (rp.probs[trk, off] * l_valid)


class TestS1Lock:
    def test_strand_and_band_are_AND_not_OR(self):
        m = _random_model(seed=7)
        fasta = _fasta()
        # (a) +len50 band0 @100; (b) +len200 out-of-band @50; (c) -len130 band1 @300
        rfa = _rfa([100, 50, 300], [150, 250, 430], ["+", "+", "-"])
        new = apply_fragment_weights(
            rfa, m, fasta, clamp=WeightClampConfig.identity(),
            drop_uncorrectable=False, contig_len=_CONTIG_LEN, tile_size=_TILE,
        )
        rp = predict_region_profiles(
            m, fasta, _CONTIG, _START, _STOP,
            contig_len=_CONTIG_LEN, tile_size=_TILE,
        )
        b0, b1 = (40, 65), (120, 175)
        # (a) +band0: 'first' weight from the (+, band0, first) track ONLY.
        ref_a = _reference_weight(rp, "+", b0, "first", 100)
        np.testing.assert_allclose(new.first_covered_base_weights[0], ref_a, rtol=1e-6)
        # NOT the wrong-strand track (v1's OR bug would let (-,band0) win).
        wrong = _reference_weight(rp, "-", b0, "first", 100)
        assert not np.isclose(new.first_covered_base_weights[0], wrong, rtol=1e-6)
        # (b) out-of-band → all three coverage weights 0.
        assert new.first_covered_base_weights[1] == 0.0
        assert new.last_covered_base_weights[1] == 0.0
        assert new.weights[1] == 0.0
        # (c) -band1 @300: 'first' from (-, band1, first).
        ref_c = _reference_weight(rp, "-", b1, "first", 300)
        np.testing.assert_allclose(new.first_covered_base_weights[2], ref_c, rtol=1e-6)

    def test_out_of_band_dropped(self):
        m = _uniform_model()
        rfa = _rfa([100, 50], [150, 250], ["+", "+"])   # in-band, out-of-band
        new = apply_fragment_weights(
            rfa, m, _fasta(), clamp=WeightClampConfig.identity(),
            drop_uncorrectable=True, contig_len=_CONTIG_LEN, tile_size=_TILE,
        )
        # only the in-band fragment survives.
        assert new.n_fragments == 1
        np.testing.assert_allclose(new.first_covered_base_weights, [1.0], atol=1e-6)

    @pytest.mark.parametrize("length", [39, 65, 100, 119, 175])
    def test_band_membership_no_track_lengths(self, length):
        # below band0 / band0 open-upper / inter-band gap / gap / band1 open-upper.
        m = _uniform_model()
        rfa = _rfa([100], [100 + length], ["+"])
        new = apply_fragment_weights(
            rfa, m, _fasta(), clamp=WeightClampConfig.identity(),
            drop_uncorrectable=False, contig_len=_CONTIG_LEN, tile_size=_TILE,
        )
        assert new.first_covered_base_weights[0] == 0.0
        assert new.last_covered_base_weights[0] == 0.0
        assert new.weights[0] == 0.0

    @pytest.mark.parametrize("length", [40, 64, 120, 174])
    def test_band_membership_in_band_lengths(self, length):
        # band0 lo, band0 hi-1, band1 lo, band1 hi-1 → nonzero (uniform ⇒ 1).
        m = _uniform_model()
        rfa = _rfa([100], [100 + length], ["+"])
        new = apply_fragment_weights(
            rfa, m, _fasta(), clamp=WeightClampConfig.identity(),
            drop_uncorrectable=False, contig_len=_CONTIG_LEN, tile_size=_TILE,
        )
        np.testing.assert_allclose(new.first_covered_base_weights, [1.0], atol=1e-6)

    def test_minus_strand_region_refused(self):
        m = _uniform_model()
        rfa = _rfa([100], [150], ["+"],
                   region=_region(strand="-"), is_flipped=True)
        with pytest.raises(AssertionError, match="strandless"):
            apply_fragment_weights(
                rfa, m, _fasta(), clamp=WeightClampConfig.identity(),
                contig_len=_CONTIG_LEN, tile_size=_TILE,
            )

    def test_minus_strand_unflipped_refused_strand_half(self):
        # Isolated STRAND half: a '-' region with is_flipped=False must trip the
        # strand assert (message names the strand requirement, NOT is_flipped),
        # locking the split from the is_flipped half below.
        m = _uniform_model()
        rfa = _rfa([100], [150], ["+"],
                   region=_region(strand="-"), is_flipped=False)
        with pytest.raises(AssertionError, match="strandless"):
            apply_fragment_weights(
                rfa, m, _fasta(), clamp=WeightClampConfig.identity(),
                contig_len=_CONTIG_LEN, tile_size=_TILE,
            )

    def test_plus_region_but_is_flipped_refused(self):
        # dual assert: a '+' region can still be is_flipped via the reverse op.
        # Isolated IS_FLIPPED half: strand assert passes ('+'), so the SECOND
        # (is_flipped) assert must fire — its message names is_flipped, NOT
        # strandless.
        m = _uniform_model()
        rfa = _rfa([100], [150], ["+"],
                   region=_region(strand="+"), is_flipped=True)
        with pytest.raises(AssertionError, match="is_flipped"):
            apply_fragment_weights(
                rfa, m, _fasta(), clamp=WeightClampConfig.identity(),
                contig_len=_CONTIG_LEN, tile_size=_TILE,
            )

    def test_strandless_region_accepted(self):
        # Region normalizes '.' → None; the applier must accept the happy path.
        m = _uniform_model()
        rfa = _rfa([100], [150], ["+"], region=_region(strand="."))
        assert rfa.region.strand is None
        new = apply_fragment_weights(
            rfa, m, _fasta(), clamp=WeightClampConfig.identity(),
            drop_uncorrectable=False, contig_len=_CONTIG_LEN, tile_size=_TILE,
        )
        np.testing.assert_allclose(new.first_covered_base_weights, [1.0], atol=1e-6)

    def test_non_plus_minus_strand_fragment_zeroed(self):
        # Strand gate (correction.py `elig`): a fragment whose strand is neither
        # '+' nor '-' maps to no track (trk stays -1).  Without the gate it is
        # still in-band/in-grid/unmasked, so it would silently gather
        # probs_w[-1] (the LAST track).  It must instead get weight 0 on every
        # coverage type, and a neighbouring valid fragment must be untouched.
        m = _uniform_model()
        # frag 0: valid +len50 band0 @100 (weight 1 under uniform).
        # frag 1: bad strand '.', but otherwise in-band/on-grid/unmasked @300.
        rfa = RegionFragmentArray(
            starts_0=np.array([100, 300], dtype=np.int64),
            stops_0=np.array([150, 350], dtype=np.int64),
            region=_region(),
            max_frag_len=511,
            fragment_strands=np.array(["+", "."]),
            validate_data=False,
        )
        new = apply_fragment_weights(
            rfa, m, _fasta(), clamp=WeightClampConfig.identity(),
            drop_uncorrectable=False, contig_len=_CONTIG_LEN, tile_size=_TILE,
        )
        # bad-strand fragment: zero on all three coverage types.
        assert new.first_covered_base_weights[1] == 0.0
        assert new.last_covered_base_weights[1] == 0.0
        assert new.weights[1] == 0.0
        # valid neighbour untouched.
        np.testing.assert_allclose(new.first_covered_base_weights[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(new.last_covered_base_weights[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(new.weights[0], 1.0, atol=1e-6)

    @pytest.mark.skipif(
        not os.path.exists(_GOLDEN_H5), reason=f"golden h5 missing ({_GOLDEN_H5})"
    )
    def test_minus_strand_region_via_from_fragments_h5_refused(self):
        # Exercise is_flipped via the REAL path (not a constructor kwarg): a
        # '-'-strand region drives from_fragments_h5 to set is_flipped =
        # region.is_minus_strand() (fragment_array.py:1843).  The applier must
        # refuse it up front.
        from fragments_h5 import FragmentsH5

        h5 = FragmentsH5(_GOLDEN_H5, cache_pointers=False)
        region = Region(chrom="chr6", start=84_150_000, stop=84_152_000, strand="-")
        rfa = RegionFragmentArray.from_fragments_h5(
            h5, region, min_mapq=10, max_frag_len=175
        )
        h5.close()
        assert rfa.is_flipped  # set by the real path, not a constructor kwarg
        m = _uniform_model()
        with pytest.raises(AssertionError):
            apply_fragment_weights(
                rfa, m, _fasta(), clamp=WeightClampConfig.identity(),
                contig_len=_CONTIG_LEN, tile_size=_TILE,
            )


# ── T7: single-track reciprocal parity (numeric core) ─────────────────────

class TestReciprocalParity:
    def test_first_weight_equals_inline_reciprocal(self):
        m = _random_model(seed=11)
        fasta = _fasta()
        # all +band0 so strand/band selection is unambiguous (v1's OR collapses).
        rfa = _rfa([100, 130, 300], [150, 194, 350], ["+", "+", "+"])
        new = apply_fragment_weights(
            rfa, m, fasta, clamp=WeightClampConfig.identity(),
            drop_uncorrectable=False, contig_len=_CONTIG_LEN, tile_size=_TILE,
        )
        rp = predict_region_profiles(
            m, fasta, _CONTIG, _START, _STOP,
            contig_len=_CONTIG_LEN, tile_size=_TILE,
        )
        b0 = (40, 65)
        for f, first_off in enumerate([100, 130, 300]):
            ref = _reference_weight(rp, "+", b0, "first", first_off)
            np.testing.assert_allclose(
                new.first_covered_base_weights[f], ref, rtol=1e-6
            )


# ── T8 / masked-endpoint handling & expected interface ────────────────────

class TestMaskedEndpoint:
    def test_masked_endpoint_zeros_only_that_coverage(self):
        m = _uniform_model()
        # +len50 @100: first=100, last=149, mid=125. Mask ONLY position 100
        # (expansion 0) so 'first' → weight 0 but last/mid survive.
        bl = _bl_rdf([(_START + 100, _START + 101)])
        rfa = _rfa([100], [150], ["+"])
        new = apply_fragment_weights(
            rfa, m, _fasta(), clamp=WeightClampConfig.identity(),
            drop_uncorrectable=False, blacklist_rdf=bl, blacklist_expansion=0,
            contig_len=_CONTIG_LEN, tile_size=_TILE,
        )
        assert new.first_covered_base_weights[0] == 0.0      # endpoint masked
        assert new.last_covered_base_weights[0] > 0.0        # 149 still valid
        assert new.weights[0] > 0.0                          # 125 still valid

    def test_expected_wrong_shape_raises(self):
        m = _uniform_model()
        bad = np.zeros((12, 10), dtype=np.float64)
        with pytest.raises(AssertionError, match="observed_counts"):
            expected_profile(
                m, _fasta(), _CONTIG, _START, _STOP, bad,
                contig_len=_CONTIG_LEN, tile_size=_TILE,
            )


# ── T9: clamp hook ────────────────────────────────────────────────────────

class TestClamp:
    def test_clamp_is_required(self):
        m = _uniform_model()
        rfa = _rfa([100], [150], ["+"])
        with pytest.raises(TypeError):
            apply_fragment_weights(
                rfa, m, _fasta(), contig_len=_CONTIG_LEN, tile_size=_TILE,
            )

    def test_identity_passthrough_vs_bounds(self):
        m = _random_model(seed=5)
        fasta = _fasta()
        rfa = _rfa([100, 300], [150, 430], ["+", "-"])
        w_id = apply_fragment_weights(
            rfa, m, fasta, clamp=WeightClampConfig.identity(),
            drop_uncorrectable=False, contig_len=_CONTIG_LEN, tile_size=_TILE,
        ).first_covered_base_weights
        w_cl = apply_fragment_weights(
            rfa, m, fasta, clamp=WeightClampConfig(0.5, 2.0),
            drop_uncorrectable=False, contig_len=_CONTIG_LEN, tile_size=_TILE,
        ).first_covered_base_weights
        # valid positions (w_id > 0) are clipped; uncorrectable (0) stay 0.
        expected = np.where(w_id > 0, np.clip(w_id, 0.5, 2.0), 0.0)
        np.testing.assert_allclose(w_cl, expected)
        assert w_cl[w_cl > 0].max() <= 2.0 + 1e-12
        assert w_cl[w_cl > 0].min() >= 0.5 - 1e-12

    def test_fragment_strands_none_raises(self):
        m = _uniform_model()
        rfa = RegionFragmentArray(
            starts_0=np.array([100], dtype=np.int64),
            stops_0=np.array([150], dtype=np.int64),
            region=_region(), max_frag_len=511, fragment_strands=None,
        )
        with pytest.raises(ValueError, match="stranded"):
            apply_fragment_weights(
                rfa, m, _fasta(), clamp=WeightClampConfig.identity(),
                contig_len=_CONTIG_LEN, tile_size=_TILE,
            )


# ── T11: tiling contract — determinism + window-relative Limitation ───────

class TestTilingContract:
    def test_repeated_calls_are_bit_identical_determinism(self):
        # Determinism half of the tiling contract (design §5.2): the SAME
        # (region, fragments, model) at the same anchor/tile_size ⇒ bit-identical
        # weights across independent calls.
        m = _random_model(seed=13)
        fasta = _fasta()
        rfa = _rfa([100, 300], [150, 430], ["+", "-"])
        kw = dict(clamp=WeightClampConfig.identity(), drop_uncorrectable=False,
                  contig_len=_CONTIG_LEN, tile_size=_TILE)
        a = apply_fragment_weights(rfa, m, fasta, **kw)
        b = apply_fragment_weights(rfa, m, fasta, **kw)
        np.testing.assert_array_equal(
            a.first_covered_base_weights, b.first_covered_base_weights
        )
        np.testing.assert_array_equal(
            a.last_covered_base_weights, b.last_covered_base_weights
        )
        np.testing.assert_array_equal(a.weights, b.weights)

    def test_shifted_anchor_changes_weight_window_relative_limitation(self):
        # Negative Limitation lock (design §5.2): weights are window-relative,
        # NOT position-intrinsic.  The SAME GENOMIC fragment
        # [_START+100, _START+150) under a region anchored half a tile
        # earlier (local coords shifted to compensate) lands on a different
        # window grid ⇒ a DIFFERENT weight.  A position-intrinsic
        # implementation would give the identical genomic fragment the
        # identical weight — this test must fail such a refactor.  A
        # non-uniform (seeded random) model makes the difference real.
        m = _random_model(seed=17)
        fasta = _fasta()
        kw = dict(clamp=WeightClampConfig.identity(), drop_uncorrectable=False,
                  contig_len=_CONTIG_LEN, tile_size=_TILE)
        rfa_a = _rfa([100], [150], ["+"],
                     region=_region(start=_START, stop=_STOP))
        shift = _TILE // 2
        rfa_b = _rfa([100 + shift], [150 + shift], ["+"],
                     region=_region(start=_START - shift,
                                    stop=_START - shift + 2 * _TILE))
        wa = apply_fragment_weights(rfa_a, m, fasta, **kw).first_covered_base_weights
        wb = apply_fragment_weights(rfa_b, m, fasta, **kw).first_covered_base_weights
        assert wa[0] > 0 and wb[0] > 0            # both corrected (nonzero)
        assert not np.allclose(wa, wb)            # window-relative ⇒ differ


# ── failure modes (design §7) ─────────────────────────────────────────────

class TestFailureModes:
    def test_all_masked_window_weights_zero_expected_nan(self):
        m = _uniform_model()
        # blacklist spanning the whole region → every position masked.
        bl = _bl_rdf([(_START - 200, _STOP + 200)])
        rfa = _rfa([100, 300], [150, 430], ["+", "-"])
        new = apply_fragment_weights(
            rfa, m, _fasta(), clamp=WeightClampConfig.identity(),
            drop_uncorrectable=False, blacklist_rdf=bl,
            contig_len=_CONTIG_LEN, tile_size=_TILE,
        )
        assert np.all(new.first_covered_base_weights == 0.0)
        assert np.all(new.last_covered_base_weights == 0.0)
        assert np.all(new.weights == 0.0)

        obs = np.ones((12, 2 * _TILE), dtype=np.float64)
        ep = expected_profile(
            m, _fasta(), _CONTIG, _START, _STOP, obs,
            blacklist_rdf=bl, contig_len=_CONTIG_LEN, tile_size=_TILE,
        )
        assert np.all(np.isnan(ep.expected))
        assert not ep.mask.any()
        assert np.all(ep.N == 0.0)

    def test_N_zero_window_expected_zero_at_valid(self):
        m = _uniform_model()
        obs = np.zeros((12, 2 * _TILE), dtype=np.float64)   # no observed counts
        ep = expected_profile(
            m, _fasta(), _CONTIG, _START, _STOP, obs,
            contig_len=_CONTIG_LEN, tile_size=_TILE,
        )
        assert np.all(ep.N == 0.0)
        # no blacklist → all positions valid → expected == 0 (finite), no NaN.
        assert np.all(ep.expected == 0.0)
        assert not np.any(np.isnan(ep.expected))

    def test_six_track_model_trips_12_track_precondition(self):
        # A valid 6-track subset of DEFAULT_OUTPUT_TRACKS builds a legal model,
        # but apply_fragment_weights gathers probs by the 12-entry TRACK_INDEX;
        # the up-front precondition (correction.py:200) must refuse it before any
        # (wrong-track) gather.
        from background_model_core import DEFAULT_OUTPUT_TRACKS

        torch.manual_seed(0)
        m = BackgroundModel(
            output_tracks=list(DEFAULT_OUTPUT_TRACKS[:6]),
            n_kernels=8, kernel_size=4, num_residual_layers=1, loss="multinomial",
        )
        assert len(m.output_tracks) == 6
        rfa = _rfa([100], [150], ["+"])
        with pytest.raises(AssertionError, match="12-track"):
            apply_fragment_weights(
                rfa, m, _fasta(), clamp=WeightClampConfig.identity(),
                contig_len=_CONTIG_LEN, tile_size=_TILE,
            )

    def test_contig_edge_region_runs_and_masks_right(self):
        # Force a right edge inside the region via a small contig_len; positions
        # past it are masked, endpoints there → weight 0, but the call succeeds.
        m = _uniform_model()
        # window 0 fully in-contig; window 1 keeps only its first 40 positions,
        # so fragment 1's first endpoint (local j=44) is past the contig end.
        clen = _START + _TILE + 40
        rfa = _rfa([100, 300], [150, 430], ["+", "-"])
        new = apply_fragment_weights(
            rfa, m, _fasta(), clamp=WeightClampConfig.identity(),
            drop_uncorrectable=False, contig_len=clen, tile_size=_TILE,
        )
        w = np.concatenate([
            new.first_covered_base_weights,
            new.last_covered_base_weights,
            new.weights,
        ])
        assert np.all(np.isfinite(w))
        # fragment 0 (window 0, in-contig) still corrected; fragment 1's first
        # endpoint (local j=44 ≥ 40 valid) is past clen → masked → weight 0.
        assert new.first_covered_base_weights[0] == 1.0
        assert new.first_covered_base_weights[1] == 0.0


# ── loss-type independence (design §5.2 — dispersion head unused) ─────────

class TestLossIndependence:
    @pytest.mark.parametrize("loss", ["multinomial", "dirichlet_multinomial",
                                      "nb_offset"])
    def test_runs_for_every_loss(self, loss):
        m = _random_model(loss=loss, seed=2)
        fasta = _fasta()
        rfa = _rfa([100, 300], [150, 430], ["+", "-"])
        new = apply_fragment_weights(
            rfa, m, fasta, clamp=WeightClampConfig(1e-6, 10.0),
            contig_len=_CONTIG_LEN, tile_size=_TILE,
        )
        cov = new.build_coverage_counts(
            fl_bands=[(40, 65), (120, 175)], split_strand=True,
            return_sparse=False,
        )
        for vec in cov:
            arr = np.asarray(vec)
            assert np.all(np.isfinite(arr)) and np.all(arr >= 0)

        obs = np.ones((12, 2 * _TILE), dtype=np.float64)
        ep = expected_profile(
            m, fasta, _CONTIG, _START, _STOP, obs,
            contig_len=_CONTIG_LEN, tile_size=_TILE,
        )
        # finite at valid positions; NaN only at masked ones.
        assert np.all(np.isfinite(ep.expected[:, ep.mask]))
        assert np.all(np.isnan(ep.expected[:, ~ep.mask]))
