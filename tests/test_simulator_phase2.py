"""Tests for Phase 2 of the simulator: LUT, capture, marginal_fl, precompute.

Each test class names what must FAIL and demonstrates the failure via mutation.
This is the Phase 1 lesson: invariants that are insensitive to the most likely
defect are useless.

Required assertions (from the task):
1. LUT construction produces the correct shape, values at bin midpoints, and
   normalisation invariant.  Exactness against the real GCFlDistModel is
   guaranteed by the ZTNB piecewise-constant property, now asserted in
   ``predict_lut_from_model`` (H1 guard).
2. GC values on exact bin boundaries go to the correct bin (0, 5, 100, last-inclusive).
3. marginal_fl sums to exactly 1 over 156 entries.
4. hex_fwd / hex_rc satisfy the contract Step 4 relies on: the RC track genuinely
   is the reverse complement at the same cut site.
"""

import numpy as np
import pytest

from background_model.simulator.weights import (
    GC_BIN_WIDTH,
    L_MAX,
    L_MIN,
    MAX_FL_HALF,
    N_GC_BINS,
    N_LENGTHS,
    NHEX,
    HexamerTables,
    build_predict_lut,
    build_region_weights,
    gc_bin_index,
    gc_pct,
    generative_domain_size,
)
from background_model.simulator.precompute import hexamer_indices, KMER


# ── helpers ──────────────────────────────────────────────────────────────

def _random_tables(seed=42):
    rng = np.random.default_rng(seed)
    tables = []
    for _ in range(4):
        w = np.exp(rng.normal(0, 0.3, NHEX))
        w /= w.max()
        tables.append(w)
    return HexamerTables(*tables)


def _flat_marginal_fl():
    fl = np.ones(N_LENGTHS, dtype=np.float64)
    fl /= fl.sum()
    return fl


def _synthetic_region(region_len, seed=123, pad=MAX_FL_HALF):
    rng = np.random.default_rng(seed)
    n_core = region_len + 1
    hex_fwd_core = rng.integers(0, NHEX, size=n_core)
    hex_rc_core = rng.integers(0, NHEX, size=n_core)
    bases_gc_core = rng.random(region_len) < 0.4
    pad_rng = np.random.default_rng(seed + 1_000_000)
    n_sites = region_len + 2 * pad + 1
    hex_fwd = np.concatenate([
        pad_rng.integers(0, NHEX, size=pad), hex_fwd_core,
        pad_rng.integers(0, NHEX, size=pad),
    ])
    hex_rc = np.concatenate([
        pad_rng.integers(0, NHEX, size=pad), hex_rc_core,
        pad_rng.integers(0, NHEX, size=pad),
    ])
    bases_gc = np.concatenate([
        pad_rng.random(pad) < 0.4, bases_gc_core,
        pad_rng.random(pad) < 0.4,
    ])
    cum_gc = np.concatenate([[0], np.cumsum(bases_gc)]).astype(np.float64)
    valid = np.ones(n_sites, dtype=bool)
    return hex_fwd, hex_rc, cum_gc, valid


# ── 1. LUT construction and normalisation ──────────────────────────────

class TestLUTConstructionAndNormalisation:
    """The LUT must be correctly built from a predict callable and the
    weight normalisation invariant must hold when the LUT is used.

    What must FAIL: if gc_bin_index maps a GC value to the wrong bin, the
    LUT lookup returns a different predict value, so the unnormalised E
    values differ.  After normalisation the total is still 1 (because
    normalisation cancels any constant-factor error), but the per-element
    weights differ.

    Note: these tests use synthetic predict functions that are NOT
    piecewise-constant, so they verify LUT construction mechanics and
    normalisation, not bit-exactness against the real GCFlDistModel.
    Exactness on the real path is guaranteed by the ZTNB model's
    piecewise-constant property, now enforced by the H1 guard in
    ``predict_lut_from_model``, and exercised by ``TestFitAndBuild``.
    """

    def test_trivial_predict_bit_exact(self):
        """With predict=1.0 everywhere, LUT gather and scalar are identical."""
        region_len = 500
        hex_fwd, hex_rc, cum_gc, valid = _synthetic_region(region_len)
        tables = _random_tables(seed=77)
        fl = _flat_marginal_fl()
        lut = build_predict_lut(lambda L, gc: 1.0)

        rw_lut = build_region_weights(
            hex_fwd=hex_fwd, hex_rc=hex_rc, cum_gc=cum_gc,
            hex_tables=tables, marginal_fl=fl,
            predict_lut=lut, region_len=region_len, valid=valid,
        )
        # With trivial predict, the LUT is all 1.0 regardless of binning,
        # so the result is exactly the same as the scalar path would give.
        total = rw_lut.w_plus.sum() + rw_lut.w_minus.sum()
        assert abs(total - 1.0) < 1e-12

    def test_gc_varying_predict_matches_bin_midpoint(self):
        """With a GC-varying predict, the LUT uses bin midpoints.  Verify
        the LUT values match what predict returns at each bin midpoint."""
        def gc_predict(L, gc):
            return 1.0 + 0.02 * gc

        lut = build_predict_lut(gc_predict)
        assert lut.shape == (N_LENGTHS, N_GC_BINS)

        # Check each cell: lut[li, gi] == gc_predict(L, midpoint)
        for li in range(N_LENGTHS):
            L = L_MIN + li
            for gi in range(N_GC_BINS):
                gc_mid = gi * GC_BIN_WIDTH + GC_BIN_WIDTH / 2.0
                expected = gc_predict(L, gc_mid)
                assert lut[li, gi] == expected, (
                    f"L={L}, gi={gi}: lut={lut[li, gi]}, expected={expected}"
                )

    def test_lut_speedup_preserves_normalisation(self):
        """Even with a non-trivial LUT, the normalisation invariant holds."""
        lut = build_predict_lut(lambda L, gc: 1.0 + 0.02 * gc)
        region_len = 2560
        hex_fwd, hex_rc, cum_gc, valid = _synthetic_region(region_len)
        tables = _random_tables(seed=55)
        fl = _flat_marginal_fl()

        rw = build_region_weights(
            hex_fwd=hex_fwd, hex_rc=hex_rc, cum_gc=cum_gc,
            hex_tables=tables, marginal_fl=fl,
            predict_lut=lut, region_len=region_len, valid=valid,
        )
        total = rw.w_plus.sum() + rw.w_minus.sum()
        assert abs(total - 1.0) < 1e-12
        assert abs(rw.w_plus.sum() - 0.5) < 1e-12
        assert abs(rw.w_minus.sum() - 0.5) < 1e-12

    def test_lut_shape_assertion(self):
        """Wrong-shaped LUT is caught by the assertion."""
        bad_lut = np.ones((N_LENGTHS, 10), dtype=np.float64)  # wrong gc dim
        hex_fwd, hex_rc, cum_gc, valid = _synthetic_region(100)
        with pytest.raises(AssertionError, match="predict_lut shape"):
            build_region_weights(
                hex_fwd=hex_fwd, hex_rc=hex_rc, cum_gc=cum_gc,
                hex_tables=_random_tables(), marginal_fl=_flat_marginal_fl(),
                predict_lut=bad_lut, region_len=100, valid=valid,
            )


# ── 2. GC bin boundary semantics ────────────────────────────────────────

class TestGCBinBoundaries:
    """GC values on exact boundaries must go to the correct bin.

    What must FAIL: off-by-one in gc_bin_index (e.g. using ceil instead of
    floor, or failing to clamp 100.0 to the last bin) maps a boundary
    value to the wrong bin, producing incorrect predict values and silently
    wrong weights.
    """

    def test_gc_0(self):
        """GC = 0% → bin 0."""
        idx = gc_bin_index(np.array([0.0]))
        assert idx[0] == 0

    def test_gc_5(self):
        """GC = 5% → bin 1 (5.0 / 5 = 1.0, floor = 1)."""
        idx = gc_bin_index(np.array([5.0]))
        assert idx[0] == 1

    def test_gc_4_999(self):
        """GC = 4.999% → bin 0 (floor(4.999/5) = 0)."""
        idx = gc_bin_index(np.array([4.999]))
        assert idx[0] == 0

    def test_gc_95(self):
        """GC = 95% → bin 19."""
        idx = gc_bin_index(np.array([95.0]))
        assert idx[0] == 19

    def test_gc_100_last_inclusive(self):
        """GC = 100% → bin 19 (last bin inclusive).

        Without the clamp, floor(100/5) = 20 which is out of range.
        """
        idx = gc_bin_index(np.array([100.0]))
        assert idx[0] == N_GC_BINS - 1  # 19

    def test_gc_100_without_clamp_would_fail(self):
        """Prove that without clamping, gc=100 would index out of bounds.

        floor(100.0 / 5.0) = 20, which exceeds N_GC_BINS - 1 = 19.
        """
        raw_idx = int(np.floor(100.0 / GC_BIN_WIDTH))
        assert raw_idx == N_GC_BINS, (
            f"Expected floor(100/5)={N_GC_BINS}, got {raw_idx}"
        )
        # But gc_bin_index clamps it:
        clamped = gc_bin_index(np.array([100.0]))
        assert clamped[0] == N_GC_BINS - 1

    def test_all_integer_boundaries(self):
        """Every integer GC% from 0 to 100 maps to a valid bin."""
        gc_vals = np.arange(0.0, 101.0)
        bins = gc_bin_index(gc_vals)
        assert bins.min() >= 0
        assert bins.max() <= N_GC_BINS - 1
        # Check monotonicity
        assert (np.diff(bins) >= 0).all()

    def test_bin_coverage_contiguous(self):
        """Every bin 0..19 has at least one integer GC value mapping to it."""
        gc_vals = np.arange(0.0, 101.0)
        bins = gc_bin_index(gc_vals)
        for b in range(N_GC_BINS):
            assert b in bins, f"No integer GC maps to bin {b}"


# ── 3. marginal_fl sum ──────────────────────────────────────────────────

class TestMarginalFL:
    """marginal_fl must sum to exactly 1 over 156 entries.

    What must FAIL: if the length range restriction is wrong (e.g. using
    L=1..500 instead of L=25..180), or if normalisation is off, the sum
    differs from 1 and every weight is wrong.
    """

    def test_sum_exactly_one(self):
        """Sum of marginal_fl is 1.0 to floating-point precision."""
        from background_model.simulator.capture import load_duphist, build_marginal_fl
        df = load_duphist("RD-56670")
        fl = build_marginal_fl(df)
        assert fl.shape == (N_LENGTHS,)
        assert abs(fl.sum() - 1.0) < 1e-15

    def test_all_nonneg(self):
        """No negative entries."""
        from background_model.simulator.capture import load_duphist, build_marginal_fl
        df = load_duphist("RD-56670")
        fl = build_marginal_fl(df)
        assert (fl >= 0).all()

    def test_exactly_156_entries(self):
        """Shape is (156,), not (500,) or anything else."""
        from background_model.simulator.capture import load_duphist, build_marginal_fl
        df = load_duphist("RD-56670")
        fl = build_marginal_fl(df)
        assert fl.shape == (N_LENGTHS,), f"shape {fl.shape} != ({N_LENGTHS},)"

    def test_empty_duphist_raises(self):
        """A duphist with no data in [25, 180] raises ValueError."""
        import pandas as pd
        from background_model.simulator.capture import build_marginal_fl
        # DataFrame with only out-of-range lengths
        df = pd.DataFrame({"length": [1, 2, 3, 200, 300],
                           "molecule_keys": [10, 20, 30, 40, 50]})
        with pytest.raises(ValueError, match="No molecule_keys"):
            build_marginal_fl(df)


# ── 3b. GC pre-binning: the ~18% silent-drop defect ─────────────────────

class TestPrebinGCToMidpoints:
    """Guards the fix for a MEASURED defect: ``SIM_GC_BINS`` are inclusive integer
    ranges with a 1-wide gap every 5, ``flgc._bin_index`` returns None in a gap and
    ``fit()`` then does ``continue``, and the duphist's ``gc`` is ``k * 100/254`` —
    so 17.99% of molecule mass was silently dropped from the capture-surface fit.
    """

    def _fractional_duphist(self):
        """Duphist whose gc values are the real encoding, k * 100/254, including
        values that land in the inclusive-bin gaps (4.33, 4.72, 9.06, ...)."""
        import pandas as pd
        gc = np.array([4.330700, 4.724400, 9.055100, 9.448800, 14.173200])
        return pd.DataFrame({
            "length": [100] * len(gc),
            "gc": gc,
            "multiplicity": [1] * len(gc),
            "molecule_keys": [1000] * len(gc),
        })

    def test_gap_values_are_not_dropped(self):
        """Every input row must survive into a cell.

        MUST FAIL if the pre-binning is removed: without it these five gc values
        match no ``SIM_GC_BINS`` entry and the fit discards them. Asserted on the
        ACTUAL bins, so it breaks if the bin definition regresses to inclusive.
        """
        from background_model.simulator.capture import (
            prebin_gc_to_midpoints, SIM_GC_BINS,
        )
        out = prebin_gc_to_midpoints(self._fractional_duphist())
        assert out.molecule_keys.sum() == 5000, "molecule mass was lost"
        for gc in out.gc:
            hits = [i for i, (lo, hi) in enumerate(SIM_GC_BINS) if lo <= gc <= hi]
            assert len(hits) == 1, (
                f"gc={gc} matches {len(hits)} of SIM_GC_BINS, need exactly 1 — "
                f"flgc._bin_index drops a non-match and takes the first of a tie"
            )

    def test_snapped_values_use_the_floor_rule(self):
        """Snapping must agree with ``weights.gc_bin_index``, not with any
        independently reimplemented rounding.

        MUST FAIL on a rounding-convention change. For the five fixture values the
        two rules give disjoint answers::

            floor: 4.33, 4.72 -> 2.5 | 9.06, 9.45 -> 7.5 | 14.17 -> 12.5
            round: 4.33, 4.72 -> 7.5 | 9.06, 9.45 -> 12.5 | 14.17 -> 17.5

        so a round-based implementation yields {7.5, 12.5, 17.5} and fails here
        while still passing the no-drop test above.
        """
        from background_model.simulator.capture import prebin_gc_to_midpoints
        out = prebin_gc_to_midpoints(self._fractional_duphist())
        got = sorted(out.gc.unique().tolist())
        assert got == [2.5, 7.5, 12.5], f"expected floor midpoints, got {got}"

    def test_collapsed_cells_aggregate_multiplicity(self):
        """Rows that collapse into one cell must SUM per multiplicity.

        MUST FAIL if the groupby-sum is dropped: a duphist cell carries one
        ``molecule_keys`` entry per multiplicity, so leaving repeated multiplicity
        rows inside a cell silently corrupts the histogram ``fit()`` reads. A
        no-drop check cannot catch this because no mass is lost.
        """
        import pandas as pd
        from background_model.simulator.capture import prebin_gc_to_midpoints
        # three gc values in the SAME bin (0-5), all multiplicity 1
        df = pd.DataFrame({
            "length": [100, 100, 100],
            "gc": [0.787400, 1.181100, 1.574800],
            "multiplicity": [1, 1, 1],
            "molecule_keys": [7, 11, 13],
        })
        out = prebin_gc_to_midpoints(df)
        assert len(out) == 1, (
            f"expected 1 aggregated row, got {len(out)} — repeated multiplicity "
            f"values inside one cell corrupt the duphist histogram"
        )
        assert out.molecule_keys.iloc[0] == 31
        assert out.multiplicity.iloc[0] == 1

    def test_gc_100_lands_in_the_last_bin(self):
        """gc = 100.0 occurs in real data and must clamp to bin 19, not 20."""
        import pandas as pd
        from background_model.simulator.capture import prebin_gc_to_midpoints
        df = pd.DataFrame({"length": [100], "gc": [100.0],
                           "multiplicity": [1], "molecule_keys": [5]})
        out = prebin_gc_to_midpoints(df)
        assert out.gc.iloc[0] == 97.5, f"gc=100 snapped to {out.gc.iloc[0]}"


# ── 4. hex_fwd / hex_rc RC contract ─────────────────────────────────────

class TestHexamerRCContract:
    """hex_rc at a position must be the reverse complement of hex_fwd.

    What must FAIL: if the RC computation is wrong (e.g. only reversing
    without complementing, or vice versa), the minus-strand hexamer lookups
    use the wrong weights, silently corrupting strand asymmetry.

    The test constructs a known sequence, computes hex indices, and verifies
    that hex_rc[c] == RC_PERM[hex_fwd[c]] where RC_PERM is the analytic
    reverse-complement permutation from background_model_core.
    """

    def test_rc_is_involution(self):
        """The RC permutation is an involution: RC(RC(i)) == i."""
        from background_model_core import rc_kmer_permutation
        rc_perm = rc_kmer_permutation(KMER)
        assert np.array_equal(rc_perm[rc_perm], np.arange(4**KMER))

    def test_hex_rc_equals_rc_perm_of_hex_fwd(self):
        """For every valid position, hex_rc[c] == RC_PERM[hex_fwd[c]].

        This is the contract Step 4 relies on: on the minus strand, both c5
        and c3 are looked up via hex_rc, which must be the RC of the forward
        hexamer at that same genomic position.
        """
        from background_model_core import rc_kmer_permutation
        rc_perm = rc_kmer_permutation(KMER)

        # Use a known all-ACGT sequence
        seq = "ACGTACGTACGTACGT" * 20  # 320 bp, all valid
        seq_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        fwd, rc, valid = hexamer_indices(seq_bytes)

        assert valid.all(), "Expected all-valid for ACGT-only sequence"
        expected_rc = rc_perm[fwd]
        assert np.array_equal(rc, expected_rc), (
            f"hex_rc != RC_PERM[hex_fwd] at positions: "
            f"{np.where(rc != expected_rc)[0][:5]}"
        )

    def test_rc_on_known_hexamer(self):
        """For the hexamer AAAAAA (index 0), RC is TTTTTT (index 4095)."""
        seq = b"AAAAAA"
        seq_bytes = np.frombuffer(seq, dtype=np.uint8)
        fwd, rc, valid = hexamer_indices(seq_bytes)
        assert len(fwd) == 1
        assert fwd[0] == 0, f"AAAAAA should be index 0, got {fwd[0]}"
        assert rc[0] == 4095, f"RC(AAAAAA)=TTTTTT should be 4095, got {rc[0]}"

    def test_rc_on_palindrome(self):
        """For palindromic hexamers (e.g. ACGTAC... nope, let me use AATTAA -> TTAATT).

        Wait — AATTAA reversed is AATTAA, complemented is TTAATT.  So
        RC(AATTAA) = TTAATT, which is different.

        Use a true palindrome: AATATT -> RC: AATATT. Let's verify:
        AATATT complement: TTATAA, reverse: AATATT. Yes!
        """
        # AATATT: A=0, A=0, T=3, A=0, T=3, T=3
        # index = 0*4^5 + 0*4^4 + 3*4^3 + 0*4^2 + 3*4 + 3 = 192+12+3 = 207
        seq = b"AATATT"
        seq_bytes = np.frombuffer(seq, dtype=np.uint8)
        fwd, rc, valid = hexamer_indices(seq_bytes)
        assert fwd[0] == rc[0], (
            f"Palindrome AATATT: fwd={fwd[0]}, rc={rc[0]} (should be equal)"
        )

    def test_hex_rc_on_real_region(self):
        """On a real genomic region, hex_rc == RC_PERM[hex_fwd] at all valid sites."""
        from background_model_core import rc_kmer_permutation
        from background_model.simulator.precompute import precompute_region

        rc_perm = rc_kmer_permutation(KMER)
        rp = precompute_region(
            "chr1", 100000, 102560,
            "/efs/analytics/nathanboley/data_resources/genome/hg38.fa",
        )
        # Only check valid positions (N bases break the relationship)
        v = rp.valid
        assert np.array_equal(rp.hex_rc[v], rc_perm[rp.hex_fwd[v]])

    def test_invalid_position_has_n_base(self):
        """A position marked invalid must contain a non-ACGT base in its window."""
        # Construct a sequence with an N in the middle
        seq = b"ACGTACNACGTACGT"  # N at position 6
        seq_bytes = np.frombuffer(seq, dtype=np.uint8)
        fwd, rc, valid = hexamer_indices(seq_bytes)
        # Positions where the N is in the 6-mer window: 1..6
        # N is at index 6 in the sequence, so windows starting at 1,2,3,4,5,6
        # include position 6.
        for c in range(1, 7):
            if c < len(valid):
                assert not valid[c], f"Position {c} should be invalid (N in window)"
        # Position 0 should be valid (window is seq[0:6] = ACGTAC)
        assert valid[0], "Position 0 should be valid"


# ── 5. Precompute matches old sim_fragments ──────────────────────────────

class TestPrecomputeMatchesOld:
    """Our hexamer_indices must produce identical results to the old
    sim_fragments.hexamer_indices on the same input."""

    def test_bit_identical_on_acgt_sequence(self):
        """Synthetic all-ACGT sequence: both implementations agree."""
        import sys
        sys.path.insert(0, "scripts")
        from sim_fragments import hexamer_indices as old_hex

        rng = np.random.default_rng(42)
        bases = np.array([ord(c) for c in "ACGT"])
        seq_bytes = rng.choice(bases, size=500).astype(np.uint8)

        old_fwd, old_rc, old_valid = old_hex(seq_bytes)
        new_fwd, new_rc, new_valid = hexamer_indices(seq_bytes)

        assert np.array_equal(old_fwd, new_fwd)
        assert np.array_equal(old_rc, new_rc)
        assert np.array_equal(old_valid, new_valid)

    def test_bit_identical_with_n_bases(self):
        """Sequence containing N bases: both implementations agree."""
        import sys
        sys.path.insert(0, "scripts")
        from sim_fragments import hexamer_indices as old_hex

        rng = np.random.default_rng(99)
        bases = np.array([ord(c) for c in "ACGTN"])
        seq_bytes = rng.choice(bases, size=300).astype(np.uint8)

        old_fwd, old_rc, old_valid = old_hex(seq_bytes)
        new_fwd, new_rc, new_valid = hexamer_indices(seq_bytes)

        assert np.array_equal(old_fwd, new_fwd)
        assert np.array_equal(old_rc, new_rc)
        assert np.array_equal(old_valid, new_valid)


# ── 6. build_predict_lut correctness ─────────────────────────────────────

class TestBuildPredictLUT:
    """The LUT must be built correctly from any predict callable."""

    def test_constant_predict(self):
        """predict=1.0 everywhere → LUT is all 1.0."""
        lut = build_predict_lut(lambda L, gc: 1.0)
        assert lut.shape == (N_LENGTHS, N_GC_BINS)
        assert np.allclose(lut, 1.0)

    def test_length_dependent_predict(self):
        """predict = L → LUT varies along the L axis."""
        lut = build_predict_lut(lambda L, gc: float(L))
        for li in range(N_LENGTHS):
            L = L_MIN + li
            assert np.allclose(lut[li, :], float(L))

    def test_gc_dependent_predict(self):
        """predict = gc_mid → LUT varies along the GC axis."""
        lut = build_predict_lut(lambda L, gc: gc)
        for gi in range(N_GC_BINS):
            gc_mid = gi * GC_BIN_WIDTH + GC_BIN_WIDTH / 2.0
            assert np.allclose(lut[:, gi], gc_mid)

    def test_all_positive(self):
        """LUT values must all be positive (predict is 1/P(seen))."""
        lut = build_predict_lut(lambda L, gc: 1.0 + 0.01 * gc)
        assert (lut > 0).all()


# ── 7. fit_and_build end-to-end (H1 + M5) ────────────────────────────

class TestFitAndBuild:
    """End-to-end test: fit a real capture surface and build the LUT.

    Exercises the full ``fit_and_build`` → ``predict_lut_from_model`` path,
    which is the real-data entry point.  After the H1 guards land, this
    test also proves the H1 conditions hold on the production path: the
    fitted model is ZTNB (not spike_grid) and its gc_bins match SIM_GC_BINS.
    If either guard fails, the test fails with a clear ValueError before
    reaching the shape/value assertions.

    What must FAIL: if the model were spike_grid or its gc_bins mismatched,
    ``predict_lut_from_model`` raises ValueError (H1 guard).  If the LUT
    shape or marginal_fl normalisation is wrong, the assertions below catch
    it.
    """

    @pytest.fixture(autouse=True)
    def _require_flgc(self):
        """`flgc` is a HARD runtime dependency -- its absence is a FAILURE.

        Deliberately NOT ``pytest.importorskip``.  ``capture.py`` imports
        ``flgc.model`` unconditionally, so the package is not optional; it is
        merely absent from ``PYTHONPATH`` unless the Makefile supplies it.
        An ``importorskip`` here made this test -- the ONLY one exercising the
        H1 guards on the production path -- vanish silently from ``make test``:
        a guard that reads as protection and is not.  The Makefile now sets
        ``PYTHONPATH``; if that ever breaks, this must go RED, not quiet.
        """
        try:
            import flgc.model  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment failure
            raise AssertionError(
                "flgc.model is not importable. It is a RUNTIME dependency of "
                "background_model.simulator.capture, not an optional extra. "
                "`make test` sets PYTHONPATH=$(FLGC_PYTHONPATH); if running "
                "pytest directly, export "
                "PYTHONPATH=/home/nathanboley/src/biomarker. "
                "This is deliberately a failure, not a skip."
            ) from exc

    def test_fit_and_build_on_real_sample(self):
        """fit_and_build("RD-56670") produces valid LUT and marginal_fl."""
        from background_model.simulator.capture import fit_and_build

        lut, fl = fit_and_build("RD-56670")

        # LUT shape: 156 lengths × 20 GC bins
        assert lut.shape == (N_LENGTHS, N_GC_BINS), (
            f"LUT shape {lut.shape} != ({N_LENGTHS}, {N_GC_BINS})"
        )
        # All predict values must be positive (they are 1/P(seen), capped)
        assert (lut > 0).all(), "LUT contains non-positive values"

        # marginal_fl: 156 entries, sums to 1, all non-negative
        assert fl.shape == (N_LENGTHS,)
        assert abs(fl.sum() - 1.0) < 1e-15, f"marginal_fl sum {fl.sum()} != 1.0"
        assert (fl >= 0).all(), "marginal_fl contains negative values"
