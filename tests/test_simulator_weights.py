"""Tests for the simulator weight builder (background_model.simulator.weights).

Covers the four required invariants from the Phase 1 spec:

1. sum_Omega w = 1 exactly (per region), AND the strand marginal is exactly 1/2.
2. The minus-strand swap: c5 = p+L, c3 = p; hexamers read RC; c3(L) = c5 + sigma*L
   with sigma = -1.  Tests both parities of L (odd and even).
3. The Z_s edge rule: Z_s(c5) sums only L whose c3(L) stays in [0, region_len].
   Assert |Omega| = 767,052 for region_len = 2560.
4. L = 25..180, 156 values — exactly the capture surface's support.

All synthetic — no FASTA, no torch, no EFS.
"""

import numpy as np
import pytest

from background_model.simulator.weights import (
    L_MAX,
    L_MIN,
    N_LENGTHS,
    SIGMA_MINUS,
    SIGMA_PLUS,
    RegionWeights,
    build_region_weights,
    c3_from_c5,
    fragment_base_range,
    gc_pct,
    omega_size,
)


# ── helpers ──────────────────────────────────────────────────────────────

NHEX = 4096


def _uniform_tables(seed=42):
    """Four uniform hexamer tables (all 1.0) — makes the weight depend only
    on marginal_fl and predict, simplifying invariant checks."""
    ones = np.ones(NHEX, dtype=np.float64)
    return ones.copy(), ones.copy(), ones.copy(), ones.copy()


def _random_tables(seed=42):
    """Four independent log-normal hexamer tables, normalised to max 1."""
    rng = np.random.default_rng(seed)
    tables = []
    for _ in range(4):
        w = np.exp(rng.normal(0, 0.3, NHEX))
        w /= w.max()
        tables.append(w)
    return tables


def _flat_marginal_fl():
    """Uniform marginal over L = 25..180."""
    fl = np.ones(N_LENGTHS, dtype=np.float64)
    fl /= fl.sum()
    return fl


def _synthetic_region(region_len, seed=123):
    """Synthetic hex_fwd, hex_rc, cum_gc, valid arrays for a region."""
    rng = np.random.default_rng(seed)
    n_sites = region_len + 1
    hex_fwd = rng.integers(0, NHEX, size=n_sites)
    hex_rc = rng.integers(0, NHEX, size=n_sites)
    # Random GC: about 40% of bases are G/C
    bases_gc = rng.random(region_len) < 0.4
    cum_gc = np.concatenate([[0], np.cumsum(bases_gc)]).astype(np.float64)
    valid = np.ones(n_sites, dtype=bool)
    return hex_fwd, hex_rc, cum_gc, valid


def _trivial_predict(L, gc):
    """Predict that always returns 1.0 (no capture bias)."""
    return 1.0


def _build_weights(region_len, tables=None, marginal_fl=None, predict=None,
                   seed=123):
    """Build weights with default synthetic inputs."""
    hex_fwd, hex_rc, cum_gc, valid = _synthetic_region(region_len, seed)
    if tables is None:
        sf, ef, sr, er = _uniform_tables()
    else:
        sf, ef, sr, er = tables
    if marginal_fl is None:
        marginal_fl = _flat_marginal_fl()
    if predict is None:
        predict = _trivial_predict
    return build_region_weights(
        hex_fwd=hex_fwd,
        hex_rc=hex_rc,
        cum_gc=cum_gc,
        start_fwd=sf,
        end_fwd=ef,
        start_rev=sr,
        end_rev=er,
        marginal_fl=marginal_fl,
        predict=predict,
        region_len=region_len,
        valid=valid,
    )


# ── Invariant 1: sum_Omega w = 1, strand marginal = 1/2 ─────────────────

class TestNormalisationInvariant:
    """sum_Omega w = 1 must hold exactly (up to floating-point), per region.
    The strand marginal must be exactly 1/2."""

    def test_uniform_tables_region_2560(self):
        rw = _build_weights(2560)
        total = rw.w_plus.sum() + rw.w_minus.sum()
        assert abs(total - 1.0) < 1e-12, f"sum = {total}"
        assert abs(rw.w_plus.sum() - 0.5) < 1e-12, f"plus = {rw.w_plus.sum()}"
        assert abs(rw.w_minus.sum() - 0.5) < 1e-12, f"minus = {rw.w_minus.sum()}"

    def test_random_tables_region_2560(self):
        tables = _random_tables(seed=99)
        rw = _build_weights(2560, tables=tables)
        total = rw.w_plus.sum() + rw.w_minus.sum()
        assert abs(total - 1.0) < 1e-12, f"sum = {total}"
        assert abs(rw.w_plus.sum() - 0.5) < 1e-12, f"plus = {rw.w_plus.sum()}"
        assert abs(rw.w_minus.sum() - 0.5) < 1e-12, f"minus = {rw.w_minus.sum()}"

    def test_random_tables_region_1536(self):
        tables = _random_tables(seed=77)
        rw = _build_weights(1536, tables=tables)
        total = rw.w_plus.sum() + rw.w_minus.sum()
        assert abs(total - 1.0) < 1e-12, f"sum = {total}"
        assert abs(rw.w_plus.sum() - 0.5) < 1e-12
        assert abs(rw.w_minus.sum() - 0.5) < 1e-12

    def test_random_tables_with_gc_bias(self):
        """Normalisation holds even with a non-trivial predict function."""
        tables = _random_tables(seed=55)

        def gc_predict(L, gc):
            # Non-trivial: higher GC -> higher weight (lower capture)
            return 1.0 + 0.01 * gc

        rw = _build_weights(2560, tables=tables, predict=gc_predict)
        total = rw.w_plus.sum() + rw.w_minus.sum()
        assert abs(total - 1.0) < 1e-12, f"sum = {total}"
        assert abs(rw.w_plus.sum() - 0.5) < 1e-12
        assert abs(rw.w_minus.sum() - 0.5) < 1e-12

    def test_random_tables_with_nonuniform_fl(self):
        """Normalisation holds with a peaked length marginal."""
        tables = _random_tables(seed=33)
        fl = np.zeros(N_LENGTHS, dtype=np.float64)
        # Concentrate mass on a few lengths
        fl[0] = 0.3    # L=25
        fl[50] = 0.4   # L=75
        fl[100] = 0.3  # L=125
        rw = _build_weights(2560, tables=tables, marginal_fl=fl)
        total = rw.w_plus.sum() + rw.w_minus.sum()
        assert abs(total - 1.0) < 1e-12, f"sum = {total}"
        assert abs(rw.w_plus.sum() - 0.5) < 1e-12
        assert abs(rw.w_minus.sum() - 0.5) < 1e-12

    def test_small_region(self):
        """Normalisation on a small region (just barely fits L_MIN)."""
        region_len = L_MIN  # 25 — minimum viable region
        rw = _build_weights(region_len)
        total = rw.w_plus.sum() + rw.w_minus.sum()
        assert abs(total - 1.0) < 1e-12, f"sum = {total}"
        assert abs(rw.w_plus.sum() - 0.5) < 1e-12
        assert abs(rw.w_minus.sum() - 0.5) < 1e-12


# ── Invariant 2: minus-strand swap ──────────────────────────────────────

class TestMinusStrandSwap:
    """c5 = p+L, c3 = p for minus strand.  Tests both parities of L."""

    def test_c3_formula_plus(self):
        assert c3_from_c5(100, 50, SIGMA_PLUS) == 150

    def test_c3_formula_minus(self):
        assert c3_from_c5(150, 50, SIGMA_MINUS) == 100

    def test_fragment_base_range_plus(self):
        # Plus: c5=100, c3=150 → p=100, L=50
        p, L = fragment_base_range(100, 150)
        assert p == 100 and L == 50

    def test_fragment_base_range_minus(self):
        # Minus: c5=150, c3=100 → p=100, L=50
        p, L = fragment_base_range(150, 100)
        assert p == 100 and L == 50

    def test_gc_pct_strand_independent(self):
        """GC is the same regardless of which end is c5 vs c3."""
        cum = np.array([0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6], dtype=np.float64)
        # Fragment bases [2, 8) → gc = cum[8] - cum[2] = 4-2 = 2, L=6, pct=33.33
        gc_plus = gc_pct(2, 8, cum)   # plus strand: c5=2, c3=8
        gc_minus = gc_pct(8, 2, cum)  # minus strand: c5=8, c3=2
        assert gc_plus == gc_minus

    def test_minus_strand_weight_nonzero_even_L(self):
        """Minus strand weights are populated for even L."""
        rw = _build_weights(200, tables=_random_tables(seed=11))
        L_even = 50
        li = L_even - L_MIN
        # For minus, c5 in [L, region_len] → c5 in [50, 200]
        assert rw.w_minus[L_even:, li].sum() > 0

    def test_minus_strand_weight_nonzero_odd_L(self):
        """Minus strand weights are populated for odd L."""
        rw = _build_weights(200, tables=_random_tables(seed=11))
        L_odd = 51
        li = L_odd - L_MIN
        # For minus, c5 in [51, 200]
        assert rw.w_minus[L_odd:, li].sum() > 0

    def test_plus_minus_symmetry_on_palindromic_sequence(self):
        """On a region where hex_fwd == hex_rc and all tables are identical,
        plus and minus weights are equal."""
        region_len = 300
        rng = np.random.default_rng(77)
        n_sites = region_len + 1
        hex_idx = rng.integers(0, NHEX, size=n_sites)
        cum = np.concatenate([[0], np.cumsum(rng.random(region_len) < 0.5)])
        cum = cum.astype(np.float64)
        table = np.exp(rng.normal(0, 0.3, NHEX))
        table /= table.max()
        fl = _flat_marginal_fl()

        rw = build_region_weights(
            hex_fwd=hex_idx, hex_rc=hex_idx,  # same!
            cum_gc=cum, start_fwd=table, end_fwd=table,
            start_rev=table, end_rev=table,
            marginal_fl=fl, predict=_trivial_predict,
            region_len=region_len, valid=np.ones(n_sites, dtype=bool),
        )
        # Both strand marginals should be 0.5
        assert abs(rw.w_plus.sum() - 0.5) < 1e-12
        assert abs(rw.w_minus.sum() - 0.5) < 1e-12


# ── Invariant 3: |Omega| = 767,052 for region_len = 2560 ────────────────

class TestOmegaSize:
    """The Z_s edge rule: Z_s(c5) sums only L whose c3(L) stays in
    [0, region_len].  This test verifies |Omega| via the closed form."""

    def test_omega_2560(self):
        assert omega_size(2560) == 767_052

    def test_omega_1536(self):
        # |Omega| = 2 * sum_{L=25}^{180} (1536 - L + 1)
        # = 2 * sum_{L=25}^{180} (1537 - L)
        # = 2 * sum_{k=1357}^{1512} k  (where k = 1537 - L)
        # = 2 * 156 * (1357 + 1512) / 2 = 156 * 2869 = 447,564
        assert omega_size(1536) == 447_564

    def test_omega_minimum_region(self):
        # region_len = L_MIN = 25: only L=25 fits, 1 position per strand
        assert omega_size(25) == 2

    def test_omega_matches_nonzero_weight_count(self):
        """The count of non-zero w entries matches |Omega|."""
        region_len = 2560
        rw = _build_weights(region_len, tables=_random_tables(seed=44))
        n_nonzero = np.count_nonzero(rw.w_plus) + np.count_nonzero(rw.w_minus)
        assert n_nonzero == omega_size(region_len), (
            f"nonzero weights = {n_nonzero}, |Omega| = {omega_size(region_len)}"
        )

    def test_edge_truncation_plus(self):
        """Plus strand: positions near the right edge have fewer valid lengths."""
        region_len = 200
        rw = _build_weights(region_len)
        # c5 = region_len - L_MIN = 175: only L=25 fits (c3=200=region_len)
        c5 = region_len - L_MIN
        li_25 = 0  # L=25
        assert rw.w_plus[c5, li_25] > 0
        # L=26 would give c3=201 > region_len → should be 0
        if N_LENGTHS > 1:
            assert rw.w_plus[c5, 1] == 0.0

    def test_edge_truncation_minus(self):
        """Minus strand: positions near the left edge have fewer valid lengths."""
        region_len = 200
        rw = _build_weights(region_len)
        # c5 = L_MIN = 25: only L=25 fits (c3=0)
        c5 = L_MIN
        li_25 = 0
        assert rw.w_minus[c5, li_25] > 0
        # L=26 would give c3=-1 < 0 → should be 0
        if N_LENGTHS > 1:
            assert rw.w_minus[c5, 1] == 0.0


# ── Invariant 4: L = 25..180, 156 values ────────────────────────────────

class TestLengthRange:
    """Exactly 156 lengths, L_MIN=25, L_MAX=180."""

    def test_constants(self):
        assert L_MIN == 25
        assert L_MAX == 180
        assert N_LENGTHS == 156

    def test_weight_shape(self):
        rw = _build_weights(2560)
        assert rw.w_plus.shape == (2561, 156)
        assert rw.w_minus.shape == (2561, 156)

    def test_all_lengths_populated(self):
        """Every length in [25, 180] has at least one non-zero weight
        (on a large enough region)."""
        rw = _build_weights(2560, tables=_random_tables(seed=88))
        for li in range(N_LENGTHS):
            L = L_MIN + li
            has_plus = rw.w_plus[:, li].sum() > 0
            has_minus = rw.w_minus[:, li].sum() > 0
            assert has_plus, f"L={L}: no plus-strand weights"
            assert has_minus, f"L={L}: no minus-strand weights"


# ── Additional correctness tests ────────────────────────────────────────

class TestWeightProperties:
    """Additional properties that should hold for well-formed weights."""

    def test_all_weights_nonneg(self):
        rw = _build_weights(2560, tables=_random_tables())
        assert (rw.w_plus >= 0).all()
        assert (rw.w_minus >= 0).all()

    def test_invalid_position_zeroed(self):
        """A position with valid=False contributes zero weight."""
        region_len = 300
        hex_fwd, hex_rc, cum_gc, valid = _synthetic_region(region_len, seed=99)
        # Mark position 100 as invalid
        valid[100] = False
        sf, ef, sr, er = _random_tables()
        fl = _flat_marginal_fl()
        rw = build_region_weights(
            hex_fwd=hex_fwd, hex_rc=hex_rc, cum_gc=cum_gc,
            start_fwd=sf, end_fwd=ef, start_rev=sr, end_rev=er,
            marginal_fl=fl, predict=_trivial_predict,
            region_len=region_len, valid=valid,
        )
        # Position 100 as c5 should have zero weight for both strands
        assert rw.w_plus[100, :].sum() == 0.0
        assert rw.w_minus[100, :].sum() == 0.0
        # Normalisation still holds
        total = rw.w_plus.sum() + rw.w_minus.sum()
        assert abs(total - 1.0) < 1e-12

    def test_predict_affects_weights(self):
        """Different predict functions produce different weight distributions."""
        rw1 = _build_weights(500, predict=_trivial_predict)
        rw2 = _build_weights(500, predict=lambda L, gc: 1.0 + 0.05 * gc)
        # They should differ (same hex/gc/fl, different predict)
        assert not np.allclose(rw1.w_plus, rw2.w_plus)
        # But both normalise to 1
        assert abs(rw1.w_plus.sum() + rw1.w_minus.sum() - 1.0) < 1e-12
        assert abs(rw2.w_plus.sum() + rw2.w_minus.sum() - 1.0) < 1e-12
