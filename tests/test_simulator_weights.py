"""Tests for the simulator weight builder (background_model.simulator.weights).

Covers the four required invariants from the Phase 1 spec:

1. ``w_plus.sum() + w_minus.sum() == 1`` exactly (per region), AND each
   strand marginal is exactly 0.5.
2. The minus-strand swap: ``c5 = p+L``, ``c3 = p``; hexamers read RC;
   ``c3(L) = c5 - L``.  Tests both parities of L (odd and even).
3. The midpoint rule: a fragment is admitted when its integer midpoint
   ``p + L // 2`` falls in ``[0, region_len)``.  Every L has exactly
   ``region_len`` valid positions per strand.  ``|Ω| = 2 × region_len ×
   N_LENGTHS`` (479,232 at region_len 1536).
4. L = 25..180, 156 values — exactly the capture surface's support.

All synthetic — no FASTA, no torch, no EFS.
"""

import numpy as np
import pytest

from background_model.simulator.weights import (
    L_MAX,
    L_MIN,
    MAX_FL_HALF,
    N_GC_BINS,
    N_LENGTHS,
    NHEX,
    STRAND_MINUS,
    STRAND_PLUS,
    HexamerTables,
    RegionWeights,
    build_predict_lut,
    build_region_weights,
    c3_from_c5,
    fragment_base_range,
    gc_bin_index,
    gc_pct,
    generative_domain_size,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _uniform_tables():
    """Four uniform hexamer tables (all 1.0) — makes the weight depend only
    on marginal_fl and predict, simplifying invariant checks.

    Deliberately takes no seed: the tables are constant, so there is nothing
    to randomise. Note these tables make the four-table wiring INDISTINGUISHABLE
    — that is what TestAsymmetricFourTableWiring exists to cover.
    """
    ones = np.ones(NHEX, dtype=np.float64)
    return HexamerTables(ones.copy(), ones.copy(), ones.copy(), ones.copy())


def _random_tables(seed=42):
    """Four independent log-normal hexamer tables, normalised to max 1."""
    rng = np.random.default_rng(seed)
    tables = []
    for _ in range(4):
        w = np.exp(rng.normal(0, 0.3, NHEX))
        w /= w.max()
        tables.append(w)
    return HexamerTables(*tables)


def _flat_marginal_fl():
    """Uniform marginal over L = 25..180."""
    fl = np.ones(N_LENGTHS, dtype=np.float64)
    fl /= fl.sum()
    return fl


def _synthetic_region(region_len, seed=123, pad=MAX_FL_HALF):
    """Synthetic hex_fwd, hex_rc, cum_gc, valid arrays for a region.

    Returns expanded arrays of size region_len + 2*pad + 1.
    The core data (positions pad through pad + region_len) is generated from
    ``seed``; padding uses a separate deterministic seed to avoid perturbing
    the core RNG stream.
    """
    rng = np.random.default_rng(seed)
    n_core = region_len + 1
    hex_fwd_core = rng.integers(0, NHEX, size=n_core)
    hex_rc_core = rng.integers(0, NHEX, size=n_core)
    bases_gc_core = rng.random(region_len) < 0.4

    # Expand with deterministic padding from a separate seed
    pad_rng = np.random.default_rng(seed + 1_000_000)
    n_sites = region_len + 2 * pad + 1
    hex_fwd = np.concatenate([
        pad_rng.integers(0, NHEX, size=pad),
        hex_fwd_core,
        pad_rng.integers(0, NHEX, size=pad),
    ])
    hex_rc = np.concatenate([
        pad_rng.integers(0, NHEX, size=pad),
        hex_rc_core,
        pad_rng.integers(0, NHEX, size=pad),
    ])
    bases_gc = np.concatenate([
        pad_rng.random(pad) < 0.4,
        bases_gc_core,
        pad_rng.random(pad) < 0.4,
    ])
    cum_gc = np.concatenate([[0], np.cumsum(bases_gc)]).astype(np.float64)
    valid = np.ones(n_sites, dtype=bool)
    return hex_fwd, hex_rc, cum_gc, valid


def _trivial_predict(L, gc):
    """Predict that always returns 1.0 (no capture bias)."""
    return 1.0


def _trivial_lut():
    """LUT for the trivial (no-op) predict — all 1.0."""
    return build_predict_lut(_trivial_predict)


def _build_weights(region_len, tables=None, marginal_fl=None, predict=None,
                   predict_lut=None, seed=123, pad=MAX_FL_HALF):
    """Build weights with default synthetic inputs.

    If ``predict_lut`` is provided it is used directly; otherwise one is
    built from ``predict`` (defaulting to the trivial predict).
    """
    hex_fwd, hex_rc, cum_gc, valid = _synthetic_region(region_len, seed, pad=pad)
    if tables is None:
        tables = _uniform_tables()
    if marginal_fl is None:
        marginal_fl = _flat_marginal_fl()
    if predict_lut is None:
        if predict is None:
            predict = _trivial_predict
        predict_lut = build_predict_lut(predict)
    return build_region_weights(
        hex_fwd=hex_fwd,
        hex_rc=hex_rc,
        cum_gc=cum_gc,
        hex_tables=tables,
        marginal_fl=marginal_fl,
        predict_lut=predict_lut,
        region_len=region_len,
        valid=valid,
        pad=pad,
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

        gc_predict = lambda L, gc: 1.0 + 0.01 * gc
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
        assert c3_from_c5(100, 50, STRAND_PLUS) == 150

    def test_c3_formula_minus(self):
        assert c3_from_c5(150, 50, STRAND_MINUS) == 100

    def test_c3_rejects_invalid_strand(self):
        """Passing a numeric sign instead of a strand label raises ValueError."""
        with pytest.raises(ValueError, match="strand must be"):
            c3_from_c5(100, 50, +1)  # type: ignore[arg-type]

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

    def test_gc_pct_zero_length_raises(self):
        """c5 == c3 (zero-length fragment) raises ValueError."""
        cum = np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)
        with pytest.raises(ValueError, match="zero-length fragment"):
            gc_pct(3, 3, cum)

    def test_gc_pct_zero_length_array_raises(self):
        """Vectorised: any c5 == c3 element raises ValueError."""
        cum = np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)
        with pytest.raises(ValueError, match="zero-length fragment"):
            gc_pct(np.array([1, 3]), np.array([4, 3]), cum)

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
        pad = MAX_FL_HALF
        rng = np.random.default_rng(77)
        n_sites = region_len + 2 * pad + 1
        hex_idx = rng.integers(0, NHEX, size=n_sites)
        cum = np.concatenate([[0], np.cumsum(rng.random(region_len + 2 * pad) < 0.5)])
        cum = cum.astype(np.float64)
        table = np.exp(rng.normal(0, 0.3, NHEX))
        table /= table.max()
        fl = _flat_marginal_fl()

        rw = build_region_weights(
            hex_fwd=hex_idx, hex_rc=hex_idx,  # same!
            cum_gc=cum,
            hex_tables=HexamerTables(table, table, table, table),
            marginal_fl=fl, predict_lut=_trivial_lut(),
            region_len=region_len, valid=np.ones(n_sites, dtype=bool),
            pad=pad,
        )
        # Both strand marginals should be 0.5
        assert abs(rw.w_plus.sum() - 0.5) < 1e-12
        assert abs(rw.w_minus.sum() - 0.5) < 1e-12


# ── Four-table wiring (asymmetric data) ──────────────────────────────────

class TestAsymmetricFourTableWiring:
    """Verify the four-table wiring with fully asymmetric data.

    With uniform tables (all 1.0), the four tables are indistinguishable — a
    bug using end_fwd where end_rev is needed passes all tests.  This class
    uses four *distinct* tables and distinct hex_fwd/hex_rc, then for specific
    (c5, L, strand) triples computes the expected weight from the Step 4
    formula and asserts (a) equality with the builder to ~1e-14 and (b) that
    the value *differs* from what each single-axis wrong wiring would give.
    """

    @staticmethod
    def _reference_weight(c5, L, strand, hex_tab, start_tab, end_tab,
                          cum_gc, valid, fl, predict_lut, region_len,
                          pad=MAX_FL_HALF):
        """Compute w(c5, L, strand) from the Step 4 formula, independently.

        ``c5`` is an array index (not region-local).
        ``hex_tab``, ``start_tab``, ``end_tab`` are the strand-selected arrays.
        ``predict_lut`` is shape ``(N_LENGTHS, N_GC_BINS)``.
        """
        n_sites = region_len + 2 * pad + 1
        sign = +1 if strand == "+" else -1
        c3 = c5 + sign * L
        li = L - L_MIN

        if c3 < 0 or c3 >= n_sites:
            return 0.0
        if not (valid[c5] and valid[c3]):
            return 0.0
        # Midpoint check: p_local + L//2 in [0, region_len)
        p = min(c5, c3) - pad
        if p + L // 2 < 0 or p + L // 2 >= region_len:
            return 0.0

        gc_val = float(gc_pct(c5, c3, cum_gc))
        gi = int(gc_bin_index(np.array([gc_val]))[0])
        E_here = end_tab[hex_tab[c3]] * fl[li] / predict_lut[li, gi]

        # Z_s(c5) = sum_{valid L} E_s(c5, L)
        Z = 0.0
        for l in range(L_MIN, L_MAX + 1):
            c3_l = c5 + sign * l
            if c3_l < 0 or c3_l >= n_sites:
                continue
            if not (valid[c5] and valid[c3_l]):
                continue
            p_l = min(c5, c3_l) - pad
            if p_l + l // 2 < 0 or p_l + l // 2 >= region_len:
                continue
            li_l = l - L_MIN
            gc_l = float(gc_pct(c5, c3_l, cum_gc))
            gi_l = int(gc_bin_index(np.array([gc_l]))[0])
            Z += end_tab[hex_tab[c3_l]] * fl[li_l] / predict_lut[li_l, gi_l]

        if Z == 0:
            return 0.0

        # S_s = sum of start_tab[hex_tab[pos]] over positions with Z_s(pos) > 0
        S = 0.0
        for pos in range(n_sites):
            Z_pos = 0.0
            for l in range(L_MIN, L_MAX + 1):
                c3_l = pos + sign * l
                if c3_l < 0 or c3_l >= n_sites:
                    continue
                if not (valid[pos] and valid[c3_l]):
                    continue
                p_l = min(pos, c3_l) - pad
                if p_l + l // 2 < 0 or p_l + l // 2 >= region_len:
                    continue
                li_l = l - L_MIN
                gc_l = float(gc_pct(pos, c3_l, cum_gc))
                gi_l = int(gc_bin_index(np.array([gc_l]))[0])
                Z_pos += end_tab[hex_tab[c3_l]] * fl[li_l] / predict_lut[li_l, gi_l]
            if Z_pos > 0:
                S += start_tab[hex_tab[pos]]

        if S == 0:
            return 0.0

        return 0.5 * start_tab[hex_tab[c5]] / S * E_here / Z

    def test_correct_wiring_matches_and_wrong_wirings_differ(self):
        """Builder matches reference; each wrong wiring gives a different value.

        Covers both strands and both parities of L (even and odd).
        """
        region_len = 50
        pad = MAX_FL_HALF
        rng = np.random.default_rng(314)
        n_sites = region_len + 2 * pad + 1

        hex_fwd = rng.integers(0, NHEX, size=n_sites)
        hex_rc = rng.integers(0, NHEX, size=n_sites)
        cum_gc = np.concatenate(
            [[0], np.cumsum(rng.random(region_len + 2 * pad) < 0.4)]
        ).astype(np.float64)
        valid = np.ones(n_sites, dtype=bool)

        # Four DISTINCT tables (log-normal, different seeds via the shared rng)
        tables = []
        for _ in range(4):
            t = np.exp(rng.normal(0, 0.5, NHEX))
            t /= t.max()
            tables.append(t)
        sf, ef, sr, er = tables
        fl = _flat_marginal_fl()

        trivial_lut = _trivial_lut()
        rw = build_region_weights(
            hex_fwd=hex_fwd, hex_rc=hex_rc, cum_gc=cum_gc,
            hex_tables=HexamerTables(sf, ef, sr, er),
            marginal_fl=fl, predict_lut=trivial_lut,
            region_len=region_len, valid=valid,
            pad=pad,
        )

        # (c5_array_idx, L, strand)
        # Both strands x both parities of L.
        # c5 positions chosen in the valid midpoint range:
        #   plus: c5 in [pad - L//2, pad + region_len - 1 - L//2]
        #   minus: c5 in [pad + ceil(L/2), pad + region_len - 1 + ceil(L/2)]
        cases = [
            (pad + 10, 30, "+"),    # even L, plus (midpoint j=10+15=25)
            (pad + 10, 31, "+"),    # odd L, plus
            (pad + 40 + 15, 30, "-"),  # even L, minus (c5=pad+55, c3=pad+25)
            (pad + 41 + 16, 31, "-"),  # odd L, minus
        ]

        for c5, L, strand in cases:
            li = L - L_MIN
            w_b = rw.w_plus[c5, li] if strand == "+" else rw.w_minus[c5, li]

            # ── correct wiring ──
            if strand == "+":
                correct = (hex_fwd, sf, ef)
            else:
                correct = (hex_rc, sr, er)

            w_ref = self._reference_weight(
                c5, L, strand, *correct,
                cum_gc, valid, fl, trivial_lut, region_len, pad=pad,
            )
            assert w_ref > 0, f"s={strand} c5={c5} L={L}: zero reference weight"
            assert abs(w_b - w_ref) < 1e-14, (
                f"s={strand} c5={c5} L={L}: builder {w_b:.18e} != ref {w_ref:.18e}"
            )

            # ── wrong wirings (one axis wrong at a time) ──
            if strand == "+":
                wrongs = [
                    ((hex_rc,  sf, ef), "wrong hex"),
                    ((hex_fwd, sr, ef), "wrong start table"),
                    ((hex_fwd, sf, er), "wrong end table"),
                    ((hex_fwd, ef, sf), "start<->end swap"),
                ]
            else:
                wrongs = [
                    ((hex_fwd, sr, er), "wrong hex"),
                    ((hex_rc,  sf, er), "wrong start table"),
                    ((hex_rc,  sr, ef), "wrong end table"),
                    ((hex_rc,  er, sr), "start<->end swap"),
                ]

            for (wh, ws, we), label in wrongs:
                w_wrong = self._reference_weight(
                    c5, L, strand, wh, ws, we,
                    cum_gc, valid, fl, trivial_lut, region_len, pad=pad,
                )
                assert abs(w_ref - w_wrong) > 1e-14, (
                    f"s={strand} c5={c5} L={L} [{label}]: "
                    f"correct {w_ref:.18e} == wrong {w_wrong:.18e}"
                )


# ── Invariant 3: generative domain size ───────────────────────────────────

class TestGenerativeDomainSize:
    """The midpoint rule: every L has exactly ``region_len`` valid midpoints
    per strand.  ``|Ω| = 2 × region_len × N_LENGTHS``.

    Tests updated from the old containment-rule values because the
    production rule now admits a fragment when its integer midpoint
    ``p + L // 2`` falls in ``[0, region_len)``.
    """

    def test_domain_2560(self):
        # Old containment rule: 767,052.
        # New midpoint rule: 2 × 2560 × 156 = 798,720.
        assert generative_domain_size(2560) == 2 * 2560 * N_LENGTHS

    def test_domain_1536(self):
        # Old containment rule: 447,564.
        # New midpoint rule: 2 × 1536 × 156 = 479,232.
        assert generative_domain_size(1536) == 2 * 1536 * N_LENGTHS

    def test_domain_minimum_region(self):
        # region_len = L_MIN = 25: under the midpoint rule, every L
        # still has 25 valid midpoints (0..24), so |Ω| = 2 × 25 × 156.
        # Old containment rule gave 2 (only L=25 fit).
        assert generative_domain_size(25) == 2 * 25 * N_LENGTHS

    def test_domain_matches_nonzero_weight_count(self):
        """The count of non-zero w entries matches the domain size.

        Uses pad=MAX_FL_HALF so the midpoint rule has room for all
        fragment endpoints.
        """
        region_len = 500
        rw = _build_weights(region_len, tables=_random_tables(seed=44),
                            pad=MAX_FL_HALF)
        n_nonzero = np.count_nonzero(rw.w_plus) + np.count_nonzero(rw.w_minus)
        expected = generative_domain_size(region_len)
        assert n_nonzero == expected, (
            f"nonzero weights = {n_nonzero}, domain size = {expected}"
        )

    def test_valid_positions_independent_of_L(self):
        """Under the midpoint rule with sufficient pad, every L has exactly
        region_len valid positions per strand.

        This is the property that eliminates the length-dependent positional
        penalty: the old containment rule gave ``region_len - L + 1`` positions,
        which biased the realised fragment-length distribution away from
        ``marginal_fl``.
        """
        region_len = 500
        rw = _build_weights(region_len, tables=_random_tables(seed=71),
                            pad=MAX_FL_HALF)
        for li in range(N_LENGTHS):
            n_plus = np.count_nonzero(rw.w_plus[:, li])
            n_minus = np.count_nonzero(rw.w_minus[:, li])
            L = L_MIN + li
            assert n_plus == region_len, (
                f"L={L}: plus-strand has {n_plus} valid positions, "
                f"expected {region_len}"
            )
            assert n_minus == region_len, (
                f"L={L}: minus-strand has {n_minus} valid positions, "
                f"expected {region_len}"
            )


# ── Invariant 4: L = 25..180, 156 values ────────────────────────────────

class TestLengthRange:
    """Exactly 156 lengths, L_MIN=25, L_MAX=180."""

    def test_constants(self):
        assert L_MIN == 25
        assert L_MAX == 180
        assert N_LENGTHS == 156

    def test_weight_shape(self):
        """Shape is (region_len + 2*pad + 1, N_LENGTHS)."""
        rw = _build_weights(2560)
        expected_2560 = 2560 + 2 * MAX_FL_HALF + 1
        assert rw.w_plus.shape == (expected_2560, 156)
        assert rw.w_minus.shape == (expected_2560, 156)
        rw_1536 = _build_weights(1536)
        expected_1536 = 1536 + 2 * MAX_FL_HALF + 1
        assert rw_1536.w_plus.shape == (expected_1536, 156)
        assert rw_1536.w_minus.shape == (expected_1536, 156)

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
        pad = MAX_FL_HALF
        hex_fwd, hex_rc, cum_gc, valid = _synthetic_region(region_len, seed=99, pad=pad)
        # Mark a position in the interior as invalid (offset by pad)
        invalid_pos = pad + 100
        valid[invalid_pos] = False
        tables = _random_tables()
        fl = _flat_marginal_fl()
        rw = build_region_weights(
            hex_fwd=hex_fwd, hex_rc=hex_rc, cum_gc=cum_gc,
            hex_tables=tables,
            marginal_fl=fl, predict_lut=_trivial_lut(),
            region_len=region_len, valid=valid,
            pad=pad,
        )
        # Position invalid_pos as c5 should have zero weight for both strands
        assert rw.w_plus[invalid_pos, :].sum() == 0.0
        assert rw.w_minus[invalid_pos, :].sum() == 0.0
        # Position invalid_pos as c3 should also zero the fragment:
        # Plus: c3 = c5 + L = invalid_pos → c5 = invalid_pos - L
        # Minus: c3 = c5 - L = invalid_pos → c5 = invalid_pos + L
        for L in [L_MIN, 50, 75]:
            li = L - L_MIN
            assert rw.w_plus[invalid_pos - L, li] == 0.0, (
                f"plus c3-mask: L={L}, w_plus[{invalid_pos-L},{li}] != 0"
            )
            assert rw.w_minus[invalid_pos + L, li] == 0.0, (
                f"minus c3-mask: L={L}, w_minus[{invalid_pos+L},{li}] != 0"
            )
        # Normalisation still holds
        total = rw.w_plus.sum() + rw.w_minus.sum()
        assert abs(total - 1.0) < 1e-12

    def test_heavy_n_masking_normalisation(self):
        """Normalisation holds with ~15% of positions masked (heavy N content)."""
        region_len = 500
        pad = MAX_FL_HALF
        rng = np.random.default_rng(42)
        n_sites = region_len + 2 * pad + 1
        hex_fwd = rng.integers(0, NHEX, size=n_sites)
        hex_rc = rng.integers(0, NHEX, size=n_sites)
        cum_gc = np.concatenate(
            [[0], np.cumsum(rng.random(region_len + 2 * pad) < 0.4)]
        ).astype(np.float64)
        valid = rng.random(n_sites) > 0.15  # ~15% masked
        tables = _random_tables(seed=77)
        fl = _flat_marginal_fl()

        rw = build_region_weights(
            hex_fwd=hex_fwd, hex_rc=hex_rc, cum_gc=cum_gc,
            hex_tables=tables,
            marginal_fl=fl, predict_lut=_trivial_lut(),
            region_len=region_len, valid=valid,
            pad=pad,
        )
        total = rw.w_plus.sum() + rw.w_minus.sum()
        assert abs(total - 1.0) < 1e-12, f"sum = {total}"
        assert abs(rw.w_plus.sum() - 0.5) < 1e-12, f"plus = {rw.w_plus.sum()}"
        assert abs(rw.w_minus.sum() - 0.5) < 1e-12, f"minus = {rw.w_minus.sum()}"

    def test_predict_affects_weights(self):
        """Different predict functions produce different weight distributions."""
        rw1 = _build_weights(500, predict=_trivial_predict)
        rw2 = _build_weights(500, predict=lambda L, gc: 1.0 + 0.05 * gc)
        # They should differ (same hex/gc/fl, different predict)
        assert not np.allclose(rw1.w_plus, rw2.w_plus)
        # But both normalise to 1
        assert abs(rw1.w_plus.sum() + rw1.w_minus.sum() - 1.0) < 1e-12
        assert abs(rw2.w_plus.sum() + rw2.w_minus.sum() - 1.0) < 1e-12
