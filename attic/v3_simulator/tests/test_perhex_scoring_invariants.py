"""Bitwise invariants for the per-hexamer scoring functions.

This is the acceptance test chosen (design §6.2) INSTEAD of regenerating
``oracle_nb_perhex.json``, whose fit cost 3.1 hours. The reasoning: the three
load-bearing functions are pure — ``score_val_perhex`` and
``nb_nll_positions`` read no module globals at all, and ``fit_perhex_grid``
reads only the four grid constants — so the expensive computation is
verifiable without the expensive pipeline.

These tests pin the functions' behaviour on a small fixed input. They are not
a substitute for end-to-end verification of the Phase C/D orchestration inside
``main()``, which remains unverified and is recorded as such in the design.

Values below were captured from the current implementation. If one of these
fails after a refactor, that is a FINDING about the refactor, not a number to
re-baseline.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts._oracle_scoring import make_oracle_loss_fn
from scripts.nb_oracle_perhex import (
    fit_perhex_grid,
    nb_nll_positions,
    score_val_perhex,
)


def _pair(seed, C=2, L=16):
    """A deterministic cached val pair, in perhex's own 4-tuple layout.

    perhex stores ``(oracle_logits, uniform_logits, y, mask)`` — verified
    against the unpacking at nb_oracle_perhex.py rather than assumed. The
    uniform slot is unused by these functions but must be present or the
    positional unpack fails.
    """
    g = torch.Generator().manual_seed(seed)
    oracle_logits = torch.randn(1, C, L, generator=g, dtype=torch.float32)
    uniform_logits = torch.zeros(1, C, L, dtype=torch.float32)
    y = torch.randint(0, 5, (1, C, L), generator=g).float()
    mask = torch.ones(1, L, dtype=torch.bool)
    return oracle_logits, uniform_logits, y, mask


def _hexes(L, hexamer=0):
    """perhex stores hex_per_pair[di] as a (hex_idx, hex_valid) TUPLE.

    Verified against the unpacking inside fit_perhex_grid rather than assumed —
    an earlier version of this fixture passed a bare array and produced zero
    position counts, which looked like a production bug and was not one.
    """
    hex_idx = np.full(L, hexamer, dtype=np.int64)
    hex_valid = np.ones(L, dtype=bool)
    return hex_idx, hex_valid


class TestNbNllPositions:
    def test_matches_closed_form(self):
        """nb_nll_positions must equal the textbook NB NLL, elementwise."""
        y = np.array([0.0, 1.0, 5.0])
        r = np.array([2.0, 2.0, 2.0])
        mu = np.array([1.0, 1.0, 3.0])
        got = nb_nll_positions(y, r, mu)

        from scipy.special import gammaln
        want = -(
            gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
            + r * np.log(r / (r + mu))
            + y * np.log(mu / (r + mu))
        )
        assert np.allclose(got, want, rtol=0, atol=1e-12), (got, want)

    def test_larger_r_changes_result(self):
        """Discriminating: the dispersion argument must actually be used."""
        y = np.array([3.0]); mu = np.array([2.0])
        a = nb_nll_positions(y, np.array([1.0]), mu)
        b = nb_nll_positions(y, np.array([100.0]), mu)
        assert not np.allclose(a, b), "r had no effect — function ignores dispersion"

    def test_deterministic_bitwise(self):
        y = np.array([0.0, 2.0, 7.0]); r = np.array([3.0, 3.0, 3.0])
        mu = np.array([1.5, 2.5, 3.5])
        assert (nb_nll_positions(y, r, mu) == nb_nll_positions(y, r, mu)).all()


class TestScoreValPerhex:
    def test_falls_back_for_unfitted_hexamers(self):
        """Hexamers with no fitted r must use the scalar fallback, not NaN."""
        pairs = {0: _pair(1)}
        hex_per_pair = {0: _hexes(16)}
        hex_r = np.full(4096, np.nan)          # nothing fitted
        loss_fn = make_oracle_loss_fn()
        out = score_val_perhex(pairs, hex_per_pair, hex_r, loss_fn, 7.5)
        assert np.isfinite(out), "unfitted hexamers leaked NaN into the score"

    def test_fallback_value_is_used(self):
        """Discriminating: changing the fallback must change the score."""
        pairs = {0: _pair(2)}
        hex_per_pair = {0: _hexes(16)}
        hex_r = np.full(4096, np.nan)
        loss_fn = make_oracle_loss_fn()
        a = score_val_perhex(pairs, hex_per_pair, hex_r, loss_fn, 2.0)
        b = score_val_perhex(pairs, hex_per_pair, hex_r, loss_fn, 900.0)
        assert a != b, "scalar_r_fallback was ignored"

    def test_deterministic_bitwise(self):
        pairs = {0: _pair(3), 1: _pair(4)}
        hex_per_pair = {i: _hexes(16) for i in (0, 1)}
        hex_r = np.full(4096, 5.0)
        loss_fn = make_oracle_loss_fn()
        a = score_val_perhex(pairs, hex_per_pair, hex_r, loss_fn, 7.5)
        b = score_val_perhex(pairs, hex_per_pair, hex_r, loss_fn, 7.5)
        assert a == b, "scoring is not bitwise reproducible"


class TestFitPerhexGrid:
    def test_respects_min_positions(self):
        """Below the threshold, EVERY hexamer falls back to the scalar r.

        Note the fallback is the scalar value, not NaN — verified from the
        function's own output ("4096 fallback to scalar r=7.5") rather than
        assumed. That is the safer design: an unfitted hexamer cannot leak a
        NaN into the score.
        """
        pairs = {0: _pair(5)}
        hex_per_pair = {0: _hexes(16)}
        hex_r, counts, info = fit_perhex_grid(pairs, hex_per_pair,
                                              scalar_r=7.5,
                                              min_positions=10_000)
        # The fallback is exp(log(scalar_r)), NOT the literal scalar_r: the
        # function works in log space and exponentiates, so 7.5 round-trips to
        # 7.499999999999999. Pinned exactly, because "tidying" the code to
        # store r directly would shift every fallback hexamer by 1 ulp and a
        # bitwise acceptance test would then flag the whole artifact.
        expected = np.exp(np.log(7.5))
        assert expected != 7.5, "round-trip no longer lossy; update this test"
        assert (hex_r == expected).all(), "a hexamer was fitted below the threshold"
        assert not np.isnan(hex_r).any(), "fallback leaked NaN"
        assert counts.max() < 10_000
        assert info["min_positions_threshold"] == 10_000

    def test_fits_when_threshold_is_met(self):
        """Discriminating counterpart: lowering the threshold produces a fit.

        Exactly one hexamer has positions in this fixture, so exactly one
        should depart from the scalar fallback.
        """
        pairs = {0: _pair(6)}
        hex_per_pair = {0: _hexes(16, hexamer=0)}
        hex_r, counts, _ = fit_perhex_grid(pairs, hex_per_pair, scalar_r=7.5,
                                           min_positions=1)
        assert counts[0] == 16, "the populated hexamer lost its positions"
        assert (counts[1:] == 0).all(), "positions leaked to other hexamers"
        assert np.isfinite(hex_r[0])
        assert (hex_r[1:] == np.exp(np.log(7.5))).all(), \
            "unpopulated hexamers should fall back to exp(log(scalar_r))"

    def test_deterministic_bitwise(self):
        pairs = {0: _pair(7)}
        hex_per_pair = {0: _hexes(16)}
        a, _, _ = fit_perhex_grid(pairs, hex_per_pair, 7.5, 1)
        b, _, _ = fit_perhex_grid(pairs, hex_per_pair, 7.5, 1)
        assert np.array_equal(a, b, equal_nan=True), "fit is not reproducible"


class TestV2ReferenceIsRead:
    """The six REF_* values must come from v2's artifact, not be transcribed."""

    def test_ref_values_match_published_artifact(self):
        import json
        from scripts.nb_oracle_perhex import (
            V2_JSON, REF_SCALAR_ORACLE, REF_UNIFORM, REF_GAP,
            REF_TRAINED_KEN, REF_TRAINED_HYBRID,
            REF_UNTRAINED_KEN, REF_UNTRAINED_HYBRID,
        )
        try:
            with open(V2_JSON) as f:
                d = json.load(f)
        except FileNotFoundError:
            pytest.skip("v2 artifact not present on this host")
        m = d["models"]
        assert REF_SCALAR_ORACLE == d["oracle_nb_nll"]
        assert REF_UNIFORM == d["uniform_nb_nll"]
        assert REF_GAP == d["gap_uniform_minus_oracle"]
        assert REF_TRAINED_KEN == m["trained_ken"]["nb_loss"]
        assert REF_TRAINED_HYBRID == m["trained_hybrid"]["nb_loss"]
        assert REF_UNTRAINED_KEN == m["untrained_ken"]["nb_loss"]
        assert REF_UNTRAINED_HYBRID == m["untrained_hybrid"]["nb_loss"]
