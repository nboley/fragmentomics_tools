"""Unit tests for scripts/_oracle_scoring.py.

Tests the shared NB-offset scoring helper that all oracle scripts use.
Follows the house style from test_nb_oracle_selection.py: import from
``scripts.``, construct discriminating tests that would fail against
broken implementations.
"""

import numpy as np
import pytest
import torch

from scripts._oracle_scoring import (
    ORACLE_DISPERSION_WINDOW_SIZE,
    ORACLE_LOSS_KWARGS,
    eval_loss_at_log_r,
    make_oracle_loss_fn,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

def _make_pair(C, L, count_scale=10, seed=None):
    """Create a single (logits, y_t, m_t) tuple with controlled counts."""
    if seed is not None:
        rng = np.random.RandomState(seed)
        logits = torch.from_numpy(rng.randn(1, C, L).astype(np.float32)) * 0.1
        y = torch.from_numpy(rng.randint(0, count_scale, (1, C, L)).astype(np.float32))
    else:
        logits = torch.randn(1, C, L) * 0.1
        y = torch.randint(0, count_scale, (1, C, L)).float()
    m = torch.ones(1, L, dtype=torch.bool)
    return logits, y, m


# ── make_oracle_loss_fn ──────────────────────────────────────────────────

class TestMakeOracleLossFn:
    """Verify make_oracle_loss_fn returns a properly configured loss."""

    def test_returns_correct_class(self):
        loss_fn = make_oracle_loss_fn()
        assert type(loss_fn).__name__ == "MaskedNegativeBinomialOffsetNLLLoss"

    def test_carries_oracle_kwargs(self):
        """The loss object must carry every key from ORACLE_LOSS_KWARGS."""
        loss_fn = make_oracle_loss_fn()
        for key, expected in ORACLE_LOSS_KWARGS.items():
            actual = getattr(loss_fn, key)
            assert actual == expected, (
                f"loss_fn.{key} = {actual}, expected {expected}"
            )

    def test_kwargs_do_not_include_dispersion_window_size(self):
        """dispersion_window_size is a model arg, not a loss arg."""
        assert "dispersion_window_size" not in ORACLE_LOSS_KWARGS, (
            "ORACLE_LOSS_KWARGS should not contain dispersion_window_size; "
            "it is a model constructor arg, not a loss parameter"
        )

    def test_dispersion_window_size_constant_is_one(self):
        """The oracle protocol uses per-position dispersion (window=1)."""
        assert ORACLE_DISPERSION_WINDOW_SIZE == 1


# ── eval_loss_at_log_r ──────────────────────────────────────────────────

class TestEvalLossAtLogR:
    """Verify eval_loss_at_log_r computes the mean correctly."""

    C, L = 18, 64

    def test_returns_mean_not_sum(self):
        """The function must return the MEAN, not the sum or first/last."""
        loss_fn = make_oracle_loss_fn()
        p0 = _make_pair(self.C, self.L, count_scale=10, seed=0)
        p1 = _make_pair(self.C, self.L, count_scale=100, seed=1)

        mean_loss = eval_loss_at_log_r(5.0, {0: p0, 1: p1}, loss_fn)

        # Evaluate each individually
        loss_0 = eval_loss_at_log_r(5.0, {0: p0}, loss_fn)
        loss_1 = eval_loss_at_log_r(5.0, {1: p1}, loss_fn)

        expected_mean = (loss_0 + loss_1) / 2.0
        assert abs(mean_loss - expected_mean) < 1e-10, (
            f"Expected mean {expected_mean:.15f}, got {mean_loss:.15f}"
        )
        # Discriminator: verify it is NOT the sum
        assert abs(mean_loss - (loss_0 + loss_1)) > 0.01, (
            f"Loss {mean_loss} looks like the sum, not the mean"
        )

    def test_returns_mean_not_first_or_last(self):
        """Three pairs with different counts — mean differs from first/last."""
        loss_fn = make_oracle_loss_fn()
        pairs = {}
        for i in range(3):
            pairs[i] = _make_pair(self.C, self.L,
                                  count_scale=10 * (i + 1), seed=i + 10)

        mean_loss = eval_loss_at_log_r(5.0, pairs, loss_fn)
        first_loss = eval_loss_at_log_r(5.0, {0: pairs[0]}, loss_fn)
        last_loss = eval_loss_at_log_r(5.0, {2: pairs[2]}, loss_fn)

        assert mean_loss != first_loss, "mean_loss equals first pair's loss"
        assert mean_loss != last_loss, "mean_loss equals last pair's loss"

    def test_respects_sorted_iteration(self):
        """Result must be identical regardless of dict insertion order."""
        loss_fn = make_oracle_loss_fn()
        pairs_fwd = {}
        pairs_rev = {}
        for i in range(5):
            p = _make_pair(self.C, self.L, count_scale=20, seed=i + 20)
            pairs_fwd[i] = p
            pairs_rev[4 - i] = p

        loss_fwd = eval_loss_at_log_r(3.0, pairs_fwd, loss_fn)
        loss_rev = eval_loss_at_log_r(3.0, pairs_rev, loss_fn)
        assert loss_fwd == loss_rev, (
            f"Different insertion orders gave different results: "
            f"{loss_fwd} vs {loss_rev}"
        )

    def test_log_r_affects_loss(self):
        """Different log_r values must produce different losses."""
        loss_fn = make_oracle_loss_fn()
        pairs = {0: _make_pair(self.C, self.L, count_scale=50, seed=30)}

        loss_low_r = eval_loss_at_log_r(0.0, pairs, loss_fn)
        loss_high_r = eval_loss_at_log_r(10.0, pairs, loss_fn)
        assert loss_low_r != loss_high_r, (
            "Loss should vary with log_r; got identical values"
        )

    def test_single_pair(self):
        """Single-pair case: mean of one element equals the element."""
        loss_fn = make_oracle_loss_fn()
        p = _make_pair(self.C, self.L, count_scale=30, seed=40)
        loss = eval_loss_at_log_r(5.0, {42: p}, loss_fn)
        assert isinstance(loss, float)
        assert np.isfinite(loss)
