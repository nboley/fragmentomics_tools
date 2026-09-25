"""Regression test for the NB oracle anchor-selection logic.

The min-over-union selection (sweep grid ∪ scipy result) was introduced in
commit 7451548 to fix a bug where scipy's bounded optimizer converged to a
local bump on the loss plateau while a lower grid point existed.

This test constructs a synthetic loss surface that reproduces that failure
mode and verifies the selection logic returns the true argmin.

The selection function is imported from scripts.nb_oracle_v2 — a change in
production that breaks the selection will break these tests.
"""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from scripts.nb_oracle_v2 import select_min_over_union


def _synthetic_loss(log_r: float) -> float:
    """Loss surface with a smooth bowl AND a sharp dip that scipy misses.

    The smooth bowl has its minimum at log_r ~ 3.5 (loss ~ 4.0).
    A sharp Gaussian dip at log_r = 1.0 brings the loss to ~ 3.9875.
    The dip is narrow enough (sigma=0.01) that scipy's bounded optimizer
    never samples near it and converges to the smooth bowl instead.
    """
    smooth = 4.0 + 0.01 * (log_r - 3.5) ** 2
    dip = -0.075 * np.exp(-((log_r - 1.0) / 0.01) ** 2)
    return smooth + dip


def _select_scipy_only(loss_fn, bounds):
    """Old behavior: take scipy's result, ignore the grid."""
    result = minimize_scalar(
        loss_fn, bounds=bounds, method="bounded",
        options={"xatol": 0.01},
    )
    return result.fun, result.x


class TestMinOverUnionSelection:
    """Anchor-selection logic: min over union of (grid, scipy)."""

    GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    BOUNDS = (0.0, 5.0)

    def test_scipy_misses_true_minimum(self):
        """Scipy alone converges to the smooth bowl, not the sharp dip."""
        loss, log_r = _select_scipy_only(_synthetic_loss, self.BOUNDS)
        # Scipy should find the smooth bowl minimum near log_r ~ 3.5
        assert log_r > 2.0, (
            f"scipy unexpectedly found the dip (log_r={log_r:.4f}); "
            f"test setup is invalid"
        )
        true_min = _synthetic_loss(1.0)
        assert loss > true_min, (
            f"scipy loss ({loss:.9f}) should be above true min ({true_min:.9f})"
        )

    def test_min_over_union_finds_true_minimum(self):
        """Min-over-union correctly selects the grid point at the global min."""
        loss, log_r = select_min_over_union(
            _synthetic_loss, self.GRID, self.BOUNDS,
        )
        true_min = _synthetic_loss(1.0)
        assert loss == true_min, (
            f"Min-over-union should find true minimum ({true_min:.9f}), "
            f"got {loss:.9f}"
        )
        assert abs(log_r - 1.0) < 0.01, (
            f"Should select log_r=1.0, got {log_r:.4f}"
        )

    def test_improvement_is_material(self):
        """The difference between old and new selection is not rounding noise."""
        old_loss, _ = _select_scipy_only(_synthetic_loss, self.BOUNDS)
        new_loss, _ = select_min_over_union(
            _synthetic_loss, self.GRID, self.BOUNDS,
        )
        improvement = old_loss - new_loss
        assert improvement > 0.01, (
            f"Improvement ({improvement:.6f}) must be material (>0.01) "
            f"to prove the test discriminates"
        )

    def test_scipy_wins_when_it_should(self):
        """When scipy finds the true minimum, min-over-union still works."""
        def unimodal(log_r):
            return 4.0 + 0.01 * (log_r - 2.3) ** 2

        coarse_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        loss, log_r = select_min_over_union(
            unimodal, coarse_grid, self.BOUNDS,
        )
        # Scipy should find log_r ~ 2.3, which is between grid points
        assert abs(log_r - 2.3) < 0.1, (
            f"Should find minimum near 2.3, got {log_r:.4f}"
        )
        # And it should beat the nearest grid point (2.0)
        grid_at_2 = unimodal(2.0)
        assert loss < grid_at_2, (
            f"Scipy result ({loss:.9f}) should beat grid at 2.0 ({grid_at_2:.9f})"
        )
