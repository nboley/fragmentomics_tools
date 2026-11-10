"""Shared NB-offset scoring helper for the oracle scripts.

Centralises the pattern used by nb_oracle_v2_compute.py, nb_oracle_perhex.py,
and nb_oracle_sweep.py: evaluate the frozen-core
MaskedNegativeBinomialOffsetNLLLoss at a given scalar log_r across a set of
cached (logits, y, mask) tuples, plus the anchor selection built on top of it.

This is plumbing, not part of the statistical specification.  It lives here
(not in background_model_core.py) by explicit owner ruling — see §4 of
docs/pending/oracle_compute_render_split.md.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from scipy.optimize import minimize_scalar

# The loss config that all oracle scripts and training runs share.
ORACLE_LOSS_KWARGS = dict(max_dispersion_ratio=2.0, clamp_margin=1.0)

# dispersion_window_size is a MODEL constructor arg, not a loss parameter.
# Recorded here as the single source for the oracle protocol config: the
# oracle uses per-position dispersion (window_size=1), matching the training
# runs' frozen config.
ORACLE_DISPERSION_WINDOW_SIZE = 1


def make_oracle_loss_fn():
    """Construct the frozen-core NB-offset loss with the standard oracle config."""
    from background_model_core import MaskedNegativeBinomialOffsetNLLLoss
    return MaskedNegativeBinomialOffsetNLLLoss(**ORACLE_LOSS_KWARGS)


# Type alias for cached val pairs: di -> (logits, y_t, m_t)
OraclePairs = Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def eval_loss_at_log_r(
    log_r_val: float,
    pairs: OraclePairs,
    loss_fn,
) -> float:
    """Mean NB-offset loss across all cached pairs at a given scalar log_r.

    Args:
        log_r_val: scalar log-dispersion value (broadcast to all positions).
        pairs: dict mapping index -> (logits, y_t, m_t), where each tensor
            has a leading batch dim of 1.
        loss_fn: a MaskedNegativeBinomialOffsetNLLLoss instance.

    Returns:
        float — the mean loss across all pairs.
    """
    losses = []
    with torch.no_grad():
        for di in sorted(pairs):
            logits, y_t, m_t = pairs[di]
            B, C, L = logits.shape
            ld = torch.full((1, C, L), log_r_val, dtype=torch.float32)
            loss_val = loss_fn(logits, ld, y_t, m_t).item()
            losses.append(loss_val)
    return float(np.mean(losses))


def select_min_over_union(loss_fn, grid, bounds):
    """Select the minimum loss over the union of grid evaluations and scipy.

    Evaluates loss_fn at every point in grid, runs scipy minimize_scalar
    on the same bounds, and returns whichever produced the lower loss.

    Returns (best_loss, best_log_r).
    """
    grid_losses = np.array([loss_fn(lr) for lr in grid])
    grid_best_idx = int(np.argmin(grid_losses))
    grid_best_loss = float(grid_losses[grid_best_idx])
    grid_best_log_r = float(grid[grid_best_idx])

    result = minimize_scalar(
        loss_fn, bounds=bounds, method="bounded",
        options={"xatol": 0.01},
    )
    scipy_loss = result.fun
    scipy_log_r = result.x

    if grid_best_loss < scipy_loss:
        return grid_best_loss, grid_best_log_r
    return scipy_loss, scipy_log_r
