"""Shared NB-offset scoring helper for the oracle scripts.

Centralises the pattern used by nb_oracle_v2.py, nb_oracle_perhex.py, and
nb_oracle_sweep.py: evaluate the frozen-core MaskedNegativeBinomialOffsetNLLLoss
at a given scalar log_r across a set of cached (logits, y, mask) tuples.

This is plumbing, not part of the statistical specification.  It lives here
(not in background_model_core.py) by explicit owner ruling — see §4 of
docs/pending/oracle_compute_render_split.md.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

# The loss config that all oracle scripts and training runs share.
ORACLE_LOSS_KWARGS = dict(max_dispersion_ratio=2.0, clamp_margin=1.0)


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
