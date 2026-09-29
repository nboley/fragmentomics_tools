"""Fragment sampler — Step 5 of the simulator.

Draws fragments from ``build_region_weights``'s factors using the sequential
sampling procedure from the design doc (§ Step 5):

    1. s  ~ Bernoulli(½)
    2. c5 ~ start_s[hex(c5)] / S_s
    3. c3 ~ E_s(c5,L) / Z_s(c5)   over L = 25..180,  c3 = c5 + σ·L

The sampler draws from the SAME factors that ``build_region_weights`` computes,
not a reimplementation.  This single shared call is the only thing preventing
sampler/scorer drift (design doc § Step 4, "one implementation" constraint).

Output: per-region fragment tuples ``(start, stop, strand_str)`` in BED
coordinates (0-based half-open ``[start, stop)``), ready for serialisation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from background_model.simulator.weights import (
    L_MAX,
    L_MIN,
    N_LENGTHS,
    HexamerTables,
    build_region_weights,
)


def draw_fragments_for_region(
    *,
    hex_fwd: np.ndarray,
    hex_rc: np.ndarray,
    cum_gc: np.ndarray,
    valid: np.ndarray,
    hex_tables: HexamerTables,
    marginal_fl: np.ndarray,
    predict_lut: np.ndarray,
    region_len: int,
    n_fragments: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw ``n_fragments`` from the generative model for one region.

    Calls ``build_region_weights`` to get the normalised weight factors, then
    samples using the sequential procedure.

    Parameters
    ----------
    hex_fwd, hex_rc, cum_gc, valid
        Per-region precompute arrays (from ``precompute_region``).
    hex_tables : HexamerTables
        The four hexamer weight tables.
    marginal_fl : ndarray, shape (N_LENGTHS,)
        Marginal fragment-length distribution.
    predict_lut : ndarray, shape (N_LENGTHS, N_GC_BINS)
        Capture-surface LUT.
    region_len : int
        Region length in bp.
    n_fragments : int
        Number of fragments to draw.
    rng : numpy Generator
        RNG instance for reproducibility.

    Returns
    -------
    starts : ndarray, shape (n_fragments,), int
        BED start (0-based, the lower genomic coordinate).
    stops : ndarray, shape (n_fragments,), int
        BED stop (exclusive, the higher genomic coordinate).
    strands : ndarray, shape (n_fragments,), str
        ``"+"`` or ``"-"`` for each fragment.
    """
    rw = build_region_weights(
        hex_fwd=hex_fwd,
        hex_rc=hex_rc,
        cum_gc=cum_gc,
        hex_tables=hex_tables,
        marginal_fl=marginal_fl,
        predict_lut=predict_lut,
        region_len=region_len,
        valid=valid,
    )

    # Precompute per-strand sampling distributions from the weight factors.
    # For plus: start_vals_plus[c5] = start_fwd[hex_fwd[c5]] (zeroed where invalid)
    # For minus: start_vals_minus[c5] = start_rev[hex_rc[c5]] (zeroed where invalid)
    n_sites = region_len + 1

    start_vals_plus = np.where(valid, hex_tables.start_fwd[hex_fwd], 0.0)
    start_vals_minus = np.where(valid, hex_tables.start_rev[hex_rc], 0.0)

    # Z_s(c5) = sum over L of w_plus/w_minus unnormalised E values.
    # But we can get the conditional E_s(c5,L) / Z_s(c5) from the normalised
    # weights directly: w[c5, :] = 0.5 * start/S * E/Z, and the conditional
    # over L given c5 is E(c5,L)/Z(c5) = w[c5,:] / w[c5,:].sum() when w[c5,:].sum()>0.
    #
    # Similarly c5 ~ start_s[hex(c5)] / S_s.

    # Plus strand start distribution: start_fwd[hex_fwd[c5]] restricted to
    # positions with Z_plus(c5) > 0.
    Z_plus = rw.w_plus.sum(axis=1)  # (n_sites,)
    has_frags_plus = Z_plus > 0
    start_prob_plus = np.where(has_frags_plus, start_vals_plus, 0.0)
    total_start_plus = start_prob_plus.sum()
    if total_start_plus > 0:
        start_prob_plus /= total_start_plus

    Z_minus = rw.w_minus.sum(axis=1)
    has_frags_minus = Z_minus > 0
    start_prob_minus = np.where(has_frags_minus, start_vals_minus, 0.0)
    total_start_minus = start_prob_minus.sum()
    if total_start_minus > 0:
        start_prob_minus /= total_start_minus

    # Per-c5 conditional over L: w[c5, :] / sum(w[c5, :])
    # Precompute these as 2D arrays. For positions with no fragments, leave as 0.
    cond_L_plus = np.zeros_like(rw.w_plus)
    mask_p = Z_plus > 0
    cond_L_plus[mask_p] = rw.w_plus[mask_p] / Z_plus[mask_p, np.newaxis]

    cond_L_minus = np.zeros_like(rw.w_minus)
    mask_m = Z_minus > 0
    cond_L_minus[mask_m] = rw.w_minus[mask_m] / Z_minus[mask_m, np.newaxis]

    Ls = np.arange(L_MIN, L_MAX + 1)  # (N_LENGTHS,)

    starts = np.empty(n_fragments, dtype=np.int64)
    stops = np.empty(n_fragments, dtype=np.int64)
    strands = np.empty(n_fragments, dtype="U1")

    for i in range(n_fragments):
        # 1. s ~ Bernoulli(1/2)
        is_plus = rng.random() < 0.5

        if is_plus:
            # 2. c5 ~ start_fwd[hex_fwd[c5]] / S_plus
            c5 = rng.choice(n_sites, p=start_prob_plus)
            # 3. L ~ E_plus(c5, L) / Z_plus(c5)
            li = rng.choice(N_LENGTHS, p=cond_L_plus[c5])
            L = Ls[li]
            # Plus: c5 = p, c3 = p + L => start=c5, stop=c5+L
            starts[i] = c5
            stops[i] = c5 + L
            strands[i] = "+"
        else:
            # 2. c5 ~ start_rev[hex_rc[c5]] / S_minus
            c5 = rng.choice(n_sites, p=start_prob_minus)
            # 3. L ~ E_minus(c5, L) / Z_minus(c5)
            li = rng.choice(N_LENGTHS, p=cond_L_minus[c5])
            L = Ls[li]
            # Minus: c5 = p+L, c3 = p => start=c5-L, stop=c5
            starts[i] = c5 - L
            stops[i] = c5
            strands[i] = "-"

    return starts, stops, strands


# ── per-region target counts ─────────────────────────────────────────────

# Layer 1 constants: fixed fragment count per region (design doc § Step 5).
_REGION_COUNTS = {
    2560: 54,
    1536: 37,
}


def target_count_for_region(region_len: int) -> int:
    """Return the per-region target fragment count for Layer 1.

    Raises KeyError if ``region_len`` is not a known Layer 1 geometry.
    """
    try:
        return _REGION_COUNTS[region_len]
    except KeyError:
        raise KeyError(
            f"No Layer 1 target count for region_len={region_len}. "
            f"Known: {_REGION_COUNTS}"
        ) from None
