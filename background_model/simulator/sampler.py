"""Fragment sampler — Step 5 of the simulator.

Draws fragments from ``build_region_weights``'s factors:

    1. s  ~ Bernoulli(½)
    2. c5 ~ start_s[hex(c5)] / S_s
    3. L  ~ E_s(c5,L) / Z_s(c5)   over L = 25..180

The draw is vectorised: all strand choices are drawn first, then all
plus-strand (c5, L) pairs, then all minus-strand pairs.  This consumes
the RNG stream in a different order from the original per-fragment loop
(commit ``c7848e5``), so outputs are **not** bit-identical given the same
seed, but the joint distribution ``P(s) · P(c5|s) · P(L|c5,s)`` is
identical — each fragment is drawn independently from the same
conditional distributions.

The sampler draws from the SAME factors that ``build_region_weights`` computes,
not a reimplementation.  This single shared call is the only thing preventing
sampler/scorer drift (design doc § Step 4, "one implementation" constraint).

Output: per-region fragment tuples ``(start, stop, strand_str)`` in BED
coordinates (0-based half-open ``[start, stop)``), ready for serialisation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from background_model.simulator.weights import (
    L_MAX,
    L_MIN,
    N_LENGTHS,
    HexamerTables,
    RegionWeights,
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
    region_weights: Optional[RegionWeights] = None,
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
    region_weights : RegionWeights, optional
        An already-built ``RegionWeights`` for THIS region, to sample from
        instead of rebuilding it.  ``build_region_weights`` is ~63 ms per
        call at ``region_len`` 2560 and dominates the simulator: a caller
        that already built ``w`` (to check the ``Sum_Omega w = 1``
        invariants, say) would otherwise pay for a second, identical build,
        which measured as 40% of the whole per-region cost.

        Passing it is purely an elision of recomputation -- the object is
        the same one this function would have constructed -- so the draw is
        bit-identical, including the RNG stream.  It is the caller's
        responsibility that the weights were built from the same
        ``hex_fwd`` / ``hex_rc`` / ``cum_gc`` / ``valid`` / tables as are
        passed here; a mismatch would sample from one region's weights
        while reporting another's coordinates, and no normalisation
        invariant would notice (``Sum_Omega w = 1`` is a property of the
        weights, not of the draw).

    Returns
    -------
    starts : ndarray, shape (n_fragments,), int
        BED start (0-based, the lower genomic coordinate).
    stops : ndarray, shape (n_fragments,), int
        BED stop (exclusive, the higher genomic coordinate).
    strands : ndarray, shape (n_fragments,), str
        ``"+"`` or ``"-"`` for each fragment.
    """
    if region_weights is None:
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
    else:
        rw = region_weights
        # Guard: verify the pre-built weights are consistent with the
        # arrays passed to this call.  A mismatch means the caller built
        # weights from one region and is drawing fragments for another —
        # the draw would sample from one region's distribution while
        # reporting another region's coordinates, and Sum_Omega w = 1 (a
        # property of the weights) would not notice.
        n_sites = region_len + 1
        if rw.w_plus.shape != (n_sites, N_LENGTHS):
            raise ValueError(
                f"region_weights.w_plus has shape {rw.w_plus.shape} but "
                f"region_len={region_len} requires ({n_sites}, {N_LENGTHS}). "
                f"The weights were built for a different region."
            )
        # S_plus and S_minus depend on hex_fwd, hex_rc, valid, and the
        # hex tables — all of which are region-specific.  Both checks are
        # needed: S_plus validates the plus-strand start weights
        # (start_fwd[hex_fwd]), S_minus validates the minus-strand start
        # weights (start_rev[hex_rc]).  A caller passing hex_fwd from
        # region A with hex_rc from region B would pass one check but
        # fail the other.
        start_vals_plus_check = np.where(valid, hex_tables.start_fwd[hex_fwd], 0.0)
        Z_plus_check = rw.w_plus.sum(axis=1)
        S_plus_check = float(start_vals_plus_check[Z_plus_check > 0].sum())
        if abs(S_plus_check - rw.S_plus) > 1e-10:
            raise ValueError(
                f"region_weights.S_plus={rw.S_plus} but the provided "
                f"hex_fwd/hex_tables/valid arrays give S_plus={S_plus_check}. "
                f"The weights were built for a different region."
            )
        start_vals_minus_check = np.where(valid, hex_tables.start_rev[hex_rc], 0.0)
        Z_minus_check = rw.w_minus.sum(axis=1)
        S_minus_check = float(start_vals_minus_check[Z_minus_check > 0].sum())
        if abs(S_minus_check - rw.S_minus) > 1e-10:
            raise ValueError(
                f"region_weights.S_minus={rw.S_minus} but the provided "
                f"hex_rc/hex_tables/valid arrays give S_minus={S_minus_check}. "
                f"The weights were built for a different region."
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

    # ── Vectorised draw ────────────────────────────────────────────────
    # All strand choices, then all plus-strand (c5, L), then minus.
    # Each fragment is drawn independently from the same conditional
    # distributions as the original per-fragment loop.

    starts = np.empty(n_fragments, dtype=np.int64)
    stops = np.empty(n_fragments, dtype=np.int64)
    strands = np.empty(n_fragments, dtype="U1")

    # 1. s ~ Bernoulli(½) for all fragments at once
    is_plus = rng.random(n_fragments) < 0.5
    n_plus = int(is_plus.sum())
    n_minus = n_fragments - n_plus

    # 2-3. Plus strand: c5 ~ start_prob_plus, then L ~ cond_L_plus[c5]
    if n_plus > 0:
        c5_plus = rng.choice(n_sites, size=n_plus, p=start_prob_plus)
        # CDF inversion: draw uniform, find first CDF bin that exceeds it.
        # (cum <= u).sum() matches numpy's searchsorted(side='right').
        u_L = rng.random(n_plus)
        cum = np.cumsum(cond_L_plus[c5_plus], axis=1)
        li_plus = np.minimum(
            (cum <= u_L[:, np.newaxis]).sum(axis=1), N_LENGTHS - 1
        )
        L_plus = Ls[li_plus]
        starts[is_plus] = c5_plus
        stops[is_plus] = c5_plus + L_plus
        strands[is_plus] = "+"

    # 2-3. Minus strand: c5 ~ start_prob_minus, then L ~ cond_L_minus[c5]
    if n_minus > 0:
        c5_minus = rng.choice(n_sites, size=n_minus, p=start_prob_minus)
        u_L = rng.random(n_minus)
        cum = np.cumsum(cond_L_minus[c5_minus], axis=1)
        li_minus = np.minimum(
            (cum <= u_L[:, np.newaxis]).sum(axis=1), N_LENGTHS - 1
        )
        L_minus = Ls[li_minus]
        starts[~is_plus] = c5_minus - L_minus
        stops[~is_plus] = c5_minus
        strands[~is_plus] = "-"

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
