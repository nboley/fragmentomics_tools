"""Region weight builder for the cfDNA fragment simulator.

Implements Step 4 of docs/pending/simulator_and_fragment_nll.md:
``build_region_weights`` computes the fully normalised probability ``w``
over the generative domain Omega for one region.

Conventions (Appendix D of the design doc):
  - A fragment occupying bases [p, p+L) has **cut sites** at p and p+L.
    (NOT endpoint bases p and p+L-1.)
  - **Plus strand:**  c5 = p,   c3 = p + L.   sigma = +1.
  - **Minus strand:** c5 = p+L, c3 = p.       sigma = -1.
    The 5' end is at the higher coordinate; hexamers read reverse-complemented.
  - Length L = |c3 - c5|, restricted to L_MIN..L_MAX (25..180), 156 values.
  - GC is over the genomic span, strand-independent:
    gc_pct = 100 * (cum_gc[max(c5,c3)] - cum_gc[min(c5,c3)]) / L.
  - ``c3(L) = c5 + sigma * L``.

The edge rule: Z_s(c5) sums only those L whose c3(L) stays in
[0, region_len].  Off the region, hex(c3) is undefined.

Hexamer index conventions:
  - ``hex_fwd[c]``: forward hexamer at cut site c (6-mer spanning the cut).
  - ``hex_rc[c]``: reverse-complement hexamer at the same position.
  - Plus strand fragments read forward: both c5 and c3 use ``hex_fwd``.
  - Minus strand fragments read RC: both c5 and c3 use ``hex_rc``.
  - The four tables ``{start,end} × {fwd,rev}`` are UNTIED; strand selects
    which ``(start_s, end_s)`` pair applies.

Invariants (Appendix A):
  - sum_Omega w = 1 exactly.
  - Strand marginal is exactly 1/2 for each strand.
  - |Omega| = 2 * sum_{L=L_MIN}^{L_MAX} (region_len - L + 1).
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import numpy as np

# ── constants ────────────────────────────────────────────────────────────

L_MIN: int = 25
"""Minimum fragment length (inclusive). Lower bound of the capture surface."""

L_MAX: int = 180
"""Maximum fragment length (inclusive). Upper bound of the capture surface."""

N_LENGTHS: int = L_MAX - L_MIN + 1  # 156

SIGMA_PLUS: int = +1
SIGMA_MINUS: int = -1


# ── orientation / geometry helpers ───────────────────────────────────────

def c3_from_c5(c5: int, L: int, sigma: int) -> int:
    """Compute the 3' cut site from the 5' cut site, length, and strand sign.

    ``sigma = +1`` for plus strand, ``-1`` for minus strand.
    """
    return c5 + sigma * L


def fragment_base_range(c5: int, c3: int):
    """Return (p, L) for the half-open base range [p, p+L) occupied by the fragment.

    Works for either strand: p = min(c5, c3), L = |c3 - c5|.
    """
    p = min(c5, c3)
    L = abs(c3 - c5)
    return p, L


def gc_pct(c5: int, c3: int, cum_gc: np.ndarray) -> float:
    """GC percentage over the genomic span of a fragment.

    Strand-independent: always uses [min(c5,c3), max(c5,c3)) which is [p, p+L).
    ``cum_gc`` is the cumulative GC count array of length ``region_len + 1``
    where ``cum_gc[i]`` = number of G/C bases in positions [0, i).
    """
    lo = min(c5, c3)
    hi = max(c5, c3)
    L = hi - lo
    if L == 0:
        return 0.0
    return 100.0 * (cum_gc[hi] - cum_gc[lo]) / L


def omega_size(region_len: int, L_min: int = L_MIN, L_max: int = L_MAX) -> int:
    """Cardinality of the generative domain Omega.

    |Omega| = 2 * sum_{L=L_min}^{L_max} (region_len - L + 1).

    For each strand and length L, the number of valid c5 positions is
    (region_len - L + 1).
    Plus strand: c5 in [0, region_len - L], that's region_len - L + 1 values.
    Minus strand: c5 in [L, region_len], that's region_len - L + 1 values.
    Same count for both strands — hence the factor of 2.
    """
    total = 0
    for L in range(L_min, L_max + 1):
        n_positions = region_len - L + 1
        if n_positions <= 0:
            break
        total += n_positions
    return 2 * total


# ── weight builder result ────────────────────────────────────────────────

class RegionWeights(NamedTuple):
    """Result of ``build_region_weights`` for one region.

    Attributes
    ----------
    w_plus : ndarray, shape (region_len + 1, N_LENGTHS)
        ``w_plus[c5, li]`` = w(c5, c3, s=+) where ``li = L - L_MIN`` and
        ``c3 = c5 + L``.  Zero where c3 > region_len (edge truncation).
    w_minus : ndarray, shape (region_len + 1, N_LENGTHS)
        ``w_minus[c5, li]`` = w(c5, c3, s=-) where ``c3 = c5 - L``.
        Zero where c3 < 0 (edge truncation).
    S_plus : float
        Start-hexamer normaliser for plus strand.
    S_minus : float
        Start-hexamer normaliser for minus strand.
    """
    w_plus: np.ndarray
    w_minus: np.ndarray
    S_plus: float
    S_minus: float


# ── the builder ──────────────────────────────────────────────────────────

def build_region_weights(
    *,
    hex_fwd: np.ndarray,
    hex_rc: np.ndarray,
    cum_gc: np.ndarray,
    start_fwd: np.ndarray,
    end_fwd: np.ndarray,
    start_rev: np.ndarray,
    end_rev: np.ndarray,
    marginal_fl: np.ndarray,
    predict: Callable[[int, float], float],
    region_len: int,
    valid: Optional[np.ndarray] = None,
) -> RegionWeights:
    """Build the fully normalised weight w over the generative domain Omega.

    Parameters
    ----------
    hex_fwd : ndarray, shape (region_len + 1,)
        Forward hexamer index at each cut site position.
    hex_rc : ndarray, shape (region_len + 1,)
        Reverse-complement hexamer index at each cut site position.
    cum_gc : ndarray, shape (region_len + 1,)
        Cumulative GC count; ``cum_gc[i]`` = #(G or C) in bases [0, i).
    start_fwd, end_fwd : ndarray, shape (4096,)
        Hexamer weight tables for the plus strand (start and end).
    start_rev, end_rev : ndarray, shape (4096,)
        Hexamer weight tables for the minus strand (start and end).
    marginal_fl : ndarray, shape (N_LENGTHS,)
        ``marginal_fl[li]`` = P(L = L_MIN + li), normalised to sum 1 over
        L = L_MIN..L_MAX.
    predict : callable (L: int, gc_pct: float) -> float
        Capture-surface inverse: ``1 / P(seen | L, gc)``.  Dividing by
        ``predict`` is multiplying by capture.
    region_len : int
        Region length in bp.
    valid : ndarray, shape (region_len + 1,), optional
        Boolean mask; False where the hexamer window contains a non-ACGT base.
        If None, all positions are treated as valid.

    Returns
    -------
    RegionWeights
        Normalised weights with ``sum(w_plus) + sum(w_minus) = 1`` and
        each strand marginal = 1/2.
    """
    n_sites = region_len + 1  # cut sites 0..region_len
    assert hex_fwd.shape == (n_sites,), f"hex_fwd shape {hex_fwd.shape} != ({n_sites},)"
    assert hex_rc.shape == (n_sites,), f"hex_rc shape {hex_rc.shape} != ({n_sites},)"
    assert cum_gc.shape == (n_sites,), f"cum_gc shape {cum_gc.shape} != ({n_sites},)"
    assert marginal_fl.shape == (N_LENGTHS,), (
        f"marginal_fl shape {marginal_fl.shape} != ({N_LENGTHS},)"
    )

    if valid is None:
        valid = np.ones(n_sites, dtype=bool)

    w_plus = np.zeros((n_sites, N_LENGTHS), dtype=np.float64)
    w_minus = np.zeros((n_sites, N_LENGTHS), dtype=np.float64)

    # Per-position start weights, zeroed where the hexamer is invalid.
    start_vals_plus = np.where(valid, start_fwd[hex_fwd], 0.0)
    start_vals_minus = np.where(valid, start_rev[hex_rc], 0.0)

    Ls = np.arange(L_MIN, L_MAX + 1)  # (N_LENGTHS,)

    # ── Plus strand: c5 in [0, region_len - L], c3 = c5 + L ─────────────
    for li, L in enumerate(Ls):
        max_c5 = region_len - L
        if max_c5 < 0:
            continue
        c5s = np.arange(0, max_c5 + 1)
        c3s = c5s + L
        vmask = valid[c5s] & valid[c3s]
        end_vals = end_fwd[hex_fwd[c3s]]
        gc_counts = cum_gc[c3s] - cum_gc[c5s]
        gc_pcts = 100.0 * gc_counts / L
        predict_vals = np.array([predict(int(L), float(g)) for g in gc_pcts])
        E = np.where(vmask, end_vals * marginal_fl[li] / predict_vals, 0.0)
        w_plus[c5s, li] = E

    # ── Minus strand: c5 in [L, region_len], c3 = c5 - L ────────────────
    for li, L in enumerate(Ls):
        if L > region_len:
            continue
        c5s = np.arange(L, region_len + 1)
        c3s = c5s - L
        vmask = valid[c5s] & valid[c3s]
        end_vals = end_rev[hex_rc[c3s]]
        gc_counts = cum_gc[c5s] - cum_gc[c3s]
        gc_pcts = 100.0 * gc_counts / L
        predict_vals = np.array([predict(int(L), float(g)) for g in gc_pcts])
        E = np.where(vmask, end_vals * marginal_fl[li] / predict_vals, 0.0)
        w_minus[c5s, li] = E

    # ── Normalisation ────────────────────────────────────────────────────
    # w(c5,c3,s) = 1/2 * start_s[hex(c5)] / S_s * E_s(c5,L) / Z_s(c5)
    #
    # Z_s(c5) = sum over L of E_s(c5,L).  Positions where Z=0 (all c3
    # out of range OR all c3 hexamers invalid) have no fragments in Omega.
    # S_s sums start values only over positions with Z > 0, so the
    # Appendix A cancellation (Z/Z -> 1, sum start/S -> 1) holds exactly
    # even when N-masking removes entire c5 positions.

    Z_plus = w_plus.sum(axis=1)
    has_frags_plus = Z_plus > 0
    S_plus = float(start_vals_plus[has_frags_plus].sum())
    safe_Z_plus = np.where(has_frags_plus, Z_plus, 1.0)
    w_plus = (
        0.5
        * (start_vals_plus / S_plus)[:, np.newaxis]
        * (w_plus / safe_Z_plus[:, np.newaxis])
    )

    Z_minus = w_minus.sum(axis=1)
    has_frags_minus = Z_minus > 0
    S_minus = float(start_vals_minus[has_frags_minus].sum())
    safe_Z_minus = np.where(has_frags_minus, Z_minus, 1.0)
    w_minus = (
        0.5
        * (start_vals_minus / S_minus)[:, np.newaxis]
        * (w_minus / safe_Z_minus[:, np.newaxis])
    )

    return RegionWeights(
        w_plus=w_plus,
        w_minus=w_minus,
        S_plus=S_plus,
        S_minus=S_minus,
    )
