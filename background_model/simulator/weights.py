"""Region weight builder for the cfDNA fragment simulator.

Implements Step 4 of docs/pending/simulator_and_fragment_nll.md:
``build_region_weights`` computes a normalised probability distribution
over every fragment the simulator can generate in one region.

Conventions (Appendix D of the design doc)
------------------------------------------
A fragment is ``(c5, c3, strand)``:

  - **Cut sites, not bases.**  A fragment occupying bases ``[p, p+L)`` has
    cut sites at ``p`` and ``p+L`` — NOT endpoint bases ``p`` and ``p+L-1``.
  - **Plus strand:**  ``c5 = p``,   ``c3 = p + L``.
  - **Minus strand:** ``c5 = p+L``, ``c3 = p``.
    The 5' end is at the higher coordinate; hexamers read reverse-complemented.
  - **Length** ``L = |c3 - c5|``, restricted to 25..180 (156 values).
  - **GC** over the genomic span, strand-independent:
    ``gc_pct = 100 * (cum_gc[max(c5,c3)] - cum_gc[min(c5,c3)]) / L``.
  - ``c3(L) = c5 + strand_sign * L``, where ``strand_sign = +1`` (plus) or
    ``-1`` (minus).

Midpoint rule (replaces the old containment edge rule)
------------------------------------------------------
A fragment is admitted if its **integer midpoint** ``p + L // 2`` (floor
division) falls in the half-open interval ``[0, region_len)``.  The region
is expanded by ``pad = L_MAX // 2 = 90`` on each side so that hexamer and
GC context are available for fragments whose endpoints extend beyond the
original region boundary.

Under this rule every fragment length L has exactly ``region_len`` valid
midpoint positions per strand, eliminating the length-dependent positional
penalty that the old containment rule imposed (where length L had only
``region_len - L + 1`` valid positions).

The midpoint convention matches the per-region count files, which define
membership as ``midpoint in [start, stop)`` using integer floor division
for the midpoint.

When ``pad = 0`` (the default for backward compatibility), the midpoint
rule with array-bounds clipping reproduces the old containment rule
exactly, so callers passing arrays of size ``region_len + 1`` get the
original behaviour with no code-path divergence.

Hexamer index conventions:
  - ``hex_fwd[c]``: forward hexamer at cut site ``c`` (6-mer spanning the cut).
  - ``hex_rc[c]``: reverse-complement hexamer at the same position.
  - Plus strand uses ``hex_fwd`` for both ``c5`` and ``c3``.
  - Minus strand uses ``hex_rc`` for both ``c5`` and ``c3``.
  - The four tables ``{start,end} × {fwd,rev}`` are UNTIED; strand selects
    which ``(start_s, end_s)`` pair applies.  Grouped in ``HexamerTables``.

Invariants (Appendix A):
  - ``w_plus.sum() + w_minus.sum() == 1`` exactly.
  - Each strand marginal is exactly ``0.5``.
  - The generative domain has ``|Ω| = 2 × region_len × N_LENGTHS``
    elements (479,232 at region_len 1536) under the midpoint rule.
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

MAX_FL_HALF: int = L_MAX // 2  # 90
"""Half of the maximum fragment length — the region expansion needed for the
midpoint rule.  Under the midpoint rule, a fragment's endpoints can extend
up to ``MAX_FL_HALF`` beyond the original region boundary."""

NHEX: int = 4096
"""Number of distinct hexamer indices (4^6)."""

STRAND_PLUS: str = "+"
"""Plus strand label.  ``c5 = p``, ``c3 = p + L``."""

STRAND_MINUS: str = "-"
"""Minus strand label.  ``c5 = p + L``, ``c3 = p``."""

_STRAND_SIGN = {STRAND_PLUS: +1, STRAND_MINUS: -1}
"""Internal mapping from strand label to the sign used in ``c3 = c5 + sign * L``."""

GC_BIN_WIDTH: float = 5.0
"""Width of each GC-percent bin."""

N_GC_BINS: int = 20
"""Number of GC-percent bins: [0,5), [5,10), ..., [95,100]."""


# ── GC binning and predict LUT ─────────────────────────────────────────


def gc_bin_index(gc_pct_values: np.ndarray) -> np.ndarray:
    """Map GC-percent values to bin indices.

    Bins: ``[0,5), [5,10), ..., [95,100]`` — contiguous 5% bins, last
    inclusive.  At an exact boundary (e.g. 5.0) the upper bin wins:
    ``floor(5.0 / 5.0) = 1``, placing 5.0 in the ``[5,10)`` bin —
    correct, since ``[0,5)`` is half-open on the right.  The exception
    is 100.0, which is clamped into the last bin (index 19) by the
    ``last-inclusive`` rule.

    Boundary semantics (Appendix F):
      - ``_bin_index`` in ``flgc.model`` uses ``lo <= v <= hi`` (inclusive on
        both sides) and returns the *first* match.  With contiguous bins
        ``[0,4], [5,9], ...`` an integer boundary falls into the lower bin.
      - Here we use ``floor(gc / 5)`` clamped to [0, 19].  ``gc = 0`` →
        bin 0; ``gc = 5`` → bin 1; ``gc = 95`` → bin 19; ``gc = 100`` →
        bin 19 (clamped).  This is equivalent for all integer values, and
        also correct for non-integer GC (which ``_bin_index`` would drop).
    """
    idx = np.floor(gc_pct_values / GC_BIN_WIDTH).astype(np.intp)
    return np.clip(idx, 0, N_GC_BINS - 1)


def build_predict_lut(
    predict: Callable[[int, float], float],
) -> np.ndarray:
    """Build the ``(N_LENGTHS, N_GC_BINS)`` predict lookup table.

    For each ``(L, gc_bin)`` pair, calls ``predict(L, gc_mid)`` where
    ``gc_mid`` is the bin midpoint (2.5, 7.5, ..., 97.5).  The result is
    cached once and gathered from inside ``build_region_weights``.

    Parameters
    ----------
    predict : callable (L: int, gc_pct: float) -> float
        Capture-surface inverse: ``min(1/P(seen), max_weight)``.
        GC in percent 0-100.

    Returns
    -------
    ndarray, shape (N_LENGTHS, N_GC_BINS)
        ``lut[li, gi]`` = ``predict(L_MIN + li, gc_midpoint_of_bin_gi)``.
    """
    gc_mids = np.arange(N_GC_BINS) * GC_BIN_WIDTH + GC_BIN_WIDTH / 2.0
    lut = np.empty((N_LENGTHS, N_GC_BINS), dtype=np.float64)
    for li in range(N_LENGTHS):
        L = L_MIN + li
        for gi in range(N_GC_BINS):
            lut[li, gi] = predict(int(L), float(gc_mids[gi]))
    return lut


# ── orientation / geometry helpers ───────────────────────────────────────

def c3_from_c5(c5: int, L: int, strand: str) -> int:
    """Compute the 3' cut site from the 5' cut site, length, and strand.

    Parameters
    ----------
    c5 : int
        5' cut-site position.
    L : int
        Fragment length in bp.
    strand : ``"+"`` or ``"-"``
        Strand label.  Plus: ``c3 = c5 + L``.  Minus: ``c3 = c5 - L``.

    Raises
    ------
    ValueError
        If *strand* is not ``"+"`` or ``"-"``.
    """
    try:
        sign = _STRAND_SIGN[strand]
    except KeyError:
        raise ValueError(
            f"strand must be '+' or '-', got {strand!r}"
        ) from None
    return c5 + sign * L


def fragment_base_range(c5: int, c3: int):
    """Return ``(p, L)`` for the half-open genomic span ``[p, p+L)`` of a fragment.

    Strand-independent: ``p = min(c5, c3)``, ``L = |c3 - c5|``.  Works for
    both plus (``c5 < c3``) and minus (``c5 > c3``) strand fragments.
    """
    p = min(c5, c3)
    L = abs(c3 - c5)
    return p, L


def gc_pct(c5, c3, cum_gc):
    """GC percentage over the genomic span of a fragment.

    Strand-independent: always uses ``[min(c5,c3), max(c5,c3))`` which is
    ``[p, p+L)``.  ``cum_gc`` is the cumulative GC count array of length
    ``region_len + 1`` where ``cum_gc[i]`` = number of G/C bases in
    positions ``[0, i)``.

    Accepts scalar or array ``c5``/``c3`` (uses ``np.minimum``/``np.maximum``).

    Raises
    ------
    ValueError
        If any element has ``c5 == c3`` (zero-length fragment).  All
        fragments in the generative domain have ``L >= 25``, so this
        condition cannot arise in normal operation.
    """
    c5 = np.asarray(c5)
    c3 = np.asarray(c3)
    zero_mask = c5 == c3
    if np.any(zero_mask):
        bad = np.argwhere(zero_mask.ravel())
        raise ValueError(
            f"gc_pct: c5 == c3 (zero-length fragment) at "
            f"index {bad.ravel()[0]}, value c5=c3={int(np.asarray(c5).ravel()[bad.ravel()[0]])}"
        )
    lo = np.minimum(c5, c3)
    hi = np.maximum(c5, c3)
    L = hi - lo
    return 100.0 * (cum_gc[hi] - cum_gc[lo]) / L


def generative_domain_size(
    region_len: int, L_min: int = L_MIN, L_max: int = L_MAX,
) -> int:
    """Count of all possible fragments the simulator can generate in a region.

    Under the midpoint rule, the generative domain ``Ω`` is the set of all
    valid ``(c5, c3, strand)`` triples where the integer midpoint
    ``p + L // 2`` (floor division) falls in ``[0, region_len)``.

    Every fragment length has exactly ``region_len`` valid midpoint positions
    per strand, so::

        |Ω| = 2 × region_len × (L_max − L_min + 1)

    This is the property that eliminates the length-dependent positional
    penalty of the old containment rule.

    Verified value: 479,232 at ``region_len = 1536``.
    """
    n_lengths = L_max - L_min + 1
    if n_lengths <= 0 or region_len <= 0:
        return 0
    return 2 * region_len * n_lengths


# ── hexamer tables ──────────────────────────────────────────────────────

class HexamerTables(NamedTuple):
    """Four hexamer cut-site weight tables: ``{start, end} × {fwd, rev}``.

    Each table is a 1-D array of shape ``(4096,)`` mapping a hexamer index
    to a non-negative weight.  The four tables are **untied** — they are
    fitted independently because single-stranded sequencing means the
    5' and 3' ends see different hexamer preferences.

    Strand selects which pair applies:

    - **Plus strand:** ``start_fwd[hex_fwd[c5]]`` × ``end_fwd[hex_fwd[c3]]``
    - **Minus strand:** ``start_rev[hex_rc[c5]]`` × ``end_rev[hex_rc[c3]]``

    Attributes
    ----------
    start_fwd, end_fwd : ndarray, shape (4096,)
        Plus-strand start and end hexamer weights.
    start_rev, end_rev : ndarray, shape (4096,)
        Minus-strand start and end hexamer weights.
    """
    start_fwd: np.ndarray
    end_fwd: np.ndarray
    start_rev: np.ndarray
    end_rev: np.ndarray


# ── weight builder result ────────────────────────────────────────────────

class RegionWeights(NamedTuple):
    """Normalised probability distribution over every fragment the simulator
    can generate in one region.

    The two weight arrays together form a proper probability distribution:
    ``w_plus.sum() + w_minus.sum() == 1`` exactly, and each strand sums
    to exactly ``0.5``.  An entry is zero where the fragment's midpoint
    falls outside ``[0, region_len)`` or where the hexamer window contains
    a non-ACGT base (N-masking).

    Attributes
    ----------
    w_plus : ndarray, shape ``(n_sites, N_LENGTHS)``
        Plus-strand weight matrix.  ``n_sites = region_len + 2*pad + 1``.
        Axis 0 is the array index for the 5' cut site; the region-local
        c5 position is ``array_index - pad``.  Axis 1 is the length index
        ``li = L - L_MIN`` for ``L`` in 25..180.
    w_minus : ndarray, shape ``(n_sites, N_LENGTHS)``
        Minus-strand weight matrix.  Same axes and offset convention.
    S_plus : float
        Sum of start-hexamer weights over plus-strand ``c5`` positions that
        have at least one achievable fragment.  Exposed for the sampler
        (Step 5), which draws ``c5`` from ``start_fwd[hex_fwd[c5]] / S_plus``.
    S_minus : float
        Same, for the minus strand.
    pad : int
        Offset from array index to region-local c5 coordinate:
        ``c5_local = array_index - pad``.  With ``pad = 0`` (default),
        the arrays are indexed directly by region-local c5 as before.
    """
    w_plus: np.ndarray
    w_minus: np.ndarray
    S_plus: float
    S_minus: float
    pad: int = 0


# ── the builder ──────────────────────────────────────────────────────────

def build_region_weights(
    *,
    hex_fwd: np.ndarray,
    hex_rc: np.ndarray,
    cum_gc: np.ndarray,
    hex_tables: HexamerTables,
    marginal_fl: np.ndarray,
    predict_lut: np.ndarray,
    region_len: int,
    valid: Optional[np.ndarray] = None,
    pad: int = 0,
) -> RegionWeights:
    """Build the fully normalised weight w over the generative domain Omega.

    Under the **midpoint rule**, a fragment ``(c5, c3, strand)`` is admitted
    when its integer midpoint ``p + L // 2`` (floor division, matching the
    per-region count files) falls in ``[0, region_len)``.  The ``pad``
    parameter controls how far beyond the original ``[0, region_len)``
    interval the input arrays extend; with ``pad = MAX_FL_HALF = 90`` every
    admissible fragment has its endpoints covered.

    When ``pad = 0`` the array-bounds clipping reproduces the old
    containment rule (``c3 in [0, region_len]``) exactly — no separate
    code path.

    Parameters
    ----------
    hex_fwd : ndarray, shape ``(n_sites,)``
        Forward hexamer index at each cut site.  ``n_sites = region_len + 2*pad + 1``.
    hex_rc : ndarray, shape ``(n_sites,)``
        Reverse-complement hexamer index.
    cum_gc : ndarray, shape ``(n_sites,)``
        Cumulative GC count over the expanded region.
    hex_tables : HexamerTables
        Four hexamer weight tables.
    marginal_fl : ndarray, shape ``(N_LENGTHS,)``
        Marginal fragment-length distribution, summing to 1.
    predict_lut : ndarray, shape ``(N_LENGTHS, N_GC_BINS)``
        Pre-built capture-surface lookup table.
    region_len : int
        Region length in bp (the original, unexpanded region).
    valid : ndarray, shape ``(n_sites,)``, optional
        Boolean mask; False where the hexamer window contains non-ACGT.
    pad : int
        Number of extra cut-site positions on each side of the original
        region.  Use ``MAX_FL_HALF`` (90) for the midpoint rule; 0 for
        backward-compatible containment-rule behaviour.

    Returns
    -------
    RegionWeights
        Normalised weights with ``sum(w_plus) + sum(w_minus) = 1`` and
        each strand marginal = 1/2.
    """
    n_sites = region_len + 2 * pad + 1
    assert hex_fwd.shape == (n_sites,), f"hex_fwd shape {hex_fwd.shape} != ({n_sites},)"
    assert hex_rc.shape == (n_sites,), f"hex_rc shape {hex_rc.shape} != ({n_sites},)"
    assert cum_gc.shape == (n_sites,), f"cum_gc shape {cum_gc.shape} != ({n_sites},)"
    assert marginal_fl.shape == (N_LENGTHS,), (
        f"marginal_fl shape {marginal_fl.shape} != ({N_LENGTHS},)"
    )
    assert predict_lut.shape == (N_LENGTHS, N_GC_BINS), (
        f"predict_lut shape {predict_lut.shape} != ({N_LENGTHS}, {N_GC_BINS})"
    )
    # Bind by NAME, not by position. Positional unpacking would silently
    # re-introduce the misrouting hazard that HexamerTables exists to remove.
    start_fwd = hex_tables.start_fwd
    end_fwd = hex_tables.end_fwd
    start_rev = hex_tables.start_rev
    end_rev = hex_tables.end_rev
    assert start_fwd.shape == (NHEX,), f"start_fwd shape {start_fwd.shape} != ({NHEX},)"
    assert end_fwd.shape == (NHEX,), f"end_fwd shape {end_fwd.shape} != ({NHEX},)"
    assert start_rev.shape == (NHEX,), f"start_rev shape {start_rev.shape} != ({NHEX},)"
    assert end_rev.shape == (NHEX,), f"end_rev shape {end_rev.shape} != ({NHEX},)"

    if valid is None:
        valid = np.ones(n_sites, dtype=bool)

    w_plus = np.zeros((n_sites, N_LENGTHS), dtype=np.float64)
    w_minus = np.zeros((n_sites, N_LENGTHS), dtype=np.float64)

    # Per-position start weights, zeroed where the hexamer is invalid.
    start_vals_plus = np.where(valid, start_fwd[hex_fwd], 0.0)
    start_vals_minus = np.where(valid, start_rev[hex_rc], 0.0)

    Ls = np.arange(L_MIN, L_MAX + 1)  # (N_LENGTHS,)

    # ── Plus strand ──────────────────────────────────────────────────────
    # Midpoint rule: midpoint = c5_local + L // 2 must be in [0, region_len).
    # c5_local in [-L//2, region_len - 1 - L//2].
    # Array index = c5_local + pad.
    # c3 array index = c5 array index + L.
    # With pad=0 the bounds clipping reproduces the old containment rule.
    for li, L in enumerate(Ls):
        half_down = L // 2
        # Ideal c5 array-index range from the midpoint constraint
        c5_lo = max(0, pad - half_down)
        c5_hi = min(n_sites - 1, pad + region_len - 1 - half_down)
        if c5_lo > c5_hi:
            continue
        c5s = np.arange(c5_lo, c5_hi + 1)
        c3s = c5s + L
        # Clip: c3 must also be a valid array index
        in_bounds = c3s < n_sites
        c5s = c5s[in_bounds]
        c3s = c3s[in_bounds]
        if len(c5s) == 0:
            continue
        vmask = valid[c5s] & valid[c3s]
        end_vals = end_fwd[hex_fwd[c3s]]
        gc_pcts_arr = gc_pct(c5s, c3s, cum_gc)
        gc_bins = gc_bin_index(gc_pcts_arr)
        predict_vals = predict_lut[li, gc_bins]
        E = np.where(vmask, end_vals * marginal_fl[li] / predict_vals, 0.0)
        w_plus[c5s, li] = E

    # ── Minus strand ─────────────────────────────────────────────────────
    # Midpoint = c5_local - (L - L//2) must be in [0, region_len).
    # c5_local in [L - L//2, region_len - 1 + L - L//2].
    # c3 array index = c5 array index - L.
    for li, L in enumerate(Ls):
        half_up = L - L // 2  # ceil(L/2)
        c5_lo = max(0, pad + half_up)
        c5_hi = min(n_sites - 1, pad + region_len - 1 + half_up)
        if c5_lo > c5_hi:
            continue
        c5s = np.arange(c5_lo, c5_hi + 1)
        c3s = c5s - L
        # Clip: c3 must be a valid array index (>= 0)
        in_bounds = c3s >= 0
        c5s = c5s[in_bounds]
        c3s = c3s[in_bounds]
        if len(c5s) == 0:
            continue
        vmask = valid[c5s] & valid[c3s]
        end_vals = end_rev[hex_rc[c3s]]
        gc_pcts_arr = gc_pct(c5s, c3s, cum_gc)
        gc_bins = gc_bin_index(gc_pcts_arr)
        predict_vals = predict_lut[li, gc_bins]
        E = np.where(vmask, end_vals * marginal_fl[li] / predict_vals, 0.0)
        w_minus[c5s, li] = E

    # ── Normalisation ────────────────────────────────────────────────────
    # w(c5,c3,s) = 1/2 * start_s[hex(c5)] / S_s * E_s(c5,L) / Z_s(c5)
    #
    # Z_s(c5) = sum over L of E_s(c5,L).  Positions where Z=0 (all
    # midpoints out of range OR all hexamers invalid) have no fragments
    # in Omega.  S_s sums start values only over positions with Z > 0,
    # so the Appendix A cancellation holds exactly.

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
        pad=pad,
    )
