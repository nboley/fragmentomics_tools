"""Correction outputs — the two locked product interfaces (Phase 2).

(a) ``apply_fragment_weights`` — per-fragment inverse-probability weights on a
    ``RegionFragmentArray`` (the fixed version of v1's
    ``_set_fragment_array_weights_from_weights_record``, whose strand-mask OR/AND
    precedence bug is documented in BIAS_CORRECTION_REVIEW.md S1).  The S1 bug is
    IMPOSSIBLE-BY-CONSTRUCTION here: each fragment maps to exactly one strand
    (its own) and exactly one band (disjoint half-open intervals), hence exactly
    one track per coverage type — no OR/AND boolean mask, no overwrite ordering.

(b) ``expected_profile`` — per-position expected-count vectors ``N_w · probs``
    for consumers that divide themselves.

Both interfaces are pure functions of ``(probs, N)`` from the model's SHAPE head;
the dispersion head is unused this phase, so a model trained under any of the
three losses (multinomial / dirichlet_multinomial / nb_offset) is fully
supported (design §5.2).

N is the plug-in scale (design §0 contract 1) and the masked-target contract
(contract 2) is enforced BY CONSTRUCTION: every N applies the window mask, and
masked output positions are ``NaN`` (expected) / weight-0 (fragments).

Coordinate-frame precondition (invariant §0.3): ``apply_fragment_weights``
asserts ``region.strand in {'.', '+'}`` AND ``not rfa.is_flipped`` — correction
queries strandless (``.``) regions so ``from_fragments_h5`` never flips.  A
``'+'`` rfa can still be flipped via the explicit reverse op
(``fragment_array.py:706-719``), so BOTH halves of the assert are required.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from background_model.config import TILE
from background_model.inference import (
    WindowGeometry,
    _iter_window_predictions,
)
from background_model.preprocess import FL_BANDS, TRACK_INDEX


@dataclass(frozen=True)
class WeightClampConfig:
    """Explicit clamp bounds for the inverse-probability weights.

    There is NO default — ``apply_fragment_weights`` REQUIRES a clamp argument,
    forcing the caller to choose consciously.  Clamp SEMANTICS remain a deferred
    owner decision (brief §Correction outputs: clamping DEFERRED); this class
    only provides the mechanism.  ``WeightClampConfig.identity()`` is the
    explicit opt-in for NO clamp — but note that with a TRAINED (peaky) model
    identity can produce UNBOUNDED weights (``1/(probs·L_valid)`` as
    ``probs → 0``; design §7).  v1-parity is
    ``WeightClampConfig(min_weight=1e-6, max_weight=10.0)``.
    """

    min_weight: Optional[float] = None
    max_weight: Optional[float] = None

    @classmethod
    def identity(cls) -> "WeightClampConfig":
        """Explicit NO-clamp opt-in.  WARNING: unbounded weights with a trained
        model (design §7); clamp semantics remain a deferred owner decision."""
        return cls(min_weight=None, max_weight=None)

    def apply(self, w: np.ndarray) -> np.ndarray:
        if self.min_weight is None and self.max_weight is None:
            return w
        return np.clip(w, self.min_weight, self.max_weight)


@dataclass(frozen=True)
class ExpectedProfile:
    expected: np.ndarray       # (C, L) expected counts; NaN at masked positions
    probs: np.ndarray          # (C, L) the underlying shape
    N: np.ndarray              # (n_windows, C) per-window per-track observed N
    mask: np.ndarray           # (L,) bool
    coord0: int
    output_tracks: List[str]


def expected_profile(
    model, fasta, contig: str, start: int, stop: int,
    observed_counts: np.ndarray, *,          # (C, L) raw sample counts, [start, stop)
    blacklist_rdf=None, blacklist_expansion: int = 120,
    contig_len: Optional[int] = None, tile_size: int = TILE,
    masked_fill: float = np.nan,
) -> ExpectedProfile:
    """Interface (b).  Per window ``w``:
    ``N_w[c] = sum_j observed[c, j]·mask[j]`` (mask applied — enforces the store
    contract by construction); ``expected[c, j] = N_w[c] · probs[c, j]`` for
    valid ``j``, else ``masked_fill`` (``NaN`` by default, so an ``obs/expected``
    consumer sees ``NaN`` at masked positions rather than a poisoning ``x/0``;
    design §5.1).
    """
    geom = WindowGeometry.from_model(model, tile_size)
    n_tracks = len(model.output_tracks)
    L = stop - start
    observed_counts = np.asarray(observed_counts, dtype=np.float64)
    assert observed_counts.shape == (n_tracks, L), (
        f"observed_counts has shape {observed_counts.shape}, "
        f"expected ({n_tracks}, {L})"
    )

    expected = np.full((n_tracks, L), masked_fill, dtype=np.float64)
    probs_full = np.zeros((n_tracks, L), dtype=np.float64)
    mask_full = np.zeros(L, dtype=bool)
    N_rows = []

    for w, win_start, win_stop, probs_w, mask_w, l_valid in _iter_window_predictions(
        model, fasta, contig, start, stop, geom,
        blacklist_rdf, blacklist_expansion, contig_len,
    ):
        col0 = win_start - start            # == w * tile_size
        local_hi = min(tile_size, stop - win_start)
        cols = slice(col0, col0 + local_hi)

        m_slice = mask_w[:local_hi]
        p_slice = probs_w[:, :local_hi]
        obs_slice = observed_counts[:, cols]

        # N_w over the window's valid positions (masked — store contract).
        N_w = (obs_slice * m_slice[None, :]).sum(axis=1)
        N_rows.append(N_w)

        probs_full[:, cols] = p_slice
        mask_full[cols] = m_slice
        # expected = N_w · probs at valid positions; masked positions stay
        # masked_fill (probs is 0 there anyway).
        exp_w = N_w[:, None] * p_slice
        valid_cols = np.nonzero(m_slice)[0]
        if len(valid_cols):
            expected[:, col0 + valid_cols] = exp_w[:, valid_cols]

    N = (
        np.stack(N_rows, axis=0)
        if N_rows
        else np.zeros((0, n_tracks), dtype=np.float64)
    )
    return ExpectedProfile(
        expected=expected, probs=probs_full, N=N, mask=mask_full,
        coord0=int(start), output_tracks=list(model.output_tracks),
    )


def _normalize_strands(fragment_strands) -> np.ndarray:
    """Return a ``'U1'`` str array of ``+``/``-`` from a possibly-bytes array."""
    arr = np.asarray(fragment_strands)
    if arr.dtype.kind == "S":
        arr = np.char.decode(arr, "ascii")
    return arr.astype("U1")


def apply_fragment_weights(
    rfa, model, fasta, *,
    fl_bands=FL_BANDS,
    blacklist_rdf=None, blacklist_expansion: int = 120,
    contig_len: Optional[int] = None, tile_size: int = TILE,
    clamp: WeightClampConfig,               # REQUIRED — no default (design §5.2)
    drop_uncorrectable: bool = True,
):
    """Interface (a).  Set ``rfa.first_covered_base_weights``,
    ``rfa.last_covered_base_weights``, ``rfa.weights`` to
    ``1 / (probs·L_valid)`` at each fragment's (first/last/midpoint) endpoint,
    indexed by the ``(strand, band, coverage)`` track that coverage getter
    actually reads.  Uncorrectable endpoints (out-of-band, masked, or off-grid)
    → weight 0; ``drop_uncorrectable`` drops fragments whose three weights are
    all 0.  Returns a NEW ``RegionFragmentArray``.

    ``clamp`` is REQUIRED (pass ``WeightClampConfig.identity()`` for no clamp);
    it is applied as the last step to the corrected (nonzero) weights only —
    uncorrectable endpoints stay exactly 0 so ``drop_uncorrectable`` and the
    masked-endpoint policy are preserved (design §5.2).
    """
    region = rfa.region
    # Coordinate-frame precondition (invariant §0.3): both halves required —
    # a '+' region can still be is_flipped via the explicit reverse op.
    # NOTE: Region normalizes a strandless '.' to None (region.py DataClassMixin,
    # strand_is_set treats None and '.' identically), so the strandless value
    # observed here is None; accept both None and '.' plus '+'.
    assert region.strand in {None, ".", "+"} and not rfa.is_flipped, (
        "apply_fragment_weights requires a strandless ('.'/None) or unflipped '+' "
        "region: correction queries the forward genomic frame, in which "
        "gpos = region.start + endpoint_coord and track = fragment_strands[f] "
        "are valid.  A minus-strand / is_flipped rfa reverses the local frame "
        "and swaps strands (fragment_array.py:1809-1819, :706-719).  Strand "
        "orientation is a consumer-layer operation, never a per-rfa flip "
        "(see correction_outputs_design.md §0.3 / §5.2).  Got "
        f"region.strand={region.strand!r}, is_flipped={rfa.is_flipped}."
    )
    if rfa.fragment_strands is None:
        raise ValueError(
            "apply_fragment_weights requires a stranded rfa: the model's tracks "
            "are strand-specific (design §7)."
        )

    # 12-track precondition: probs is gathered by TRACK_INDEX[(strand, band,
    # cov)] (design's 12-track strand×band×coverage constraint).  A model with
    # fewer output tracks would silently index a wrong / out-of-range track.
    assert len(model.output_tracks) == len(TRACK_INDEX), (
        f"apply_fragment_weights requires the full {len(TRACK_INDEX)}-track "
        f"model (design §7 strand×band×coverage constraint; probs is gathered "
        f"by TRACK_INDEX); got {len(model.output_tracks)} output tracks."
    )

    geom = WindowGeometry.from_model(model, tile_size)
    contig = region.chrom
    start = int(region.start)
    stop = int(region.stop)

    # per-window predictions (probs, mask, l_valid) keyed by window index.
    windows = {}
    for w, win_start, win_stop, probs_w, mask_w, l_valid in _iter_window_predictions(
        model, fasta, contig, start, stop, geom,
        blacklist_rdf, blacklist_expansion, contig_len,
    ):
        windows[w] = (probs_w, mask_w, l_valid)

    n = rfa.n_fragments
    strands = _normalize_strands(rfa.fragment_strands)
    lengths = np.asarray(rfa.fragment_lengths)
    bands = [tuple(b) for b in fl_bands]

    # band membership: explicit per-band half-open lo <= len < hi (NOT
    # searchsorted, which maps an inter-band GAP length onto a neighbour; F3).
    band_idx = np.full(n, -1, dtype=np.int64)
    for b, (lo, hi) in enumerate(bands):
        band_idx[(lengths >= lo) & (lengths < hi)] = b

    # strand gate (same guard-before-gather class as the F11 band guard, on the
    # strand axis): only '+'/'-' fragments map to a real track.  A fragment with
    # any other strand keeps trk == -1, which would silently gather probs_w[-1]
    # (the last track).  Gate it OUT of eligibility so it gets weight 0 on every
    # coverage type instead.
    strand_ok = (strands == "+") | (strands == "-")

    cov_specs = [
        ("first", np.asarray(rfa.first_covered_bases_0), "first_covered_base_weights"),
        ("last", np.asarray(rfa.last_covered_bases_0), "last_covered_base_weights"),
        ("midpoint", np.asarray(rfa.midpoints_0), "weights"),
    ]

    new_weight_vectors = {}
    for cov, endpoint_coord_0, attr in cov_specs:
        gpos = start + endpoint_coord_0
        in_grid = (gpos >= start) & (gpos < stop)
        off = gpos - start
        # F11 guard: compute the track index ONLY for in-band fragments; an
        # out-of-band band_idx == -1 must NEVER index bands[-1] (which python
        # negative-indexes to the LAST band).
        trk_arr = np.full(n, -1, dtype=np.int64)
        for b, band in enumerate(bands):
            for s in ("+", "-"):
                sel_sb = (band_idx == b) & (strands == s)
                if sel_sb.any():
                    trk_arr[sel_sb] = TRACK_INDEX[(s, band, cov)]

        elig = in_grid & (band_idx >= 0) & strand_ok
        w_out = np.zeros(n, dtype=np.float64)
        valid = np.zeros(n, dtype=bool)

        win_all = np.where(in_grid, off // tile_size, -1)
        j_all = np.where(in_grid, off - win_all * tile_size, -1)

        for w_idx, (probs_w, mask_w, l_valid) in windows.items():
            if l_valid == 0:
                continue
            sel = elig & (win_all == w_idx)
            idxs = np.nonzero(sel)[0]
            if len(idxs) == 0:
                continue
            jj = j_all[idxs]
            m_good = mask_w[jj]                     # endpoint on a valid position?
            good = idxs[m_good]
            if len(good) == 0:
                continue
            jg = j_all[good]
            tg = trk_arr[good]
            p = probs_w[tg, jg]
            w_out[good] = 1.0 / (p * l_valid)
            valid[good] = True

        # clamp the corrected (nonzero) weights only; uncorrectable stay 0.
        w_clamped = clamp.apply(w_out)
        w_out = np.where(valid, w_clamped, 0.0)
        new_weight_vectors[attr] = w_out

    new = rfa._replace(
        first_covered_base_weights=new_weight_vectors["first_covered_base_weights"],
        last_covered_base_weights=new_weight_vectors["last_covered_base_weights"],
        weights=new_weight_vectors["weights"],
        validate_data=False,
    )

    if drop_uncorrectable:
        keep = (
            (new_weight_vectors["weights"] > 1e-6)
            | (new_weight_vectors["first_covered_base_weights"] > 1e-6)
            | (new_weight_vectors["last_covered_base_weights"] > 1e-6)
        )
        new = new.mask(keep, validate_data=False)

    return new
