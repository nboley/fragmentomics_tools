"""Inference-path window assembly + seam-free profile stitching (Phase 2).

Store-free counterpart to the training data-assembly path.  Given a trained
``BackgroundModel``, a FASTA, and a genomic region, this module builds the
model input (one-hot sequence + valid mask) for each ``tile_size`` window and
runs ``predict_profile`` on it.  The input assembly is BYTE-IDENTICAL to the
val-mode ``BackgroundTileDataset`` input for the same tile (equivalence is
test-locked, see ``tests/test_bg_inference.py`` T1) because it reuses the exact
same three operations the training path performs — the same FASTA fetch + N-pad
rule (``preprocess.py`` Phase B) and the same Cython ``one_hot_encode_sequences``
encoder the Dataset calls.

Geometry (design §1.1, §3.2): the model is unpadded and trims exactly
``margin = (model_input_size - tile_size) // 2`` bp symmetrically from each side.
So the emitted output window of length ``tile_size`` is the center of the fetched
input window ``[win_start - margin, win_stop + margin)``, and output position
``j`` maps to genomic ``win_start + j``.  ``margin`` is derived from the model
(``calc_input_region_size``), NEVER hardcoded.

Coordinate frame (SYSTEM-WIDE invariant, design §0.3): everything here operates
in the forward genomic frame.  Correction queries build strandless regions and
never flip; strand orientation is a consumer-layer operation.

Seam-free stitching (of the SHAPE, design §3.3): adjacent windows abut with no
overlap in their outputs; each carries its OWN masked softmax (``probs`` sums to
1 within that window over its valid positions).  Stitching is plain
concatenation of per-window ``probs`` — there is deliberately NO cross-window
renormalization.  Expected COUNTS are NOT seam-free: they carry a per-window N
scale (see ``correction.expected_profile``).
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from background_model.config import TILE
from background_model.preprocess import _get_overlapping_blacklist_regions
from fragmentomics_tools.region import one_hot_encode_sequences


@dataclass(frozen=True)
class WindowGeometry:
    """Derived window geometry for one model + tile size.

    ``margin`` is the symmetric per-side trim of the unpadded model, derived
    from ``model.calc_input_region_size`` — never hardcoded (it is
    architecture-dependent).
    """

    tile_size: int          # L_out per window
    model_input_size: int   # = model.calc_input_region_size(tile_size)
    margin: int             # = (model_input_size - tile_size) // 2

    @classmethod
    def from_model(cls, model, tile_size: int = TILE) -> "WindowGeometry":
        mis = int(model.calc_input_region_size(tile_size))
        diff = mis - tile_size
        assert diff >= 0 and diff % 2 == 0, (
            f"model_input_size - tile_size = {diff} must be non-negative and "
            "even for a symmetric center crop (design §2.1)"
        )
        return cls(tile_size=int(tile_size), model_input_size=mis, margin=diff // 2)


def build_window_onehot(
    fasta,
    contig: str,
    win_start: int,
    win_stop: int,
    geom: WindowGeometry,
) -> np.ndarray:
    """One-hot ``(4, model_input_size)`` for window ``[win_start, win_stop)``.

    Fetches ``fasta[win_start - margin : win_stop + margin]``, N-pads at contig
    edges, ``.upper()`` → bytes → ``one_hot_encode_sequences([...])[0].T``.
    BYTE-IDENTICAL to the val-mode Dataset ``x`` for the same tile (design §3.1;
    T1).  No ``contig_len`` argument is needed: ``pysam.FastaFile.fetch`` already
    clamps the right edge to the contig end, exactly as the Phase B store build
    does (``preprocess.py:489``), so the N-padding is identical on both paths.
    """
    mis = geom.model_input_size
    seq_start = win_start - geom.margin
    seq_stop = win_stop + geom.margin
    seq_str = fasta.fetch(contig, max(0, seq_start), seq_stop)

    left_pad = max(0, -seq_start)
    right_pad = mis - len(seq_str) - left_pad
    if left_pad > 0 or right_pad > 0:
        seq_str = "N" * left_pad + seq_str + "N" * max(0, right_pad)

    seq_bytes = np.frombuffer(seq_str.upper().encode("ascii"), dtype=np.uint8)
    if len(seq_bytes) != mis:
        seq_bytes = np.pad(
            seq_bytes, (0, max(0, mis - len(seq_bytes))),
            constant_values=ord("N"),
        )[:mis]

    onehot = one_hot_encode_sequences([seq_bytes.tobytes()])[0]
    x = np.ascontiguousarray(onehot.T, dtype=np.float32)
    # Model/geometry mismatch guard (design §7): fail loudly before predict.
    assert x.shape == (4, mis), (
        f"window one-hot has shape {x.shape}, expected (4, {mis})"
    )
    return x


def build_window_mask(
    contig: str,
    win_start: int,
    win_stop: int,
    geom: WindowGeometry,
    *,
    blacklist_rdf=None,
    blacklist_expansion: int = 120,
    contig_len: Optional[int] = None,
) -> np.ndarray:
    """``(tile_size,)`` bool valid mask for window ``[win_start, win_stop)``.

    Invalid past contig ends and within ``blacklist_expansion`` bp of any
    blacklist region.  IDENTICAL to the center-tile crop of the store's
    L_TARGET mask (reuses the Phase B mask logic on the TILE extent, frame
    origin ``win_start`` instead of ``tile.start - jitter``; design §2.1, T1).
    """
    tile_size = geom.tile_size
    mask = np.ones(tile_size, dtype=bool)

    # positions past the contig start (win_start < 0 does not happen for a
    # region-anchored grid, but keep the clamp symmetric with Phase B).
    left_invalid = max(0, -win_start)
    if left_invalid > 0:
        mask[:left_invalid] = False
    if contig_len is not None:
        right_valid = contig_len - win_start  # first out-of-contig position
        if right_valid < tile_size:
            mask[max(0, right_valid):] = False

    if blacklist_rdf is not None:
        # Widen the overlap query by `expansion` so a blacklist region just
        # outside the window whose expanded zone reaches in is still caught
        # (matches Phase B, preprocess.py:538-544).
        bl_regions = _get_overlapping_blacklist_regions(
            blacklist_rdf, contig,
            win_start - blacklist_expansion, win_stop + blacklist_expansion,
        )
        for bl_reg in bl_regions:
            local_start = max(0, bl_reg.start - blacklist_expansion - win_start)
            local_stop = min(tile_size, bl_reg.stop + blacklist_expansion - win_start)
            if local_start < local_stop:
                mask[local_start:local_stop] = False
    return mask


def iter_windows(start: int, stop: int, tile_size: int = TILE):
    """Yield consecutive ``(win_start, win_stop)`` covering ``[start, stop)`` on
    the grid anchored at ``start``.

    The final window runs full ``tile_size`` even if ``win_stop > stop`` (the
    model has no shorter mode); the caller trims out-of-range output positions
    (design §4).
    """
    w0 = start
    while w0 < stop:
        yield (w0, w0 + tile_size)
        w0 += tile_size


def locate(gpos: int, start: int, stop: int, tile_size: int = TILE):
    """Map a genomic position to ``(window_index, local_position)`` on the grid
    anchored at ``start`` (design §5.2 ``locate``).

    Positions outside ``[start, stop)`` are OFF-GRID → ``(-1, -1)``.  Note a
    sub-grid ``stop`` (final window trimmed): a position in
    ``[stop, start + n_windows*tile_size)`` lands in the final window's computed
    output range but is rejected here as off-grid (design §4, T10 (iv)).
    """
    if gpos < start or gpos >= stop:
        return -1, -1
    off = gpos - start
    win = off // tile_size
    j = off - win * tile_size
    return int(win), int(j)


def _predict_window(
    model, fasta, contig, win_start, win_stop, geom,
    blacklist_rdf, blacklist_expansion, contig_len,
):
    """Predict one window: returns ``(probs (C, tile_size), mask (tile_size,),
    l_valid)``.

    All-masked guard (design §7): if the window has no valid position, softmax
    over empty support is undefined, so skip ``predict_profile`` and return
    ``probs = 0`` with ``l_valid = 0``.
    """
    mask = build_window_mask(
        contig, win_start, win_stop, geom,
        blacklist_rdf=blacklist_rdf, blacklist_expansion=blacklist_expansion,
        contig_len=contig_len,
    )
    n_tracks = len(model.output_tracks)
    l_valid = int(mask.sum())
    if l_valid == 0:
        probs = np.zeros((n_tracks, geom.tile_size), dtype=np.float64)
        return probs, mask, 0
    onehot = build_window_onehot(
        fasta, contig, win_start, win_stop, geom,
    )
    out = model.predict_profile(onehot, mask=mask)
    probs = np.asarray(out["probs"])
    return probs, mask, l_valid


def _iter_window_predictions(
    model, fasta, contig, start, stop, geom,
    blacklist_rdf, blacklist_expansion, contig_len,
):
    """Yield ``(w, win_start, win_stop, probs, mask, l_valid)`` per window."""
    for w, (win_start, win_stop) in enumerate(
        iter_windows(start, stop, geom.tile_size)
    ):
        probs, mask, l_valid = _predict_window(
            model, fasta, contig, win_start, win_stop, geom,
            blacklist_rdf, blacklist_expansion, contig_len,
        )
        yield w, win_start, win_stop, probs, mask, l_valid


@dataclass(frozen=True)
class RegionProfile:
    probs: np.ndarray          # (C, L) masked softmax, per-window, concatenated
    mask: np.ndarray           # (L,) bool valid
    window_bounds: list        # [(win_start, win_stop, local_lo, local_hi)]
    coord0: int                # genomic position of probs[:, 0] (== start)
    output_tracks: List[str]


@torch.no_grad()
def predict_region_profiles(
    model, fasta, contig: str, start: int, stop: int, *,
    blacklist_rdf=None, blacklist_expansion: int = 120,
    contig_len: Optional[int] = None, tile_size: int = TILE,
) -> RegionProfile:
    """Partition ``[start, stop)`` into ``tile_size`` windows; per window build
    the one-hot + mask, call ``predict_profile``, and CONCATENATE the per-window
    ``probs`` (the trimmed emitted slice per window).

    Seam-free in the SHAPE by construction (design §3.3): each window carries its
    own masked softmax; concatenation is the exact within-window decomposition.
    """
    geom = WindowGeometry.from_model(model, tile_size)
    probs_parts, mask_parts, window_bounds = [], [], []
    for w, win_start, win_stop, probs_w, mask_w, l_valid in _iter_window_predictions(
        model, fasta, contig, start, stop, geom,
        blacklist_rdf, blacklist_expansion, contig_len,
    ):
        local_lo = 0
        local_hi = min(tile_size, stop - win_start)
        probs_parts.append(probs_w[:, local_lo:local_hi])
        mask_parts.append(mask_w[local_lo:local_hi])
        window_bounds.append((win_start, win_stop, local_lo, local_hi))

    probs = np.concatenate(probs_parts, axis=1)
    mask = np.concatenate(mask_parts)
    return RegionProfile(
        probs=probs, mask=mask, window_bounds=window_bounds,
        coord0=int(start), output_tracks=list(model.output_tracks),
    )
