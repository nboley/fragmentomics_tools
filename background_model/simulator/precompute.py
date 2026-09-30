"""Per-region precompute: hexamer indices and cumulative GC (Step 3).

For each region, compute:
  - ``hex_fwd``: forward hexamer index at each cut site (length ``region_len + 1``).
  - ``hex_rc``: reverse-complement hexamer index at the same positions.
  - ``cum_gc``: cumulative G/C count (length ``region_len + 1``).
  - ``valid``: boolean mask (False where the hexamer window contains non-ACGT).

The hexamer at cut site ``c`` spans ``seq[c:c+6]`` — 3 bases inside and 3 bases
outside the fragment at each end (the 3-in/3-out convention from the existing
simulator).  This means we need 3 extra bases of sequence flanking on each side
of the region, which is handled by ``precompute_region``.

The algorithm is the same as ``scripts/sim_fragments.py::hexamer_indices`` and
``precompute_region`` (design doc: "reuse where useful").  Reimplemented here
because the scripts are not importable library code; the logic is small and
self-contained.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import only for the type annotation
    import pysam

# ── hexamer encoding constants ────────────────────────────────────────────

KMER: int = 6
HEX_HALF: int = 3  # 3-in / 3-out

# base -> 2-bit code; anything else (N) -> 255 sentinel
_BASE_LUT = np.full(256, 255, dtype=np.uint8)
_BASE_LUT[ord("A")] = 0
_BASE_LUT[ord("C")] = 1
_BASE_LUT[ord("G")] = 2
_BASE_LUT[ord("T")] = 3

# big-endian positional weights: [4^5, 4^4, ..., 4^0]
_POW = (4 ** np.arange(KMER - 1, -1, -1)).astype(np.int64)


# ── hexamer indexing ──────────────────────────────────────────────────────

def hexamer_indices(seq_bytes: np.ndarray):
    """Sliding 6-mer indices over an ASCII-uint8 sequence.

    Returns ``(fwd_idx, rc_idx, valid)`` each of length ``len(seq_bytes) - 5``.

    - ``fwd_idx[c]``: forward hexamer index for ``seq[c:c+6]``.
    - ``rc_idx[c]``: reverse-complement hexamer index at the same position.
    - ``valid[c]``: False if the 6-mer window contains a non-ACGT base.

    Invalid windows carry index 0 (callers must gate on ``valid``).
    """
    codes = _BASE_LUT[seq_bytes].astype(np.int64)
    win = np.lib.stride_tricks.sliding_window_view(codes, KMER)  # (L-5, 6)
    valid = ~(win == 255).any(axis=1)
    safe = np.where(win == 255, 0, win)
    fwd = safe @ _POW
    rc = (3 - safe)[:, ::-1] @ _POW
    return fwd.astype(np.int64), rc.astype(np.int64), valid


NHEX: int = 4 ** KMER  # 4096


def hexamer_vocabulary() -> np.ndarray:
    """``vocab[i]`` is the 6-mer whose forward index is ``i``, as ``S6`` bytes.

    Derived **from** ``hexamer_indices`` rather than by reimplementing its
    encoding backwards.  All 4096 6-mers are laid end to end and passed through
    the production indexer in one call; taking every 6th sliding window recovers
    each 6-mer's own index, which is then used to place it.  A second,
    hand-written base-4 decoder here would be exactly the shared-contract
    problem this output format exists to remove.

    >>> v = hexamer_vocabulary()
    >>> v.shape, v[0].decode(), v[-1].decode()
    ((4096,), 'AAAAAA', 'TTTTTT')
    """
    grid = np.indices((4,) * KMER).reshape(KMER, -1).T
    letters = np.frombuffer(b"ACGT", dtype=np.uint8)[grid].astype(np.uint8)

    fwd, _rc, valid = hexamer_indices(letters.reshape(-1))
    starts = np.arange(0, NHEX * KMER, KMER)
    idx = fwd[starts]
    assert valid[starts].all()
    assert np.unique(idx).size == NHEX, "hexamer index is not a bijection"

    strings = np.frombuffer(letters.tobytes(), dtype=f"S{KMER}")
    vocab = np.empty(NHEX, dtype=f"S{KMER}")
    vocab[idx] = strings
    return vocab


# ── per-region result ─────────────────────────────────────────────────────

class RegionPrecompute(NamedTuple):
    """Per-region arrays for ``build_region_weights``.

    All arrays have length ``region_len + 1`` (one entry per cut site,
    including position ``region_len``).
    """
    hex_fwd: np.ndarray
    hex_rc: np.ndarray
    cum_gc: np.ndarray
    valid: np.ndarray


def precompute_region(
    contig: str,
    gstart: int,
    gstop: int,
    fasta_path: str,
    fasta: "Optional[pysam.FastaFile]" = None,
) -> RegionPrecompute:
    """Compute hexamer indices and cumulative GC for one region.

    Fetches the region plus ``HEX_HALF`` (3 bp) flanking on each side so
    that hexamer windows for cut sites at positions 0 through ``region_len``
    are fully resolved.

    Parameters
    ----------
    contig : str
        Chromosome name.
    gstart, gstop : int
        0-based half-open genomic coordinates ``[gstart, gstop)``.
    fasta_path : str
        Path to an indexed FASTA file.  Used only when *fasta* is None.
    fasta : pysam.FastaFile, optional
        An already-open handle to reuse.  Opening and closing a
        ``FastaFile`` costs ~5 ms against the NFS-hosted hg38, which is
        **89% of this function's cost** -- the actual work (hexamer
        indexing, cumulative GC, the fetch itself) is ~0.5 ms.  A caller
        looping over many regions should open the handle once and pass it
        here.  The sequence fetched is identical either way, so this
        changes cost only, never the returned arrays.

    Returns
    -------
    RegionPrecompute
        Named tuple with ``hex_fwd``, ``hex_rc``, ``cum_gc``, ``valid``.
    """
    import pysam

    region_len = gstop - gstart

    if gstart < HEX_HALF:
        raise ValueError(
            f"gstart={gstart} < HEX_HALF={HEX_HALF}: cannot fetch the "
            f"{HEX_HALF}-bp left flank needed for cut-site hexamers. "
            f"Regions must start at least {HEX_HALF} bp from the "
            f"chromosome start."
        )

    # Fetch with HEX_HALF flanking on each side for cut-site hexamers
    if fasta is not None:
        seq = fasta.fetch(contig, gstart - HEX_HALF, gstop + HEX_HALF).upper()
    else:
        with pysam.FastaFile(fasta_path) as fa:
            seq = fa.fetch(contig, gstart - HEX_HALF, gstop + HEX_HALF).upper()
    seq_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)

    # Cut-site hexamers: seq_bytes[c : c+6] for c in 0..region_len
    # The fetched seq covers [gstart - 3, gstop + 3), so seq_bytes has
    # length region_len + 2*HEX_HALF.  Sliding window produces
    # region_len + 2*HEX_HALF - 5 = region_len + 1 entries — exactly
    # one per cut site.
    fwd_cut, rc_cut, valid = hexamer_indices(seq_bytes)
    assert len(fwd_cut) == region_len + 1, (
        f"hex length {len(fwd_cut)} != {region_len + 1}"
    )

    # Cumulative GC over the core region bases [gstart, gstop)
    core = seq_bytes[HEX_HALF : HEX_HALF + region_len]
    is_gc = (core == ord("G")) | (core == ord("C"))
    cum_gc = np.empty(region_len + 1, dtype=np.float64)
    cum_gc[0] = 0
    np.cumsum(is_gc, out=cum_gc[1:])

    return RegionPrecompute(
        hex_fwd=fwd_cut,
        hex_rc=rc_cut,
        cum_gc=cum_gc,
        valid=valid,
    )
