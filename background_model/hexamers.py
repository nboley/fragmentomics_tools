"""Cut-site hexamer encoding -- the one encoder in ``background_model``.

numpy only, so anything that needs hexamer indices can import it without
pulling in the fragment library, pandas or the simulator.  Measurement on real
data is ``background_model.cut_site_stats``; the draw is
``background_model.simulator.draw``.  Authority: ``docs/pending/simulator_spec.md``.

The hexamer at cut site ``c`` spans genomic ``[c - HEX_HALF, c + HEX_HALF)``
-- 3 bases inside the fragment and 3 outside.

``hexamer_indices`` is the encoder, and everything else here is DERIVED from
it rather than restating its convention: ``hexamer_vocabulary`` decodes by
pushing all 4096 6-mers through it, and the reverse complement
(``rc_permutation``) is a permutation read off its own reverse-complement
output.  So no second base-4 or complement convention exists anywhere.

Case is folded in the lookup table, so soft-masked (lowercase) reference
sequence encodes like uppercase.  An invalid window -- one holding a
non-ACGTacgt base -- carries the index of its N-as-A reading, not a sentinel;
callers must gate on ``valid``.  See ``hexamer_indices``.
"""

from __future__ import annotations

import functools

import numpy as np

# ── encoding, owned here ──────────────────────────────────────────────────
#
# These were once imported from simulator.precompute and simulator.weights,
# the previous-generation simulator now in attic/pre_rewrite_simulator/.
# Owning them here is what let that package be retired.
#
# test_encoder_matches_oracle_all_4096 guards this encoder by checking all 4096
# hexamers against an INDEPENDENT oracle -- which is stronger than pinning two
# copies to each other, since those could drift in step and still agree.

KMER: int = 6
HEX_HALF: int = 3            # 3 bases in, 3 out, around a cut site
NHEX: int = 4 ** KMER        # 4096

# base -> 2-bit code; anything else (N) -> 255 sentinel.
#
# BOTH cases are mapped. hg38 is soft-masked, so repeat bases arrive lowercase,
# and an uppercase-only table sends every one of them to the 255 sentinel --
# silently discarding exactly the repeat-rich positions. Folding case here
# rather than at each call site means one caller cannot forget it while another
# remembers, and it covers callers that pass a uint8 array, which a string
# .upper() cannot reach.
_BASE_LUT = np.full(256, 255, dtype=np.uint8)
for _code, _base in enumerate("ACGT"):
    _BASE_LUT[ord(_base)] = _code
    _BASE_LUT[ord(_base.lower())] = _code

# big-endian positional weights: [4^5, 4^4, ..., 4^0]
_POW = (4 ** np.arange(KMER - 1, -1, -1)).astype(np.int64)


def hexamer_indices(seq):
    """Sliding 6-mer indices over a sequence.

    ``seq`` may be ``str``, ``bytes`` or an ASCII ``uint8`` array -- the
    conversion lives here so no caller repeats it, and case is folded by
    ``_BASE_LUT`` so soft-masked reference sequence needs no ``.upper()``.

    Returns ``(fwd_idx, rc_idx, valid)``, each of length ``len(seq) - 5``.

    - ``fwd_idx[c]``: forward hexamer index for ``seq[c:c+6]``.
    - ``rc_idx[c]``: reverse-complement index at the same position.
    - ``valid[c]``: False if the window contains a non-ACGTacgt base.

    **An invalid window carries the index of its N-as-A reading, NOT index 0.**
    ``safe = np.where(win == 255, 0, win)`` zeroes the offending BASE, not the
    window, so ``ACGTAN`` returns the index of ``ACGTAA`` (432) and only an
    all-N window returns 0.  A caller that does not gate on ``valid`` therefore
    miscounts an N-containing window into a NEIGHBOURING hexamer -- one
    differing only at the N positions -- which is harder to notice than
    everything piling into a single cell.  Measured, after the docstring here
    claimed index 0 for months and that claim was repeated downstream.

    >>> fwd, rc, valid = hexamer_indices("acgtAC")
    >>> int(fwd[0]) == int(hexamer_indices("ACGTAC")[0][0]), bool(valid[0])
    (True, True)
    """
    if isinstance(seq, str):
        seq = seq.encode("ascii")
    if isinstance(seq, (bytes, bytearray, memoryview)):
        seq = np.frombuffer(seq, dtype=np.uint8)
    codes = _BASE_LUT[seq].astype(np.int64)
    win = np.lib.stride_tricks.sliding_window_view(codes, KMER)  # (L-5, 6)
    valid = ~(win == 255).any(axis=1)
    safe = np.where(win == 255, 0, win)
    fwd = safe @ _POW
    rc = (3 - safe)[:, ::-1] @ _POW
    return fwd.astype(np.int64), rc.astype(np.int64), valid


def hexamer_vocabulary() -> np.ndarray:
    """``vocab[i]`` is the 6-mer whose forward index is ``i``, as ``S6`` bytes.

    Derived **from** ``hexamer_indices`` rather than by inverting its encoding:
    all 4096 6-mers are laid end to end, pushed through the indexer in one
    call, and every 6th sliding window recovers that 6-mer's own index.  A
    hand-written base-4 decoder here would be the shared-contract problem this
    function exists to remove.

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


@functools.lru_cache(maxsize=1)
def rc_permutation() -> np.ndarray:
    """``perm[i]`` is the index of the reverse complement of hexamer ``i``.

    Derived FROM the production encoder rather than from a complement table:
    the vocabulary is laid end to end and pushed through ``hexamer_indices``,
    which reports both the forward and the reverse-complement index of every
    window, so ``perm[fwd] = rc``.  A hand-written complement is the
    shared-contract problem ``hexamer_vocabulary`` exists to remove.

    Verified a true permutation of ``0..4095`` and an involution
    (``perm[perm] == identity``), with ``AAAAAA -> TTTTTT``.
    """
    vocab = hexamer_vocabulary()
    fwd, rc, valid = hexamer_indices(
        np.frombuffer(b"".join(vocab.tolist()), dtype=np.uint8)
    )
    take = np.arange(0, NHEX * KMER, KMER)
    if not valid[take].all():
        raise AssertionError("vocabulary contains an invalid hexamer")
    perm = np.empty(NHEX, dtype=np.int64)
    perm[fwd[take]] = rc[take]
    if np.unique(perm).size != NHEX:
        raise AssertionError("reverse-complement map is not a permutation")
    return perm


def _hexamers_at(seq: np.ndarray, pos: np.ndarray):
    """``(index, valid)`` for the 6-mer at each cut site in ``pos``.

    ``pos`` is region-local.  With ``left_pad == HEX_HALF`` the window covering
    genomic ``[c-3, c+3)`` begins at sequence offset ``c``, so no pad term is
    needed -- the region-local coordinate indexes the sequence directly.

    Encoding goes through ``hexamer_indices`` via the stride trick
    ``hexamer_vocabulary`` uses: lay the 6-mers end to end, slide, keep every
    ``KMER``-th.  An out-of-range ``pos`` raises ``IndexError`` from NumPy,
    which is the caller's contract to satisfy.

    **``valid`` is returned because it cannot be inferred from the index.**
    ``hexamer_indices`` gives an invalid window -- one containing a non-ACGT
    base -- the index of its **N-as-A reading**, not a sentinel and not ``0``:
    ``ACGTAN`` returns the index of ``ACGTAA``, and only an all-N window
    returns 0.  So a caller that ignores ``valid`` does not *lose* N-containing
    cut sites, it miscounts each one into a NEIGHBOURING hexamer -- one
    differing from the truth only at the N positions -- which is
    indistinguishable from a genuine observation of that hexamer.  The flag is
    the only way to tell them apart.  Both of a fragment's cut sites must be
    valid for it to count, which is why this returns the flag rather than
    filtering: the caller has to AND the two together.

    (This docstring said "index 0 / ``AAAAAA``" until 2026-10-07. The sibling
    claim in ``hexamer_indices`` was corrected first and this one was missed,
    so the wrong version survived one round of fixing it.)
    """
    if isinstance(seq, str):
        seq = seq.encode("ascii")
    if isinstance(seq, (bytes, bytearray, memoryview)):
        seq = np.frombuffer(seq, dtype=np.uint8)
    fwd, _rc, valid = hexamer_indices(
        seq[pos[:, None].astype(np.int64) + np.arange(KMER)[None, :]].reshape(-1)
    )
    return fwd[::KMER], valid[::KMER]
