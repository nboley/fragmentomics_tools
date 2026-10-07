"""Cut-site hexamer counts, observed and expected, via the fragmentomics_tools API.

Two artifacts, with different lifetimes:

- ``C(h)`` -- observed cut-site counts, **per sample**.  ``count_sample``, or
  the three passes by hand.  Parallel.  The same pass also returns the
  per-region admitted counts (the spec's stage 4), since the frame it already
  built is the only place that count exists; see ``count_srdf`` for the
  distinction between those and the hexamer-valid totals.
- ``N(h)`` -- EXPECTED hexamer counts under a uniform start/end null.
  ``uniform_hexamer_counts``.  Serial.

``propensities(C, N)`` divides them into the per-site rate ``r(h) = C(h)/N(h)``,
which is what the sampler needs.  Counts alone are ``r(h)·N(h)``, so feeding
them to a sampler that re-enumerates candidate positions applies hexamer
abundance twice.  The spread in ``N`` across hexamers is large, so this is not
a small correction.

The ``C(h)`` path is three chained passes over a ``SampleAndRegionDataFrame``:

1. ``attach_fragment_arrays(min_mapq=..., fragment_array_callback=filter_fragments)``
   fetch and MAPQ from the library, then dedup, the length filter and
   start-in-region admission, all in the loading worker.
2. ``attach_sequence(left_pad=HEX_HALF, right_pad=L_MAX + HEX_HALF)``.  Must
   follow pass 1: ``expand_regions`` refuses a frame that already carries
   fragment arrays.
3. ``count_srdf`` -- one ``parallel_apply`` of ``cut_site_hexamers``, reduced by
   four ``bincount`` calls.

The fragment half is the library's; the sequence half is not, because
``fragmentomics_tools`` has no hexamer machinery.  ``hexamer_indices`` from
``simulator.precompute`` stays the encoder, and the reverse complement is a
permutation DERIVED from it, so no second base-4 or complement convention
exists anywhere here.

Geometry
--------
Admission is **start-in-region**: a fragment belongs to the region containing
its genomic start.  Tiles are contiguous, so any single-point rule assigns each
fragment to exactly one tile; start is chosen because midpoint makes the
admissible LENGTH set depend on the start position, which breaks the
``P(start)`` normalisation at region edges (9.7% of positions affected, 1 of 156
lengths admissible at the extreme edge).

The sequence frame is therefore **asymmetric**: ``left_pad = HEX_HALF`` and
``right_pad = L_MAX + HEX_HALF``, since starts need only 3 bases of left context
while an end reaches ``start + L_MAX``.  A symmetric ``MAX_FL_HALF`` flank is
NOT enough.  Keeping ``left_pad == HEX_HALF`` is what makes a hexamer index
equal its region-local coordinate, so no pad term appears in the lookup at all.
``uniform_hexamer_counts`` uses the same frame, so one attached ``sequence`` column
serves both.

Two library details that bite if assumed rather than checked:

- ``rfa.fragment_strands`` is ``<U1`` (``'+'``), NOT the ``b'+'`` bytes
  ``FragmentsH5.fetch_array`` returns.  Comparing against bytes silently
  yields an all-False mask and an empty minus-strand table.
- ``starts_0``/``stops_0`` are **not clipped** to the region.  A fragment
  reaching in from the previous tile has a NEGATIVE start and belongs to that
  tile; one admitted here may overhang the far end by up to ``L_MAX``, which is
  what the right flank exists to cover.
- ``subset_fragment_lengths`` is half-open ``[min, max)`` while ``L_MAX`` is an
  inclusive bound, hence ``l_max + 1`` at the call site.

The per-region return is three columns -- ``start_hex``, ``stop_hex``,
``strand`` -- one row per fragment, naming the GENOMIC ends with no strand
convention applied; ``counts_from_hexamers`` owns that mapping.  A dense
4x4096 per region would be ~200x larger than the fragments it summarises, and
1.09e9 values over the full region set.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import functools

import numpy as np
import pandas as pd

from fragmentomics_tools.dataframe import (
    SampleAndRegionDataFrame,
    SampleDataFrame,
)

# ── geometry and encoding, owned here ─────────────────────────────────────
#
# These were imported from simulator.precompute and simulator.weights, which
# belong to the simulator being replaced.  Owning them makes this file
# independent of that package so it survives its deletion.
#
# While BOTH copies exist they are a shared-contract hazard: two encoders free
# to drift, with counts diverging silently.  test_encoder_matches_precompute
# pins them bit-for-bit until the old package goes.

KMER: int = 6
HEX_HALF: int = 3            # 3 bases in, 3 out, around a cut site
NHEX: int = 4 ** KMER        # 4096
L_MIN: int = 25              # INCLUSIVE length bounds -- the library's
L_MAX: int = 180             # subset_fragment_lengths is half-open
N_LENGTHS: int = L_MAX - L_MIN + 1   # 156

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

TABLE_NAMES = ("start_fwd", "end_fwd", "start_rev", "end_rev")


def hexamer_indices(seq):
    """Sliding 6-mer indices over a sequence.

    ``seq`` may be ``str``, ``bytes`` or an ASCII ``uint8`` array -- the
    conversion lives here so no caller repeats it, and case is folded by
    ``_BASE_LUT`` so soft-masked reference sequence needs no ``.upper()``.

    Returns ``(fwd_idx, rc_idx, valid)``, each of length ``len(seq) - 5``.

    - ``fwd_idx[c]``: forward hexamer index for ``seq[c:c+6]``.
    - ``rc_idx[c]``: reverse-complement index at the same position.
    - ``valid[c]``: False if the window contains a non-ACGTacgt base.

    **Invalid windows carry index 0**, so a caller that does not gate on
    ``valid`` miscounts every N-containing window as ``AAAAAA`` rather than
    losing it.

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


def empty_counts() -> Dict[str, np.ndarray]:
    """Four zeroed ``(4096,)`` int64 tables."""
    return {k: np.zeros(NHEX, dtype=np.int64) for k in TABLE_NAMES}


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


def load_sample_dataframe(samples: Iterable[Tuple[str, str]]):
    """A ``SampleDataFrame`` from explicit ``(sample_id, h5_path)`` pairs.

    ``sample_id`` and ``frag_h5`` are its required columns.  Paths are kept as
    strings, not opened: ``SampleDataFrame`` can hold live ``FragmentsH5``
    handles, but those break pickling until ``detach_h5()`` is called, and
    ``parallel_apply`` has to move this frame to workers.
    """
    pairs = list(samples)
    if not pairs:
        raise ValueError("no samples given")
    return SampleDataFrame(pd.DataFrame({
        "sample_id": [s for s, _ in pairs],
        "frag_h5": [p for _, p in pairs],
    }))


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
    base -- the index ``0``, not a sentinel.  A caller that ignores ``valid``
    therefore does not *lose* N-containing cut sites, it miscounts every one of
    them as ``AAAAAA``, inflating a real cell with garbage.  Index 0 is
    indistinguishable from a genuine ``AAAAAA``, so the flag is the only way to
    tell them apart.  Both of a fragment's cut sites must be valid for it to
    count, which is why this returns the flag rather than filtering: the caller
    has to AND the two together.
    """
    if isinstance(seq, str):
        seq = seq.encode("ascii")
    if isinstance(seq, (bytes, bytearray, memoryview)):
        seq = np.frombuffer(seq, dtype=np.uint8)
    fwd, _rc, valid = hexamer_indices(
        seq[pos[:, None].astype(np.int64) + np.arange(KMER)[None, :]].reshape(-1)
    )
    return fwd[::KMER], valid[::KMER]


def cut_site_hexamers(rfa, sequence) -> "pd.DataFrame":
    """One region's fragments as ``start_hex``, ``stop_hex``, ``strand``.

    One row per admitted fragment.  ``start_hex``/``stop_hex`` are the forward
    hexamer indices at the fragment's GENOMIC start and stop -- purely
    positional, with no strand convention applied.  Mapping those onto the four
    untied tables (which 5' cut site, and whether to reverse-complement) is
    ``counts_from_hexamers``' job, so every convention lives in one place.

    **This function does no filtering.** Dedup, the length filter and
    start-in-region admission are the caller's responsibility -- see
    ``filter_fragments``, passed as the ``fragment_array_callback``
    so they run in the loading worker.  The contract here is that every fragment
    handed in fits inside ``sequence``; a position that does not raises
    ``IndexError`` from NumPy rather than being silently skipped.

    ``sequence`` must carry ``left_pad = HEX_HALF`` and
    ``right_pad >= L_MAX + HEX_HALF``, which is what lets a region-local
    coordinate index it directly with no pad term.

    Invalid windows -- those containing a non-ACGT base -- are dropped, since a
    fragment is only usable when BOTH its cut sites encode.
    """
    if rfa.n_frags == 0:
        return pd.DataFrame({"start_hex": np.empty(0, np.int32),
                             "stop_hex": np.empty(0, np.int32),
                             "strand": np.empty(0, "<U1")})
    s_hex, s_ok = _hexamers_at(sequence, rfa.starts_0)
    e_hex, e_ok = _hexamers_at(sequence, rfa.stops_0)
    ok = s_ok & e_ok
    return pd.DataFrame({
        "start_hex": s_hex[ok].astype(np.int32),
        "stop_hex": e_hex[ok].astype(np.int32),
        "strand": np.asarray(rfa.fragment_strands)[ok],
    })


def counts_from_hexamers(df) -> Dict[str, np.ndarray]:
    """Map genomic-start/stop hexamers onto the four untied count tables.

    Every strand convention lives here. For a plus-strand fragment the 5' cut
    site is the genomic START; for a minus-strand fragment it is the genomic
    STOP, and the hexamer is read on the other strand -- so the minus tables
    index through ``rc_permutation()``. Getting that pair backwards silently
    destroys the strand asymmetry the four untied tables exist to capture.
    """
    counts = empty_counts()
    if not len(df):
        return counts
    plus = df["strand"].to_numpy() == "+"
    s = df["start_hex"].to_numpy()
    e = df["stop_hex"].to_numpy()
    perm = rc_permutation()
    counts["start_fwd"] = np.bincount(s[plus], minlength=NHEX)
    counts["end_fwd"] = np.bincount(e[plus], minlength=NHEX)
    counts["start_rev"] = np.bincount(perm[e[~plus]], minlength=NHEX)
    counts["end_rev"] = np.bincount(perm[s[~plus]], minlength=NHEX)
    return counts


class FragmentLengthDist:
    """Fragment-length distribution over ``[min_fl, max_fl]``.

    Attributes: ``counts`` (int64), ``densities`` (float64, sums to 1),
    ``min_fl``, ``max_fl``.  Both vectors have length ``max_fl - min_fl + 1``
    and are indexed by ``L - min_fl``.

    The support is whatever the caller supplies.  Built from an SRDF after
    ``filter_fragments`` it is bounded by the admission filter, but the class
    does not assume that and does not impose ``L_MIN`` / ``L_MAX``.
    """

    __slots__ = ("counts", "densities", "min_fl", "max_fl", "_cdf")

    def __init__(self, counts, min_fl: int):
        # copy=True so the instance owns its buffer: the arrays are frozen
        # below, and freezing a caller's array would be a surprising side
        # effect.
        counts = np.array(counts, dtype=np.int64, copy=True)
        if counts.ndim != 1 or counts.size == 0:
            raise ValueError(f"counts must be 1-D and non-empty, got {counts.shape}")
        if (counts < 0).any():
            raise ValueError("counts has negative entries")
        total = int(counts.sum())
        if total == 0:
            raise ValueError("counts sum to 0; densities would be undefined")
        self.counts = counts
        self.min_fl = int(min_fl)
        self.max_fl = int(min_fl) + counts.size - 1
        self.densities = counts / np.float64(total)

        # Cumulative density, built once. cdf_at is called twice per region by
        # fl_end_weight, so recomputing here would mean ~133k cumsums over the
        # 66,649-region set, inside the loop.
        #
        # _cdf[k] is the mass of the first k lengths, so _cdf[0] == 0 and
        # _cdf[-1] == 1. Both vectors are then made READ-ONLY: an in-place
        # edit of counts or densities would leave this cache stale, and a
        # stale CDF fails silently.
        self._cdf = np.concatenate([[0.0], np.cumsum(self.densities)])
        self.counts.setflags(write=False)
        self.densities.setflags(write=False)
        self._cdf.setflags(write=False)

    @classmethod
    def from_dataframe(cls, df) -> "FragmentLengthDist":
        """From a frame with columns ``fragment_length`` and ``count``.

        Lengths absent from the frame between the min and max become zero
        counts.  Densifying here is the point: a caller that indexed a frame
        with gaps by ``L - min_fl`` would read a neighbouring length's count
        and never find out.
        """
        missing = {"fragment_length", "count"} - set(df.columns)
        if missing:
            raise ValueError(f"frame is missing column(s) {sorted(missing)}")
        fl = np.asarray(df["fragment_length"], dtype=np.int64)
        ct = np.asarray(df["count"], dtype=np.int64)
        if fl.size == 0:
            raise ValueError("frame is empty")
        if np.unique(fl).size != fl.size:
            raise ValueError(
                "fragment_length has duplicate values; aggregate before "
                "constructing, since silently summing them would hide a "
                "double-counted population"
            )
        lo, hi = int(fl.min()), int(fl.max())
        counts = np.zeros(hi - lo + 1, dtype=np.int64)
        counts[fl - lo] = ct
        return cls(counts, lo)

    @classmethod
    def from_srdf(cls, srdf) -> "FragmentLengthDist":
        """From an SRDF that already carries filtered ``fragment_array``s.

        Call AFTER ``filter_fragments``, so this is the same population
        ``C(h)`` counts.
        """
        if "fragment_array" not in srdf.columns:
            raise ValueError(
                "srdf has no 'fragment_array' column -- call "
                "attach_fragment_arrays first"
            )
        fas = [fa for fa in srdf["fragment_array"] if fa.n_frags]
        if not fas:
            raise ValueError("no fragments in any region")
        hi = max(int(fa.lengths.max()) for fas_ in (fas,) for fa in fas_)
        counts = np.zeros(hi + 1, dtype=np.int64)
        for fa in fas:
            counts += np.bincount(fa.lengths, minlength=hi + 1)
        nz = np.nonzero(counts)[0]
        lo = int(nz[0])
        return cls(counts[lo:], lo)

    def cdf_at(self, L):
        """Total density with ``L' <= L``. Reads the cached CDF.

        Clipped outside the support: below ``min_fl`` returns 0, at or above
        ``max_fl`` returns 1.  Accepts a scalar or an array.
        """
        return self._cdf[np.clip(np.asarray(L) - self.min_fl + 1, 0,
                                 self.counts.size)]

    def __repr__(self):
        return (
            f"FragmentLengthDist(min_fl={self.min_fl}, max_fl={self.max_fl}, "
            f"n={int(self.counts.sum())})"
        )


def fl_end_weight(n_hex: int, region_len: int, fl: "FragmentLengthDist"):
    """Expected end-position weight per hexamer index, under a uniform start.

    Start and end expectations are NOT symmetric.  With starts uniform over
    ``[gstart, gstop)``, an end at offset ``i`` is reachable from ``s = i - L``
    for every ``L`` whose start still lands in the region, and each route
    carries FL mass ``f(L)``.  Interior ends are reachable by every length and
    so carry total weight 1; ends within ``max_fl`` of either edge carry a
    partial sum.  Weighting every end equally would understate the denominator
    at the edges and inflate ``r_end`` there.

    One closed form rather than one window per length:

        w(i) = F(min(i, max_fl)) - F(max(min_fl - 1, i - region_len))

    for FL CDF ``F``, so the cost is ``O(region_len + n_lengths)``.
    """
    i = np.arange(n_hex)
    return (
        fl.cdf_at(np.minimum(i, fl.max_fl))
        - fl.cdf_at(np.maximum(fl.min_fl - 1, i - region_len))
    )


def uniform_hexamer_counts(
    rdf,
    fasta_path: str,
    fl: "FragmentLengthDist",
    *,
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """``N(h)``: EXPECTED hexamer counts under a uniform start/end null.

    Returns ``({"start": N_start, "end": N_end}, meta)``.  ``N_start`` is
    int64; ``N_end`` is float64, since FL weights are fractional.

    The same tabulation ``cut_site_hexamers`` performs on real fragments, but
    with every start site taken as equally likely, so ``r(h) = C(h)/N(h)`` is
    observed over expected.  Two consequences of that framing:

    - **``C`` and ``N`` must cover the same candidate set**, since they are one
      object counted two ways.  A uniform scale factor is harmless --
      ``P(start)`` renormalises within each region -- but the region set is
      quiet, repeat- and blacklist-filtered, so its composition differs from
      the genome's PER HEXAMER.  A genome-wide ``N`` against a region-set
      ``C`` reintroduces the bias the division removes.
    - **The asymmetric start/end weighting is the null's own prediction**, not
      a correction imposed on it.  A uniform start gives every position in
      ``[gstart, gstop)`` weight 1; an end at offset ``i`` is reached from
      ``s = i - L`` for each admissible ``L``, with probability ``f(L)`` per
      route.

    ``N_start`` is sequence-only, so it is a property of
    ``(region set, reference)``.  ``N_end`` is ``f(L)``-weighted and therefore
    keyed to ``(region set, reference, FL)``.  Note ``f`` only reaches the
    edge ramp: interior positions carry weight 1 under any normalised ``f``.

    Serial by design: returning per-region tables through ``parallel_apply``
    would move gigabytes to reduce 32 KB, the mirror image of the fragment
    counting where the payload is tiny and parallelism wins.

    Only the FORWARD tables are returned.  The minus-strand expectation is the
    forward table permuted -- ``N_rc == N_fwd[rc_permutation()]`` -- because a
    reverse-complement hexamer at a position is a relabelling, not a different
    position.
    """
    import pysam

    if "fragment_array" in getattr(rdf, "columns", ()):
        raise ValueError(
            "pass a region frame WITHOUT fragment arrays: the uniform counts are "
            "sequence-only and region-set-scoped, and attaching fragments "
            "first blocks the padded sequence fetch (expand_regions refuses a "
            "frame that already carries them)"
        )

    # Starts need only HEX_HALF of left context. An end's hexamer window
    # reaches gstop-1 + max_fl + HEX_HALF, so the RIGHT flank must cover that
    # -- asymmetric. left_flank = HEX_HALF makes a hexamer index equal its
    # region-local coordinate, which is why it is not widened for symmetry.
    left_flank = HEX_HALF
    right_flank = fl.max_fl + HEX_HALF

    N_start = np.zeros(NHEX, dtype=np.int64)
    N_end = np.zeros(NHEX, dtype=np.float64)
    meta = dict(n_regions=0, n_start_positions=0, n_start_invalid=0,
                end_weight_total=0.0)

    fasta = pysam.FastaFile(fasta_path)
    try:
        rows = rdf.itertuples()
        if verbose:
            from tqdm import tqdm
            rows = tqdm(rows, total=len(rdf), desc="uniform counts")
        for row in rows:
            gstart, gstop = int(row.start), int(row.stop)
            region_len = gstop - gstart
            seq = fasta.fetch(
                row.contig, gstart - left_flank, gstop + right_flank
            )
            fwd, _rc, valid = hexamer_indices(seq)
            n_hex = len(fwd)
            if n_hex < region_len + fl.max_fl:
                raise ValueError(
                    f"{row.contig}:{gstart}-{gstop}: {n_hex} hexamer windows, "
                    f"need at least {region_len + fl.max_fl} -- the FASTA "
                    f"fetch was truncated, which happens within "
                    f"{right_flank}bp of a contig end."
                )

            # Starts: uniform, one unit per position in [gstart, gstop).
            s_idx = slice(0, region_len)
            s_valid = valid[s_idx]
            N_start += np.bincount(fwd[s_idx][s_valid], minlength=NHEX)
            meta["n_start_positions"] += region_len
            meta["n_start_invalid"] += int((~s_valid).sum())

            # Ends: FL-weighted. The validity gate must match the fragment
            # pass, or N is inflated at exactly the repeat-rich and
            # gap-adjacent positions where r would then be depressed.
            w = fl_end_weight(n_hex, region_len, fl)
            m = valid & (w > 0)
            N_end += np.bincount(fwd[m], weights=w[m], minlength=NHEX)
            meta["end_weight_total"] += float(w[m].sum())
            meta["n_regions"] += 1
    finally:
        fasta.close()

    return {"start": N_start, "end": N_end}, meta


def sample_region(
    sequence,
    region_len: int,
    n: int,
    *,
    r: Dict[str, np.ndarray],
    fl: "FragmentLengthDist",
    p_plus: float,
    rng,
):
    """Draw ``n`` fragments in one region. Returns ``(starts_0, lengths, is_plus)``.

    Per fragment: strand, then start, then length.

    - **Strand** ~ Bernoulli(``p_plus``).  It selects the tables, because a
      minus-strand fragment's genomic start is its 3' end: plus reads
      ``start_fwd`` / ``end_fwd`` off ``hex_fwd``, minus reads ``end_rev`` /
      ``start_rev`` off ``hex_rc``.
    - **Start** ``i`` over ``[0, region_len)``, proportional to the start-side
      propensity, normalised within the region.
    - **Length** over ``[min_fl, max_fl]``, proportional to
      ``end_p x f(L)``, normalised over ``L``.

    Vectorised over fragments: all ``n`` share the region's arrays, so the
    length draw is one ``(n, n_lengths)`` block.
    """
    seq = np.frombuffer(bytes(sequence).upper(), dtype=np.uint8)
    fwd, rc, valid = hexamer_indices(seq)
    Ls = np.arange(fl.min_fl, fl.max_fl + 1)

    n_plus = int(rng.binomial(n, p_plus))
    out_s, out_L, out_p = [], [], []

    for is_plus, k in ((True, n_plus), (False, n - n_plus)):
        if k == 0:
            continue
        track = fwd if is_plus else rc
        s_tab = r["start_fwd"] if is_plus else r["end_rev"]
        e_tab = r["end_fwd"] if is_plus else r["start_rev"]

        # Start: propensity over in-region positions, gated on a valid hexamer.
        pos = np.arange(region_len)
        w_s = s_tab[track[pos]] * valid[pos]
        tot = w_s.sum()
        if tot <= 0:
            continue
        starts = rng.choice(pos, size=k, replace=True, p=w_s / tot)

        # Length: one (k, n_lengths) block.
        ends = starts[:, None] + Ls[None, :]
        w = e_tab[track[ends]] * valid[ends] * fl.densities[None, :]

        tot = w.sum(axis=1)
        live = tot > 0
        if not live.any():
            continue
        w, starts, tot = w[live], starts[live], tot[live]
        cdf = np.cumsum(w / tot[:, None], axis=1)
        # NOT clamped, deliberately. float64 throughout (e_tab and fl.densities
        # are both float64), but cumsum is a SEQUENTIAL accumulation, so
        # cdf[-1] lands just under 1.0 in ~43% of rows, by at most ~6.5 eps
        # (measured). `pick` can therefore reach len(Ls) when u > cdf[-1] --
        # probability 1.3e-16 per draw, about one occurrence per 3e9 full runs.
        # If that ever happens we want the IndexError: clamping it would hand
        # one fragment the longest length silently, and a loud failure at
        # 1-in-3e9 is worth more than a quiet wrong value.
        pick = (cdf < rng.random((len(starts), 1))).sum(axis=1)

        out_s.append(starts)
        out_L.append(Ls[pick])
        out_p.append(np.full(len(starts), is_plus))

    if not out_s:
        z = np.zeros(0, dtype=np.int64)
        return z, z, np.zeros(0, dtype=bool)
    return (np.concatenate(out_s).astype(np.int64),
            np.concatenate(out_L).astype(np.int64),
            np.concatenate(out_p))


def simulate_fragments_to_bed(
    srdf,
    out_path: str,
    *,
    r: Dict[str, np.ndarray],
    fl: "FragmentLengthDist",
    region_counts,
    rng,
    p_plus: float = 0.5,
    l_max: int = L_MAX,
    mapq: int = 60,
) -> Dict[str, int]:
    """Draw fragments for every region and write an 8-column BED.

    The BED is the input to ``fragments_h5.build_fragments_h5``, which needs it
    bgzipped and tabix-indexed -- that is the caller's next step, not this
    function's.

    ``region_counts`` is the per-row ``n``, as returned by ``count_srdf``.
    ``rng`` is required rather than defaulted: an unseeded run cannot be
    reproduced, and the seed is this artifact's only provenance.

    Column layout, verified against ``fragments_h5.fragment.tsv_to_fragments``:

        contig  start  stop  <empty>  0  strand  mapq  mapq

    - **Exactly 8 columns.** 7 is rejected outright by the reader, and a row
      whose column count differs from the first row's is skipped.
    - **Column 4 is EMPTY, not** ``"."``. The reader takes
      ``parts[3] if parts[3] else None``, and ``"."`` is truthy -- it would
      write a literal ``"."`` cell barcode into the h5. Empty writes none.
    - **MAPQ is written explicitly** in columns 7 and 8, and must be 0-255.
      Omitting it stores a 255 sentinel that reads back as ``-1``, and the
      model's ``min_mapq=10`` then drops **every** fragment. The default 60
      clears that with room to spare.
    - 0-based half-open, matching ``starts_0``/``stops_0``.

    Output is sorted by ``(contig, start, stop)`` because tabix requires each
    contig's records contiguous and position-ordered. Sorting here is why
    ``sample_region``'s draw order is not part of its contract.

    Next step, verified end to end against the real reader::

        gz = pysam.tabix_index(bed_path, preset="bed", force=True)
        build_fragments_h5(gz, out_h5, fasta_filename=...)   # FASTA required

    **``pysam.tabix_index`` CONSUMES the plain BED** -- measured: after the call
    only ``sim.bed.gz`` and ``sim.bed.gz.tbi`` remain. Do not plan to re-read or
    hash the plain file afterwards. Convenient for the two-output contract,
    since the intermediate deletes itself, but surprising if unexpected.

    Returns a stats dict. ``n_drawn < n_requested`` is possible and is reported
    rather than raised -- see the spec's Settled note on the dropped-start
    shortfall.
    """
    if out_path.endswith(".gz"):
        raise ValueError(
            f"write a PLAIN bed, got {out_path!r}. pandas would gzip it, and "
            f"tabix needs BGZIP, which gzip is not -- the index step would fail "
            f"on a file that looks correct. bgzip it as a separate step."
        )
    region_counts = np.asarray(region_counts)
    if region_counts.shape != (len(srdf),):
        raise ValueError(
            f"region_counts has shape {region_counts.shape}, expected "
            f"({len(srdf)},) -- one entry per row of srdf, in row order"
        )
    for col in ("contig", "start", "stop", "fragment_array", "sequence"):
        if col not in srdf.columns:
            raise ValueError(f"srdf has no {col!r} column")

    # The frame sample_region assumes: left_pad = HEX_HALF makes a hexamer index
    # equal its region-local coordinate, and right_pad = l_max + HEX_HALF covers
    # a fragment that overhangs the far edge. Checking the exact length pins
    # BOTH pads, and is the only thing standing between us and a silently
    # truncated sequence at a contig end -- which would shift every hexamer
    # index without any other symptom.
    expected_seq_len_extra = 2 * HEX_HALF + l_max

    chunks = []
    n_requested = n_drawn = n_short = 0
    for i, (contig, gstart, gstop, fa, seq) in enumerate(zip(
        srdf["contig"], srdf["start"], srdf["stop"],
        srdf["fragment_array"], srdf["sequence"],
    )):
        region_len = int(gstop) - int(gstart)
        if fa.length != region_len:
            raise AssertionError(
                f"row {i}: fragment_array.length {fa.length} != stop-start "
                f"{region_len}. Admission used the former and the sequence "
                f"frame the latter, so they must agree."
            )
        if len(seq) != region_len + expected_seq_len_extra:
            raise AssertionError(
                f"row {i} ({contig}:{gstart}-{gstop}): sequence is {len(seq)} b, "
                f"expected {region_len + expected_seq_len_extra} "
                f"(region {region_len} + left_pad {HEX_HALF} + right_pad "
                f"{l_max + HEX_HALF}). A short sequence means the flank was "
                f"truncated -- a region within {l_max + HEX_HALF} b of a contig "
                f"end cannot supply it -- and every hexamer index would shift."
            )

        n = int(region_counts[i])
        n_requested += n
        if n == 0:
            continue
        starts_0, lengths, is_plus = sample_region(
            seq, region_len, n, r=r, fl=fl, p_plus=p_plus, rng=rng,
        )
        n_drawn += len(starts_0)
        if len(starts_0) < n:
            n_short += 1
        if not len(starts_0):
            continue

        starts = int(gstart) + starts_0
        chunks.append(pd.DataFrame({
            "contig": contig,
            "start": starts,
            "stop": starts + lengths,
            "name": "",
            "score": 0,
            "strand": np.where(is_plus, "+", "-"),
            "mapq1": mapq,
            "mapq2": mapq,
        }))

    if chunks:
        bed = pd.concat(chunks, ignore_index=True)
        bed.sort_values(["contig", "start", "stop"], kind="stable", inplace=True)
    else:
        bed = pd.DataFrame(columns=["contig", "start", "stop", "name", "score",
                                    "strand", "mapq1", "mapq2"])
    bed.to_csv(out_path, sep="\t", header=False, index=False)

    return dict(
        n_regions=len(srdf),
        n_requested=n_requested,
        n_drawn=n_drawn,
        n_short_regions=n_short,
        n_rows_written=len(bed),
    )


def propensities(
    counts: Dict[str, np.ndarray],
    expected: Dict[str, np.ndarray],
    *,
    min_expected: float = 0.0,
) -> Dict[str, np.ndarray]:
    """``r(h) = C(h) / N(h)``, observed over expected, for all four tables.

    The minus-strand tables divide by the PERMUTED expectation: ``start_rev``
    counts reverse-complement hexamer indices, so its denominator must be the
    expected count of the same relabelled hexamer.  Using the forward table
    there would misalign every cell.

    Cells with ``N <= min_expected`` yield 0 rather than a divide -- a hexamer
    the null never places cannot have a measured rate, and letting it become
    ``inf`` or ``nan`` would propagate into the sampler.
    """
    perm = rc_permutation()
    n_start, n_end = expected["start"], expected["end"]
    denom = {
        "start_fwd": n_start.astype(np.float64),
        "end_fwd": n_end.astype(np.float64),
        "start_rev": n_start.astype(np.float64)[perm],
        "end_rev": n_end.astype(np.float64)[perm],
    }
    out = {}
    for name in TABLE_NAMES:
        d = denom[name]
        ok = d > min_expected
        r = np.zeros(NHEX, dtype=np.float64)
        r[ok] = counts[name][ok] / d[ok]
        out[name] = r
    return out


def count_srdf(
    srdf,
    *,
    strand_tol: float = 0.1,
    n_workers: int | None = None,
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, int]]:
    """Count over a frame that already has ``fragment_array`` and ``sequence``.

    One ``parallel_apply`` returning three columns per fragment, which
    ``parallel_apply`` concatenates; then four ``bincount`` calls.  The frame is
    expected to be already deduplicated and length-filtered by the
    ``fragment_array_callback`` used at attach time.

    Returns ``(counts, region_counts, stats)``.

    ``region_counts``
        int64, one entry **per ROW, in row order** -- entry ``i`` is the number
        of fragments admitted to row ``i``.  This is the spec's stage-4
        "fragments per region", and it is what the sampler's ``n`` is drawn
        against.

        "Per region" holds because ``count_sample`` builds the frame from a
        single sample, so rows are regions.  An ``srdf`` carrying several
        samples has one row per ``(sample, region)`` PAIR -- ``len`` is rows,
        not regions and not fragments -- and then these counts are per pair.
        The same caveat applies to ``stats["n_regions"]``.

        It counts fragments that survived, in order: MAPQ at fetch, dedup on
        ``(starts_0, stops_0)``, the length filter, and start-in-region
        admission.  **It is NOT the hexamer-valid count.**
        ``cut_site_hexamers`` additionally drops a fragment whose start or stop
        hexamer contains a non-ACGT base, so

            region_counts.sum() >= stats["n_counted"]

        and the gap is the N-containing cut sites.  The two differ per region,
        not by a global factor, because N content is not uniform across the
        region set.  **The owner has accepted this divergence** (2026-10-06):
        the sampler's ``n`` is the post-admission count, so a simulated region
        gets as many fragments as were admitted, including the few whose real
        counterparts carried an N in a cut site and so never reached ``C(h)``.
        Do not "fix" this to the hexamer-valid count without asking.

        Read off the frame rather than reduced through ``parallel_apply``,
        matching how the scalar stats below are produced: the reduction
        concatenates per-region frames and so discards region identity, and
        recovering it would mean giving ``cut_site_hexamers`` a region key it
        does not currently carry.

        ``stats["n_after_filters"]`` is this vector's sum **by construction**
        (computed from it, not alongside it), so the scalar and the vector
        cannot drift apart.
    """

    for col in ("fragment_array", "sequence"):
        if col not in srdf.columns:
            raise ValueError(
                f"srdf has no {col!r} column -- call "
                f"{'attach_fragment_arrays' if col == 'fragment_array' else 'attach_sequence'}"
                f" first"
            )

    # parallel_apply concatenates the per-region frames for us.
    res = srdf.parallel_apply(
        lambda row: cut_site_hexamers(row["fragment_array"], row["sequence"]),
        n_workers=n_workers,
        verbose=verbose,
    )
    counts = counts_from_hexamers(res)

    # Stats are READ OFF THE FRAME, not smuggled through the reduction.
    # n_after_mapq is what from_fragments_h5 left on each row; n_counted
    # follows from the tables, since every fragment contributes exactly one
    # start and one end.
    total = sum(int(v.sum()) for v in counts.values())
    if total % 2:
        raise AssertionError(
            f"odd hexamer total {total} -- every fragment must contribute one "
            f"start and one end"
        )

    # Per-strand totals are EXACT IDENTITIES of the tables, not estimates:
    # counts_from_hexamers puts one start and one end per fragment into its own
    # strand's pair, so each pair's two sums must agree to the fragment. A
    # mismatch means the strand routing is broken -- which the `total % 2`
    # check above cannot see, since it passes for any even total.
    n_plus, n_plus_end = int(counts["start_fwd"].sum()), int(counts["end_fwd"].sum())
    n_minus, n_minus_end = int(counts["start_rev"].sum()), int(counts["end_rev"].sum())
    if n_plus != n_plus_end or n_minus != n_minus_end:
        raise AssertionError(
            f"strand routing is broken: start/end totals disagree within a "
            f"strand pair -- plus {n_plus} vs {n_plus_end}, minus {n_minus} vs "
            f"{n_minus_end}. Every fragment contributes exactly one start and "
            f"one end to ITS OWN strand's pair, so these are identities."
        )
    # Per-region admitted counts, in row order. Length follows from the column
    # itself, so there is nothing to assert: it is this frame's own column.
    region_counts = np.array(
        [fa.n_frags for fa in srdf["fragment_array"]], dtype=np.int64
    )
    stats = dict(
        n_regions=len(srdf),
        # Post-callback: MAPQ at load, then dedup and the length filter. NOT
        # the raw fetched count -- that is gone by the time the frame exists.
        # DERIVED from region_counts so the scalar cannot disagree with the
        # vector.
        n_after_filters=int(region_counts.sum()),
        n_counted=total // 2,
        n_plus=n_plus,
        n_minus=n_minus,
        plus_frac=(n_plus / (n_plus + n_minus)) if (n_plus + n_minus) else float("nan"),
    )

    # Strand balance. A cfDNA fragment is double-stranded and has no intrinsic
    # orientation: the strand label records which of its two ends became read 1,
    # and adapter ligation is symmetric, so the label is a fair coin independent
    # of sequence. p_plus is therefore 0.5 BY CONSTRUCTION, not by fitting, and
    # this is an assertion rather than a measurement.
    #
    # What it is really guarding: `rfa.fragment_strands` is `<U1` ('+'), not the
    # `b'+'` bytes FragmentsH5.fetch_array returns. Comparing against bytes
    # yields an all-False plus mask and so an EMPTY minus or plus table -- and
    # sample_region skips a strand whose start weights are all zero *silently*,
    # which would make the simulator emit strand-pure data with no error
    # anywhere. This is the cheapest place to catch that.
    #
    # The tolerance is deliberately loose. It exists to catch a table that is
    # empty or grossly lopsided, not to police a few percent of real skew, and a
    # tight bound would false-positive on ordinary variation. Note the fraction
    # is over the hexamer-VALID population (n_counted), not the admitted one.
    if (n_plus + n_minus) and abs(stats["plus_frac"] - 0.5) > strand_tol:
        raise AssertionError(
            f"strand fraction {stats['plus_frac']:.4f} is more than {strand_tol} "
            f"from 0.5 ({n_plus} plus, {n_minus} minus). Strand carries no "
            f"sequence information, so this is not biology. Check that "
            f"fragment_strands was compared against '+' and not b'+'."
        )
    if stats["n_regions"] and not stats["n_after_filters"]:
        raise ValueError(
            "MAPQ removed ALL fragments. Unknown MAPQ is stored as -1, so "
            "this is almost certainly the '-1 >= min_mapq' trap: MAPQ was "
            "never carried into the h5."
        )
    return counts, region_counts, stats


def filter_fragments(fa, *, l_min: int = L_MIN, l_max: int = L_MAX):
    """Dedup, length-filter, then admit on start-in-region.

    Runs as ``attach_fragment_arrays``' ``fragment_array_callback``, i.e. inside
    the loading worker, so the attached column carries only fragments that
    survive.  MAPQ is already applied by then -- ``from_fragments_h5`` does it
    at fetch -- and dedup after MAPQ is the store's order.

    ALL filtering lives here rather than in ``cut_site_hexamers``, which
    assumes every fragment it is handed fits inside the sequence it is given.

    ``l_max + 1`` because ``L_MAX`` is an INCLUSIVE bound while
    ``subset_fragment_lengths`` is half-open ``[min, max)`` -- passing ``l_max``
    would silently drop every fragment of exactly ``L_MAX``.

    Admission is ``starts_0 in [0, region_len)``.  ``starts_0`` is NOT clipped
    to the region, so a fragment reaching in from the previous tile has a
    NEGATIVE start and belongs to that tile, not this one.  Tiles are
    contiguous, so each fragment is admitted exactly once across the set.
    """
    fa = fa.drop_duplicate_fragments().subset_fragment_lengths(l_min, l_max + 1)
    return fa.mask((fa.starts_0 >= 0) & (fa.starts_0 < fa.length),
                   validate_data=False)


def count_sample(
    rdf,
    sample_id: str,
    h5_path: str,
    fasta_path: str,
    *,
    min_mapq: int = 10,
    l_min: int = L_MIN,
    l_max: int = L_MAX,
    strand_tol: float = 0.1,

    n_workers: int | None = None,
    verbose: bool = True,
) -> Tuple[
    Dict[str, np.ndarray], np.ndarray, Dict[str, int], "SampleAndRegionDataFrame"
]:
    """The three passes end to end for one sample.

    Returns ``(counts, region_counts, stats, srdf)`` -- ``C(h)``, the per-region
    admitted counts in the row order of ``rdf``, the stats, and the frame that
    produced them.  See ``count_srdf`` for what ``region_counts`` does and does
    not count.

    **The frame is returned rather than discarded** because it is the only
    place three later inputs exist:

    - ``f(L)`` -- ``FragmentLengthDist.from_srdf`` reads the ``fragment_array``
      column, and f(L) must come from the same admitted population as ``C``.
    - the **padded sequences** the sampler draws against, already attached with
      the asymmetric frame.
    - the **region coordinates**, needed to lift a region-local draw to a
      genomic one.

    Rebuilding it would cost a second fetch and a second serial FASTA walk.
    Note ``uniform_hexamer_counts`` takes the ``rdf``, NOT this frame -- it
    refuses one carrying fragment arrays -- so ``N(h)`` is a separate pass by
    design, and f(L) must be built before it, since its end weights are
    f(L)-weighted.

    Pair with ``uniform_hexamer_counts`` and ``propensities`` to get ``r(h)``:
    counts alone are not weights.

    ``attach_sequence`` is the library's and is serial -- one ``FastaFile``
    for the whole loop, no ``n_workers``.  Its cost is negligible against the
    fragment pass, and not worth a second hand-rolled fetch loop to avoid.
    """
    # ASYMMETRIC, and that is deliberate. Admission is start-in-region, so
    # starts need only HEX_HALF of left context, while an end reaches
    # start + l_max and so needs l_max + HEX_HALF on the right. Keeping
    # left_pad == HEX_HALF is what makes the hexamer index equal the
    # region-local coordinate, removing the pad offset from
    # cut_site_hexamers. It also matches uniform_hexamer_counts's frame, so one
    # attached sequence column serves both passes.
    left_flank, right_flank = HEX_HALF, l_max + HEX_HALF
    srdf = (
        SampleAndRegionDataFrame
        .init_from_rdf_and_sdf(rdf, load_sample_dataframe([(sample_id, h5_path)]))
        # Dedup and length-filter in the LOADING worker, so the attached column
        # carries only surviving fragments (~41 of the ~275 that clear MAPQ).
        # The callback runs inside load_fragment_arrays' parallel_apply, which
        # fork-inherits rather than pickles fn, so a lambda is safe.
        #
        # l_max + 1: L_MAX is an INCLUSIVE bound while
        # subset_fragment_lengths is half-open [min, max), so passing l_max
        # would silently drop every fragment of exactly L_MAX. Half-open is the
        # codebase convention -- CLAUDE.md states it, and FL_BANDS share the
        # 110 boundary because of it -- so the +1 is absorbed here rather than
        # by changing the library.
        .attach_fragment_arrays(
            min_mapq=min_mapq,
            fragment_array_callback=(
                lambda fa: filter_fragments(fa, l_min=l_min, l_max=l_max)
            ),
            n_workers=n_workers,
            verbose=verbose,
        )
        # Must come AFTER the fragment arrays: expand_regions refuses to run on
        # a frame that already has them, and attach_sequence pads via the same
        # resize machinery.
        .attach_sequence(
            fasta_path, left_pad=left_flank, right_pad=right_flank,
            verbose=verbose,
        )
    )
    counts, region_counts, stats = count_srdf(
        srdf, strand_tol=strand_tol, n_workers=n_workers, verbose=verbose
    )
    return counts, region_counts, stats, srdf
