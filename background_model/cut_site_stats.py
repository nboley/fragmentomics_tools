"""Cut-site hexamer counts, observed and expected, via the fragmentomics_tools API.

Measurement on real data.  Encoding is ``background_model.hexamers``; the
draw that consumes these tables is ``background_model.simulator.draw``.
Authority: ``docs/pending/simulator_spec.md``.

Two artifacts, with different lifetimes:

- ``C(h)`` -- observed cut-site counts, **per sample**.  ``count_sample``, or
  the three passes by hand.  Parallel.  The same pass also returns the
  per-region admitted counts (the spec's stage 4), since the frame it already
  built is the only place that count exists; see ``count_srdf`` for the
  distinction between those and the hexamer-valid totals.
- ``N(h)`` -- EXPECTED hexamer counts under a uniform start/end null.
  ``uniform_hexamer_counts``.  Parallel over fixed-size region blocks, and
  **byte-identical across worker counts** (owner decision 166): the float64
  reduction is grouped by ``UNIFORM_BLOCK_SIZE``, independently of
  ``n_workers``.

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
``fragmentomics_tools`` has no hexamer machinery.  That half is
``background_model.hexamers``.

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

import numpy as np
import pandas as pd

from fragmentomics_tools.dataframe import (
    DataFrameBase,
    SampleAndRegionDataFrame,
    SampleDataFrame,
)

from background_model.hexamers import (
    HEX_HALF,
    NHEX,
    _hexamers_at,
    hexamer_indices,
    rc_permutation,
)

# ── geometry, owned here ──────────────────────────────────────────────────

L_MIN: int = 25              # INCLUSIVE length bounds -- the library's
L_MAX: int = 180             # subset_fragment_lengths is half-open
N_LENGTHS: int = L_MAX - L_MIN + 1   # 156

TABLE_NAMES = ("start_fwd", "end_fwd", "start_rev", "end_rev")

# Regions per task in uniform_hexamer_counts.  A property of the REDUCTION, not
# of the execution: N_end is a float64 sum, so its last bits depend on how the
# per-region terms are grouped, and r(h) -- hence every draw -- inherits them.
# Fixing the grouping here, rather than deriving it from n_workers, is what
# makes N(h) identical for any worker count.  Changing it changes N_end's last
# bits.
UNIFORM_BLOCK_SIZE: int = 256


def empty_counts() -> Dict[str, np.ndarray]:
    """Four zeroed ``(4096,)`` int64 tables."""
    return {k: np.zeros(NHEX, dtype=np.int64) for k in TABLE_NAMES}


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


def _uniform_block(contigs, starts, stops, lo: int, hi: int,
                   fasta_path: str, fl: "FragmentLengthDist") -> dict:
    """``N(h)`` partial tables for regions ``[lo, hi)``.  Runs in a worker.

    Regions are accumulated in row order, so the partial depends only on
    ``(lo, hi)`` and never on which process ran it.
    """
    import pysam

    # Starts need only HEX_HALF of left context. An end's hexamer window
    # reaches gstop-1 + max_fl + HEX_HALF, so the RIGHT flank must cover that
    # -- asymmetric. left_flank = HEX_HALF makes a hexamer index equal its
    # region-local coordinate, which is why it is not widened for symmetry.
    left_flank = HEX_HALF
    right_flank = fl.max_fl + HEX_HALF

    N_start = np.zeros(NHEX, dtype=np.int64)
    N_end = np.zeros(NHEX, dtype=np.float64)
    n_start_positions = n_start_invalid = 0
    end_weight_total = 0.0

    # Opened HERE, inside the worker, and closed before returning: a pysam
    # handle must never be held across a fork.
    with pysam.FastaFile(fasta_path) as fasta:
        for k in range(lo, hi):
            contig, gstart, gstop = contigs[k], int(starts[k]), int(stops[k])
            region_len = gstop - gstart
            seq = fasta.fetch(contig, gstart - left_flank, gstop + right_flank)
            fwd, _rc, valid = hexamer_indices(seq)
            n_hex = len(fwd)
            if n_hex < region_len + fl.max_fl:
                raise ValueError(
                    f"{contig}:{gstart}-{gstop}: {n_hex} hexamer windows, "
                    f"need at least {region_len + fl.max_fl} -- the FASTA "
                    f"fetch was truncated, which happens within "
                    f"{right_flank}bp of a contig end."
                )

            # Starts: uniform, one unit per position in [gstart, gstop).
            s_idx = slice(0, region_len)
            s_valid = valid[s_idx]
            N_start += np.bincount(fwd[s_idx][s_valid], minlength=NHEX)
            n_start_positions += region_len
            n_start_invalid += int((~s_valid).sum())

            # Ends: FL-weighted. The validity gate must match the fragment
            # pass, or N is inflated at exactly the repeat-rich and
            # gap-adjacent positions where r would then be depressed.
            w = fl_end_weight(n_hex, region_len, fl)
            m = valid & (w > 0)
            N_end += np.bincount(fwd[m], weights=w[m], minlength=NHEX)
            end_weight_total += float(w[m].sum(dtype=np.float64))

    return dict(N_start=N_start, N_end=N_end, n_regions=hi - lo,
                n_start_positions=n_start_positions,
                n_start_invalid=n_start_invalid,
                end_weight_total=end_weight_total)


def uniform_hexamer_counts(
    rdf,
    fasta_path: str,
    fl: "FragmentLengthDist",
    *,
    n_workers: int | None = None,
    block_size: int = UNIFORM_BLOCK_SIZE,
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

    **Parallel over fixed blocks of ``block_size`` regions, not over
    regions.**  Returning a dense per-region table through ``parallel_apply``
    would move ~4.4 GB (66,649 regions x two 32 KB tables) to reduce 64 KB.
    Each task instead walks a block of regions and returns ONE partial pair,
    so the transfer is ``ceil(n_regions / block_size)`` pairs -- 261 x 64 KB,
    about 17 MB, at the default block size.  The blocks are rows of a small
    frame handed to ``parallel_apply``, so the fork, main-thread and tqdm
    monitor guards are the library's rather than a second pool.

    **Byte-identical for every ``n_workers``.**  ``N_end`` is a float64 sum,
    and a reduction grouped by worker would change its last bits with the
    worker count, then r(h), then every draw.  Here the grouping is fixed by
    ``block_size`` alone: each block accumulates its regions in row order,
    and the parent adds the block partials in block order.  ``n_workers=1``
    runs the same blocks in-process.  ``block_size`` is therefore part of the
    result -- changing it moves ``N_end`` in the last bits -- which is why it
    is a module constant and not derived from the worker count.

    Only the FORWARD tables are returned.  The minus-strand expectation is the
    forward table permuted -- ``N_rc == N_fwd[rc_permutation()]`` -- because a
    reverse-complement hexamer at a position is a relabelling, not a different
    position.
    """
    if "fragment_array" in getattr(rdf, "columns", ()):
        raise ValueError(
            "pass a region frame WITHOUT fragment arrays: the uniform counts are "
            "sequence-only and region-set-scoped, and attaching fragments "
            "first blocks the padded sequence fetch (expand_regions refuses a "
            "frame that already carries them)"
        )
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    # Plain arrays, built before the fork and inherited by the workers.
    contigs = rdf["contig"].to_numpy()
    starts = rdf["start"].to_numpy(dtype=np.int64)
    stops = rdf["stop"].to_numpy(dtype=np.int64)
    n = len(contigs)
    los = np.arange(0, n, block_size, dtype=np.int64)
    blocks = DataFrameBase(pd.DataFrame({
        "lo": los, "hi": np.minimum(los + block_size, n),
    }))

    N_start = np.zeros(NHEX, dtype=np.int64)
    N_end = np.zeros(NHEX, dtype=np.float64)
    meta = dict(n_regions=0, n_start_positions=0, n_start_invalid=0,
                end_weight_total=0.0)
    if not len(blocks):
        return {"start": N_start, "end": N_end}, meta

    res = blocks.parallel_apply(
        lambda row: _uniform_block(contigs, starts, stops, int(row["lo"]),
                                   int(row["hi"]), fasta_path, fl),
        n_workers=n_workers,
        verbose=verbose,
    )
    # parallel_apply returns the records in ROW order, so this sequential
    # loop adds the partials in block order whatever process produced them.
    for part in res.itertuples(index=False):
        N_start += part.N_start
        N_end += part.N_end
        meta["n_regions"] += int(part.n_regions)
        meta["n_start_positions"] += int(part.n_start_positions)
        meta["n_start_invalid"] += int(part.n_start_invalid)
        meta["end_weight_total"] += float(part.end_weight_total)

    return {"start": N_start, "end": N_end}, meta


def propensities(
    counts: Dict[str, np.ndarray],
    expected: Dict[str, np.ndarray],
    *,
    min_expected: float = 0.0,
) -> Dict[str, np.ndarray]:
    """``r(h) = C(h) / N(h)``, observed over expected, for all four tables.

    Two independent things decide a denominator, and conflating them is how
    this function was wrong until 2026-10-07.

    **1. Which expectation: table names are MOLECULE-relative, denominators
    must be POSITION-relative.** ``start``/``end`` name the molecule's 5' and
    3' cut site, but a minus-strand fragment's 5' cut site sits at its GENOMIC
    STOP.  ``counts_from_hexamers`` therefore tallies genomic stops into
    ``start_rev`` and genomic starts into ``end_rev``, and ``sample_region``
    applies them that way round too.  The null expectation has to match the
    positions actually tallied, so:

        start_fwd <- genomic starts -> N_start
        end_fwd   <- genomic stops  -> N_end
        start_rev <- genomic STOPS  -> N_end    (not N_start)
        end_rev   <- genomic STARTS -> N_start  (not N_end)

    This is not a relabelling: ``N_start`` weights every position 1 while
    ``N_end`` carries ``fl_end_weight``'s edge ramp, so the two differ for
    every position within ``max_fl`` of a region edge -- **11.7% of a 1536 bp
    tile**, by up to 156x at offset 25 and 2.05x at offset 100, and offsets
    below ``min_fl`` have ``N_end == 0`` where ``N_start`` is positive.
    Measured by drawing 6e6 fragments from the uniform null and asking which
    denominator makes ``r`` constant: the correct pairing gives CV 0.041 (pure
    sampling noise), the swapped one 0.064-0.067.

    **2. Which indexing: the ``_rev`` tables are permuted.** They tally
    reverse-complement hexamer indices, so the denominator must be the expected
    count of the same RELABELLED hexamer -- hence ``[perm]``.  A
    reverse-complement hexamer at a position is a relabelling, not a different
    position, which is why one permutation serves both ``_rev`` tables.

    Cells with ``N <= min_expected`` yield 0 rather than a divide -- a hexamer
    the null never places cannot have a measured rate, and letting it become
    ``inf`` or ``nan`` would propagate into the sampler.
    """
    perm = rc_permutation()
    n_start, n_end = expected["start"], expected["end"]
    denom = {
        "start_fwd": n_start.astype(np.float64),
        "end_fwd": n_end.astype(np.float64),
        # Crossed on purpose -- see point 1 above. start_rev tallies genomic
        # STOPS, so it needs the END expectation, and vice versa.
        "start_rev": n_end.astype(np.float64)[perm],
        "end_rev": n_start.astype(np.float64)[perm],
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

    # Per-strand totals must agree within a pair, because counts_from_hexamers
    # puts one start and one end per fragment into its own strand's pair.
    #
    # DO NOT MISTAKE THIS FOR ROUTING PROTECTION. I originally claimed it
    # "catches broken strand routing"; it does not, and that was measured:
    # feeding counts_from_hexamers a start/stop SWAP leaves both identities
    # true, because each sum is just the number of rows of that strand however
    # the hexamers are routed. Together with the `total % 2` check above it is
    # a TAUTOLOGY for any `counts` this module produced.
    # What it does still buy: a malformed `counts` dict from somewhere else --
    # hand-built, or loaded from a stored artifact -- fails here rather than
    # silently downstream. That is the only reason it stays.
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

    # Strand balance, in two separate checks because they catch different things
    # and only one of them can be stated without a tolerance.
    #
    # Why 0.5 is expected at all: a cfDNA fragment is double-stranded and has no
    # intrinsic orientation. The strand label records which of its two ends
    # became read 1, and adapter ligation is symmetric, so the label carries no
    # sequence information. p_plus is 0.5 BY CONSTRUCTION, not by fitting.
    #
    if stats["n_regions"] and not stats["n_after_filters"]:
        raise ValueError(
            "EVERY fragment was removed before counting. The admission chain "
            "is MAPQ at fetch, then dedup, the length filter and "
            "start-in-region, and ANY of them can empty the frame -- the "
            "message used to blame MAPQ alone, which made this guard read as "
            "MAPQ coverage that it does not provide. MAPQ is still the first "
            "thing to check: unknown MAPQ is stored as -1, so a h5 built "
            "without MAPQ hits the '-1 >= min_mapq' trap and loses "
            "everything. Also check that the regions overlap the h5's contigs "
            "and that the length bounds are not inverted."
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
    - a ``region_index`` column -- each region's position in ``rdf``'s
      region set, taken from ``rdf``'s index labels -- which keys the
      region's draw stream in ``simulate_fragments_to_bed``.

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
    # region_index keys each region's draw stream (owner decision 166), so it
    # must be the region's position in the INPUT region set, not its row here.
    # The cross join below discards rdf's index, so it is carried as a column.
    # from_bed gives a RangeIndex over the BED's rows and row subsetting
    # (iloc, a mask) keeps the labels, so the label IS that position. No row
    # is dropped between here and the draw -- the cross join with one sample
    # keeps every row, and attach_sequence is a left join -- but carrying the
    # label means a caller who filters rdf first still gets stable streams.
    if "region_index" not in rdf.columns:
        if not (pd.api.types.is_integer_dtype(rdf.index)
                and rdf.index.is_unique):
            raise ValueError(
                "rdf needs a unique integer index (from_bed's RangeIndex, "
                "or a subset of it) or an explicit 'region_index' column: "
                "it keys each region's draw stream"
            )
        rdf = rdf.assign(region_index=np.asarray(rdf.index, dtype=np.int64))
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
        srdf, n_workers=n_workers, verbose=verbose
    )
    return counts, region_counts, stats, srdf
