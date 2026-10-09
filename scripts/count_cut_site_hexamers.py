#!/usr/bin/env python
"""Count de-duplicated cut-site hexamer frequencies for ONE fragment h5.

This script **counts**.  It makes no normalisation, ratio or weighting
decision: it emits observed counts *and* the matching background (all-candidate)
counts, and whoever consumes the file decides what to divide by what.

What is counted
---------------
For every de-duplicated, MAPQ-passing fragment fully contained in a region of
the region set, the hexamer at each of its two cut sites is counted into one of
**four untied 4096-wide tables**::

    start_fwd, end_fwd   -- plus-strand fragments
    start_rev, end_rev   -- minus-strand fragments

The four are untied because the data is short-read *single-stranded*: the 5' and
3' ends are not related to each other by reverse complement, so tying them would
average away a real asymmetry.

Every table is additionally stratified by narrow fragment-length bands (10 bp by
default over ``L = 25..180``).  The bands are deliberately *finer* than the
model's ``FL_BANDS``: whether length stratification is needed at all should be
answerable from these counts rather than assumed, and a fine band can always be
summed into a coarse one downstream.  The band edges are written into the
output.

Geometry -- this is the part that fails silently if wrong
---------------------------------------------------------
The conventions were taken verbatim from the previous-generation simulator
(``weights.py``, now in ``attic/pre_rewrite_simulator/``; Appendix D of the
design doc) and the hexamer indexing is *imported*, not reimplemented, from
``background_model/hexamers.py::hexamer_indices``.  That encoder folds case;
``region_hexamers`` upper-cases first anyway, which is what the old,
uppercase-only encoder needed, so the output is the same under either:

* **Cut sites, not bases.**  A fragment occupying bases ``[p, p+L)`` has
  endpoint *bases* ``p`` and ``p+L-1`` but cut *sites* ``p`` and ``p+L``.
  ``fragments_h5`` stores ``stop`` exclusive (``fragment_lengths =
  stops_0 - starts_0``), so ``stop`` is already the 3' cut site of a plus-strand
  fragment -- no +/-1 adjustment anywhere.
* **Plus strand:** ``c5 = p``, ``c3 = p + L``; both read ``hex_fwd``.
* **Minus strand:** ``c5 = p + L``, ``c3 = p`` -- the 5' end sits at the
  **higher** coordinate -- and both read ``hex_rc``.
* ``hex_fwd[c]`` is the forward index of the 6-mer spanning cut site ``c``
  (3 bases inside the fragment, 3 outside); ``hex_rc[c]`` is its reverse
  complement at the same position.
* A cut site whose 6-mer window contains a non-ACGT base is ``valid == False``.
  A fragment is counted only when **both** of its cut sites are valid, matching
  the simulator's ``vmask = valid[c5] & valid[c3]``.

The background tables
---------------------
``bg_*`` holds the same four tables computed over **every candidate fragment**
in the same regions -- the simulator's generative domain

    Omega = {(c5, c3, strand) : L in [L_MIN, L_MAX], both cut sites in region}

i.e. every ``(position, length, strand)`` a fragment *could* have used, not only
those it did.  This is not optional colour: a raw cut-site hexamer count is
dominated by the region set's base composition, not by cut-site preference, and
the background is the only input that **cannot be reconstructed later** without
re-reading the whole region set.  It is computed in closed form from a cumulative
sum of ``valid`` rather than by enumerating Omega (which has ~767k members per
2560 bp region).

Note the resulting symmetry, which the tests assert: a position's candidate
count as a plus-strand *start* equals its count as a minus-strand *end* (both
are "how many lengths reach forward from here"), and likewise plus *end* ==
minus *start*.  Only the hexamer track differs.

In-region rule
--------------
A fragment is counted only if it is **fully contained**: ``gstart <= start`` and
``stop <= gstop``.  This is the same containment the generative domain assumes
(``hex(c3)`` is undefined off the region), so observed and background cover
exactly the same support.  A fragment straddling a tile boundary is counted in
neither tile -- deliberately, since counting it in one would have no matching
background entry.

Filters and de-duplication
--------------------------
* ``min(mapq_read1, mapq_read2) >= --min-mapq`` (inclusive), matching
  ``from_fragments_h5``'s ``mapq_vals >= min_mapq`` and hence the store's
  ``PlumbingConfig.min_mapq`` (10).  Unknown MAPQ is stored as ``-1``, so a
  file that did not carry MAPQ through will have *everything* removed; the
  script detects that and aborts loudly rather than writing an empty table.
* De-duplication collapses fragments sharing ``(contig, start, stop)`` to one
  molecule -- exactly ``FragmentArray.drop_duplicate_fragments()``, which the
  store uses.  Strand is **not** part of the key, so a plus and a minus fragment
  with identical coordinates collapse to one.  The strand-inclusive duplicate
  rate is measured and recorded alongside as a diagnostic, so the cost of that
  choice is visible rather than hidden.
  The h5 does *not* pre-de-duplicate; the measured rate is in the output.

Output: a dataframe keyed by the hexamer STRING
-----------------------------------------------
A single self-describing Parquet file in long form, one row per
``(hexamer, table, band)``::

    hexamer  table      band_lo  band_hi  observed  background
    AAAAAA   start_fwd       25       35       137     1092341
    ...

The key is the literal 6-mer, **not** an integer code, and that is the point.
A bare 4096-element array ordered by an implicit integer code makes the k-mer
ordering and the RC convention a *contract* between producer and consumer.  If
a table is ever produced under a different convention every downstream weight
is wrong, and the normalisation invariant ``sum_Omega w == 1`` still holds,
because normalisation cannot see a relabelling of the table -- a silent failure
with no detector.  Keyed by the string, that failure is not expressible: a
consumer builds its own index-ordered array by looking each string up through
its own indexer, so a mismatched table fails to *join* rather than silently
misaligning.  The integer code is now a private implementation detail on each
side rather than a shared contract.

**Which strand the string is written in.**  The string removes the *index*
ambiguity; it does not by itself remove the *orientation* one.  In every table
the 6-mer is written **5'->3' along the strand of the fragment that produced
it**:

* ``start_fwd`` / ``end_fwd`` -- plus-strand fragments, so the string is the
  reference-forward 6-mer spanning the cut site.
* ``start_rev`` / ``end_rev`` -- minus-strand fragments, so the string is the
  **reverse complement** of the reference-forward 6-mer at that cut site
  (these tables index ``hex_rc``).

So ``start_rev`` row ``"AAAAAA"`` counts cut sites whose *reference* sequence
is ``TTTTTT``.  Do not reverse-complement these tables again when joining.

**Long, not wide.**  A wide layout -- 4096 hexamer columns, or one column per
``(table, band)`` -- would push the table name and the band edges back out of
the data and into *column names*, which is the same implicit schema contract
the string key was adopted to remove.  In long form every row carries its own
``(hexamer, table, band_lo, band_hi)`` as values, a band re-scheme changes no
schema, and ``pyarrow``'s dictionary encoding makes the repeated strings nearly
free.

Both count columns are ``int64``.  ``background`` is accumulated in float64
only because ``np.bincount`` weights are float64; the values are exact integers
by construction and the writer refuses to write a non-integral one rather than
rounding it.

Provenance travels in the Parquet file-level metadata under the key
``count_cut_site_hexamers_meta``: region-set identity (path + md5 + region
count), sample/h5 identity, ``min_mapq``, the dedup rule, the in-region rule,
the geometry constants and every fragment tally.  A count table without that is
unusable six weeks later.  ``read_output(path)`` returns ``(dataframe, meta)``.

Example
-------
::

    python scripts/count_cut_site_hexamers.py \\
        --fragments-h5 /efs/.../<md5>-RD-56413-Lib1.hg38.fragments.h5 \\
        --regions-bed data/region_sets/quiet_v2_pad1200_repeats_removed_tile2560.bed \\
        --fasta /efs/analytics/nathanboley/data_resources/genome/hg38.fa \\
        --output /efs/.../RD-56413-Lib1.cut_site_hexamers.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ``background_model`` is not an installed package -- it is importable only with
# the repo root on sys.path.  Pin it to THIS FILE's repo at position 0 rather
# than relying on CWD or PYTHONPATH: running a script from another directory
# otherwise puts that directory on sys.path[0] and the import falls through to
# whatever checkout happens to be installed, on whatever branch it happens to
# be, which fails silently rather than loudly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from background_model.config import PlumbingConfig  # noqa: E402
from background_model.cut_site_stats import L_MAX, L_MIN  # noqa: E402
from background_model.hexamers import HEX_HALF, KMER, NHEX, hexamer_indices, hexamer_vocabulary  # noqa: E402
from fragments_h5 import FragmentsH5  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_BAND_WIDTH = 10

TABLE_NAMES = ("start_fwd", "end_fwd", "start_rev", "end_rev")

META_KEY = b"count_cut_site_hexamers_meta"

#: The written schema, stated once so that ``write_output`` cannot drift from
#: the module docstring by accident.  ``band_lo``/``band_hi`` are the half-open
#: edges of the row's band, repeated on every row so that the band scheme is a
#: *value* in the data rather than a naming convention on columns.
OUTPUT_SCHEMA = pa.schema([
    ("hexamer", pa.string()),
    ("table", pa.string()),
    ("band_lo", pa.int32()),
    ("band_hi", pa.int32()),
    ("observed", pa.int64()),
    ("background", pa.int64()),
])


# ── fragment-length bands ────────────────────────────────────────────────

def make_band_edges(l_min: int, l_max: int, band_width: int) -> np.ndarray:
    """Half-open ``[lo, hi)`` band edges tiling ``[l_min, l_max]``.

    Half-open to match the repo convention for fl bands.  The final band is
    truncated at ``l_max + 1`` when ``band_width`` does not divide the range,
    so ``(25, 180, 10)`` gives 16 bands ending ``[175, 181)`` -- 6 lengths wide,
    not 10.  Band *widths* are therefore not all equal and a consumer must use
    the recorded edges rather than assume uniformity.

    >>> make_band_edges(25, 180, 10)[[0, -1]].tolist()
    [[25, 35], [175, 181]]
    >>> len(make_band_edges(25, 180, 10))
    16
    """
    if band_width < 1:
        raise ValueError(f"band_width must be >= 1, got {band_width}")
    if l_max < l_min:
        raise ValueError(f"l_max ({l_max}) < l_min ({l_min})")
    los = np.arange(l_min, l_max + 1, band_width, dtype=np.int64)
    his = np.minimum(los + band_width, l_max + 1)
    return np.stack([los, his], axis=1)


def band_index(lengths: np.ndarray, l_min: int, band_width: int) -> np.ndarray:
    """Band index for each length. Caller must already have clipped to range.

    >>> band_index(np.array([25, 34, 35, 180]), 25, 10).tolist()
    [0, 0, 1, 15]
    """
    return (lengths - l_min) // band_width


# ── per-region hexamer tracks ────────────────────────────────────────────

def region_hexamers(fasta, contig: str, gstart: int, gstop: int):
    """``(hex_fwd, hex_rc, valid)`` at every cut site of ``[gstart, gstop)``.

    Each array has length ``gstop - gstart + 1`` -- one entry per cut site,
    positions ``0..region_len`` inclusive.

    This duplicated the sequence-fetch half of the previous-generation
    ``precompute_region`` (now in ``attic/pre_rewrite_simulator/``; same
    ``HEX_HALF`` flanking, same ``hexamer_indices`` call) but takes an
    already-open ``pysam.FastaFile`` instead of opening one per call.
    ``precompute_region`` opened and closed the 3 GB hg38 FASTA on every
    invocation, which measured **20x slower** on EFS (13.6 ms vs 0.66 ms per
    region; 157 s vs 7.6 s over the 11,505-tile region set).  The part that
    would fail silently if it diverged -- the hexamer indexing itself -- is
    imported, not copied, and
    ``test_region_hexamers_matches_precompute_region_exactly`` checks every cut
    site against an independent string encoder.  ``cum_gc`` is not computed
    because nothing here uses GC.
    """
    region_len = gstop - gstart
    seq = fasta.fetch(contig, gstart - HEX_HALF, gstop + HEX_HALF).upper()
    seq_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    hex_fwd, hex_rc, valid = hexamer_indices(seq_bytes)
    if len(hex_fwd) != region_len + 1:
        raise ValueError(
            f"{contig}:{gstart}-{gstop}: got {len(hex_fwd)} cut sites, expected "
            f"{region_len + 1}. The FASTA fetch was truncated -- this happens "
            f"when a region runs off the end of a contig."
        )
    return hex_fwd, hex_rc, valid


# ── background: candidate cut sites, in closed form ──────────────────────

def _valid_cumsum(valid: np.ndarray) -> np.ndarray:
    """``cv[i] = valid[:i].sum()``, length ``len(valid) + 1``."""
    cv = np.zeros(len(valid) + 1, dtype=np.int64)
    np.cumsum(valid, out=cv[1:])
    return cv


def candidates_forward(cv, valid, lo: int, hi: int, region_len: int) -> np.ndarray:
    """For each cut site ``c``: how many lengths in ``[lo, hi]`` reach *forward*.

    ``= valid[c] * #{L in [lo, hi] : c + L <= region_len and valid[c + L]}``.

    This is the candidate count for a position used as a plus-strand **start**
    (``c3 = c5 + L``) and equally as a minus-strand **end** (``c5 = c3 + L``).
    Evaluated by differencing ``cv`` rather than looping over ``L``.
    """
    n = region_len + 1
    c = np.arange(n)
    first = c + lo
    last = np.minimum(c + hi, region_len)
    reachable = first <= last
    w = np.where(
        reachable,
        cv[np.clip(last + 1, 0, n)] - cv[np.clip(first, 0, n)],
        0,
    )
    return np.where(valid, w, 0).astype(np.float64)


def candidates_backward(cv, valid, lo: int, hi: int, region_len: int) -> np.ndarray:
    """For each cut site ``c``: how many lengths in ``[lo, hi]`` reach *backward*.

    ``= valid[c] * #{L in [lo, hi] : c - L >= 0 and valid[c - L]}``.

    The candidate count for a position used as a plus-strand **end**
    (``c5 = c3 - L``) and equally as a minus-strand **start** (``c3 = c5 - L``).
    """
    n = region_len + 1
    c = np.arange(n)
    first = np.maximum(c - hi, 0)
    last = c - lo
    reachable = last >= first
    w = np.where(
        reachable,
        cv[np.clip(last + 1, 0, n)] - cv[np.clip(first, 0, n)],
        0,
    )
    return np.where(valid, w, 0).astype(np.float64)


def accumulate_background(bg, hex_fwd, hex_rc, valid, band_edges, region_len):
    """Add one region's candidate counts into the four ``bg`` tables in place.

    ``bg`` maps each of ``TABLE_NAMES`` to a ``(n_bands, NHEX)`` float64 array.
    The values are exact integers; float64 is used only because
    ``np.bincount`` weights are float64, and stays exact well past the ~1e10
    totals this reaches.
    """
    cv = _valid_cumsum(valid)
    for bi, (lo_edge, hi_edge) in enumerate(band_edges):
        lo, hi = int(lo_edge), int(hi_edge) - 1  # half-open -> inclusive
        fwd_reach = candidates_forward(cv, valid, lo, hi, region_len)
        bwd_reach = candidates_backward(cv, valid, lo, hi, region_len)
        bg["start_fwd"][bi] += np.bincount(hex_fwd, weights=fwd_reach, minlength=NHEX)
        bg["end_fwd"][bi] += np.bincount(hex_fwd, weights=bwd_reach, minlength=NHEX)
        bg["start_rev"][bi] += np.bincount(hex_rc, weights=bwd_reach, minlength=NHEX)
        bg["end_rev"][bi] += np.bincount(hex_rc, weights=fwd_reach, minlength=NHEX)


# ── observed ─────────────────────────────────────────────────────────────

def accumulate_observed(
    obs, hex_fwd, hex_rc, valid, starts, stops, is_plus, gstart,
    l_min: int, band_width: int, n_bands: int,
) -> int:
    """Add one region's fragments into the four ``obs`` tables in place.

    ``starts``/``stops`` are genomic, already filtered to fragments fully inside
    ``[gstart, gstart + region_len)`` with length in range.  Returns the number
    of fragments actually counted (those whose *both* cut sites are valid);
    each contributes exactly one start and one end count.
    """
    lengths = (stops - starts).astype(np.int64)
    p = starts.astype(np.int64) - gstart
    q = stops.astype(np.int64) - gstart

    # Cut sites, not bases: plus is (p, p+L) = (p, q); minus swaps them so that
    # the 5' end is at the HIGHER coordinate.
    c5 = np.where(is_plus, p, q)
    c3 = np.where(is_plus, q, p)

    keep = valid[c5] & valid[c3]
    if not keep.any():
        return 0
    c5, c3, is_plus = c5[keep], c3[keep], is_plus[keep]
    bidx = band_index(lengths[keep], l_min, band_width)

    flat_len = n_bands * NHEX
    for strand_mask, track, s_name, e_name in (
        (is_plus, hex_fwd, "start_fwd", "end_fwd"),
        (~is_plus, hex_rc, "start_rev", "end_rev"),
    ):
        if not strand_mask.any():
            continue
        b = bidx[strand_mask] * NHEX
        obs[s_name] += np.bincount(
            b + track[c5[strand_mask]], minlength=flat_len
        ).reshape(n_bands, NHEX)
        obs[e_name] += np.bincount(
            b + track[c3[strand_mask]], minlength=flat_len
        ).reshape(n_bands, NHEX)

    return int(keep.sum())


# ── filtering / dedup ────────────────────────────────────────────────────

def filter_and_dedup(starts, stops, mapqs, strands, min_mapq: int):
    """MAPQ filter then ``(start, stop)`` de-duplication.

    Returns ``(starts, stops, is_plus, n_mapq_pass, n_dup_coord, n_dup_coord_strand)``.

    Order matters and matches the store: ``from_fragments_h5`` applies the MAPQ
    filter, then ``drop_duplicate_fragments()`` runs on what survives.

    ``n_dup_coord`` is the number of fragments removed by the ``(start, stop)``
    key that is actually applied.  ``n_dup_coord_strand`` is what a
    strand-inclusive key would have removed -- reported so the cost of
    collapsing a plus/minus coordinate pair into one molecule is visible.
    """
    min_mapq_vals = mapqs.min(axis=1)
    mask = min_mapq_vals >= min_mapq
    starts, stops, strands = starts[mask], stops[mask], strands[mask]
    n_mapq_pass = int(mask.sum())

    is_plus = strands == b"+"

    _, idx = np.unique(np.array([starts, stops]), axis=1, return_index=True)
    n_dup_coord = n_mapq_pass - len(idx)
    _, idx_s = np.unique(
        np.array([starts, stops, is_plus.astype(starts.dtype)]),
        axis=1, return_index=True,
    )
    n_dup_coord_strand = n_mapq_pass - len(idx_s)

    return (
        starts[idx], stops[idx], is_plus[idx],
        n_mapq_pass, n_dup_coord, n_dup_coord_strand,
    )


# ── BED ──────────────────────────────────────────────────────────────────

def contained_in_region(starts, stops, gstart: int, gstop: int) -> np.ndarray:
    """Mask of fragments lying **entirely** inside ``[gstart, gstop)``.

    Containment, not overlap and not midpoint: a fragment's 3' cut site must be
    a cut site of this region or ``hex(c3)`` is undefined, which is also the
    edge rule the generative domain uses.  A fragment straddling a tile boundary
    therefore falls out of *both* neighbouring tiles rather than being counted in
    one of them -- counting it would put an observation where the background has
    no matching candidate.

    ``stop`` is exclusive, so ``stop == gstop`` is contained: it names cut site
    ``region_len``, the last one the region has.
    """
    return (starts >= gstart) & (stops <= gstop)


def load_regions(bed_path: str):
    """Load a 3+-column BED as a list of ``(contig, start, stop)``.

    Hand-rolled rather than going through ``RegionDataFrame.from_bed``: that
    path requires a ``ref`` and pulls in ``pybedtools``, whose ``bedtools``
    binary is frequently absent from PATH in sandboxes and Batch containers
    (see CLAUDE.md).  Nothing here needs interval arithmetic -- the regions are
    consumed one at a time, verbatim -- so there is no rule that could drift.
    """
    regions = []
    with open(bed_path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.split("\t")
            regions.append((f[0], int(f[1]), int(f[2])))
    return regions


def _file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── driver ───────────────────────────────────────────────────────────────

def count_sample(
    h5_path: str,
    regions,
    fasta_path: str,
    *,
    min_mapq: int,
    band_edges: np.ndarray,
    l_min: int = L_MIN,
    l_max: int = L_MAX,
    band_width: int = DEFAULT_BAND_WIDTH,
    with_background: bool = True,
    log_every: int = 2000,
):
    """Count one sample over ``regions``. Returns ``(obs, bg, stats)``."""
    import pysam

    n_bands = len(band_edges)
    obs = {k: np.zeros((n_bands, NHEX), dtype=np.int64) for k in TABLE_NAMES}
    bg = {k: np.zeros((n_bands, NHEX), dtype=np.float64) for k in TABLE_NAMES}

    stats = dict(
        n_regions=0, n_regions_skipped_missing_contig=0,
        n_fetched=0, n_mapq_pass=0, n_dup_removed=0,
        n_dup_removed_strand_key=0, n_deduped=0,
        n_contained=0, n_in_length_range=0, n_counted=0,
    )

    h5 = FragmentsH5(h5_path, cache_pointers=False)
    fasta = pysam.FastaFile(fasta_path)
    t0 = time.monotonic()
    try:
        for i, (contig, gstart, gstop) in enumerate(regions):
            if contig not in h5.contig_lengths:
                stats["n_regions_skipped_missing_contig"] += 1
                continue
            region_len = gstop - gstart
            hex_fwd, hex_rc, valid = region_hexamers(fasta, contig, gstart, gstop)

            starts, stops, supp = h5.fetch_array(
                contig, gstart, gstop, return_mapqs=True, return_strand=True,
            )
            stats["n_fetched"] += len(starts)
            stats["n_regions"] += 1

            if len(starts):
                (starts, stops, is_plus, n_mapq_pass,
                 n_dup, n_dup_s) = filter_and_dedup(
                    starts, stops, supp["mapq"], supp["strand"], min_mapq,
                )
                stats["n_mapq_pass"] += n_mapq_pass
                stats["n_dup_removed"] += n_dup
                stats["n_dup_removed_strand_key"] += n_dup_s
                stats["n_deduped"] += len(starts)

                contained = contained_in_region(starts, stops, gstart, gstop)
                starts, stops, is_plus = (
                    starts[contained], stops[contained], is_plus[contained],
                )
                stats["n_contained"] += len(starts)

                lengths = stops - starts
                in_range = (lengths >= l_min) & (lengths <= l_max)
                starts, stops, is_plus = (
                    starts[in_range], stops[in_range], is_plus[in_range],
                )
                stats["n_in_length_range"] += len(starts)

                if len(starts):
                    stats["n_counted"] += accumulate_observed(
                        obs, hex_fwd, hex_rc, valid, starts, stops, is_plus,
                        gstart, l_min, band_width, n_bands,
                    )

            if with_background:
                accumulate_background(
                    bg, hex_fwd, hex_rc, valid, band_edges, region_len,
                )

            if log_every and (i + 1) % log_every == 0:
                logger.info(
                    "  %d/%d regions (%.1fs, %d counted)",
                    i + 1, len(regions), time.monotonic() - t0,
                    stats["n_counted"],
                )
    finally:
        fasta.close()
        h5.close()

    stats["elapsed_s"] = time.monotonic() - t0

    if stats["n_fetched"] > 0 and stats["n_mapq_pass"] == 0:
        raise SystemExit(
            f"MAPQ filter removed ALL {stats['n_fetched']} fetched fragments at "
            f"min_mapq={min_mapq}. Unknown MAPQ is stored as -1, so this is "
            f"almost certainly the '-1 >= {min_mapq}' trap: MAPQ was never "
            f"carried into {h5_path}. Refusing to write an empty table."
        )

    total_obs = sum(int(v.sum()) for v in obs.values())
    if total_obs != 2 * stats["n_counted"]:
        raise AssertionError(
            f"observed counts {total_obs} != 2 x counted fragments "
            f"{2 * stats['n_counted']} -- fragments were silently dropped"
        )
    return obs, bg, stats


def _exact_int64(name: str, values: np.ndarray) -> np.ndarray:
    """``values`` as int64, refusing to round.

    ``bg`` is float64 only because ``np.bincount`` weights are; its entries are
    counts and so exact integers.  If one is ever not, that is a bug upstream
    and silently truncating it here would hide it.
    """
    if not np.all(values == np.rint(values)):
        bad = values[values != np.rint(values)][:3]
        raise ValueError(
            f"{name} holds non-integral counts (e.g. {bad.tolist()}); refusing "
            f"to round them into the int64 output column"
        )
    if values.size and values.max() > np.iinfo(np.int64).max:
        raise ValueError(f"{name} overflows int64")
    return values.astype(np.int64)


def build_dataframe(obs, bg, band_edges) -> pd.DataFrame:
    """The long-form table: one row per ``(hexamer, table, band)``.

    Row order is table-major, then band, then forward hexamer index -- but that
    is an implementation detail and a consumer must join on the ``hexamer``
    *string*, never on row position.  Rebuilding an index-ordered array by
    looking each string up through the consumer's own indexer is what turns a
    convention mismatch into a failed join instead of a silent relabelling.
    """
    vocab = hexamer_vocabulary().astype("U")
    n_bands = len(band_edges)
    n_tables = len(TABLE_NAMES)
    band_edges = np.asarray(band_edges)

    return pd.DataFrame({
        "hexamer": np.tile(vocab, n_tables * n_bands),
        "table": np.repeat(np.asarray(TABLE_NAMES), n_bands * NHEX),
        "band_lo": np.tile(np.repeat(band_edges[:, 0], NHEX), n_tables),
        "band_hi": np.tile(np.repeat(band_edges[:, 1], NHEX), n_tables),
        "observed": np.concatenate(
            [_exact_int64(f"obs[{k}]", obs[k]).reshape(-1) for k in TABLE_NAMES]
        ),
        "background": np.concatenate(
            [_exact_int64(f"bg[{k}]", bg[k]).reshape(-1) for k in TABLE_NAMES]
        ),
    })


def write_output(out_path, obs, bg, band_edges, meta):
    """Write the long-form Parquet file, with ``meta`` as file-level metadata."""
    table = pa.Table.from_pandas(
        build_dataframe(obs, bg, band_edges),
        schema=OUTPUT_SCHEMA,
        preserve_index=False,
    ).replace_schema_metadata(
        {META_KEY: json.dumps(meta, indent=2, sort_keys=True).encode("utf-8")}
    )
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd")


def read_output(path):
    """Read a file written by this script: ``(dataframe, meta)``.

    The metadata is not optional.  A Parquet file of these columns without the
    provenance block is not one of ours -- it could have been produced under a
    different region set, MAPQ cut or hexamer convention -- so this raises
    rather than returning counts whose origin is unknown.
    """
    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    if META_KEY not in metadata:
        raise ValueError(
            f"{path} carries no {META_KEY.decode()} file-level metadata, so its "
            f"region set, MAPQ cut and geometry are unknown. Refusing to return "
            f"counts without their provenance."
        )
    return table.to_pandas(), json.loads(metadata[META_KEY])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fragments-h5", required=True)
    parser.add_argument("--regions-bed", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--output", required=True, help="Output .parquet path")
    parser.add_argument(
        "--sample-name", default=None,
        help="Identity recorded in the output; defaults to the h5 basename.",
    )
    parser.add_argument(
        "--min-mapq", type=int, default=PlumbingConfig.min_mapq,
        help="Inclusive lower bound (>=) on min(mapq_read1, mapq_read2). "
             "Default is the store's PlumbingConfig.min_mapq.",
    )
    parser.add_argument("--l-min", type=int, default=L_MIN)
    parser.add_argument("--l-max", type=int, default=L_MAX,
                        help="INCLUSIVE, matching cut_site_stats.L_MAX.")
    parser.add_argument("--band-width", type=int, default=DEFAULT_BAND_WIDTH)
    parser.add_argument(
        "--no-background", action="store_true",
        help="Skip the candidate-cut-site tables (they roughly double runtime).",
    )
    parser.add_argument("--max-regions", type=int, default=None,
                        help="Limit regions (smoke tests only).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    sample_name = args.sample_name or os.path.basename(args.fragments_h5)
    regions = load_regions(args.regions_bed)
    if args.max_regions is not None:
        regions = regions[: args.max_regions]
    band_edges = make_band_edges(args.l_min, args.l_max, args.band_width)
    logger.info(
        "%s: %d regions, %d bands (%s..%s), min_mapq=%d",
        sample_name, len(regions), len(band_edges),
        band_edges[0].tolist(), band_edges[-1].tolist(), args.min_mapq,
    )

    obs, bg, stats = count_sample(
        args.fragments_h5, regions, args.fasta,
        min_mapq=args.min_mapq, band_edges=band_edges,
        l_min=args.l_min, l_max=args.l_max, band_width=args.band_width,
        with_background=not args.no_background,
    )

    meta = {
        "script": os.path.basename(__file__),
        "sample_name": sample_name,
        "fragments_h5": os.path.abspath(args.fragments_h5),
        "regions_bed": os.path.abspath(args.regions_bed),
        "regions_bed_md5": _file_md5(args.regions_bed),
        "n_regions_in_bed": len(regions),
        "fasta": os.path.abspath(args.fasta),
        "min_mapq": args.min_mapq,
        "mapq_rule": "min(mapq_read1, mapq_read2) >= min_mapq (inclusive)",
        "dedup_rule": "collapse fragments sharing (contig, start, stop) to one "
                      "molecule; strand NOT in the key "
                      "(== FragmentArray.drop_duplicate_fragments)",
        "in_region_rule": "fully contained: gstart <= start and stop <= gstop",
        "l_min": args.l_min,
        "l_max_inclusive": args.l_max,
        "band_width": args.band_width,
        "n_bands": len(band_edges),
        "band_edges_are": "half-open [lo, hi); last band truncated at l_max+1",
        "hexamer_convention": (
            "hex at cut site c spans seq[c-3:c+3] (3 in / 3 out); "
            "plus: c5=p,c3=p+L via hex_fwd; minus: c5=p+L,c3=p via hex_rc; "
            "counted only when valid[c5] and valid[c3]"
        ),
        "hexamer_orientation": (
            "the 'hexamer' column is written 5'->3' along the strand of the "
            "fragment that produced it: start_fwd/end_fwd are reference-forward, "
            "start_rev/end_rev are the REVERSE COMPLEMENT of the "
            "reference-forward 6-mer at that cut site. Do not reverse-complement "
            "them again when joining."
        ),
        "background_domain": (
            "all (c5, c3, strand) with L in [l_min, l_max] and both cut sites "
            "inside the region (simulator generative domain Omega)"
        ),
        "background_included": not args.no_background,
        "n_hexamers": NHEX,
        "stats": stats,
        "stats_note": (
            "n_fetched / n_mapq_pass / n_dup_removed / n_deduped are summed over "
            "per-region fetches, so a fragment overlapping two adjacent tiles is "
            "seen twice. It is counted in NEITHER (see in_region_rule), so "
            "n_contained onwards are exact. The duplicate rate is unaffected: "
            "exact-coordinate copies always land in the same fetch."
        ),
    }
    if stats["n_mapq_pass"]:
        meta["duplicate_rate"] = stats["n_dup_removed"] / stats["n_mapq_pass"]
        meta["duplicate_rate_strand_key"] = (
            stats["n_dup_removed_strand_key"] / stats["n_mapq_pass"]
        )

    write_output(args.output, obs, bg, band_edges, meta)
    logger.info(
        "%s: fetched %d, mapq-pass %d, deduped %d (dup rate %.4f), "
        "contained %d, counted %d in %.1fs -> %s",
        sample_name, stats["n_fetched"], stats["n_mapq_pass"],
        stats["n_deduped"], meta.get("duplicate_rate", float("nan")),
        stats["n_contained"], stats["n_counted"], stats["elapsed_s"], args.output,
    )


if __name__ == "__main__":
    main()
