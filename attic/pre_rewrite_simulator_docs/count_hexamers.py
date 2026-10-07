"""Count de-duplicated cut-site hexamers from a fragment h5.

Emits the four untied tables the simulator consumes, as RAW COUNTS of shape
``(4096,)`` each.  Turning counts into weights is a separate decision and is
deliberately not made here.

Geometry is the simulator's own, not a parallel convention:

- A fragment is admitted if its **midpoint** lies in ``[gstart, gstop)`` --
  the same admission rule as ``weights.midpoint_index_arrays``.
- Cut sites may therefore fall up to ``MAX_FL_HALF`` outside the region, so
  sequence is fetched with ``MAX_FL_HALF + HEX_HALF`` flanking and array index
  ``= genomic - gstart + MAX_FL_HALF``.  This is the same padded frame
  ``build_region_weights`` uses.
- ``hexamer_indices`` is imported, never reimplemented: a second base-4 encoder
  is the shared-contract problem ``hexamer_vocabulary`` exists to remove.
- MAPQ then dedup, in that order, matching ``from_fragments_h5`` followed by
  ``drop_duplicate_fragments()``.  Dedup is on ``(start, stop)`` only, so a
  coordinate pair differing just in strand collapses to one molecule.

Unknown MAPQ is stored as ``-1``, so ``min_mapq=0`` still excludes it.

Parallelism
-----------
``count_regions_parallel`` goes through ``RegionDataFrame.parallel_apply``,
which forks.  **Handles are opened lazily per worker process, never inherited
across the fork**: an HDF5 handle opened before a fork is not safe to use in
the child, and CLAUDE.md records a ``parallel_apply`` fork deadlock that ran 12
hours emitting nothing at all.  ``_handles`` therefore caches on ``os.getpid()``
so each worker opens its own on first use.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Tuple

import numpy as np

from background_model.simulator.precompute import HEX_HALF, NHEX, hexamer_indices
from background_model.simulator.weights import L_MAX, L_MIN, MAX_FL_HALF

TABLE_NAMES = ("start_fwd", "end_fwd", "start_rev", "end_rev")

# pid -> (FragmentsH5, pysam.FastaFile). Keyed by pid so a forked worker never
# reuses the parent's handles; see the module docstring.
_HANDLE_CACHE: dict = {}


def _handles(h5_path: str, fasta_path: str):
    """Per-process handles, opened on first use in whichever process asks."""
    key = (os.getpid(), h5_path, fasta_path)
    if key not in _HANDLE_CACHE:
        import pysam
        from fragments_h5 import FragmentsH5

        _HANDLE_CACHE[key] = (
            FragmentsH5(h5_path, cache_pointers=False),
            pysam.FastaFile(fasta_path),
        )
    return _HANDLE_CACHE[key]


def empty_counts() -> Dict[str, np.ndarray]:
    """Four zeroed ``(4096,)`` int64 tables."""
    return {k: np.zeros(NHEX, dtype=np.int64) for k in TABLE_NAMES}


def filter_fragments(
    starts: np.ndarray,
    stops: np.ndarray,
    mapqs: np.ndarray,
    strands: np.ndarray,
    gstart: int,
    gstop: int,
    *,
    min_mapq: int = 10,
    l_min: int = L_MIN,
    l_max: int = L_MAX,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """MAPQ -> dedup -> midpoint admission -> length band.

    Returns ``(starts, stops, is_plus, stats)``, all genomic coordinates.

    Order matters and matches the store: ``from_fragments_h5`` applies MAPQ
    then ``drop_duplicate_fragments()`` runs on the survivors.

    - **MAPQ** is ``mapqs.min(axis=1) >= min_mapq``, i.e. the min over a
      fragment's two reads, which is byte-for-byte what ``from_fragments_h5``
      does (``fragment_array.py``, the ``ndim == 2`` branch). The reduction is
      not cosmetic: fragments whose two reads straddle the threshold are kept
      or dropped by this choice alone. ``mapqs`` must be 2-D ``(n, 2)``; a 1-D
      array raises rather than quietly filtering per-read.
    - **Dedup** is on ``(start, stop)`` ONLY, so a coordinate pair differing
      just in strand collapses to one molecule -- matching
      ``drop_duplicate_fragments()``.
    - **Admission** is midpoint in ``[gstart, gstop)``, the same rule as
      ``weights.midpoint_index_arrays``, NOT containment. Tiles are contiguous
      and the fetch is overlap-based, so containment would both admit a
      neighbour's fragments and reject this region's overhanging ones.
    """
    stats = dict(n_mapq_pass=0, n_deduped=0, n_admitted=0)
    empty = np.zeros(0, dtype=bool)

    keep = mapqs.min(axis=1) >= min_mapq
    starts, stops, strands = starts[keep], stops[keep], strands[keep]
    stats["n_mapq_pass"] = int(keep.sum())
    if not len(starts):
        return starts, stops, empty, stats

    _, idx = np.unique(np.array([starts, stops]), axis=1, return_index=True)
    starts, stops = starts[idx], stops[idx]
    is_plus = strands[idx] == b"+"
    stats["n_deduped"] = len(idx)

    lengths = (stops - starts).astype(np.int64)
    mids = starts.astype(np.int64) + lengths // 2
    adm = (
        (mids >= gstart) & (mids < gstop)
        & (lengths >= l_min) & (lengths <= l_max)
    )
    stats["n_admitted"] = int(adm.sum())
    return starts[adm], stops[adm], is_plus[adm], stats


def count_region(
    contig: str,
    gstart: int,
    gstop: int,
    h5,
    fasta,
    *,
    min_mapq: int = 10,
    l_min: int = L_MIN,
    l_max: int = L_MAX,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Count one region. Returns ``(counts, stats)``.

    ``h5`` and ``fasta`` are already-open handles -- opening the 3 GB hg38
    FASTA per call measured 20x slower on EFS.
    """
    pad = MAX_FL_HALF
    counts = empty_counts()
    stats = dict(n_fetched=0, n_mapq_pass=0, n_deduped=0, n_admitted=0,
                 n_counted=0)

    seq = fasta.fetch(
        contig, gstart - pad - HEX_HALF, gstop + pad + HEX_HALF
    ).upper()
    hex_fwd, hex_rc, valid = hexamer_indices(
        np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    )
    n_sites = (gstop - gstart) + 2 * pad + 1
    if len(hex_fwd) != n_sites:
        raise ValueError(
            f"{contig}:{gstart}-{gstop}: {len(hex_fwd)} cut sites, expected "
            f"{n_sites} -- the FASTA fetch was truncated, which happens within "
            f"{pad + HEX_HALF}bp of a contig end."
        )

    starts, stops, supp = h5.fetch_array(
        contig, gstart, gstop, return_mapqs=True, return_strand=True,
    )
    stats["n_fetched"] = len(starts)
    if not len(starts):
        return counts, stats

    starts, stops, is_plus, fstats = filter_fragments(
        starts, stops, supp["mapq"], supp["strand"], gstart, gstop,
        min_mapq=min_mapq, l_min=l_min, l_max=l_max,
    )
    stats.update(fstats)
    if not len(starts):
        return counts, stats

    # Cut sites in the padded frame. Minus strand puts the 5' end at the
    # HIGHER coordinate, so the pair swaps.
    lo = starts.astype(np.int64) - gstart + pad
    hi = stops.astype(np.int64) - gstart + pad
    c5 = np.where(is_plus, lo, hi)
    c3 = np.where(is_plus, hi, lo)

    ok = valid[c5] & valid[c3]
    if not ok.any():
        return counts, stats
    stats["n_counted"] = int(ok.sum())

    for strand_mask, track, s_name, e_name in (
        (is_plus, hex_fwd, "start_fwd", "end_fwd"),
        (~is_plus, hex_rc, "start_rev", "end_rev"),
    ):
        sel = strand_mask & ok
        if not sel.any():
            continue
        counts[s_name] += np.bincount(track[c5[sel]], minlength=NHEX)
        counts[e_name] += np.bincount(track[c3[sel]], minlength=NHEX)

    return counts, stats


def _check_totals(counts, stats, h5_path, min_mapq):
    if stats["n_fetched"] and not stats["n_mapq_pass"]:
        raise ValueError(
            f"MAPQ removed ALL {stats['n_fetched']} fragments at "
            f"min_mapq={min_mapq}. Unknown MAPQ is stored as -1, so this is "
            f"almost certainly the '-1 >= {min_mapq}' trap: MAPQ was never "
            f"carried into {h5_path}."
        )
    total = sum(int(v.sum()) for v in counts.values())
    if total != 2 * stats["n_counted"]:
        raise AssertionError(
            f"counted {total} hexamers != 2 x {stats['n_counted']} fragments "
            f"-- fragments were silently dropped"
        )


def count_regions(
    h5_path: str,
    regions: Iterable[Tuple[str, int, int]],
    fasta_path: str,
    *,
    min_mapq: int = 10,
    l_min: int = L_MIN,
    l_max: int = L_MAX,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Serial accumulation over ``regions``. Returns ``(counts, stats)``."""
    h5, fasta = _handles(h5_path, fasta_path)
    counts = empty_counts()
    stats = dict(n_regions=0, n_fetched=0, n_mapq_pass=0, n_deduped=0,
                 n_admitted=0, n_counted=0)
    for contig, gstart, gstop in regions:
        c, s = count_region(
            contig, gstart, gstop, h5, fasta,
            min_mapq=min_mapq, l_min=l_min, l_max=l_max,
        )
        stats["n_regions"] += 1
        for k in TABLE_NAMES:
            counts[k] += c[k]
        for k in s:
            stats[k] += s[k]
    _check_totals(counts, stats, h5_path, min_mapq)
    return counts, stats


def count_regions_parallel(
    rdf,
    h5_path: str,
    fasta_path: str,
    *,
    min_mapq: int = 10,
    l_min: int = L_MIN,
    l_max: int = L_MAX,
    n_workers: int | None = None,
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Same result as ``count_regions``, fanned out over ``rdf`` rows.

    ``rdf`` is a ``RegionDataFrame``.  Each row returns only its NON-ZERO
    ``(table, hexamer, count)`` triples plus its stats: returning the full
    4x4096 tables per region would materialise 66,649 x 16,384 values in the
    parent, so the reduction is a groupby-sum over sparse triples instead.
    """
    import pandas as pd

    def fn(row):
        h5, fasta = _handles(h5_path, fasta_path)
        c, s = count_region(
            row["contig"], int(row["start"]), int(row["stop"]), h5, fasta,
            min_mapq=min_mapq, l_min=l_min, l_max=l_max,
        )
        frames = []
        for ti, name in enumerate(TABLE_NAMES):
            nz = np.nonzero(c[name])[0]
            if len(nz):
                frames.append(pd.DataFrame({
                    "table": np.full(len(nz), ti, dtype=np.int8),
                    "hexamer": nz.astype(np.int32),
                    "count": c[name][nz],
                }))
        # One stats row ALWAYS, marked table=-1. A region that fetched
        # fragments but counted none still has to report its n_fetched, or the
        # MAPQ-trap guard goes blind exactly when it is needed.
        stats_row = pd.DataFrame({
            "table": np.array([-1], dtype=np.int8),
            "hexamer": np.array([-1], dtype=np.int32),
            "count": np.array([0], dtype=np.int64),
            **{k: np.array([v], dtype=np.int64) for k, v in s.items()},
        })
        if not frames:
            return stats_row
        counts_df = pd.concat(frames, ignore_index=True)
        for k in s:
            counts_df[k] = 0
        return pd.concat([stats_row, counts_df], ignore_index=True)

    res = rdf.parallel_apply(fn, n_workers=n_workers, verbose=verbose)

    counts = empty_counts()
    stats = dict(n_regions=len(rdf), n_fetched=0, n_mapq_pass=0, n_deduped=0,
                 n_admitted=0, n_counted=0)
    if len(res):
        real = res[res["table"] >= 0]
        if len(real):
            agg = real.groupby(["table", "hexamer"], sort=False)["count"].sum()
            for (ti, hexi), v in agg.items():
                counts[TABLE_NAMES[ti]][hexi] += int(v)
        for k in ("n_fetched", "n_mapq_pass", "n_deduped", "n_admitted",
                  "n_counted"):
            stats[k] = int(res.loc[res["table"] < 0, k].sum())
    _check_totals(counts, stats, h5_path, min_mapq)
    return counts, stats
