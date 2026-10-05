"""Count cut-site hexamers via the fragmentomics_tools region/fragment API.

Three passes over a ``SampleAndRegionDataFrame``, each parallel:

1. ``attach_fragment_arrays(min_mapq=...)`` -- fetch + MAPQ, from the library.
2. ``attach_sequence`` -- the padded region sequence, one column.
3. ``count_srdf`` -- dedup, midpoint admission, hexamer lookup, bincount.

The fragment half is the library's; the sequence half is not, because
``fragmentomics_tools`` has no hexamer machinery.  ``hexamer_indices`` from
``simulator.precompute`` stays the encoder and the ``MAX_FL_HALF`` padded frame
is managed here.

What the library supplies, and why taking it matters:

- **MAPQ** via ``attach_fragment_arrays(min_mapq=...)``, reaching
  ``from_fragments_h5``, which applies ``mapqs.min(axis=1) >= min_mapq`` -- the
  min over a fragment's two reads.  Using the library's filter rather than
  repeating it is the point: fragments whose two reads straddle the threshold
  are kept or dropped by that reduction alone.
- **Dedup** via ``RegionFragmentArray.drop_duplicate_fragments()``, keyed on
  ``(starts_0, stops_0)``, so a coordinate pair differing only in strand
  collapses to one molecule.  MAPQ at load, dedup after -- store order.
- **Midpoints** via ``rfa.midpoints_0``, verified to use the same
  ``start + L // 2`` floor convention as ``weights.midpoint_index_arrays``.

Two library details that bite if assumed rather than checked:

- ``rfa.fragment_strands`` is ``<U1`` (``'+'``), NOT the ``b'+'`` bytes
  ``FragmentsH5.fetch_array`` returns.  Comparing against bytes silently
  yields an all-False mask and an empty minus-strand table.
- ``starts_0``/``stops_0`` are **not clipped** to the region: a fragment
  admitted on its midpoint may overhang, and ``stops_0.max()`` measured 1591
  for a 1536 bp region.  That overhang is what the padded sequence frame
  exists to resolve, so it must not be filtered away.

The per-region return is **always sparse** -- nonzero ``(table, hexamer,
count)`` triples.  A dense 4x4096 per region would be ~200x larger than the
fragments it summarises and 1.09e9 values over the full region set.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

from background_model.simulator.precompute import HEX_HALF, NHEX, hexamer_indices
from background_model.simulator.weights import L_MAX, L_MIN, MAX_FL_HALF

TABLE_NAMES = ("start_fwd", "end_fwd", "start_rev", "end_rev")


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
    import pandas as pd

    from fragmentomics_tools.dataframe import SampleDataFrame

    pairs = list(samples)
    if not pairs:
        raise ValueError("no samples given")
    return SampleDataFrame(pd.DataFrame({
        "sample_id": [s for s, _ in pairs],
        "frag_h5": [p for _, p in pairs],
    }))


def build_srdf(rdf, sdf):
    """Cross ``rdf`` with ``sdf`` via the sanctioned constructor."""
    from fragmentomics_tools.dataframe import SampleAndRegionDataFrame

    return SampleAndRegionDataFrame.init_from_rdf_and_sdf(rdf, sdf)


def attach_sequence(srdf, fasta_path: str, *, pad: int = MAX_FL_HALF,
                    verbose: bool = False):
    """Add a ``sequence`` column: the region plus ``pad + HEX_HALF`` flanking.

    Thin wrapper over the library's ``attach_sequence``, which gained
    ``left_pad``/``right_pad`` for exactly this. It is serial (one
    ``FastaFile`` for the whole loop, no ``n_workers``), which costs ~45s over
    66,649 regions -- negligible against the rest of the pipeline, and not
    worth a second hand-rolled fetch loop to avoid.

    The raw sequence is stored rather than the hexamer arrays: ~1.7 KB per
    region against ~29 KB for ``(hex_fwd, hex_rc, valid)`` as int64, and
    ``hexamer_indices`` is cheap to run in the counting pass.
    """
    flank = pad + HEX_HALF
    return srdf.attach_sequence(
        fasta_path, left_pad=flank, right_pad=flank, verbose=verbose,
    )


def count_hexamers(
    rfa,
    sequence: str,
    gstart: int,
    gstop: int,
    *,
    l_min: int = L_MIN,
    l_max: int = L_MAX,
    pad: int = MAX_FL_HALF,
) -> Dict[str, np.ndarray]:
    """Bincount one region's cut-site hexamers into four ``(4096,)`` tables.

    ``rfa`` is a ``RegionFragmentArray`` already MAPQ-filtered at load;
    ``sequence`` is the padded region sequence from ``attach_sequence``.
    """
    counts = empty_counts()
    if rfa.n_frags == 0:
        return counts

    rfa = rfa.drop_duplicate_fragments()
    region_len = gstop - gstart
    lengths = rfa.lengths.astype(np.int64)
    mids = rfa.midpoints_0.astype(np.int64)
    adm = (
        (mids >= 0) & (mids < region_len)
        & (lengths >= l_min) & (lengths <= l_max)
    )
    if not adm.any():
        return counts

    # <U1, not bytes -- see module docstring.
    is_plus = np.asarray(rfa.fragment_strands)[adm] == "+"
    lo = rfa.starts_0.astype(np.int64)[adm] + pad
    hi = rfa.stops_0.astype(np.int64)[adm] + pad

    # The library's get_sequence returns BYTES and does NOT upper-case, while
    # hg38 is soft-masked: lowercase repeat bases miss the base LUT and would
    # be silently counted as invalid, which would drop exactly the repeat-rich
    # positions.
    raw = sequence if isinstance(sequence, bytes) else sequence.encode("ascii")
    hex_fwd, hex_rc, valid = hexamer_indices(
        np.frombuffer(raw.upper(), dtype=np.uint8)
    )
    n_sites = region_len + 2 * pad + 1
    if len(hex_fwd) != n_sites:
        raise ValueError(
            f"{gstart}-{gstop}: {len(hex_fwd)} cut sites, expected {n_sites} "
            f"-- the FASTA fetch was truncated, which happens within "
            f"{pad + HEX_HALF}bp of a contig end."
        )
    if lo.min() < 0 or hi.max() >= n_sites:
        raise ValueError(
            f"{gstart}-{gstop}: cut site outside the padded frame "
            f"([{lo.min()}, {hi.max()}] vs [0, {n_sites - 1}]) -- a fragment "
            f"overhangs by more than pad={pad}."
        )

    # Minus strand puts the 5' end at the HIGHER coordinate, so the pair swaps.
    c5 = np.where(is_plus, lo, hi)
    c3 = np.where(is_plus, hi, lo)
    ok = valid[c5] & valid[c3]
    if not ok.any():
        return counts

    for strand_mask, track, s_name, e_name in (
        (is_plus, hex_fwd, "start_fwd", "end_fwd"),
        (~is_plus, hex_rc, "start_rev", "end_rev"),
    ):
        sel = strand_mask & ok
        if not sel.any():
            continue
        counts[s_name] += np.bincount(track[c5[sel]], minlength=NHEX)
        counts[e_name] += np.bincount(track[c3[sel]], minlength=NHEX)
    return counts


def count_srdf(
    srdf,
    *,
    l_min: int = L_MIN,
    l_max: int = L_MAX,
    pad: int = MAX_FL_HALF,
    n_workers: int | None = None,
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Count over a frame that already has ``fragment_array`` and ``sequence``.

    Each row bincounts locally and returns only its NONZERO ``(table, hexamer,
    count)`` triples; the reduction is a groupby-sum.
    """
    import pandas as pd

    for col in ("fragment_array", "sequence"):
        if col not in srdf.columns:
            raise ValueError(
                f"srdf has no {col!r} column -- call "
                f"{'attach_fragment_arrays' if col == 'fragment_array' else 'attach_sequence'}"
                f" first"
            )

    def fn(row):
        c = count_hexamers(
            row["fragment_array"], row["sequence"],
            int(row["start"]), int(row["stop"]),
            l_min=l_min, l_max=l_max, pad=pad,
        )
        frames = []
        for ti, name in enumerate(TABLE_NAMES):
            nz = np.nonzero(c[name])[0]
            if len(nz):
                frames.append(pd.DataFrame({
                    "table": np.full(len(nz), ti, dtype=np.int8),
                    "hexamer": nz.astype(np.int32),
                    "count": c[name][nz].astype(np.int64),
                }))
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame({
            "table": np.empty(0, np.int8),
            "hexamer": np.empty(0, np.int32),
            "count": np.empty(0, np.int64),
        })

    res = srdf.parallel_apply(fn, n_workers=n_workers, verbose=verbose)

    counts = empty_counts()
    if len(res):
        agg = res.groupby(["table", "hexamer"], sort=False)["count"].sum()
        for (ti, hexi), v in agg.items():
            counts[TABLE_NAMES[ti]][hexi] += int(v)

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
    stats = dict(
        n_regions=len(srdf),
        n_after_mapq=int(sum(fa.n_frags for fa in srdf["fragment_array"])),
        n_counted=total // 2,
    )
    if stats["n_regions"] and not stats["n_after_mapq"]:
        raise ValueError(
            "MAPQ removed ALL fragments. Unknown MAPQ is stored as -1, so "
            "this is almost certainly the '-1 >= min_mapq' trap: MAPQ was "
            "never carried into the h5."
        )
    return counts, stats


def count_sample(
    rdf,
    sample_id: str,
    h5_path: str,
    fasta_path: str,
    *,
    min_mapq: int = 10,
    l_min: int = L_MIN,
    l_max: int = L_MAX,
    pad: int = MAX_FL_HALF,
    n_workers: int | None = None,
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """The three passes end to end for one sample."""
    sdf = load_sample_dataframe([(sample_id, h5_path)])
    srdf = build_srdf(rdf, sdf)
    srdf = srdf.attach_fragment_arrays(
        min_mapq=min_mapq, n_workers=n_workers, verbose=verbose,
    )
    srdf = attach_sequence(srdf, fasta_path, pad=pad, verbose=verbose)
    return count_srdf(
        srdf, l_min=l_min, l_max=l_max, pad=pad,
        n_workers=n_workers, verbose=verbose,
    )
