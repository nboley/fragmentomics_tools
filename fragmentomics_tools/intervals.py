"""Interval algebra for region frames.

Five free functions over ``RegionDataFrame`` backed by ``bioframe``.
No ``bioframe`` name, argument or column convention appears in any signature.

Layer 2 of the three-layer model (load → operate → write).  Takes and
returns ``RegionDataFrame``; imports it lazily so that ``import
fragmentomics_tools.intervals`` does not pull in the heavy stack.

See ``docs/pending/interval_api_design.md`` for the design rationale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from fragmentomics_tools.dataframe import RegionDataFrame

# Column convention used by every RegionDataFrame in this codebase.
_COLS = ("contig", "start", "stop")

_VALID_HOW = frozenset({"inner", "left", "right", "outer", "anti"})


def _assert_same_ref(a: "RegionDataFrame", b: "RegionDataFrame") -> None:
    if a.ref != b.ref:
        raise ValueError(
            f"RegionDataFrames must have the same reference: "
            f"{a.ref!r} vs {b.ref!r}"
        )


def _strand_mask(a, b, idx, same_strand: bool) -> pd.Series:
    """Return a boolean mask selecting rows that satisfy the strand constraint.

    When *same_strand* is False every row passes.  When True, rows match
    only when both strands are in {"+", "-"} and are equal.  In particular,
    ``"."`` vs ``"."`` and ``None`` vs ``None`` are **not** matches — this
    is the bedtools ``-s`` convention, and the divergence from bioframe's
    equality join that the design document mandates.
    """
    if not same_strand:
        return pd.Series(True, index=idx.index)

    # Positional: callers pass frames already normalised by `_positional`.
    a_strand = a["strand"].values[idx["a_pos"].values.astype(int)]
    b_strand = b["strand"].values[idx["b_pos"].values.astype(int)]

    stranded_a = (a_strand == "+") | (a_strand == "-")
    stranded_b = (b_strand == "+") | (b_strand == "-")
    return pd.Series(
        stranded_a & stranded_b & (a_strand == b_strand),
        index=idx.index,
    )


def _positional(*frames):
    """Return positionally-indexed copies of the input frames.

    **This is the module's central invariant.** ``bioframe`` reports matches as
    the *labels* of whatever index the input frames carried.  Resetting to
    0..n-1 makes those labels equal to positions, so everything downstream
    is unambiguously positional.

    The returned ``a_pos``/``b_pos`` columns in ``overlap_indices`` and
    ``nearest`` ARE the public contract — callers use
    ``a.iloc[result.a_pos]`` — so the position-space boundary is the whole
    API, not an internal detail mapped back to labels on exit.
    """
    return tuple(f.reset_index(drop=True) for f in frames)


def _deterministic_order(idx):
    """Sort pair results by ``(a_pos, b_pos)`` and renumber.

    Row order here is **not** incidental. The Phase 0 design made it part of
    the contract deliberately: two implementations can agree on the set of
    pairs and still differ on ordering, which silently breaks any caller that
    zips or positionally indexes the result.

    Without this, the order is **not stable across processes**. Measured on
    964,593 CTCF regions against the hg38 blacklist: three runs of identical
    code on identical input produced three different digests, while pinning
    `PYTHONHASHSEED` made them identical — so something upstream iterates a
    set or dict whose order follows Python's hash randomisation. It is stable
    *within* a process, which is exactly what makes it easy to miss.

    This predates the position-space refactor: the original implementation
    shows the same instability. A library that returns rows in a different
    order on every run cannot be used as a baseline and cannot be reasoned
    about, so it is sorted here rather than documented as a caveat.

    Sorted on positions, which are always plain integers — no mixed-type
    comparison risk.
    """
    return idx.sort_values(
        ["a_pos", "b_pos"], na_position="last", kind="mergesort"
    ).reset_index(drop=True)


# ── overlap_indices ──────────────────────────────────────────────────

def overlap_indices(
    a: "RegionDataFrame",
    b: "RegionDataFrame",
    *,
    how: str = "inner",
    pad: int = 0,
    min_frac_a: float = 0.0,
    min_frac_b: float = 0.0,
    reciprocal: bool = False,
    same_strand: bool = False,
) -> pd.DataFrame:
    """Return ``(a_pos, b_pos, overlap_bases)`` for overlapping pairs.

    Parameters
    ----------
    a, b : RegionDataFrame
        The two region sets.  Must share the same ``ref``.
    how : {"inner", "left", "right", "outer", "anti"}
        Join type.  ``"anti"`` returns A rows with **no** match in B.
    pad : int
        Slack for overlap testing.  A pair matches when the edge-to-edge
        gap is **strictly less than** ``pad``.  ``pad=0`` is strict overlap
        (book-ended intervals do **not** match).  Equivalent to
        ``bedtools window -w pad``: gap 0 first matches at ``pad=1``,
        gap G first matches at ``pad=G+1``.
    min_frac_a, min_frac_b : float
        Minimum fraction of A (or B) that must be covered.
    reciprocal : bool
        bedtools ``-r``. If True, ``min_frac_a`` must be satisfied against
        **both** A and B, and ``min_frac_b`` is ignored. Requires
        ``min_frac_a > 0``.
    same_strand : bool
        If True, only match when both strands are in {"+", "-"} and equal.
        ``"."`` vs ``"."`` is **not** a match (bedtools ``-s`` convention).

    Returns
    -------
    DataFrame with columns ``a_pos``, ``b_pos``, ``overlap_bases``.
    ``a_pos`` and ``b_pos`` are 0-based row positions into *a* and *b*
    respectively, so ``a.iloc[result.a_pos]`` is always unambiguous —
    even when the input carries duplicate index labels (e.g. from
    ``bin_regions_into_windows``).
    For ``how="anti"``, ``b_pos`` is always ``pd.NA`` and
    ``overlap_bases`` is 0.
    """
    import bioframe

    if how not in _VALID_HOW:
        raise ValueError(
            f"how={how!r} is not valid; must be one of {sorted(_VALID_HOW)}"
        )

    # Validated at entry, not inside the fraction block -- that block is
    # guarded by `min_frac_a > 0 or min_frac_b > 0`, so a check placed there
    # never fires for the one case it exists to catch.
    if pad < 0:
        raise ValueError(f"pad must be >= 0, got {pad}")

    if reciprocal and min_frac_a <= 0:
        raise ValueError(
            "reciprocal=True requires min_frac_a > 0; it applies min_frac_a "
            "to both A and B (bedtools -r)"
        )

    _assert_same_ref(a, b)

    # Pad: expand B intervals so a gap < pad still produces an overlap.
    # Half-open overlap: [a_start, a_stop) and [b_start, b_stop) overlap iff
    # b_start < a_stop AND a_start < b_stop.  Expanding B by `pad` on each
    # side makes a gap of G overlap iff G < pad — the `bedtools window -w`
    # convention.
    # Enter position space. Everything below indexes positionally, which is
    # correct BECAUSE of this line -- see _positional.
    a, b = _positional(a, b)

    b_orig_start = b["start"].values.copy()
    b_orig_stop = b["stop"].values.copy()
    if pad > 0:
        b = b.copy()
        b["start"] = b["start"] - pad
        b["stop"] = b["stop"] + pad

    bf_how = how
    if how == "anti":
        bf_how = "left"
    elif how == "outer":
        bf_how = "outer"

    result = bioframe.overlap(
        a, b,
        how=bf_how,
        return_index=True,
        return_overlap=True,
        return_input=False,
        cols1=_COLS,
        cols2=_COLS,
    )

    idx = pd.DataFrame({
        "a_pos": result["index"],
        "b_pos": result["index_"],
    })

    # Compute overlap_bases from the overlap coordinates bioframe returns.
    has_overlap = idx["b_pos"].notna()
    overlap_bases = pd.array([0] * len(result), dtype="Int64")
    if has_overlap.any():
        if pad > 0:
            # Compute actual overlap against original (un-expanded) B intervals.
            a_pos = idx.loc[has_overlap, "a_pos"].values.astype(int)
            b_pos = idx.loc[has_overlap, "b_pos"].values.astype(int)
            a_starts = a["start"].values[a_pos]
            a_stops = a["stop"].values[a_pos]
            b_starts = b_orig_start[b_pos]
            b_stops = b_orig_stop[b_pos]
            o_start = np.maximum(a_starts, b_starts)
            o_stop = np.minimum(a_stops, b_stops)
            raw = np.maximum(o_stop - o_start, 0)
            overlap_bases[has_overlap.values] = raw
        else:
            o_start = result.loc[has_overlap, "overlap_start"]
            o_stop = result.loc[has_overlap, "overlap_stop"]
            overlap_bases[has_overlap.values] = (o_stop - o_start).values

    idx["overlap_bases"] = overlap_bases

    # Strand filter — applied before fraction filters and before anti.
    if same_strand and has_overlap.any():
        matched_rows = idx[has_overlap].copy()
        strand_ok = _strand_mask(a, b, matched_rows, same_strand=True)
        fail_mask = has_overlap & ~strand_ok.reindex(idx.index, fill_value=True)
        if how in ("inner", "right"):
            idx = idx[~fail_mask]
        else:
            idx.loc[fail_mask, "b_pos"] = pd.NA
            idx.loc[fail_mask, "overlap_bases"] = 0
        has_overlap = idx["b_pos"].notna()

    # Fraction filters (applied on matched rows only).
    if (min_frac_a > 0 or min_frac_b > 0) and has_overlap.any():
        matched = idx[has_overlap]
        a_lens = (a["stop"].values - a["start"].values)[
            matched["a_pos"].values.astype(int)
        ]
        b_lens = (b_orig_stop - b_orig_start)[matched["b_pos"].values.astype(int)]
        ob = matched["overlap_bases"].values

        # bedtools `-r`: the fraction requirement applies reciprocally, i.e.
        # `min_frac_a` must be met against BOTH A and B. Without it the two
        # thresholds are independent. An earlier version made `reciprocal`
        # compute `frac_a_ok & frac_b_ok`, which is identical to the branch
        # below it whenever both thresholds are set and a no-op otherwise --
        # the flag could not change any result.
        if reciprocal:
            keep = (ob >= (min_frac_a * a_lens)) & (ob >= (min_frac_a * b_lens))
        else:
            frac_a_ok = ob >= (min_frac_a * a_lens) if min_frac_a > 0 else True
            frac_b_ok = ob >= (min_frac_b * b_lens) if min_frac_b > 0 else True
            keep = frac_a_ok & frac_b_ok

        fail = has_overlap.copy()
        fail[has_overlap] = ~keep
        if how in ("inner", "right"):
            idx = idx[~fail]
        else:
            idx.loc[fail, "b_pos"] = pd.NA
            idx.loc[fail, "overlap_bases"] = 0
        has_overlap = idx["b_pos"].notna()

    # Anti: keep only A rows with no match in B.
    if how == "anti":
        no_match = ~has_overlap
        idx = idx[no_match].copy()
        idx["b_pos"] = pd.NA
        idx["overlap_bases"] = 0

    return _deterministic_order(idx)


# ── overlaps ─────────────────────────────────────────────────────────

def overlaps(
    a: "RegionDataFrame",
    b: "RegionDataFrame",
    *,
    pad: int = 0,
    same_strand: bool = False,
) -> pd.Series:
    """Boolean mask: which rows of *a* overlap at least one row in *b*.

    The most-used operation in the family, which is why it has its own name
    rather than being spelled out at each call site.

    Computed **per row**, not per index label. An earlier version was
    `a.index.isin(pairs.a_index)`, which is label membership: when two rows
    share an index label and only one of them overlaps, `isin` marks *both*
    True. `pd.concat` without `ignore_index=True` produces exactly that frame.
    Building the mask positionally makes the result independent of whatever
    index the caller happens to carry.

    Parameters
    ----------
    a, b : RegionDataFrame
    pad : int
        Slack for overlap testing.  A pair matches when the gap is
        strictly less than ``pad``.  ``pad=0`` is strict overlap.
        See ``overlap_indices`` for the full convention.
    same_strand : bool
        If True, only match when both strands are in {"+", "-"} and equal.

    Returns
    -------
    pd.Series[bool], index-aligned to *a*, one entry per ROW of *a*.
    """
    idx = overlap_indices(a, b, how="inner", pad=pad, same_strand=same_strand)
    mask = np.zeros(len(a), dtype=bool)
    matched = idx["a_pos"].dropna()
    if len(matched):
        mask[matched.values.astype(int)] = True
    return pd.Series(mask, index=a.index, dtype=bool)


# ── nearest ──────────────────────────────────────────────────────────

def nearest(
    a: "RegionDataFrame",
    b: "RegionDataFrame",
    *,
    k: int = 1,
    ignore_overlaps: bool = False,
    direction: str | None = None,
    same_strand: bool = False,
) -> pd.DataFrame:
    """For each row in *a*, find the *k* nearest rows in *b*.

    Parameters
    ----------
    a, b : RegionDataFrame
    k : int
        Number of nearest neighbours to return per A row.
    ignore_overlaps : bool
        If True, skip B intervals that overlap A.
    direction : {None, "upstream", "downstream"}
        Restrict search direction.  None means both.
    same_strand : bool
        If True, only match same-strand intervals.

    Returns
    -------
    DataFrame with columns ``a_pos``, ``b_pos``, ``distance``.
    ``a_pos`` and ``b_pos`` are 0-based row positions.
    """
    import bioframe

    _assert_same_ref(a, b)

    # Enter position space -- same invariant as overlap_indices.
    a, b = _positional(a, b)

    bf_kwargs = dict(
        k=k,
        ignore_overlaps=ignore_overlaps,
        return_index=True,
        return_distance=True,
        return_input=False,
        cols1=_COLS,
        cols2=_COLS,
    )

    if direction == "upstream":
        bf_kwargs["ignore_downstream"] = True
    elif direction == "downstream":
        bf_kwargs["ignore_upstream"] = True
    elif direction is not None:
        raise ValueError(
            f"direction={direction!r} is not valid; must be None, "
            f"'upstream' or 'downstream'"
        )

    result = bioframe.closest(a, b, **bf_kwargs)

    idx = pd.DataFrame({
        "a_pos": result["index"],
        "b_pos": result["index_"],
        "distance": result["distance"],
    })

    if same_strand:
        has_match = idx["b_pos"].notna()
        if has_match.any():
            strand_ok = _strand_mask(a, b, idx[has_match], same_strand=True)
            fail = has_match & ~strand_ok.reindex(idx.index, fill_value=True)
            idx = idx[~fail]

    return _deterministic_order(idx)


# ── cluster ──────────────────────────────────────────────────────────

def cluster(
    a: "RegionDataFrame",
    b: "RegionDataFrame | None" = None,
    *,
    min_dist: int | None = 0,
    same_strand: bool = False,
) -> pd.Series:
    """Label connected components among overlapping intervals.

    Parameters
    ----------
    a : RegionDataFrame
        The primary region set.  When *b* is None, clusters within *a*.
    b : RegionDataFrame or None
        If given, compute transitive clustering across both frames.
        The returned Series is aligned to *a* (labels for *b* are
        not returned; use the two-frame form when you need to know which
        A-intervals are transitively connected *through* B).
    min_dist : int or None
        Join distance.  ``min_dist=0`` joins book-ended intervals
        (``bedtools merge -d 0``).  ``min_dist=N`` joins when the gap is
        ``<= N``.  ``min_dist=None`` joins only genuinely overlapping
        intervals (book-ended stay separate).  Passed straight through to
        ``bioframe.cluster``.
    same_strand : bool
        If True, only connect same-strand intervals.

    Returns
    -------
    pd.Series[int], index-aligned to *a*.
    """
    _check_min_dist(min_dist)

    if b is not None:
        _assert_same_ref(a, b)

    df = a if b is None else pd.concat([
        a[list(_COLS) + (["strand"] if "strand" in a.columns else [])].reset_index(drop=True),
        b[list(_COLS) + (["strand"] if "strand" in b.columns else [])].reset_index(drop=True),
    ], ignore_index=True)

    labels = _cluster_df(df, min_dist, same_strand)

    n_a = len(a)
    return pd.Series(labels[:n_a], index=a.index, name="cluster")


def _check_min_dist(min_dist):
    """Reject a negative ``min_dist`` with OUR message, not bioframe's.

    bioframe already rejects it, so this is not a behaviour fix — it is an
    error-surface fix. Without it the caller sees
    ``min_dist>=0 currently required``, which is bioframe's wording: it leaks
    the backend through our API, contradicts the rule that backend arguments
    never appear in our signatures, reads inconsistently beside ``pad``'s
    message, and would change under us if bioframe reworded it.
    """
    if min_dist is not None and min_dist < 0:
        raise ValueError(
            f"min_dist must be >= 0 or None, got {min_dist}. "
            f"Use min_dist=None for strictly-overlapping-only."
        )


def _cluster_df(df, min_dist, same_strand):
    """Assign connected-component labels to rows of *df*.

    Delegates entirely to ``bioframe.cluster(min_dist=...)``.
    ``min_dist=None`` gives strictly-overlapping-only (book-ended stay
    separate); ``min_dist=0`` joins book-ended (the ``bedtools merge -d 0``
    convention).

    **The per-strand split is deliberate and must not be replaced by
    bioframe's ``on=['strand']``.** That parameter is an equality join, so it
    treats ``"."`` as equal to ``"."`` and clusters strandless rows together;
    ``bedtools -s`` treats ``"."`` as *no strand* and never calls two
    strandless features same-stranded. Measured across all nine
    ``(A strand, B strand)`` combinations, those two agree on eight and
    diverge on exactly this one — and strandless is the DEFAULT path in this
    codebase, since ``Region(strand=".")`` normalises to ``None``. Delegating
    here would invert the result on the common case with no error, which is
    why the loop below iterates only ``+`` and ``-`` and leaves every
    unstranded row in a cluster of its own.
    """
    import bioframe

    if same_strand:
        labels = np.full(len(df), -1, dtype=int)
        offset = 0
        for strand_val in ("+", "-"):
            mask = (df["strand"] == strand_val).values
            if not mask.any():
                continue
            sub = df[mask].copy()
            result = bioframe.cluster(
                sub.reset_index(drop=True), min_dist=min_dist, cols=_COLS,
            )
            sub_labels = result["cluster"].values
            labels[mask] = sub_labels + offset
            offset += sub_labels.max() + 1 if len(sub_labels) > 0 else 0
        # Unstranded rows each get their own cluster.
        unstranded = labels == -1
        if unstranded.any():
            labels[unstranded] = np.arange(offset, offset + unstranded.sum())
        return labels
    else:
        result = bioframe.cluster(
            df.reset_index(drop=True), min_dist=min_dist, cols=_COLS,
        )
        return result["cluster"].values


# ── merge ────────────────────────────────────────────────────────────

def merge(
    a: "RegionDataFrame",
    *,
    min_dist: int | None = 0,
    same_strand: bool = False,
) -> "RegionDataFrame":
    """Merge overlapping (or within-distance) intervals.

    Parameters
    ----------
    a : RegionDataFrame
    min_dist : int or None
        Join distance.  ``min_dist=0`` merges book-ended intervals
        (``bedtools merge -d 0``).  ``min_dist=N`` merges when the gap is
        ``<= N``.  ``min_dist=None`` merges only genuinely overlapping
        intervals (book-ended do **not** merge).  Passed straight through
        to ``bioframe.cluster``.
    same_strand : bool
        If True, only merge intervals on the same strand.

    Returns
    -------
    A new ``RegionDataFrame`` with merged intervals, sorted by contig/start.
    """
    from fragmentomics_tools.dataframe import RegionDataFrame

    _check_min_dist(min_dist)

    labels = _cluster_df(
        a[list(_COLS) + (["strand"] if "strand" in a.columns else [])],
        min_dist,
        same_strand,
    )

    df = a[list(_COLS)].copy()
    if same_strand and "strand" in a.columns:
        df["strand"] = a["strand"].values
    df["_cluster"] = labels

    group_cols = ["_cluster"]
    agg = {"contig": "first", "start": "min", "stop": "max"}
    if same_strand and "strand" in df.columns:
        agg["strand"] = "first"

    result = df.groupby(group_cols, sort=False).agg(agg).reset_index(drop=True)
    result = result.sort_values(["contig", "start", "stop"]).reset_index(drop=True)

    return RegionDataFrame(result, ref=a.ref)
