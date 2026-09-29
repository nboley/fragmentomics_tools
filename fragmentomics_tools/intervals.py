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

    a_strand = a["strand"].values[idx["a_index"].values]
    b_strand = b["strand"].values[idx["b_index"].values]

    stranded_a = (a_strand == "+") | (a_strand == "-")
    stranded_b = (b_strand == "+") | (b_strand == "-")
    return pd.Series(
        stranded_a & stranded_b & (a_strand == b_strand),
        index=idx.index,
    )


def _wiggle_to_min_dist(wiggle: int) -> int:
    """Map our wiggle semantics to bioframe's min_dist.

    Our wiggle: gap <= wiggle counts as adjacent.  wiggle=0 means plain
    overlap only (book-ended do NOT match).

    bioframe's min_dist: gap <= min_dist counts as adjacent.  min_dist=0
    merges book-ended intervals.  min_dist >= 0 required.

    For wiggle >= 1 the mapping is direct (min_dist = wiggle).
    For wiggle == 0, bioframe cannot express "strictly overlapping only",
    so callers must handle that case separately.
    """
    if wiggle < 0:
        raise ValueError(f"wiggle must be >= 0, got {wiggle}")
    return wiggle


# ── overlap_indices ──────────────────────────────────────────────────

def overlap_indices(
    a: "RegionDataFrame",
    b: "RegionDataFrame",
    *,
    how: str = "inner",
    wiggle: int = 0,
    min_frac_a: float = 0.0,
    min_frac_b: float = 0.0,
    reciprocal: bool = False,
    same_strand: bool = False,
) -> pd.DataFrame:
    """Return ``(a_index, b_index, overlap_bases)`` for overlapping pairs.

    Parameters
    ----------
    a, b : RegionDataFrame
        The two region sets.  Must share the same ``ref``.
    how : {"inner", "left", "right", "outer", "anti"}
        Join type.  ``"anti"`` returns A rows with **no** match in B.
    wiggle : int
        Maximum edge-to-edge gap that still counts as overlap.
        ``wiggle=0`` is plain overlap; book-ended intervals do **not** match.
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
    DataFrame with columns ``a_index``, ``b_index``, ``overlap_bases``.
    For ``how="anti"``, ``b_index`` is always ``pd.NA`` and
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
    if reciprocal and min_frac_a <= 0:
        raise ValueError(
            "reciprocal=True requires min_frac_a > 0; it applies min_frac_a "
            "to both A and B (bedtools -r)"
        )

    _assert_same_ref(a, b)

    # Wiggle: expand B intervals so a gap <= wiggle still produces an overlap.
    # We keep the original B lengths for overlap_bases computation.
    # Expansion must be (wiggle + 1) because half-open overlap requires
    # strict inequality (start < end), so expanding by exactly `wiggle`
    # leaves a gap of `wiggle` as book-ended (no overlap).
    # Held as label-indexed Series, NOT bare arrays. `bioframe` returns the
    # input frames' index LABELS in `index`/`index_`, so any lookup here must
    # align by label. Using `.values[labels]` indexes positionally and is wrong
    # the moment a frame has a non-default index -- which is what every filter
    # or slice produces. It raised IndexError on a frame indexed 10/20.
    b_orig_start = b["start"].copy()
    b_orig_stop = b["stop"].copy()
    if wiggle > 0:
        b = b.copy()
        b["start"] = b["start"] - (wiggle + 1)
        b["stop"] = b["stop"] + (wiggle + 1)

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
        "a_index": result["index"],
        "b_index": result["index_"],
    })

    # Compute overlap_bases from the overlap coordinates bioframe returns.
    has_overlap = idx["b_index"].notna()
    overlap_bases = pd.array([0] * len(result), dtype="Int64")
    if has_overlap.any():
        if wiggle > 0:
            # Compute actual overlap against original (un-expanded) B intervals.
            a_lbl = idx.loc[has_overlap, "a_index"].values
            b_lbl = idx.loc[has_overlap, "b_index"].values
            a_starts = a["start"].loc[a_lbl].values
            a_stops = a["stop"].loc[a_lbl].values
            b_starts = b_orig_start.loc[b_lbl].values
            b_stops = b_orig_stop.loc[b_lbl].values
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
            idx.loc[fail_mask, "b_index"] = pd.NA
            idx.loc[fail_mask, "overlap_bases"] = 0
        has_overlap = idx["b_index"].notna()

    # Fraction filters (applied on matched rows only).
    if (min_frac_a > 0 or min_frac_b > 0) and has_overlap.any():
        matched = idx[has_overlap]
        # Label-aligned, not positional -- see the note on b_orig_start.
        a_lens = (a["stop"] - a["start"]).loc[matched["a_index"].values].values
        b_lens = (b_orig_stop - b_orig_start).loc[matched["b_index"].values].values
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
            idx.loc[fail, "b_index"] = pd.NA
            idx.loc[fail, "overlap_bases"] = 0
        has_overlap = idx["b_index"].notna()

    # Anti: keep only A rows with no match in B.
    if how == "anti":
        no_match = ~has_overlap
        idx = idx[no_match].copy()
        idx["b_index"] = pd.NA
        idx["overlap_bases"] = 0

    idx = idx.reset_index(drop=True)
    return idx


# ── overlaps ─────────────────────────────────────────────────────────

def overlaps(
    a: "RegionDataFrame",
    b: "RegionDataFrame",
    *,
    wiggle: int = 0,
    same_strand: bool = False,
) -> pd.Series:
    """Boolean mask: which rows of *a* overlap at least one row in *b*.

    Equivalent to ``a.index.isin(overlap_indices(a, b, ...).a_index)``
    but exists because it is the most-used operation in the family.

    Parameters
    ----------
    a, b : RegionDataFrame
    wiggle : int
        Maximum edge-to-edge gap that still counts as overlap.
    same_strand : bool
        If True, only match when both strands are in {"+", "-"} and equal.

    Returns
    -------
    pd.Series[bool], index-aligned to *a*.
    """
    idx = overlap_indices(a, b, how="inner", wiggle=wiggle, same_strand=same_strand)
    matched = set(idx["a_index"].dropna().values)
    return pd.Series(
        [i in matched for i in range(len(a))],
        index=a.index,
        dtype=bool,
    )


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
    DataFrame with columns ``a_index``, ``b_index``, ``distance``.
    """
    import bioframe

    _assert_same_ref(a, b)

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
        "a_index": result["index"],
        "b_index": result["index_"],
        "distance": result["distance"],
    })

    if same_strand:
        has_match = idx["b_index"].notna()
        if has_match.any():
            strand_ok = _strand_mask(a, b, idx[has_match], same_strand=True)
            fail = has_match & ~strand_ok.reindex(idx.index, fill_value=True)
            idx = idx[~fail]

    return idx.reset_index(drop=True)


# ── cluster ──────────────────────────────────────────────────────────

def cluster(
    a: "RegionDataFrame",
    b: "RegionDataFrame | None" = None,
    *,
    wiggle: int = 0,
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
    wiggle : int
        Maximum edge-to-edge gap that still counts as connection.
    same_strand : bool
        If True, only connect same-strand intervals.

    Returns
    -------
    pd.Series[int], index-aligned to *a*.
    """
    import bioframe

    if b is not None:
        _assert_same_ref(a, b)

    df = a if b is None else pd.concat([
        a[list(_COLS) + (["strand"] if "strand" in a.columns else [])].reset_index(drop=True),
        b[list(_COLS) + (["strand"] if "strand" in b.columns else [])].reset_index(drop=True),
    ], ignore_index=True)

    labels = _cluster_df(df, wiggle, same_strand)

    n_a = len(a)
    return pd.Series(labels[:n_a], index=a.index, name="cluster")


def _cluster_df(df, wiggle, same_strand):
    """Assign connected-component labels to rows of *df*.

    For wiggle >= 1 delegates to bioframe (min_dist = wiggle).
    For wiggle == 0, bioframe's min_dist=0 incorrectly clusters book-ended
    intervals, so we implement the sweep ourselves.
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
            sub_labels = _cluster_one_group(sub, wiggle)
            labels[mask] = sub_labels + offset
            offset += sub_labels.max() + 1 if len(sub_labels) > 0 else 0
        # Unstranded rows each get their own cluster.
        unstranded = labels == -1
        if unstranded.any():
            labels[unstranded] = np.arange(offset, offset + unstranded.sum())
        return labels
    else:
        return _cluster_one_group(df, wiggle)


def _cluster_one_group(df, wiggle):
    """Cluster a single strand-group (or all-strand) df.

    Uses bioframe for wiggle >= 1 (where min_dist = wiggle gives the right
    semantics: gap <= wiggle clusters together).  For wiggle == 0 bioframe's
    min_dist=0 incorrectly clusters book-ended intervals, so we sweep.
    """
    import bioframe

    if wiggle >= 1:
        result = bioframe.cluster(df.reset_index(drop=True), min_dist=wiggle, cols=_COLS)
        return result["cluster"].values
    else:
        # wiggle=0: only truly overlapping intervals cluster (book-ended do not).
        # Sort by contig, start, stop and sweep.
        # Reset index so positional and label indexing agree.
        df = df.reset_index(drop=True)
        sort_order = df.sort_values(list(_COLS)).index.values
        contigs = df["contig"].values
        starts = df["start"].values
        stops = df["stop"].values

        labels = np.empty(len(df), dtype=int)
        cluster_id = 0

        if len(sort_order) == 0:
            return labels

        # Sweep: maintain current cluster's extent.
        cur_contig = contigs[sort_order[0]]
        cur_stop = stops[sort_order[0]]
        labels[sort_order[0]] = cluster_id

        for i in range(1, len(sort_order)):
            ix = sort_order[i]
            c = contigs[ix]
            s = starts[ix]
            e = stops[ix]
            if c == cur_contig and s < cur_stop:
                # Overlaps current cluster (strict <, so book-ended excluded).
                labels[ix] = cluster_id
                if e > cur_stop:
                    cur_stop = e
            else:
                cluster_id += 1
                labels[ix] = cluster_id
                cur_contig = c
                cur_stop = e

        return labels


# ── merge ────────────────────────────────────────────────────────────

def merge(
    a: "RegionDataFrame",
    *,
    wiggle: int = 0,
    same_strand: bool = False,
) -> "RegionDataFrame":
    """Merge overlapping (or within-wiggle) intervals.

    Parameters
    ----------
    a : RegionDataFrame
    wiggle : int
        Maximum edge-to-edge gap that counts as adjacent.
        ``wiggle=0`` means only truly overlapping intervals merge
        (book-ended do **not** merge).
    same_strand : bool
        If True, only merge intervals on the same strand.

    Returns
    -------
    A new ``RegionDataFrame`` with merged intervals, sorted by contig/start.
    """
    from fragmentomics_tools.dataframe import RegionDataFrame

    labels = _cluster_df(
        a[list(_COLS) + (["strand"] if "strand" in a.columns else [])],
        wiggle,
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
