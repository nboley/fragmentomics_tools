from __future__ import annotations

import math
import itertools
from collections import defaultdict
from typing import TYPE_CHECKING, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from fragmentomics_tools.dataframe import DataFrameBase

offset = 2000


def cov_score(x):
    return np.sum(x)


def calc_tss_cov_bias(fa, condition, fl_frac):
    from fragmentomics_tools.fragment_array.fragment_array import (
        _switch_plus_with_minus_and_minus_with_plus,
        RegionFragmentArray,
    )
    _fa = fa.subset_fragment_lengths(*fl_frac).drop_duplicate_fragments()
    if condition == 'EMSeqMod':
        _fa = make_converted_fa(_fa)
    if condition in ['EMSeqStd']:
        _fa.fragment_strands = _switch_plus_with_minus_and_minus_with_plus(_fa.fragment_strands)
    tss_pos = _fa.length//2
    _fa_neg = _fa.subset_by_fragment_strand('-')
    post_neg_cov = cov_score(_fa_neg.get_midpoint_coverage_array()[tss_pos+offset:])
    pre_neg_cov = cov_score(_fa_neg.get_midpoint_coverage_array()[:tss_pos-offset])
    
    _fa_pos = _fa.subset_by_fragment_strand('+')
    post_pos_cov = cov_score(_fa_pos.get_midpoint_coverage_array()[tss_pos+offset:])
    pre_pos_cov = cov_score(_fa_pos.get_midpoint_coverage_array()[:tss_pos-offset])

    if isinstance(fa, RegionFragmentArray):
        if fa.strand == '-':
            post_neg_cov, post_pos_cov = post_pos_cov, post_neg_cov
        else:
            assert fa.strand == '+'

    return pd.Series(dict(anti_sense_cov=post_neg_cov, sense_cov=post_pos_cov))

    score = post_neg_cov/(post_neg_cov+post_pos_cov)
    return score


def calc_gene_cov_bias(
    fa,
    fl_frac: Tuple[int, int],
    offset: int = 0,
) -> pd.Series:
    """
    Compute strand-specific coverage bias for a single gene's fragment array.

    Parameters
    ----------
    fa
        FragmentArray-like object with:
          - .strand attribute ('+' or '-')
          - subset_fragment_lengths(min_fl, max_fl)
          - drop_duplicate_fragments()
          - subset_by_fragment_strand('+ or -')
          - get_midpoint_coverage_array()
    fl_frac : (int, int)
        Fragment length bounds (inclusive, exclusive) to keep, e.g. (40, 60).
    offset : int
        Number of bases to trim from the start of the midpoint coverage array
        before summarizing with `cov_score`. This is used to ignore coverage
        near the TSS or gene boundary.

    Returns
    -------
    pandas.Series
        Series with two entries:
          - 'anti_sense_cov'
          - 'sense_cov'

        Values are already corrected for the gene's strand, so that
        "anti_sense" is always the template strand and "sense" the coding
        strand, regardless of whether `fa.strand` is '+' or '-'.

    Notes
    -----
    This function relies on an external `cov_score` function, which should
    take a 1D coverage array and return a scalar (e.g. mean coverage in the
    region of interest).
    """
    # Filter by fragment length and deduplicate
    _fa = fa.subset_fragment_lengths(*fl_frac).drop_duplicate_fragments()

    # Negative-strand (rev / anti-sense) coverage
    _fa_neg = _fa.subset_by_fragment_strand('-')
    neg_cov_arr = _fa_neg.get_midpoint_coverage_array()[offset:]
    post_neg_cov = cov_score(neg_cov_arr)

    # Positive-strand (fwd / sense) coverage
    _fa_pos = _fa.subset_by_fragment_strand('+')
    pos_cov_arr = _fa_pos.get_midpoint_coverage_array()[offset:]
    post_pos_cov = cov_score(pos_cov_arr)

    # Re-orient so that anti_sense_cov always refers to the template strand,
    # independent of the gene's annotation strand.

    if hasattr(fa, 'strand'):
        if fa.strand == '-':
            # Swap so anti_sense is still template
            post_neg_cov, post_pos_cov = post_pos_cov, post_neg_cov
        else:
            assert fa.strand == '+', "Unexpected strand value on fragment array"

    return pd.Series(
        {
            "anti_sense_cov": post_neg_cov,
            "sense_cov": post_pos_cov,
        }
    )


def make_gene_score(
    srdf: DataFrameBase,
    label: str,
    fl_bnd: Tuple[int, int],
    exp_bnd: Tuple[float, float],
    offset: int,
) -> float:
    """
    Compute strand-bias score for one (label, fragment-length bin, expression bin).

    Parameters
    ----------
    srdf : DataFrameBase
        Long-format sample-by-gene dataframe with at least:
          - 'label' (protocol / condition)
          - 'expression' (expression value, same units as exp_bnd)
          - 'strand' (gene strand, '+' or '-')
          - 'fragment_array' (FragmentArray-like object per row)
    label : str
        Protocol / condition name (e.g. 'Double Stranded Protocol').
    fl_bnd : (int, int)
        Fragment length bin, e.g. (40, 60).
    exp_bnd : (float, float)
        Expression bin (lower, upper), e.g. (0.1, 1).
    offset : int
        Passed through to `calc_gene_cov_bias`.

    Returns
    -------
    float
        Strand bias score:
            (anti_sense_cov - sense_cov) / (anti_sense_cov + sense_cov)

        Aggregated across all genes in this (label, fl_bnd, exp_bnd) bin.
    """
    le, ue = exp_bnd

    # Subset to this label / expression window and plus-strand entries
    # (fa.strand encodes gene orientation; this avoids double-counting).
    sub_df = srdf.query(
        "expression >= @le and expression < @ue and label == @label and strand == '+'"
    )

    if sub_df.empty:
        return float("nan")

    # Sum anti- and sense coverage across all fragment arrays in this bin
    cov_sums = sub_df.fragment_array.apply(
        lambda fa: calc_gene_cov_bias(fa, fl_bnd, offset)
    ).sum(axis=0)

    anti = cov_sums["anti_sense_cov"]
    sense = cov_sums["sense_cov"]
    denom = sense + anti

    if denom == 0:
        return float("nan")

    score = (anti - sense) / denom
    return float(score)


def make_gene_body_exp_vs_bias_df(
    gene_body_srdf: DataFrameBase,
    label_col: str | None = "label",
    exp_col: str = "expression",
    fl_bnds=None,
    n_exp_bins: int = 5,
) -> pd.DataFrame:
    """
    Compute gene-body strand-bias scores across expression and fragment-length bins.

    Parameters
    ----------
    gene_body_srdf : DataFrameBase
        Dataframe with per-gene fragment arrays and metadata. Must contain:
          - 'label'          : protocol / condition name
          - 'expression'     : numeric expression value
          - 'strand'         : '+' or '-'
          - 'fragment_array' : FragmentArray-like object
    offset : int
        Position offset to apply when summarizing midpoint coverage
        (see `calc_gene_cov_bias`).
    exp_bnds : iterable of (float, float), optional
        Expression bins (lower, upper). By default:
        (0, 0.01), (0.01, 0.1), (0.1, 1), (1, 10), (10, 100), (100, 1000).

    Returns
    -------
    pandas.DataFrame
        A matrix of strand-bias scores indexed by log10(expression upper bound),
        with columns forming a MultiIndex (label, fl_bnd):

            index  -> log10(exp upper bound)
            cols   -> (label, (fl_min, fl_max))

        Values are:
            (anti_sense_cov - sense_cov) / (anti_sense_cov + sense_cov)

        summed across all genes in the given bin.
    """
    if fl_bnds is None:
        # Fragment-length bins: 20–90 bp (10-bp bins) + 90–120.
        fl_edges = list(range(20, 100, 10)) + [120]
        fl_bnds = list(zip(fl_edges[:-1], fl_edges[1:]))

    expr = gene_body_srdf[exp_col].astype(float)
    exp_bnds = np.quantile(expr, [0.] + list(np.linspace(0.5, 1, n_exp_bins)))
    exp_bnds = [(x, y) for x, y in zip(exp_bnds[:-1], exp_bnds[1:])]

    exp_bnds = (
        (0, 0.01),
        (0.01, 0.1),
        (0.1, 1),
        (1, 10),
        (10, 100),
        (100, 1000),
    )

    labels = gene_body_srdf.label.unique().tolist()

    # Build all (label, fl_bnd, exp_bnd) combinations
    from fragmentomics_tools.dataframe import DataFrameBase
    design = DataFrameBase(
        list(itertools.product(labels, fl_bnds, exp_bnds)),
        columns=["label", "fl_bnd", "exp_bnd"],
    )

    # Compute scores, in parallel if DataFrameBase.parallel_apply supports it
    scores = design.parallel_apply(
        lambda row: make_gene_score(
            gene_body_srdf,
            label=row[label_col],
            fl_bnd=row.fl_bnd,
            exp_bnd=row.exp_bnd,
            offset=offset,
        )
    )

    scores = scores[0].rename("score")
    scored_design = pd.concat([design, scores], axis=1)

    # Pivot into a 2D matrix: rows = expression bin, columns = (label, fl_bnd)
    bias_matrix = scored_design.pivot(
        index="exp_bnd",
        columns=[label_col, "fl_bnd"],
        values="score",
    )

    # Replace expression-bin tuples with log10 of the *upper* bound
    bias_matrix.index = [math.log10(ub+1e-300) for (_, ub) in bias_matrix.index.values]
    bias_matrix.index.name = "log10_exp_upper"

    return bias_matrix


def make_tss_exp_vs_bias_df(
    tss_srdf: pd.DataFrame,
    label_col: str | None = "label",
    exp_col: str = "expression",
    fl_bnds=None,
    n_exp_bins: int = 5,
):
    """
    Computes TSS strand bias scores across bins of gene expression and
    bins of fragment length.
    """

    # -----------------------
    # 1. Build adaptive expression bins
    # -----------------------
    expr = tss_srdf[exp_col].astype(float)
    if False:
        exp_bnds = np.quantile(expr, [0.] + list(np.linspace(0.5, 1, n_exp_bins)))
        exp_bnds = [(max(x, 1e-6), y) for x, y in zip(exp_bnds[:-1], exp_bnds[1:])]
    else:
        exp_bnds = list(np.quantile(expr, [0., 0.5]))
        exp_bnds.extend(np.linspace(exp_bnds[-1], np.quantile(expr, 0.90), 3))
        exp_bnds.append(expr.max())
        exp_bnds = list(zip(exp_bnds[:-1], exp_bnds[1:]))

    exp_bnds = (
        (0, 0.01),
        (0.01, 0.1),
        (0.1, 1),
        (1, 10),
        (10, 100),
        (100, 1000),
    )

    # -----------------------
    # 2. Fragment-length bins
    # -----------------------
    if fl_bnds is None:
        _ = list(range(20, 100, 10)) + [120]
        fl_bnds = list(zip(_[:-1], _[1:]))

    # -----------------------
    # 3. Labels: optional
    # -----------------------
    if label_col is None:
        label_values = [None]
    else:
        label_values = tss_srdf[label_col].unique()

    res = defaultdict(list)
    for label in label_values:
        if label_col is None:
            sub = tss_srdf
            label_key = "all"
        else:
            sub = tss_srdf[tss_srdf[label_col] == label]
            label_key = label

        if sub.empty:
            continue

        for le, ue in exp_bnds:
            mask = (
                (sub[exp_col] >= le) &
                (sub[exp_col] < ue) &
                (sub["strand"] == "+")
            )
            sub_bin = sub.loc[mask]

            if sub_bin.empty:
                for fl_bnd in fl_bnds:
                    res[(label_key, fl_bnd)].append(np.nan)
                continue

            from fragmentomics_tools.fragment_array import merge_fragment_arrays
            fa = (
                merge_fragment_arrays(sub_bin["fragment_array"])
                .drop_duplicate_fragments()
            )

            for fl_bnd in fl_bnds:
                sub_df = calc_tss_cov_bias(fa, label_key, fl_bnd)

                # Ensure we treat sense/antisense as scalars
                # (works whether sub_df is a Series or 1-row DataFrame)
                if isinstance(sub_df, pd.Series):
                    sense = float(sub_df["sense_cov"])
                    anti = float(sub_df["anti_sense_cov"])
                else:
                    sense = float(sub_df["sense_cov"].iloc[0])
                    anti = float(sub_df["anti_sense_cov"].iloc[0])

                denom = sense + anti
                if denom == 0:
                    score = np.nan
                else:
                    score = (anti - sense) / denom

                res[(label_key, fl_bnd)].append(score)

    df = pd.DataFrame(res)
    df.index = [np.log10(ue) for (_, ue) in exp_bnds]
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["label", "fl_bnd"])

    return df
