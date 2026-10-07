"""Geometry operations for region frames.

Free functions for resizing, expanding, truncating, and binning regions.
Layer 2 alongside ``intervals.py``.  Takes and returns frames; does **not**
import ``dataframe.py``.

See ``docs/pending/dataframe_layering_design.md``, Phase 3.
"""

from __future__ import annotations

import math
import logging
from typing import TYPE_CHECKING, Union, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from fragmentomics_tools.contig import CONTIG_LENGTHS

if TYPE_CHECKING:
    from fragmentomics_tools.dataframe import RegionDataFrame

logger = logging.getLogger(__name__)


# ── windowed_range ──────────────────────────────────────────────────

def windowed_range(start, stop, window_size):
    """
    >>> list(windowed_range(0, 5, 2))
    [(0, 2), (2, 4), (4, 5)]
    >>> list(windowed_range(0, 1, 2))
    [(0, 1)]
    >>> list(windowed_range(0, 11, 3))
    [(0, 3), (3, 6), (6, 9), (9, 11)]
    >>> list(windowed_range(-3, 3, 3))
    [(-3, 0), (0, 3)]
    """
    if window_size <= 0:
        raise ValueError("invalid window size")
    if stop <= start:
        raise ValueError("invalid start/stop")

    for start in range(start, stop, window_size):
        yield start, min(stop, start + window_size)


# ── region_lengths ──────────────────────────────────────────────────

def region_lengths(rdf: "RegionDataFrame") -> pd.Series:
    return rdf.stop - rdf.start


# ── center_on_summit ────────────────────────────────────────────────

def center_on_summit(rdf: "RegionDataFrame", inplace: bool = False):
    if "summit" not in rdf.columns:
        raise TypeError(
            "Must contain a 'summit' column to center on the summit."
        )

    rdf = rdf if inplace else rdf.copy()
    lengths = (rdf.stop - rdf.start).copy()

    if ((rdf.summit >= rdf.start) & (rdf.summit < rdf.stop)).all():
        rdf["start"] = rdf.summit - lengths // 2
    elif (rdf.summit <= region_lengths(rdf)).all():
        rdf["start"] = rdf.start + rdf.summit - lengths // 2
    else:
        raise ValueError(
            "summits must either be within the region interval or less "
            "than the length of the region."
        )

    rdf["stop"] = rdf.start + lengths
    return rdf.drop(columns=["summit"])


# ── boundary validation (private) ──────────────────────────────────

def _error_on_invalid_new_starts(new_start):
    from fragmentomics_tools.region import OutOfBoundsError

    if (new_start < 0).any():
        raise OutOfBoundsError(
            f"There is not enough flanking sequence to modify this region"
            f"(would result in a start coordinate of "
            f"'{new_start.min()} at idx {new_start.argmin()}')"
        )


def _error_on_invalid_new_stops(rdf, new_stop):
    from fragmentomics_tools.region import OutOfBoundsError

    if rdf.ref == "NA":
        return

    valid_contig_set = set(CONTIG_LENGTHS[rdf.ref].keys())
    for contig in sorted(set(rdf.contig)):
        new_stops_for_contig = new_stop[rdf.contig == contig]
        if (
            contig in valid_contig_set
            and new_stops_for_contig.max() > CONTIG_LENGTHS[rdf.ref][contig]
        ):
            raise OutOfBoundsError(
                f"There is not enough flanking sequence to modify a region "
                f"(would result in a stop coordinate of "
                f"'{new_stops_for_contig.max()}' at "
                f"idx {new_stops_for_contig.argmax()} but the "
                f"chrom length is '{CONTIG_LENGTHS[rdf.ref][contig]}')"
            )


def _valid_regions_mask(rdf, new_start, new_stop, discard_buffer_bp=0):
    ok = (new_start - discard_buffer_bp) >= 0
    for contig in sorted(set(rdf.contig)):
        max_len = CONTIG_LENGTHS[rdf.ref][contig]
        contig_good = (new_stop + discard_buffer_bp) <= max_len
        contig_good |= rdf.contig != contig
        ok &= contig_good

    return ok


# ── _resize_region_boundaries ──────────────────────────────────────

def _resize_region_boundaries(
    rdf: "RegionDataFrame",
    left: int = 0,
    right: int = 0,
    inplace: bool = False,
    strand_aware: bool = False,
    discard_invalid_resizes: bool = False,
):
    """Resize region boundaries by ``left``/``right``.

    ``inplace`` is ignored when ``discard_invalid_resizes=True``.
    """
    if strand_aware:
        neg_mask = rdf.strand == "-"

    new_starts = rdf.start + left
    if strand_aware:
        new_starts[neg_mask] = rdf.loc[neg_mask, "start"] - right

    new_stops = rdf.stop + right
    if strand_aware:
        new_stops[neg_mask] = rdf.loc[neg_mask, "stop"] - left

    if discard_invalid_resizes:
        valid_mask = _valid_regions_mask(rdf, new_starts, new_stops)
        result = rdf.loc[valid_mask, :].copy()
        result["start"] = new_starts[valid_mask]
        result["stop"] = new_stops[valid_mask]
    else:
        _error_on_invalid_new_starts(new_starts)
        _error_on_invalid_new_stops(rdf, new_stops)
        if inplace:
            result = rdf
        else:
            result = rdf.copy()
        result["start"] = new_starts
        result["stop"] = new_stops

    return result


# ── expand_regions ─────────────────────────────────────────────────

def check_nonneg_resize_amounts(left_amt, right_amt) -> None:
    """Shared precondition for expand/truncate.

    Lives here, and is called from BOTH this module's free functions and
    `RegionDataFrame`'s methods, because the two cannot delegate to each
    other: the methods must call `self._resize_region_boundaries` to keep
    `SampleAndRegionDataFrame`'s override in the path, while these functions
    call the module-level one. Only the *dispatch* has to be duplicated — the
    *validation* does not, and two copies of one rule is what CLAUDE.md's
    first rule exists to prevent.
    """
    assert (np.array(left_amt) >= 0).all()
    assert (np.array(right_amt) >= 0).all()


def check_truncation_fits(rdf: "RegionDataFrame", left_amt, right_amt) -> None:
    """Refuse a truncation that would consume a whole region.

    Shared for the same reason as `check_nonneg_resize_amounts`. Keeping the
    message in one place matters more here: it is prose, and prose edited in
    one of two copies drifts without anything failing.
    """
    total_truncation = np.array(left_amt) + np.array(right_amt)
    if (total_truncation >= region_lengths(rdf)).any():
        raise ValueError(
            "truncation amounts exceed region length for at least one region"
        )


def expand_regions(
    rdf: "RegionDataFrame",
    /,
    left_amt: int = 0,
    right_amt: int = 0,
    inplace: bool = False,
    strand_aware: bool = False,
    discard_invalid_resizes: bool = False,
):
    """Grow each region by `left_amt` and `right_amt`.

    **Same dispatch hazard as `truncate_regions`** — calling this directly on
    a `SampleAndRegionDataFrame` bypasses the SRDF override and so does not
    touch attached fragment arrays. In SRDF's case the method additionally
    *refuses* outright when arrays are attached (widening cannot be served
    from an in-memory array; it needs the h5 again), so going through this
    function would silently produce what the method deliberately rejects.
    """
    check_nonneg_resize_amounts(left_amt, right_amt)
    return _resize_region_boundaries(
        rdf, -left_amt, right_amt, inplace, strand_aware,
        discard_invalid_resizes,
    )


# ── truncate_regions ───────────────────────────────────────────────

def truncate_regions(
    rdf: "RegionDataFrame",
    /,
    left_amt: int = 0,
    right_amt: int = 0,
    inplace: bool = False,
    strand_aware: bool = False,
    discard_invalid_resizes: bool = False,
):
    """Shrink each region by `left_amt` and `right_amt`.

    **Calling this directly on a `SampleAndRegionDataFrame` SILENTLY SKIPS
    fragment-array resizing.** SRDF overrides `_resize_region_boundaries` so
    that attached arrays follow the geometry, and that override is only
    reached through the *method* — `srdf.truncate_regions(...)`. This free
    function calls the module-level `_resize_region_boundaries` instead, so
    the arrays keep their old extent while the region shrinks, with no error.
    Use the method on anything that might carry fragment arrays.
    `test_bypass_leaves_fragment_arrays_stale` pins this difference.
    """
    check_nonneg_resize_amounts(left_amt, right_amt)
    check_truncation_fits(rdf, left_amt, right_amt)
    return _resize_region_boundaries(
        rdf, left_amt, -right_amt, inplace, strand_aware,
        discard_invalid_resizes,
    )


# ── resize_regions ─────────────────────────────────────────────────

def resize_regions(
    rdf: "RegionDataFrame",
    new_size: Union[int, Sequence[int]],
    inplace: bool = False,
    discard_invalid_resizes: bool = False,
    discard_buffer_bp: int = 0,
):
    from fragmentomics_tools.region import Region

    if not inplace:
        rdf = rdf.copy()

    sizes = rdf.stop - rdf.start
    new_start = Region.get_resize_starts(rdf.start, sizes, new_size, rdf.strand)
    new_stop = new_start + new_size

    if discard_invalid_resizes:
        ok = (new_start - discard_buffer_bp) >= 0
        filter = None
        for contig in sorted(set(rdf.contig)):
            max_len = CONTIG_LENGTHS[rdf.ref][contig]
            contig_bad = ((new_stop + discard_buffer_bp) > max_len) & (
                rdf.contig == contig
            )
            if filter is None:
                filter = contig_bad
            else:
                filter = filter | contig_bad
        ok = ok & ~filter

        rdf = rdf.loc[ok, :]
        new_start = new_start[ok]
        new_stop = new_stop[ok]
        n_discarded = np.sum(~ok)
        if n_discarded > 0:
            logger.warning(
                f"Discarded {n_discarded} of {len(ok)} regions due to "
                f"invalid resize."
            )

    _error_on_invalid_new_starts(new_start)
    _error_on_invalid_new_stops(rdf, new_stop)
    rdf["start"] = new_start
    rdf["stop"] = new_stop
    return rdf


# ── bin_regions_into_windows ───────────────────────────────────────

def bin_regions_into_windows(
    rdf: "RegionDataFrame",
    window_size: int,
    mode: str,
    stride: int = None,
):
    """Tile windows across each region.

    :param window_size: the size of the window
    :param mode: 'full', 'valid', or 'exact'
    :param stride: stride length. ``window_size % stride`` must be 0.
    """
    from fragmentomics_tools.region import Region

    assert mode in ["full", "valid", "exact"]
    if stride is None:
        stride = window_size
    else:
        if window_size % stride != 0:
            raise ValueError(
                f"window size ({window_size}) must be evenly divisble "
                f"by stride ({stride})"
            )

    def _resize(region):
        if mode == "full":
            return region.resize(
                int(stride * math.ceil(region.length / stride))
            )
        elif mode == "valid":
            if region.length < window_size:
                raise ValueError(
                    f"region {region.chrom}:{region.start}-{region.stop} "
                    f"(length {region.length}) is shorter than window_size "
                    f"({window_size}); no valid windows can be produced"
                )
            n_windows = (region.length - window_size) // stride + 1
            extent = (n_windows - 1) * stride + window_size
            offset = (region.length - extent) // 2
            tiled_start = region.start + offset
            return Region(
                region.chrom,
                tiled_start,
                tiled_start + n_windows * stride,
                region.strand,
                region.ref,
                region.data,
            )
        elif mode == "exact":
            assert window_size % stride == 0
            if region.length % stride != 0:
                raise ValueError(
                    f"region length ({region.length}) must be evenly "
                    f"divisible by stride ({stride}) in 'exact' mode."
                )
            return region
        else:
            assert False, "UNREACHABLE"

    index_name = rdf.index.name
    rdf_copy = rdf.reset_index()
    all_windows = []
    for _, row in tqdm(
        rdf_copy.iterrows(), total=rdf_copy.shape[0], disable=False
    ):
        region = Region(row.contig, row.start, row.stop)
        region = _resize(region)
        all_windows.extend(
            (row.name, x[0], x[1])
            for x in windowed_range(region.start, region.stop, stride)
        )

    window_df = pd.DataFrame(
        all_windows, columns=["index", "new_start", "new_stop"]
    ).set_index("index")
    window_df["new_stop"] = window_df["new_stop"] + window_size - stride

    rv = (
        rdf_copy.join(window_df)
        .rename(
            columns=dict(
                new_start="start",
                new_stop="stop",
                start="old_start",
                stop="old_stop",
            )
        )
        .drop(columns=["old_start", "old_stop"])
        .set_index("index")
    )
    rv.index.rename(index_name, inplace=True)
    return rv
