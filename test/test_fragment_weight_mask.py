"""The fragment-weight mask must select on length band AND strand.

Each `pred_dist.*` column in the record is a weight track for one
(strand, fl_band, coverage_type) combination. A fragment should receive that
track's weights only if it falls in the band *and* lies on the strand.

The mask previously combined the two conditions with OR, which admitted
fragments outside the band whenever the strand matched. Because the loop
overwrites `attr[mask]` once per combination, the weight a fragment ended up
with depended on iteration order.
"""

import numpy
import pandas as pd
import pytest

from fragmentomics_tools.fragment_array.fragment_array import FragmentArray
from fragmentomics_tools.dataframe import (
    _set_fragment_array_weights_from_weights_record,
)

REGION_LEN = 1000
BAND = (40, 65)

# in-band/+, in-band/-, out-of-band/+, out-of-band/-
STARTS = [100, 200, 300, 600]
STOPS = [150, 250, 500, 800]
STRANDS = ["+", "-", "+", "-"]


def _fragment_array():
    return FragmentArray(
        starts_0=STARTS,
        stops_0=STOPS,
        length=REGION_LEN,
        max_frag_len=511,
        fragment_strands=numpy.array(STRANDS, dtype="<U1"),
    )


def _record(track_name):
    # 1/2.0 -> every selected fragment gets weight 0.5, so "weighted" is
    # unambiguous and independent of position.
    return pd.Series({track_name: numpy.full(REGION_LEN, 2.0)})


def test_only_in_band_fragments_on_the_track_strand_are_weighted():
    """The single plus-strand 40-65 track must weight exactly one fragment.

    Fragment 0 is the only one both in the band and on '+'. Fragment 1 is in
    the band but on '-'; fragment 2 is on '+' but 200bp, far outside the band.
    Under the previous OR, both were also selected.
    """
    fa = _fragment_array()
    out = _set_fragment_array_weights_from_weights_record(
        fa, _record("pred_dist.strand_+__fl_40_65__coverage_midpoint"), 0
    )

    assert out.n_fragments == 1, (
        "expected only the in-band plus-strand fragment to survive; "
        f"got {out.n_fragments}"
    )
    # Fragment 0 is the 50bp fragment at 100.
    assert out.starts_0[0] == 100
    assert out.fragment_lengths[0] == 50


def test_out_of_band_same_strand_fragment_is_excluded():
    """A fragment on the track's strand but outside its band gets no weight.

    This is the exact case the OR admitted.
    """
    fa = _fragment_array()
    out = _set_fragment_array_weights_from_weights_record(
        fa, _record("pred_dist.strand_+__fl_40_65__coverage_midpoint"), 0
    )

    # the 200bp plus-strand fragment starts at 300
    assert 300 not in list(out.starts_0)


def test_in_band_other_strand_fragment_is_excluded():
    """A fragment in the track's band but on the other strand gets no weight."""
    fa = _fragment_array()
    out = _set_fragment_array_weights_from_weights_record(
        fa, _record("pred_dist.strand_+__fl_40_65__coverage_midpoint"), 0
    )

    # the in-band minus-strand fragment starts at 200
    assert 200 not in list(out.starts_0)


def test_minus_strand_track_selects_the_minus_strand_fragment():
    """The same rule holds for the minus strand -- not a plus-strand quirk."""
    fa = _fragment_array()
    out = _set_fragment_array_weights_from_weights_record(
        fa, _record("pred_dist.strand_-__fl_40_65__coverage_midpoint"), 0
    )

    assert out.n_fragments == 1
    assert out.starts_0[0] == 200  # the in-band minus-strand fragment


def test_strandless_track_selects_on_band_alone():
    """A '.' track skips the strand clause, so the band is the only filter.

    Both in-band fragments qualify regardless of their strand. This pins the
    behaviour of the branch the fix does not touch.
    """
    fa = _fragment_array()
    out = _set_fragment_array_weights_from_weights_record(
        fa, _record("pred_dist.strand_.__fl_40_65__coverage_midpoint"), 0
    )

    assert out.n_fragments == 2
    assert sorted(out.starts_0) == [100, 200]
