"""Tests that the track constants in background_model.tracks are stable.

TRACK_INDEX ordering is baked into every on-disk zarr store and into
config_hash.  These tests catch any accidental reordering or value change.
"""

import pytest

from background_model.tracks import (
    COVERAGE_TYPES,
    FL_BANDS,
    N_TRACKS,
    STRANDS,
    TRACK_INDEX,
)


# Golden mapping — the exact TRACK_INDEX that all existing stores use.
GOLDEN_TRACK_INDEX = {
    ("+", (40, 65), "first"): 0,
    ("+", (40, 65), "last"): 1,
    ("+", (40, 65), "midpoint"): 2,
    ("+", (120, 175), "first"): 3,
    ("+", (120, 175), "last"): 4,
    ("+", (120, 175), "midpoint"): 5,
    ("-", (40, 65), "first"): 6,
    ("-", (40, 65), "last"): 7,
    ("-", (40, 65), "midpoint"): 8,
    ("-", (120, 175), "first"): 9,
    ("-", (120, 175), "last"): 10,
    ("-", (120, 175), "midpoint"): 11,
}


def test_track_index_matches_golden():
    """TRACK_INDEX must match the golden mapping exactly."""
    assert TRACK_INDEX == GOLDEN_TRACK_INDEX


def test_n_tracks():
    assert N_TRACKS == 12


def test_strands():
    assert STRANDS == ("+", "-")


def test_fl_bands():
    assert FL_BANDS == ((40, 65), (120, 175))


def test_coverage_types():
    assert COVERAGE_TYPES == ("first", "last", "midpoint")


def test_config_hash_unchanged():
    """PlumbingConfig.config_hash must not change when track constants move."""
    from background_model.config import PlumbingConfig

    cfg = PlumbingConfig(sample_sheet="dummy.tsv")
    h = cfg.config_hash(with_content_hashes=False)
    assert h == "717951cdf810ff0ae602613ba12efff50edcd4e2cb4df95175b1ac4b63a63905"


def test_all_consumers_use_canonical_tracks():
    """Every module that exposes TRACK_INDEX or FL_BANDS must re-export the
    canonical instances from background_model.tracks, not independent copies."""
    from background_model.preprocess import FL_BANDS as p_fb
    from background_model.preprocess import TRACK_INDEX as p_ti

    assert p_ti is TRACK_INDEX, "preprocess.TRACK_INDEX is a copy, not the canonical"
    assert p_fb is FL_BANDS, "preprocess.FL_BANDS is a copy, not the canonical"

    from background_model_core import FL_BANDS as c_fb
    from background_model_core import STRANDS as c_s
    from background_model_core import COVERAGE_TYPES as c_ct

    assert c_s is STRANDS, "background_model_core.STRANDS is a copy"
    assert c_fb is FL_BANDS, "background_model_core.FL_BANDS is a copy"
    assert c_ct is COVERAGE_TYPES, "background_model_core.COVERAGE_TYPES is a copy"
