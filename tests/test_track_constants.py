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


# Golden mapping — the exact TRACK_INDEX under the current band layout
# (owner decision 14, 2026-09-27: bands ((25,110),(110,180))).
GOLDEN_TRACK_INDEX = {
    ("+", (25, 110), "first"): 0,
    ("+", (25, 110), "last"): 1,
    ("+", (25, 110), "midpoint"): 2,
    ("+", (110, 180), "first"): 3,
    ("+", (110, 180), "last"): 4,
    ("+", (110, 180), "midpoint"): 5,
    ("-", (25, 110), "first"): 6,
    ("-", (25, 110), "last"): 7,
    ("-", (25, 110), "midpoint"): 8,
    ("-", (110, 180), "first"): 9,
    ("-", (110, 180), "last"): 10,
    ("-", (110, 180), "midpoint"): 11,
}


def test_track_index_matches_golden():
    """TRACK_INDEX must match the golden mapping exactly."""
    assert TRACK_INDEX == GOLDEN_TRACK_INDEX


def test_n_tracks():
    assert N_TRACKS == 12


def test_strands():
    assert STRANDS == ("+", "-")


def test_fl_bands():
    assert FL_BANDS == ((25, 110), (110, 180))


def test_coverage_types():
    assert COVERAGE_TYPES == ("first", "last", "midpoint")


def test_config_hash_unchanged():
    """PlumbingConfig.config_hash is pinned to the current band layout.

    fl_bands IS part of the config hash, so this value tracks the band layout:
    it was updated when the bands changed to ((25,110),(110,180)) (owner
    decision 14, 2026-09-27).  A drift NOT explained by an intentional layout
    change is a finding.
    """
    from background_model.band_model.config import PlumbingConfig

    cfg = PlumbingConfig(sample_sheet="dummy.tsv")
    h = cfg.config_hash(with_content_hashes=False)
    assert h == "9e56ac6b0c6127768ae591cbde7582a69b2dfc3b8b55a6c60059e08bb484f38b"


def test_all_consumers_use_canonical_tracks():
    """Every module that exposes TRACK_INDEX or FL_BANDS must re-export the
    canonical instances from background_model.tracks, not independent copies."""
    from background_model.band_model.preprocess import FL_BANDS as p_fb
    from background_model.band_model.preprocess import TRACK_INDEX as p_ti

    assert p_ti is TRACK_INDEX, "preprocess.TRACK_INDEX is a copy, not the canonical"
    assert p_fb is FL_BANDS, "preprocess.FL_BANDS is a copy, not the canonical"

    from background_model_core import FL_BANDS as c_fb
    from background_model_core import STRANDS as c_s
    from background_model_core import COVERAGE_TYPES as c_ct

    assert c_s is STRANDS, "background_model_core.STRANDS is a copy"
    assert c_fb is FL_BANDS, "background_model_core.FL_BANDS is a copy"
    assert c_ct is COVERAGE_TYPES, "background_model_core.COVERAGE_TYPES is a copy"
