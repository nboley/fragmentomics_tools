"""Canonical track constants for the background model.

This module is the single source of truth for the track layout used by the
background model's data pipeline and training code.  It imports nothing
heavier than the standard library, so it is safe to use from config.py and
other lightweight modules that must avoid pulling in torch/lightning.

TRACK_INDEX ordering is baked into the on-disk zarr store layout AND into
config_hash.  Changing the nesting order of the construction loops, or the
contents/order of STRANDS, FL_BANDS, or COVERAGE_TYPES, would silently
mis-index every existing store.  Do not change these without a migration.
"""

from typing import Dict, Tuple

# The three axes that define a track.
STRANDS: Tuple[str, ...] = ("+", "-")
FL_BANDS: Tuple[Tuple[int, int], ...] = ((40, 65), (120, 175))
COVERAGE_TYPES: Tuple[str, ...] = ("first", "last", "midpoint")

# Canonical mapping: (strand, fl_band, coverage_type) -> column index.
# Loop order is strands (outer) -> fl_bands -> coverage_types (inner).
TRACK_INDEX: Dict[Tuple[str, Tuple[int, int], str], int] = {}
_idx = 0
for _s in STRANDS:
    for _fl in FL_BANDS:
        for _c in COVERAGE_TYPES:
            TRACK_INDEX[(_s, _fl, _c)] = _idx
            _idx += 1

N_TRACKS: int = _idx  # 12 for the default layout
