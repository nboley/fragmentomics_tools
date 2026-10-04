"""Simulator package — cfDNA fragment generation with exact weight normalisation.

This package implements the generative model described in
docs/pending/simulator_and_fragment_nll.md.  The weight builder
(``build_region_weights``) is the single shared implementation used by both
the sampler and the oracle scorer; keeping one implementation prevents the
silent sampler/scorer drift that would invalidate every ``% captured`` number.
"""

from background_model.simulator.weights import (
    HexamerTables,
    build_predict_lut,
    build_region_weights,
    gc_bin_index,
    generative_domain_size,
    midpoint_index_arrays,
)

__all__ = [
    "HexamerTables",
    "build_predict_lut",
    "build_region_weights",
    "gc_bin_index",
    "generative_domain_size",
    "midpoint_index_arrays",
]

# Step 1-2 (capture surface + marginal FL) and Step 3 (precompute) are
# available via their submodules:
#   from background_model.simulator.capture import fit_and_build
#   from background_model.simulator.precompute import precompute_region
# Step 5 (sampler) and Step 6 (emit) are available via:
#   from background_model.simulator.sampler import draw_fragments_for_region
#   from background_model.simulator.emit import write_manifest, load_manifest
