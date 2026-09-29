"""Simulator package — cfDNA fragment generation with exact weight normalisation.

This package implements the generative model described in
docs/pending/simulator_and_fragment_nll.md.  The weight builder
(``build_region_weights``) is the single shared implementation used by both
the sampler and the oracle scorer; keeping one implementation prevents the
silent sampler/scorer drift that would invalidate every ``% captured`` number.
"""

from background_model.simulator.weights import HexamerTables, build_region_weights

__all__ = ["HexamerTables", "build_region_weights"]
