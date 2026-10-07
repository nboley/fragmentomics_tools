"""Simulator package — cfDNA fragment generation.

**The live simulator is ``count_hexamers_rdf.py``, and its authority is
``docs/pending/simulator_spec.md``.** Start there.

Everything else in this package is the PREVIOUS generation, kept only because
~8 files still import it, some belonging to another work stream. Do not build on
it: it is written around capture / GC modelling and a ``predict(L, gc)``
surface, both dropped by owner decision, and it admits fragments by **midpoint**
where the rewrite admits by **start-in-region**.

This docstring used to name ``docs/pending/simulator_and_fragment_nll.md`` as
the design. That document says "midpoint" ten times and "start-in-region" zero
times, so the package's own entry point was pointing readers at the reversed
admission rule. Two separate agents have already reached confident wrong
conclusions from stale simulator files sitting in live directories; this was the
most direct route to doing it again. Superseded docs are in
``attic/pre_rewrite_simulator_docs/``.
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
