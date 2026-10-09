"""Simulator package — cfDNA fragment generation.

**The live simulator's authority is ``docs/pending/simulator_spec.md``.**
Start there.

It is three modules, split along their dependency layers (owner decision 169),
each importing only from layers above it (``draw`` imports ``hexamers``
directly as well as ``cut_site_stats``):

1. ``background_model/hexamers.py`` — hexamer encoding, numpy only.
2. ``background_model/cut_site_stats.py`` — measurement on real data:
   ``C(h)``, ``N(h)``, ``f(L)`` and ``r(h)``.
3. ``background_model/simulator/draw.py`` — the draw, in this package.

The driver is ``scripts/run_cut_site_simulator.py``.

The previous generation (``capture``, ``emit``, ``precompute``, ``sampler``,
``weights``) is in ``attic/pre_rewrite_simulator/``. It was written around
capture / GC modelling and a ``predict(L, gc)`` surface, both dropped by owner
decision, and it admits fragments by **midpoint** where the live simulator
admits by **start-in-region**. Nothing is re-exported from here: two agents
have already reached confident wrong conclusions from stale simulator code
sitting in live directories.
"""
