"""Simulator package — cfDNA fragment generation.

**The live simulator's authority is ``docs/pending/simulator_spec.md``.**
Start there.

It is four modules, split along their dependency layers (owner decisions 169
and 174-177), each importing only from layers above it (``draw`` imports
``hexamers`` directly as well as ``measure``):

0. ``background_model/constants.py`` — the shared cut-site definitions
   (length bounds from ``tracks.FL_BANDS``, k-mer geometry), stdlib only.
1. ``background_model/hexamers.py`` — hexamer encoding, numpy plus
   ``constants``.
2. ``background_model/simulator/measure.py`` — measurement on real data:
   ``C(h)``, ``N(h)``, ``f(L)`` and ``r(h)``.
3. ``background_model/simulator/draw.py`` — the draw.

``measure`` was ``background_model/cut_site_stats.py`` until owner decision
187 moved it into this package, since only simulator code imports it.
``constants`` and ``hexamers`` stay outside: neither needs the simulator's
dependencies, and ``constants`` is shared with the cut-site store/model.

The driver is ``scripts/run_cut_site_simulator.py``. ``scripts/measure_cut_site_hexamers.py``
runs only the measure step for one sample and writes ``C(h)``, ``N(h)``, ``r(h)`` and
``f(L)`` to disk (owner decision 189).

The previous generation (``capture``, ``emit``, ``precompute``, ``sampler``,
``weights``) is in ``attic/pre_rewrite_simulator/``. It was written around
capture / GC modelling and a ``predict(L, gc)`` surface, both dropped by owner
decision, and it admits fragments by **midpoint** where the live simulator
admits by **start-in-region**. Nothing is re-exported from here: two agents
have already reached confident wrong conclusions from stale simulator code
sitting in live directories.
"""
