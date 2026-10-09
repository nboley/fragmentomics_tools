# attic/pre_rewrite_simulator — REFERENCE ONLY, NOT LIVE CODE

The previous-generation simulator, its scripts and its tests. Retired
2026-10-09 by owner decision 171. **Do not import from here, and do not treat
anything here as describing current behaviour.** Nothing here runs: these
files import `background_model.simulator.{weights,precompute,...}`, and those
paths no longer exist. They were moved rather than deleted so the design stays
readable next to its successor.

The live simulator is `background_model/hexamers.py`,
`background_model/cut_site_stats.py` and `background_model/simulator/draw.py`.
Its authority is `docs/pending/simulator_spec.md`. The design documents for
this generation are in `attic/pre_rewrite_simulator_docs/`.

## Why it was retired

It was written around capture / GC modelling and a `predict(L, gc)` surface,
both dropped by owner decision. It admits fragments by **midpoint**; the live
simulator admits by **start-in-region**. It also kept a second copy of the
hexamer encoder (`precompute.py`) that did not fold case.

## Layout

Paths mirror where each file lived.

| Path | What it was |
|---|---|
| `background_model/simulator/weights.py` | `HexamerTables`, `build_region_weights`, the midpoint geometry, `L_MIN`/`L_MAX` |
| `background_model/simulator/precompute.py` | per-region hexamer/GC tracks; the second encoder copy |
| `background_model/simulator/capture.py` | capture surface + marginal FL fit (Steps 1-2) |
| `background_model/simulator/sampler.py` | `draw_fragments_for_region` (Step 5) |
| `background_model/simulator/emit.py` | BED/h5 emit and the manifest (Step 6) |
| `scripts/run_simulator.py` | the driver for Steps 1-6 |
| `scripts/validate_parameter_recovery.py`, `verify_vectorize.py`, `profile_simulator.py` | checks and profiling of the above |
| `scripts/cut_site_oracle.py` | oracle NLL against an old-generation manifest + store. NOT `tests/cut_site_oracle.py`, which is live |
| `tests/test_simulator_phase2.py`, `test_simulator_phase3.py`, `test_simulator_weights.py` | 142 tests (33 + 67 + 42) |

## The tests here are not collected

`make test` collects `test/ tests/ fragmentomics_tools/`, so nothing under
`attic/` runs. Removing the three test files took 142 tests out of the default
suite. They could not stay live: every one imports the retired modules.

## What stayed live, and why

Importers that needed only shared pieces were rewired, not moved:

- `scripts/count_cut_site_hexamers.py` used `precompute`'s encoder and
  `weights`' `L_MIN`/`L_MAX`. It now uses `hexamers` and `cut_site_stats`. The
  old encoder did not fold case; the script upper-cases before encoding, so
  its output is unchanged.
- `scripts/build_hexamer_prior.py` used `weights.HexamerTables` as a plain
  container. It now defines its own.
- `tests/test_count_cut_site_hexamers.py` compared `region_hexamers` with
  `precompute_region`. It now compares with the file's independent string
  encoder.
