# attic/hexamer_prior_pipeline — REFERENCE ONLY, NOT LIVE CODE

The containment-admission cut-site hexamer counter and the analysis built on
its output: the pooled hexamer prior, the per-hexamer dispersion fit and the
disattenuation measurement. Retired 2026-10-09 by owner decision 189. **Do not
import from here, and do not treat anything here as describing current
behaviour.** Nothing here runs: the scripts import each other as
`scripts.build_hexamer_prior` and `scripts.count_cut_site_hexamers`, and those
paths no longer exist.

`make test` collects `test/ tests/ fragmentomics_tools/`, so nothing under
`attic/` runs. Moving the four test files took 67 tests out of the default
suite (36 + 11 + 8 + 12).

## Why it was retired

Per-sample hexamer counting is rebased on the simulator's measure step,
`background_model/simulator/measure.py`. Its replacement is
`scripts/simulator/measure_cut_site_hexamers.py`, which calls `count_sample`,
`FragmentLengthDist.from_srdf`, `uniform_hexamer_counts` and `propensities`
and holds no counting rule of its own.

The two are **not comparable**, so the 92-sample tables this counter wrote are
not a baseline for the replacement:

- Admission: this counter kept a fragment only if it lay wholly inside a tile
  (containment), so a fragment straddling a boundary fell out of both tiles.
  `measure` admits on start-in-region, so every fragment lands in exactly one
  tile. The owner accepted the change.
- Shape: this counter wrote 4 tables x 16 length bands x 4,096 hexamers of
  `observed` and an all-candidate `background` count. `measure` writes the four
  `C(h)` tables, the uniform-null expectation `N(h)` and `r(h) = C/N`, with no
  band split.

## What is here

| File | What it was |
|---|---|
| `scripts/count_cut_site_hexamers.py` | Per-sample counter, containment admission, Parquet output |
| `scripts/batch_count_hexamers.sh` | AWS Batch array shard that ran the counter over the 88 remaining samples |
| `scripts/build_hexamer_prior.py` | Pooled prior and per-sample posteriors from the 92 counted samples; provenance guard |
| `scripts/fit_hexamer_dispersion.py` | Per-hexamer NB2 dispersion from the production zarr store |
| `scripts/measure_hexamer_disattenuation.py` | Disattenuated per-sample vs pool correlation; reads `build_hexamer_prior`'s artifact |
| `scripts/_hexamer_corr_heatmap.py` | Correlation-structure figures for the disattenuation report; imports `build_hexamer_prior` |
| `scripts/analyze_hexamer_distributions.py` | Real vs synthetic enrichment figures for `cut_site_hexamer_counts.md`; reads the counter's Parquet and carries a pasted copy of the retired `build_w6` |
| `tests/test_count_cut_site_hexamers.py` | Tests of the counter (36) |
| `tests/test_hexamer_disattenuation.py` | Tests of the disattenuation estimator (11); imports `build_hexamer_prior` |
| `tests/test_hexamer_prior_attenuation.py` | Tests of the prior's shrinkage (8) |
| `tests/test_provenance_guard.py` | Tests of `build_hexamer_prior.verify_script_provenance` and the disattenuation script's provenance reader (12) |

## How each dependent was classified

The rule: a dependent that makes sense only with the retired scripts moved here
with them; one with an independent purpose stayed and had its reference
rewired or removed.

| Dependent | Decision | Why |
|---|---|---|
| `scripts/measure_hexamer_disattenuation.py` | moved | Consumes `build_hexamer_prior`'s artifact and imports it |
| `scripts/_hexamer_corr_heatmap.py` | moved | Imports `build_hexamer_prior`'s loaders |
| `scripts/analyze_hexamer_distributions.py` | moved (missed by the first pass, caught in review) | Reads only the counter's Parquet output, and its `build_w6` is a pasted snapshot of retired simulator code. It imports nothing from the other scripts, which is why the import-based search missed it. Its `REPO`/`PLOT_DIR` are `__file__`-relative, so from here they no longer point at `docs/pending/` |
| `tests/test_hexamer_disattenuation.py` | moved | Tests a moved script; imports `build_hexamer_prior` and the counter |
| `tests/test_hexamer_prior_attenuation.py` | moved | Tests `build_hexamer_prior` |
| `tests/test_provenance_guard.py` | moved | Tests the provenance guard inside `build_hexamer_prior` and the disattenuation script |
| `docs/pending/cut_site_hexamer_counts.md` | stayed, note added | Records findings on the 92-sample tables and embeds 7 committed figures by relative path |
| `docs/pending/hexamer_prior_report.md` | stayed, note added | Analysis results. The note says its code snippet names paths now under this directory |
| `docs/pending/hexamer_disattenuation_report.md` | stayed, note added | Analysis results with 9 embedded figures. The note says its code paths are now under this directory |
| `tests/test_hexamers.py::test_constants_single_source` | rewired | Its non-vacuity list named two moved scripts; `attic/` is outside the scan |
| `environment.yml` (`pyarrow`) | rewired | Still needed: `fragmentomics_tools/checkpoint_hooks.py` serialises to Parquet. Only the comment changed |
| `docs/pending/simulator_spec.md`, `attic/pre_rewrite_simulator/README.md` | rewired | Narrated the counter as live |

The artifacts these scripts wrote are untouched on EFS, under
`/efs/analytics/nathanboley/background_model/cut_site_hexamers/`.
