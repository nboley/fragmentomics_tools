# Handoff — the `dataframe.py` refactor

Forward-looking. Current as of 2026-10-06, branch `version_2`, HEAD `a06cdac`.

There is no separate statement of work. Scope lives in the two design
documents below.

## Why this refactor exists

`RegionDataFrame` carried 61 methods spanning ten concerns that share nothing
but the rows they sit on. It holds **44** now, after the interval work deleted
16 and renamed four. Two consequences drove the refactor:

1. Interval arithmetic — the most reusable and most testable code in the
   library — could not be imported without pulling in pysam, the motif stack
   and `pybedtools`, which additionally needs the `bedtools` binary on `PATH`.
2. Every new data source meant another method on the class, so the class only
   ever grew.

The target is four layers: data model, interval algebra, annotation, analysis.
Interval algebra becomes independently importable. Annotation becomes
composition rather than a method per source.

## The documents

| File | Role |
|---|---|
| [`dataframe_layering_design.md`](dataframe_layering_design.md) | Parent design. Layers, annotation protocol, `SampleAndRegionDataFrame`, module layout, Phases 3-4. |
| [`interval_api_design.md`](interval_api_design.md) | Carved out of the parent. Holds the **nine binding Requirements**. Phases 0-2, all complete. |
| [`bedtools_equivalence.md`](bedtools_equivalence.md) | `bedtools` command to API mapping, every row verified against the live CLI. |
| [`dataframe_critical_review.md`](dataframe_critical_review.md) | The findings ledger that started the work. |

`CLAUDE.md` at the repo root is binding convention, not design. Read it first.

`COORDINATION.version_2.md` holds process state. It is gitignored and dies with
the worktree, so nothing durable belongs only there.

## Branch and release state

| | |
|---|---|
| branch | `version_2` (renamed from `f10-test-fix`; the worktree directory keeps the old name) |
| `origin/version_2` | `a06cdac` |
| `main` | `b0cf772` |
| divergence | 25 ahead, 0 behind |

**No merge to `main` and no tag until the whole refactor completes** — owner
decision, 2026-10-05. The flow is inverted meanwhile: `main` merges *into*
`version_2` at phase boundaries.

## What is done

Interval design Phases 0-2.

| | |
|---|---|
| `test/` suite | 2 failed / 591 passed / 3 skipped |
| `make test-equivalence` | 42 passed |
| `merge` vs `bedtools`, 964,593 real regions | byte-identical (`ea10500ba93aa568`) |
| `pybedtools` in the library | removed; kept in `environment.yml` for `scripts/` |
| `import fragmentomics_tools.dataframe` | 2,779 → 1,631 modules |
| `import fragmentomics_tools.intervals` | 606 modules standalone |

The two suite failures are a known missing-data pair, not defects:
`test_slice_encode_big_wig` needs an ENCODE bigwig fetch and
`test_get_one_hot_encoded_sequence` needs the in-package GRCh38 reference.

Gates closed: design review r1-r4, implementation review r1-r5
(A-/A-/A/A/A), and a test audit that found zero wrong assertions.

## What remains

In order.

1. **Documentation update.** Move the three completed designs out of
   `docs/pending/`. Update architecture docs.
2. **Constructor fallback.** `rdf.groupby("contig").first()`, `rdf.describe()`
   and `rdf.T` all raise `AssertionError` from the required-columns check in
   `DataFrameBase.__init__`. pandas reconstructs the subclass with a reduced
   column set and the assert fires. Predates this refactor and is present on
   `main`. The geopandas constructor-fallback pattern is a candidate shape:
   `_constructor` returns a plain `DataFrame` when required columns are absent
   instead of asserting. **That pattern is unverified** — geopandas is not
   installed here.
3. **Layering Phase 3.** Geometry, resizing and binning move out alongside the
   `intervals` module, with `RegionDataFrame` delegating. `FlDist` moves out.
4. **Layering Phase 4.** Annotation sources, `on_resize` with `shrink_only`,
   `lift_over` invalidation.
5. **`joblib` to `parallel_apply` consolidation.** After the above.
6. **Merge `version_2` into `main`,** then tag.

### Deferred, with reasons

| Item | Why deferred |
|---|---|
| `fragment_matrix` migration | 17 live notebooks depend on it; needs its own design |
| plot code into `plot/` submodule | not on the critical path |
| `numba` review | two `njit` kernels, 0.80 s of import cost |
| drop `intervaltree` | only reachable once `overlaps_rdf` consumers migrate |
| re-export `TabixBedReader` | per-region BED fetch survives but is undiscoverable |
| rewrite `scripts/build_inactive_regions.py` | needs geometric `subtract` and clamped `slop`, both deliberately out of scope |

## Two things to settle before Phase 3 starts

**One design question is open.** Where `center_regions_on_tf_motif` goes. It is
the largest method on the class (~150 LOC), needs JASPAR, a scoring model and a
GPU, and defers its torch and motif imports to call time to keep them off the
import path. It *relocates* regions rather than following them, so it produces
new geometry rather than responding to it and does not fit the
`PositionalAnnotation` shape. Phase 4 must not force it into `on_resize`. See
the layering design's "Still open" section.

**Build a real-data baseline for fragment orientation first.** Phases 3-4 touch
fragment attachment and strand orientation. `from_fragments_h5` reverses
coordinates and swaps strands for minus-strand regions, and CLAUDE.md records
that getting this wrong silently destroys strand asymmetry.

The reason to insist on this: seven defects surfaced during the interval work
*after* a passing A-grade review, and every one was caught by executing against
awkward real input — a windowed frame, a non-default index, a second process,
the real CLI. None were caught by reading code, and none by the synthetic
in-process tests.

There is no external reference tool for fragment orientation, so the equivalent
of the `bedtools` CLI is a committed real-data baseline with a test that
asserts against it. `test/test_interval_real_data.py` and
`test/fixtures/interval_manifest.tsv` are the working pattern: the manifest is
regenerated by `scripts/capture_interval_fixtures.py` and compared by a test
that skips when EFS is absent, with `make test-realdata` refusing to skip.

## Test entry points

| Command | Covers |
|---|---|
| `make test` | the library suite, `test/` — wrapped in a kill-timeout |
| `make test-equivalence` | `bedtools` differential tests; errors rather than skips if the binary is absent |
| `make test-realdata` | real-data manifest; errors rather than skips if EFS is absent |

Run the suite only via `make test`, never bare `pytest`. It wraps
`timeout --signal=KILL`. This suite can *wedge* rather than fail — a fork
deadlock once ran 12 hours emitting nothing. Exit 137 means it hung, which is a
finding to diagnose with `py-spy dump --pid`, not a flake to re-run.
