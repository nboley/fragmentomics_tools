# Handoff — the `dataframe.py` refactor

Current as of 2026-10-07. Branch `version_2`, HEAD `308f84d`, pushed,
**0 behind `main`**.

## Goal

Break `RegionDataFrame` into four layers — data model, interval algebra,
annotation, analysis — so that interval and geometry code can be imported
without pulling in pysam, the motif stack and `pybedtools`, and so that adding
a data source stops meaning adding a method to the class.

Progress on the headline metric: **61 methods → 41**. `import
fragmentomics_tools.dataframe` went **2,779 → 1,633** modules.

## Reference documents

| File | Role |
|---|---|
| [`dataframe_layering_design.md`](dataframe_layering_design.md) | **The parent design.** Layers, annotation protocol, Phases 3-4. Carries inline reconciliation notes marking what is done and what was superseded. |
| [`dataframe_critical_review.md`](dataframe_critical_review.md) | Findings ledger. Its `## Open` section is the live list of remaining defects — currently just the `parallel_apply` fork exposure. |
| [`interval_api_design.md`](../architecture/interval_api_design.md) | The interval API, **implemented**. Holds the nine binding Requirements. |
| [`bedtools_equivalence.md`](../architecture/bedtools_equivalence.md) | `bedtools` → `intervals` mapping, every row verified against the live CLI. |
| `CLAUDE.md` (worktree root) | Binding convention, not design. **Read first.** |
| `COORDINATION.version_2.md` | Per-worktree process state. Gitignored, dies with the worktree, so nothing durable may live only there. |

## Where to work

```
worktree   /home/nathanboley/src/fragmentomics_tools/.claude/worktrees/f10-test-fix
branch     version_2    (HEAD 308f84d == origin/version_2, 0 behind main)
python     /home/nathanboley/miniconda3/envs/biomarker_env/bin/python
```

The directory is still named `f10-test-fix`; the branch was renamed and the
directory was not. The **repo root** is a different checkout on
`background-model-v2` holding another session's uncommitted work — never write
there. Other live worktrees: `background-model-work`, `cut-site-model`,
`fragmentomics-tools-improvements`, `recovery-threshold`.

## Baselines — measure your own, do not quote these

```
make test                            2 failed / 616 passed / 3 skipped
make test PYTEST_ARGS="tests/ -q"    494 passed          <- NOT covered by `make test`
make test-equivalence                42 passed
make test-realdata                   10 passed
import fragmentomics_tools.dataframe 1633 modules (total)
```

The 2 failures are a known missing-data pair (`test_slice_encode_big_wig`
needs an ENCODE bigwig; `test_get_one_hot_encoded_sequence` needs the
in-package GRCh38 reference), so **`make test` exits nonzero at baseline**.
Exit 137 means it HUNG — diagnose with `py-spy dump --pid`, never re-run as a
flake.

## What is done

| Phase | Goal | Outcome |
|---|---|---|
| Interval API, Phases 0-2 | Replace 13 `bedtools`-backed methods with five free functions on `bioframe`; drop `pybedtools` | Done and archived. `intervals.py` has `overlap_indices`, `overlaps`, `nearest`, `cluster`, `merge`. `merge` is byte-identical to `bedtools` on 964,593 real regions. |
| Orientation defect fix | Fix two silent-corruption defects in `from_fragments_h5` | Construction-time autoflip **removed entirely** — the block hand-duplicated `reverse_strand()` and had drifted, losing the strand reversal and never reversing `weights` at all. Merged to `main`, **released as v1.5.0**. |
| D3 — required-column deletion | Remove `_required_columns`, which meant three different things and was unenforced | Done. `groupby().first()`, `describe()` and `.T` now work. `save_as_bed` on an SRDF stopped emitting the h5 path (6 columns → 4, owner-approved). |
| D4 — docs archive | Separate implemented designs from pending ones | `docs/architecture/` created; the two complete interval docs moved there. |
| Layering Phase 3 | Move geometry/resizing/binning and `FlDist` out of `dataframe.py` | Done. New `geometry.py` and `fldist.py`, neither importing `dataframe.py` (verified by blocking the import, not by reading it). Import weight held at 1,633. |

## What is left

1. **Layering Phase 4 — annotation protocol.** Annotation sources plus
   `on_resize`/`shrink_only`. Replaces the four SRDF geometry overrides with a
   `runtime_checkable` `Protocol`. Specified in the layering design under
   "Annotation: composition, not a value protocol" and
   "`SampleAndRegionDataFrame` — the hardest problem".
   **Scope was reduced:** `lift_over` is no longer part of it.
2. **`joblib` → `parallel_apply` consolidation.** Owner decision: **collapse
   the two helpers only, keep `fork`.** Dropping `fork` would close the
   demonstrated deadlock class but costs lambda support and copy-on-write
   frames, and the exposure is no worse than it has always been.
3. **Merge `version_2` → `main`, then tag `2.0.0`.** `main` merges IN at phase
   boundaries; the reverse happens once, at the end.
4. **`lift_over` — its own phase, unscheduled, needs a design first.** The
   target is that liftover should *actually lift the fragment arrays* where
   required. The old "invalidation" answer is marked superseded in the design
   with the four questions a new design must answer. **Interim contract:
   liftover is restricted to `RegionDataFrame`; the SRDF refusal is the
   documented behaviour.**

Deferred with reasons, unchanged: `fragment_matrix` migration (17 live
notebooks, needs its own design), plot submodule, `numba` review, dropping
`intervaltree`, re-exporting `TabixBedReader`, rewriting
`scripts/build_inactive_regions.py`.

## Traps — each of these cost real time

- **`cd` into the worktree before any ad-hoc script, then assert
  `'/f10-test-fix/' in fragmentomics_tools.__file__`.** Setting `PYTHONPATH`
  is **not** sufficient: with the shell's cwd at the repo root, `''` precedes
  `PYTHONPATH` on `sys.path` and the import silently resolves to a checkout 59
  commits behind with no `intervals.py`. `make test` is unaffected, which is
  why this is easy to miss.
- **`make test` does NOT cover `tests/`.** `PYTEST_ARGS` is
  `test/ fragmentomics_tools/`, so any change under `fragment_array/` needs an
  explicit `PYTEST_ARGS="tests/ -q"` run or a `background_model` regression
  ships unseen.
- **Repo-wide greps must exclude `.claude/worktrees/`** (copies inflate counts
  ~3x; that produced five wrong findings). **The inverse also bites:** that
  exclusion also hides the worktree you are working in, so root self-greps at
  the worktree.
- **Reconcile the TOTAL first** when two suite measurements disagree. The
  total is invariant under environment: a matching total means an environment
  difference, a differing total means tests went uncollected. Two separate
  confusions would have been one-step diagnoses with that check.
- **Deleting an attribute requires asking "what reads this?" *before* "does
  the new code work?"** The second question gets asked by default. Only the
  first would have caught `RegionDataFrame.reorder_columns` silently becoming
  a no-op with two live consumers and no test.
- **Suite parity is NOT evidence that an extraction preserves behaviour.**
  Measured: reverting Phase 3's extraction while keeping all tests left **all
  612 pre-existing tests passing**. The suite exercises the method API, which
  both trees satisfy identically, so it was blind to the whole 448-line
  change. Prove discrimination by reverting; stash only production files,
  never the tests, or the new tests are not collected and you learn nothing.
- **Never regenerate a fixture manifest to clear a diff.** A `bioframe` change
  once moved `merge` output by 592 real intervals and was written up in a
  design doc as a deliberate fixture movement. `orientation_manifest.tsv` has
  a `status` column, enforced in the comparison keys, so a pinned defect
  cannot be quietly relabelled as correct.
- **`expand_regions` and `truncate_regions` cannot be thin delegators.** They
  dispatch through `self._resize_region_boundaries()`, which is polymorphic;
  on an SRDF that override is what resizes the attached fragment arrays.
  Calling the `geometry` free function directly bypasses it silently. The free
  functions say so in their docstrings and
  `test_bypass_leaves_fragment_arrays_stale` pins the difference.
- **Agent return channels fail often here.** Seven variants seen: `not_found`
  → delivered, `timeout` → committed, silent death mid-task. **Never record an
  agent as having produced nothing without `git log` + `git status` in its
  worktree.** Two agents in one worktree also race on `.pytest_cache` and
  `make test` — serialise phases.
