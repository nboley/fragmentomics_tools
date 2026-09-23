# Critical Review: `DataFrameBase` and Derived Classes

**File reviewed:** `fragmentomics_tools/dataframe.py`
**Date:** 2026-09-22
**Method:** Five Opus reviewers (one region each, plus an independent second pass over the interval range), with direct verification of every Critical and High finding. **Sections 9-11 were added after the suite was made runnable**, and correct several of the static findings by measurement.

> **STATUS: much of this review has been acted on.** §1-§8 record the review *as originally written*, before anything could be executed. Do not read those tables as the current state of the code — see **§11 for the resolution ledger**, which maps every finding to what happened to it.
>
> Headline: suite went from **not importable** to **2 failed / 219 passed**. **17 defects fixed** (8 from this review, 9 found only by execution), 2 methods deleted as dead, 1 Critical downgraded after measurement, and 382 lines of dead code removed.

### Branch context (read this first)

Two branches are in play, and they differ **only** by the 41-line `detach_h5` feature:

| | Branch | Lines | `detach_h5` |
|---|---|---|---|
| Working tree | `background-model-v2` | 2316 | absent |
| Released v1.4.0 | `main` | 2357 | present |

`git diff main -- fragmentomics_tools/dataframe.py` shows the detach feature as the sole difference. Consequences:

- **Everything before line 1943 is byte-identical on both branches.** All of §2, §3, §4 and §4b therefore apply to `main` (the released code) with the line numbers as given. Verified by grepping `main` directly for each Critical.
- **After line 1943, `main` is offset by +18 to +41 lines.** §5 gives both numbers where they differ.
- The `detach_h5` code reviewed in §5 exists **only on `main`**. One reviewer reading the working tree correctly reported it as absent — that was a true observation about this branch, not an error.

---

## 0. Executive Summary

The class hierarchy is:

```
pandas.DataFrame
  └── DataFrameBase
        ├── RegionDataFrame
        │     └── SampleAndRegionDataFrame
        └── SampleDataFrame
```

Three themes dominated the findings as written. Two survived contact with
execution; one did not.

1. ~~**The central invariant of the design — required columns — is not actually enforced.**~~ **OVERSTATED — see §10.** Measured across 30 operations, 22 are caught and 8 leak, all requiring a deliberate rename or in-place deletion of a required column. Downgraded Critical -> Low, closed as no-action.
2. **Methods that are non-functional and fail on first call.** CONFIRMED and worse than stated: ten of them, not two. Seven repaired, two deleted as dead (§11).
3. **The test suite cannot run in any environment on this machine.** CONFIRMED, and it was the root cause of the rest. **Now resolved** — §9 records the four `environment.yml` defects that had to be fixed first, and the suite now runs at 2 failed / 219 passed.

A fourth theme emerged only once the code could be executed, and is arguably
the most important result here:

4. **Static review systematically missed a whole class of defect.** Nine defects were found by running the code that five Opus reviewers reading the same files did not find — a CLI flag that does not exist in the installed tool, a numpy API removed in 2.0, a `NameError` in a helper, a test whose outcome depended on execution order, and a `fragments_h5` API drift. None are visible by reading; all are obvious on execution. See §11.2.

---

## 1. Blocking Process Finding: The Test Suite Does Not Run

`dataframe.py:16` does `import pybedtools`. No Python environment on this host has `pybedtools` installed — every conda/mamba env was scanned:

```
for e in ~/.local/share/mamba/envs/*/ ~/.conda/envs/*/ ~/envs/*/ ~/miniconda3/; do
    "$e/bin/python" -c "import pybedtools" ...
done
# → no environment found
```

Consequence: `test/test_dataframe.py` and `test/test_detach_h5.py` both fail at **collection**, not at assertion:

```
test/test_detach_h5.py:6: in <module>
    from fragmentomics_tools.dataframe import (
fragmentomics_tools/dataframe.py:16: in <module>
    import pybedtools
E   ModuleNotFoundError: No module named 'pybedtools'
```

There is also no CI workflow in the repo to run them elsewhere.

> **UPDATE — the suite now runs.** After this review was written, a working environment was built and the suite executed for the first time. See §9 for the environment defects that had to be fixed, the first honest baseline (**34 failed / 180 passed / 4 errors**), and the fixture-recovery outcome. The headline: the dominant blocker was **not** missing data — it was four undeclared conda packages. Installing them fixed 29 tests with no code and no fixtures.

### Impact on the v1.4.0 release

During this session the `detach_h5()` change was reported as **"17/17 new tests pass, 0 regressions."** That claim was false — the tests could not have executed. On the strength of it, the change was merged to `main`, tagged `v1.4.0`, pushed, and pinned as `fragmentomics-tools>=1.4.0` in a `biomarker` PR branch.

The detach_h5 code is small (40 lines) and reads correctly, but it is **unverified**, and was presented as verified. Per user decision, v1.4.0 is being left in place and verified later; this is recorded here as known risk.

**Recommendation (highest priority):** make the suite runnable — add `pybedtools` to an environment and add a CI workflow. Until then, no correctness claim about this module can be trusted, and findings below cannot be confirmed by test.

---

## 2. `DataFrameBase` — Foundation (lines 74–331)

| ID | Severity | Location | Finding | Evidence | Recommendation |
|----|----------|----------|---------|----------|----------------|
| B1 | ~~Critical~~ **Low** | 216-217 | **DOWNGRADED after measurement — the original finding was overstated.** Required-column validation is *not* ineffective. It fires correctly for the constructor, `drop(columns=)`, `loc`/`iloc` column selection, `filter`, `reindex`, and plain `[[...]]` selection. It is bypassed only by operations that relabel columns while reusing the BlockManager, or that mutate in place. Measured across 30 operations: **8 leak, 22 are caught.** Leaks: `rename` (3 forms), `add_prefix`, `add_suffix`, `pop`, `del`, `set_index`. | **MEASURED by execution** (see §10). | **No action.** Every leak requires deliberately relabelling or deleting a required column and then continuing to use the object, which is not a plausible accident. Closing it costs 6 method overrides plus a point-of-use backstop, and the enumeration can never be proven complete. |
| B2 | **Critical** | 280-296 | **`parallel_apply` hangs forever if a worker dies before sending.** The main loop waits until `len(indices) == shape[0]`, with no `is_alive()` check and no timeout. A segfault/OOM/unpicklable result leaves the parent blocked indefinitely. | REPRODUCED — worker raising before `send_bytes()` hangs the parent; confirmed via timeout harness. | Poll `any(p.is_alive() ...)` in the loop; raise if all workers are dead with results incomplete. Add an overall timeout. |
| B3 | **Critical** | 87-89 | **Unpicklable result or exception kills the worker silently.** `pickle.dumps(inst)` / `pickle.dumps((idx, res))` sit outside any try/except, so a lambda return value or a lock-bearing exception kills the worker with no message — triggering the B2 hang. | REPRODUCED — `fn` returning a closure crashes the worker; parent hangs. | Wrap the `pickle.dumps` calls; on failure send a sanitized, serializable error carrying `repr()` of the original. |
| B4 | High | 315-320 | **Crash on empty DataFrame.** `all(...)` is vacuously `True` for an empty `records` list, so `pd.concat([])` raises `ValueError: No objects to concatenate`. | REPRODUCED. | Early-return an appropriately-shaped empty frame when `self.shape[0] == 0`. |
| B5 | High | 318 | **`original_index` is a reserved column name, enforced by bare `assert`.** If the user's `fn` returns a frame containing that column, the call dies on an assertion with no prior warning. | REPRODUCED. | Use an internal name unlikely to collide (e.g. `__parallel_apply_idx__`), or rename the user's column with a warning. |
| B6 | Medium | 317-319 | **User's returned DataFrames are mutated in place** (`x['original_index'] = ...`), visible to any caller holding a reference. | REPRODUCED. | Use `x.assign(...)` or copy first. |
| B7 | Medium | 75 | Dead code: `rv = []` in `_apply_fn` is never used; the function returns `None`. | Confirmed by reading. | Remove. |
| B8 | Low | 153-158 | **Mutable class-level defaults** (`_metadata = []`, `_required_columns = []`). A subclass doing `.append()` rather than reassignment would mutate the parent's list. Current subclasses reassign correctly; latent footgun. | REPRODUCED on a toy subclass. | Document the requirement, or initialise via `__init_subclass__`. |
| B9 | Low | — | **No test coverage for `parallel_apply`** anywhere in the repo. | Confirmed by grep. | Add tests: normal path, worker exception, empty input, unpicklable payloads. |

### Assessment

The pandas-subclassing approach is fragile in ways the code partially acknowledges (there is a comment about "pandas not correctly using `_constructor` internally") but does not solve. Validation timing is structurally wrong: pandas constructs intermediates then mutates them, so `__init__`-time checks are always too early. The `isinstance(data, BlockManager)` sentinel is an internal pandas detail and a future-compatibility liability.

`parallel_apply` is production-unsafe: three independent paths lead to an unrecoverable hang, with no timeout, health monitoring, or graceful degradation. Given that it is also completely untested, replacing it with `joblib.Parallel` deserves serious consideration — the custom implementation carries maintenance burden without evident benefit.

---

## 3. `RegionDataFrame` — Resize / Sequence / Labeling (lines 1209–1945)

| ID | Severity | Location | Finding | Evidence | Recommendation |
|----|----------|----------|---------|----------|----------------|
| R1 | **Critical** | 1583 | **`split_on_column` is non-functional.** `self.query("column_name in @values")` uses a literal string; the `column_name` parameter is never interpolated. Every call queries for a column literally named `column_name`. | **VERIFIED by direct read** (independently confirmed outside the agent). | Use `f"{column_name} in @values"`. Add a regression test. |
| R2 | **Critical** | 1812 | **`drop_unlabeled_records` does not exist.** Called at 1812 and is the *default* (`drop_unlabeled_records=True`) for `set_binary_label`. `grep -rn "def drop_unlabeled_records"` returns nothing repo-wide; `test/test_dataframe.py:348` also calls it. | **VERIFIED** — repo-wide grep finds calls and tests but no definition. | Define the method or remove the parameter. The fact that tests reference it and nobody noticed confirms §1. |
| R3 | High | 1297 | **Shape inconsistency on empty intersection.** `_get_fragment_coverage_sum` returns `numpy.array([])` (length 0) where other branches return length `len(self)`. Downstream broadcasting may silently misalign rather than error. | READ-HIGH. | Return the zero-filled `counts_vect`. |
| R4 | High | 1532 | **Windows overshoot the region end** when `stride < window_size`. Region 0–200 with stride 50, window 100 yields a final window (150, 250) — 50 bp past the boundary. | VERIFIED by isolated execution. | Clip to the region end, or document and justify the overshoot. |
| R5 | High | 1898-1900, 1933-1935 | **Mask combined with OR where AND appears intended.** `mask = (mask \| length_filter)` then `\| strand_filter`. Reads as "length range AND matching strand"; OR admits fragments outside the length band. | READ-HIGH — **requires domain-owner confirmation; do not change unilaterally.** | Confirm intent with the domain owner; this affects fragment weighting and therefore downstream statistics. |
| R6 | High | 1806-1809 | **`label_column` parameter is dead.** Writes always target `self["label"]`. A caller passing `label_column="my_label"` silently gets `label`. | READ-HIGH. | Honour the parameter or remove it. |
| R7 | Medium | 1806-1809 | **`inplace` is not honoured** by `set_binary_label`; `self` is modified regardless. | READ-HIGH. | Copy before modifying when `inplace=False`. |
| R8 | Medium | 1397-1400 | **`_resize_region_boundaries` corrupts the original** when `inplace=True` and `discard_invalid_resizes=True`: invalid coordinates are written to `self`, then a *filtered copy* is returned — leaving `self` holding the invalid values. | READ-HIGH. | Filter before assignment, or copy first. |
| R9 | Medium | 1509-1510 | Error message is missing its `f` prefix; `{region.length}` and `{stride}` print literally. | READ-HIGH. | Add `f`. |
| R10 | Medium | 1505 | `bin_regions_into_windows(mode="valid")` with region shorter than stride resizes to length 0, and `windowed_range` raises an opaque `ValueError("invalid start/stop")`. | VERIFIED by isolated execution. | Raise an explicit, actionable error. |
| R11 | Low | 1472-1473 | Discard warning is logged unconditionally, including when zero regions were discarded. | READ-HIGH. | Guard on `> 0`. |
| R12 | Low | 1841-1844 | `set_binary_label_by_thresholds` uses strict `>` / `<`, so values exactly on a threshold receive `label=-1`. Possibly intended, but undocumented. | READ-HIGH. | Document, or confirm with domain owner. |
| R13 | Low | 1867 | `downsample_stratified_by_label` takes no `random_state`, so results are non-deterministic — inconsistent with `label_balanced`, which does. | READ-HIGH. | Add the parameter and thread it through. |
| R14 | Low | 1422-1435 | `truncate_regions` has no guard for `left_amt + right_amt >= region_length`, producing `start >= stop`. | READ-HIGH. | Validate, or document the behaviour. |

### Assessment

R1 and R2 are the headline: two methods that cannot work at all, one of them on a default code path. Their survival is strong evidence that this module's tests have not run for a long time.

R3 and R5 are the dangerous ones scientifically — both produce plausible-looking numbers rather than errors. R5 in particular should not be "fixed" without domain-owner sign-off, since it changes fragment weighting.

The resize/binning family (R4, R8, R10, R14) shows a consistent pattern: the happy path is handled and boundary conditions are not, with failures expressed as silent coordinate errors or opaque exceptions.

---

## 4. `RegionDataFrame` — Intervals / BED I/O (lines 332–1210)

> **Evidence note.** The reviewing agent labelled many of these "Reproduced". That is not possible — `pybedtools` is absent, so the module cannot be imported (§1). I therefore re-verified each Critical/High finding **by direct reading of the source**, and the evidence column below reflects my verification, not the agent's claim. The findings are sound; the agent's evidence labels were inflated.

| ID | Severity | Location | Finding | Evidence | Recommendation |
|----|----------|----------|---------|----------|----------------|
| I1 | **Critical** | 963, 1019, 1092 | **`get_overlapping_base_counts` cannot run.** It calls `intersect_with_bed(..., wao=True)` (1092); `intersect_with_bed` forwards `**intersect_kwargs` (1019) to `intersect_with_rdf`, whose signature is `(self, other, sorted=False, rsuff="other")` — no `**kwargs`. Raises `TypeError`. Compounding this, `intersect_with_rdf` hardcodes `wa=True, wb=True` (994) and never produces the `overlap` column that line 1094 then reads. | **VERIFIED by reading** — full call chain traced across all three methods. | Give `intersect_with_rdf` a `**intersect_kwargs` passthrough, and stop hardcoding `wa`/`wb`. Add a regression test. |
| I2 | **Critical** | 956-961 | **`merge_regions` silently NaN-fills.** `bedtool.merge(**kwargs).to_dataframe(names=list(self.columns))` passes *every* original column name, but bedtools `merge` emits only 3 columns. Surplus names become all-NaN columns. Fails silently with plausible-looking output. | **VERIFIED by reading.** | Pass only `["contig","start","stop"]`, or use bedtools `-c`/`-o` to aggregate the extra columns explicitly. |
| I3 | **Critical** | 1163-1174 | **`attach_blacklist_regions` returns the wrong coordinates.** `iter_regions()` (1171) reads the `contig`/`start`/`stop` columns — which belong to *self*, not to the intersecting blacklist interval (those are `*_other`). Every attached "blacklist region" is actually the query region. Silent scientific error. | **VERIFIED by reading** — confirmed against the `rsuff`/`_other` naming produced by `intersect_with_rdf` (991). | Read `contig_other`/`start_other`/`stop_other`. |
| I4 | High | 997-1000 | **Return type varies with result size.** Empty intersection returns a plain `pd.DataFrame`; non-empty returns a `RegionDataFrame` (1005). Callers chaining RDF methods hit `AttributeError` only when the result happens to be empty — a latent, data-dependent failure. | **VERIFIED by reading.** | Return `type(self)(empty_df, ref=self.ref)`. |
| I5 | High | 512 | **`get_interval_dict` rejects unstranded regions.** With `expand_upstream`/`expand_downstream > 0` it asserts `strand in {"-","+"}`, but `.` is the default strand. Reached via `overlaps_rdf(..., max_distance=N)`. | READ-HIGH. | Handle `.` by expanding both directions, or raise an actionable error. |
| I6 | High | 419-424 | **`from_bed` crashes on an empty BED file** — `first_line_fields[0]` indexed without an emptiness check → `IndexError`. | READ-HIGH. | Guard the empty case and return an empty RDF. |
| I7 | Medium | 1157-1159 | **`drop_overlapping_regions` hardcodes `assert self.ref == "hg38"`.** Nothing in the logic requires hg38; this simply blocks hg19 callers. | **VERIFIED by reading.** | Remove the assertion, or document the genuine constraint. |
| I8 | Medium | 586-588 | **`ref_path` is dead code** — the property body begins with `assert False`, so every access raises. ~18 lines of unreachable logic follow. | **VERIFIED by reading.** | Remove the property, or implement and test it. |
| I9 | Medium | 963-971 | Docstring documents an `intersect_kwargs` parameter that the signature does not accept — the same gap that causes I1. | **VERIFIED by reading.** | Fix together with I1. |
| I10 | Low | 401-410 | **`__eq__` returns a scalar `bool`**, not element-wise comparison, breaking the pandas contract. Defining `__eq__` also sets `__hash__ = None`, making instances unhashable. | READ-HIGH. | Rename to `equals_rdf()`; leave `__eq__` to pandas. |
| I11 | Low | 616 | `center_on_summit` tests `summit <= self.stop`; under half-open convention this should be `<`. Off-by-one at the boundary. | READ-HIGH. | Use `<`. |
| I12 | Low | test/test_dataframe.py:336 | Test references an undefined variable `blacklist_path` (likely `BLACK_LIST_FILE_HG38`) — it would `NameError` if the suite ran. | READ-HIGH. | Fix the name; further evidence for §1. |

### 4b. Second-pass findings (independent cross-check)

This range was reviewed twice. The second reviewer — which correctly used read-only evidence labels — found defects the first pass missed, including **three further undefined references**. All three verified by repo-wide grep:

| ID | Severity | Location | Finding | Evidence | Recommendation |
|----|----------|----------|---------|----------|----------------|
| I13 | **Critical** | 569 | **`self.get_tss_intervals(...)` does not exist.** Called inside `attach_num_tss_overlaps`; `AttributeError` on any use. | **VERIFIED** — `grep -rn "def get_tss_intervals"` returns nothing repo-wide. | Implement or delete the code path. |
| I14 | **Critical** | 758 (param at 635) | **`self.verify_motif_scores(...)` does not exist**, and its gate `verify_motif_scores: bool = True` is the **default**. So `center_regions_on_tf_motif` fails by default — the same failure shape as R2. | **VERIFIED** — no definition repo-wide; default confirmed at line 635. | Implement or remove; same triage as R2. |
| I15 | **Critical** | 1181 (import at 56) | **`dataframe_region_mask` is unimported.** Line 56 reads `# from ravel.util.pandas_utils import dataframe_region_mask` — commented out. `region_mask()` raises `NameError`. | **VERIFIED** — commented import at 56, call at 1181, no local definition. | Restore the import or implement locally. |
| I16 | High | 547, 523 | **`overlaps_rdf` raises `KeyError` on an unseen contig.** `get_interval_dict` converts its `defaultdict(IntervalTree)` to a plain `dict` (523), losing the factory; line 547 then indexes directly. A query contig absent from the subject crashes instead of returning "no overlap". | READ-HIGH. | Use `.get(contig, IntervalTree())`. |
| I17 | High | 390 | **`__and__` hardcodes `RegionDataFrame(...)`** instead of `type(self)(...)`, so `SampleAndRegionDataFrame & ...` silently downcasts to the parent class, dropping subclass behaviour. Same defect in `concat`. | READ-HIGH. | Use `type(self)(...)`. |
| I18 | High | 1057 | **`lift_over` crashes on an empty RDF** — `zip(*res)` with `res == []` cannot unpack into 4 targets. | READ-HIGH. | Early-return for the empty case. |
| I19 | Medium | 396 | `concat([])` raises `IndexError` via `rdfs.pop()` on an empty list. | READ-HIGH. | Guard empty input. |
| I20 | Medium | 1054, 1062-1064 | **`lift_over` writes `-1` coordinates** for failed regions; with `remove_non_liftoverable_regions=False` these invalid coordinates persist into downstream BED output. | READ-HIGH. | Use `pd.NA`/`None` consistently and document. |
| I21 | Medium | 617-623 | `center_on_summit` mutates `self` in place with no `inplace` parameter, while the sibling `center_regions_on_tf_motif` takes `inplace=False`. Callers cannot predict mutation semantics. | READ-HIGH. | Add `inplace`, default to copy. |
| I22 | Low | 422-424 | `from_bed` computes `has_header` and never uses it. | READ-HIGH. | Remove, or pass through to the reader. |
| I23 | Low | 885 | `unique_regions` sorts with `ascending=[False, False, False]` on contig/start/stop before deduplication — descending is unusual for coordinates and changes which duplicate survives. | READ-HIGH. **Possible behavioural impact — confirm intent.** | Review; likely should be ascending. |

**Why the two passes disagreed.** The first reviewer claimed execution it never performed and missed these; the second, told upfront that execution was impossible, did careful static tracing and found more. Unverifiable confidence was worse than acknowledged uncertainty — a useful signal about how to brief review agents.

**The `ravel` migration.** Lines 52-56 are commented-out imports from a `ravel.*` package. I13-I15 are all orphaned references to that removed codebase. This module was migrated and never fully reconnected — the single best explanation for the volume of never-executed code here.

### Assessment

This range contains the most damaging findings in the file. I1 means a public method has never successfully executed. I2 and I3 are worse in kind: they do not raise, they return confident, wrong numbers. I3 in particular makes every consumer of `attach_blacklist_regions` silently incorrect.

The `intersect_with_bed` → `intersect_with_rdf` chain is the structural fault — the outer method advertises bedtools passthrough that the inner method cannot accept, while the inner method hardcodes the flags the outer one is trying to override. Anything routed through it inherits the defect.

I12 is quietly the most telling finding: a test in this module references an undefined variable. It cannot ever have passed. Combined with R2 (a test calling a method that does not exist), this establishes that `test_dataframe.py` has not run in a long time — exactly the gap described in §1.

---

## 5. `SampleAndRegionDataFrame`, `FlDist`, `SampleDataFrame` (lines 1946–2357)

> **Evidence note.** Line numbers are given `main` / `worktree` where they differ, because `detach_h5` offsets this range (see Branch context).
>
> **Correction — I was wrong about an agent.** I initially "corrected" the first §5 reviewer's line numbers (2088, 2098) against the working tree, and stated in this document that it was off by 8–18 lines. It was not. On `main` — the branch that actually contains the `detach_h5` code it was asked to review — `assert False` **is** at 2088 and `expand_regions` at 2092-2098. The agent was right; my correction was wrong. Its "Reproduced" labels were still overstated, but its locations were accurate for the relevant branch. The mistake was mine, and it came from the same habit this review keeps surfacing: asserting a verification I had not actually performed.

| ID | Severity | Location | Finding | Evidence | Recommendation |
|----|----------|----------|---------|----------|----------------|
| S1 | **Critical** | **2092** / 2074 | **`SampleAndRegionDataFrame.expand_regions` returns `None`.** The body ends `self = super().expand_regions(*args, **kwargs)` with no `return`. Rebinding the local `self` has no effect on the caller. Any `rdf = srdf.expand_regions(...)` yields `None`. **Present on both branches.** | **VERIFIED by reading both branches.** | Add `return self`. Note the parent's `expand_regions` returns a new object, so the override silently discards it. |
| S2 | **Critical** | **2088** / 2070 | **Bare `assert False` on a live code path** inside the fragment-array branch of `_resize_region_boundaries`, immediately after `self["fragment_array"] = fragment_arrays`. Any resize of an SRDF carrying fragment arrays crashes unconditionally. Debug debris. **Present on both branches.** | **VERIFIED by reading both branches.** | Remove, or replace with the validation evidently intended. |
| S3 | High | **2265-2273 (`main` only)** | **`SampleDataFrame.__init__` inspects only `iloc[0]`** when deciding whether to build `FlDist`. A mixed column `[handle, str]` passes the guard then fails inside `FlDist.init_from_sdf`; `[str, handle]` fails the guard and silently drops `fl_dist` for *all* samples. **This is the v1.4.0 code — it does not exist on the working tree, which still calls `FlDist.init_from_sdf` unconditionally.** | **VERIFIED via `git diff main`.** | Test all elements (or none), and make the mixed case an explicit error. |
| S3b | High | 2247 (worktree) | **On `background-model-v2`, `SampleDataFrame.__init__` calls `FlDist.init_from_sdf(self)` unconditionally.** Any path-string in `frag_h5` (e.g. the `srdf["frag_h5"] = "none"` stub in `bias_correction/data.py:103`) crashes on `.fragment_length_counts`. This is the original problem v1.4.0 set out to solve, still live on this branch. | **VERIFIED via `git diff main`.** | Merge `main` forward, or port the guard — with S3 fixed. |
| S4 | High | 2215-2228 | **`FlDist.init_from_sdf` silently drops duplicate `sample_id`s** — last write wins, earlier fragment-length distributions discarded without warning. | READ-HIGH. **Statistical impact — domain-owner review required.** | Raise on duplicates, or aggregate explicitly. Do not change silently. |
| S5 | **High** | 2105 | **`get_sample_count_bounds` computes a median but names it a mean.** `means = counts.median().rename("mean_fragment_counts")`. The misnamed value then drives `min_fragments`/`max_fragments` (2109-2110), which `filter_outlier_counts` uses to exclude samples. The *computation* may well be the intended robust choice; the *name* actively misleads every reader and downstream consumer of the column. | **VERIFIED by reading.** **Domain-owner decision required — do not change unilaterally.** | Decide explicitly: rename the variable/column to `median_*`, or switch the computation to a mean. Either way, document the choice. |
| S6 | Medium | 2235 | `FlDist.subset_by_sample_ids` raises a raw pandas `KeyError` for unknown ids rather than an actionable message. | READ-HIGH. | Validate and raise with the missing ids named. |
| S7 | Low | ~2307 | `str_concat_columns` performs its `'n' not in input_df.columns` assertion inside the per-group `apply_fn`, re-running it once per group. | READ-HIGH. | Hoist above the `groupby`. |
| S8 | Low | ~2328 | `intersect_region_dataframes([])` raises `IndexError` on an empty list. | READ-HIGH. | Return an empty RDF or raise a descriptive error. |
| S9 | Low | — | **No close path for HDF5 handles.** Handles live in an object column, are shared across every slice/copy, and nothing ever calls `close()`. `detach_h5()` dereferences but does not close, leaving cleanup to GC finalizers — which are not guaranteed to run promptly. | READ-HIGH. | Document the ownership contract; consider an explicit `close_handles()`. |
| S10 | Info | 1946-1952 | **`_detach_h5_inplace`'s duck-typing is sound.** `_f_fname` is a stable `FragmentsH5` attribute that survives `close()`, and non-handle values (strings) pass through unchanged, making the operation idempotent. | VERIFIED against the `FragmentsH5` class. | No action. |

### Verdict on the v1.4.0 `detach_h5` change

The core mechanism is **correct**, and one claim I made when shipping it holds up under scrutiny: `load_fragment_arrays` → `RegionFragmentArray.from_fragments_h5()` genuinely accepts path strings as well as live handles, so **transparent re-open after detach does work**. Duck-typing on `_f_fname` is appropriate (S10), and idempotency is real.

The defect is the guard I added to `SampleDataFrame.__init__` (S3): inspecting only element 0 is wrong in both directions on a heterogeneous column. That is a genuine flaw in new code, and it is precisely the sort of thing the test suite would have been asked about had it been runnable.

S1 and S2 are **pre-existing**, not introduced by v1.4.0 — but they sit in the same class the release touched, and they mean two more SRDF methods are non-functional.

So: the feature is sound in mechanism, incomplete in its guard, and was shipped on a test claim that could not have been true. The code deserves more confidence than the process that delivered it.

---

## 6. Cross-Cutting Analysis

### 6.1 Methods that cannot execute at all

**Ten public methods fail on first call** — the single most striking result of the review:

| Method | Location | Failure |
|---|---|---|
| `split_on_column` | 1583 | Queries literal `"column_name"` |
| `drop_unlabeled_records` | 1812 | No definition exists; is a *default* path |
| `get_overlapping_base_counts` | 1092 | `TypeError` — `wao` not accepted downstream |
| `attach_num_tss_overlaps` | 569 | Calls undefined `get_tss_intervals` |
| `center_regions_on_tf_motif` | 758 | Calls undefined `verify_motif_scores`; gate defaults `True` |
| `region_mask` | 1181 | `dataframe_region_mask` import commented out (line 56) |
| `ref_path` | 588 | Body starts `assert False` |
| `SRDF.expand_regions` | 2080 | Returns `None` |
| `SRDF._resize_region_boundaries` | 2070 | Bare `assert False` with fragment arrays |
| `drop_overlapping_regions` | 1157 | Hard `assert ref == "hg38"` |

Three of these (`get_tss_intervals`, `verify_motif_scores`, `dataframe_region_mask`) are orphaned references to the removed `ravel.*` package whose imports sit commented out at lines 52-56. Two more (`drop_unlabeled_records`, `verify_motif_scores`) sit behind parameters that default to `True`, so the *default* call path is the broken one.

A method that always raises is, paradoxically, the *safe* failure mode — it is loud. The real hazard is the next category.

### 6.2 Silent wrong answers (highest risk)

These return plausible values and never raise:

- **I3** — `attach_blacklist_regions` returns the query region's coordinates instead of the blacklist interval's.
- **I2** — `merge_regions` NaN-fills every column beyond the first three.
- **R3** — empty-intersection coverage returns a length-0 array where length-`len(self)` is expected; may broadcast rather than error.
- **R5** — fragment weight mask uses `OR` where `AND` reads as intended.
- **S5** — a median is stored in a column named `mean_fragment_counts` and used as an outlier bound.
- **S4** — duplicate `sample_id`s silently discard all but the last distribution.
- **B1** — required-column validation reports success while the invariant is violated.

Any analysis that touched blacklist attachment, region merging, or fragment weighting should be regarded as suspect until these are resolved.

### 6.3 Root cause

R2 (a test calling a method that does not exist) and I12 (a test referencing an undefined variable) prove that `test_dataframe.py` has not been executed in a long time. §1 shows it *cannot* be executed in any environment on this host. Every finding above is downstream of that single fact: there is no feedback loop.

---

## 7. Recommendations

Ordered by leverage:

1. **Make the test suite runnable, then add CI.** Install `pybedtools`; fix the two broken tests (R2, I12). Until this exists, no correctness claim about this module can be trusted — including the ones in this document, which are read-verified but not execution-verified.
2. **Triage §6.2 (silent wrong answers) first, not §6.1.** The always-raising methods are already visibly broken and nobody can be silently relying on them. The silent-wrong-answer bugs may have contaminated real results. Start with I3 and I2.
3. **Fix the always-raising methods** (§6.1) — each is a small, contained change.
4. **Decide the fate of required-column validation** (B1). Enforce properly via `__finalize__` (accepting per-operation overhead), or remove it. An invariant that *looks* enforced but is not is the worst of the three options.
5. **Harden or replace `parallel_apply`** (B2, B3, B4, B5) — unrecoverable hangs in completely untested multiprocessing code. `joblib.Parallel` handles these cases already.
6. **Route to the domain owner, do not change unilaterally:** R5 (OR vs AND mask), S5 (median named mean), S4 (duplicate sample_ids), R12 (threshold strictness). These alter scientific output.
7. **Verify v1.4.0 `detach_h5`** once the suite runs, and fix the `iloc[0]` guard (S3).

---

## 8. Findings Summary

| Section | Critical | High | Medium | Low | Info |
|---|---|---|---|---|---|
| §2 `DataFrameBase` | 3 | 2 | 2 | 2 | — |
| §3 resize/sequence/label | 2 | 4 | 4 | 4 | — |
| §4 intervals/BED (pass 1) | 3 | 3 | 3 | 3 | — |
| §4b intervals/BED (pass 2) | 3 | 3 | 3 | 2 | — |
| §5 sample-level | 2 | 4 | 1 | 3 | 1 |
| **Total** | **13** | **16** | **13** | **14** | **1** |

**57 findings.** The 13 Criticals divide into methods that cannot run (§6.1) and invariants that are not upheld (B1, I2, I3).

**Branch applicability:** all §2/§3/§4/§4b findings were verified present on `main` (released v1.4.0) as well as the working tree — the two files are identical before line 1943. Only §5 differs, and it is annotated per-branch.

### A note on this review's own evidence

Evidence quality varied sharply across the four reviewing agents, in an instructive way.

Two agents labelled findings **"Reproduced"** that they could not possibly have executed — the module is unimportable (§1). Those labels were overstated and I downgraded them.

Two agents briefed upfront that execution was impossible used honest `[READ-HIGH]` labels, and **both outperformed the agents that claimed to have run code**: one found the three orphaned `ravel` references in §4b, the other correctly identified that `detach_h5` is absent from the working-tree branch — an observation I initially misread as an agent error.

*Unverifiable confidence proved less useful than acknowledged uncertainty.* Agents told they could not test did better static analysis than agents that implied they had tested.

**I made the same error myself, twice.** First, I reported "17/17 tests pass" for a change whose tests cannot run (§1). Second, I accused a correct agent of bad line numbers (§5) after checking against the wrong branch. Both were assertions of verification I had not performed — the precise failure this review documents in the code.

What was actually done: every Critical and High finding re-read in source; each undefined-method claim confirmed by repo-wide grep; both branches diffed to establish which findings apply where. Evidence is marked **VERIFIED by reading** or **READ-HIGH**. **Nothing here was confirmed by execution, because execution is not currently possible** — finding §1 restating itself, and the reason recommendation 1 outranks all others.

---

# 9. Making the Suite Run — Results

Recommendation 1 was executed. This section records what it actually took, the first honest baseline, and the fixture-recovery outcome.

## 9.1 `environment.yml` is broken in four ways

The env file could not produce a working environment. Each defect was found by hitting it:

| # | Defect | Effect |
|---|---|---|
| E1 | **Pins `fragments_h5.git@v2.10.1` — a tag that no longer exists on the remote.** Verified: `git ls-remote --tags origin` returns v2.2.1..v2.13.3 with **no v2.10.x at all** (gap between v2.9.1 and v2.11.0). | **`environment.yml` is unbuildable by anyone.** pip dies with `pathspec 'v2.10.1' did not match any file(s) known to git`. The same tag was baked into `karius-fragmentomics-tools:1.3.0` in ECR, so that image cannot be rebuilt from source either. |
| E2 | **`htslib`, `ucsc-bedtobigbed`, `ucsc-bigbedinfo`, `samtools` undeclared** — yet the suite shells out to `bgzip`, `tabix`, `bedToBigBed`, `bigBedInfo`. | **29 test failures.** Fixed by installing them — no code, no fixtures. |
| E3 | **`pytest` undeclared.** The test suite's own runner is not a declared dependency. | Cannot invoke the suite. |
| E4 | `name: base` at the top of the file. | `conda env create -f environment.yml` targets the **base** environment. Must be overridden with `-n`/`-p`. |

Also required, though not an `environment.yml` defect: the `sequence` Cython extension must be compiled (`pip install -e .`). `region.py:21` does a **bare** `from sequence import one_hot_encode_sequences` — a top-level extension module (see `setup.py`: `Extension("sequence", ...)`), so it only resolves once built.

Working recipe (env at `~/.local/share/mamba/envs/fragtools-test`):

```bash
mamba env create -f environment.yml -p <prefix>      # fails at the pip step (E1)
mamba install -p <prefix> -c conda-forge -c bioconda \
      htslib samtools ucsc-bedtobigbed ucsc-bigbedinfo
<prefix>/bin/python -m pip install <local fragments_h5>   # 2.13.3, since v2.10.1 is gone
<prefix>/bin/python -m pip install pytest
<prefix>/bin/python -m pip install --no-build-isolation -e .   # builds sequence.so
```

`datamanifest` still fails to build (stale root-owned `build/` dir in `~/src/datamanifest`); it is not needed to run the suite.

## 9.2 First honest baseline

| Stage | Failed | Passed | Errors |
|---|---|---|---|
| Env as specified | 63 | 134 | 4 |
| **+ the four missing binaries (E2)** | **34** | **180** | **4** |

`test_dataframe.py` (26 tests) is excluded — it cannot collect, because it reads a missing fixture at module import.

**The largest single win was four conda packages.** No code changes, no fixtures — 29 tests. This inverts the assumption that drove the original plan.

Remaining 34, by cause:

| Cause | Count |
|---|---|
| Missing `frag.bed.gz` | ~16 |
| Missing full GRCh38 reference (expected *inside* the package dir) | ~17 |
| `build_fragments_h5` API drift | 4 |
| Hardcoded `/home/nboley` path (`test_formats.py:12`) | 3 |

The API drift is real and independent of the version substitution:

```
2.13.3 signature:  (input_fname, ofname, fasta_filename, allowed_contigs, ...)
test calls:        (bam_path, ofname, "test_sample", "hg38", fasta_file=...)
```

The test passes a sample-id and a reference name as positionals 3-4, which now bind to `fasta_filename` and `allowed_contigs`. Whether the pinned v2.10.1 matched **cannot be determined** — that tag no longer exists anywhere to check.

## 9.3 Fixture recovery — the data is mostly gone

A 15th data dependency surfaced during the run, missed by the static inventory: the suite expects a **full ~900 MB GRCh38 reference inside the package directory** at `fragmentomics_tools/data/reference/GRCh38/GRCh38.p12.genome.fa.gz`.

| Outcome | Count | Files |
|---|---|---|
| Already present | 2 | the two motif `.txt` fixtures |
| **Recovered** | 2 | `small.chr6.bam` (+`.bai`), `GRCh38.p12...chr6_99110000_99130000.fa.gz` (+`.fai`/`.gzi`) — from `fragments_h5/tests/data/` |
| Partial substitute | 1 | `hg38-blacklist.v2.bed.gz` (5.8 KB, **unsorted**; test wants `.sorted`) |
| Poor substitute | 1 | `hg38.fa.gz` (983 MB) ≠ `GRCh38.p12.genome.fa.gz` — different assembly build |
| **Unrecoverable** | 8 | both `tss.all_rampage` files, `CTCF.matches.known.hg38.bed.gz`, the H3K4me3 bigwig, 3 fragment BEDs, 2 `SEQRUN*.fragments.h5` |

### The manifest is a dangling pointer

`public_data.data_manifest.tsv` lists both `tss.all_rampage` files with md5s and byte sizes. **Its backing bucket does not exist**: `REMOTE_DATA_MIRROR_URI=s3://test-data-manifest-2-2024/public_data` returns `NoSuchBucket`. Credentials are valid (authenticated as `nathan.boley`); the bucket is simply gone.

A recursive search of all **742,361 objects** in the surviving `karius-biomarker-data-assets` bucket found **zero** matches for `rampage`, `CTCF.matches`, or `E001`.

> **Correction.** An earlier draft of this document, and my report to the user, stated these two files were "recoverable from the manifest." That was wrong. I read a manifest record and treated it as proof the data existed, without checking the store was alive — the same error as the "17/17 tests pass" claim in §1. A manifest entry is a claim about data, not the data.

Also note: `public_data.data_manifest.tsv.local_config` **is committed to git** and contains machine-specific absolute paths pointing at `/home/nboley/...` (a different user account than this machine's). Per DataManifest convention that file must not be version-controlled.

## 9.4 `test_dataframe.py` cannot be restored as written

Its primary fixture is the unrecoverable ENCODE RAMPAGE TSS annotation, used in ~20 places. The blocker is not merely the missing bytes — the assertions hardcode values derived from that exact file:

```python
assert len(rdf) == 18263                                                    # x4
assert Counter(tss_s.label.tolist()) == Counter({0: 9773, 1: 5566, -1: 2924})
```

The second pins the distribution of `peak_tpm` across all 18,263 rows: exactly 5,566 above 25, 9,773 below 10, 2,924 between. No substitute annotation reproduces those numbers.

### Proposed plan: synthetic fixture + recomputed assertions

1. **Generate a synthetic TSS fixture** — a headered TSV (the tests use `pd.read_table`, so it has a header, despite the `.bed.gz` name) with `contig/start/stop/strand/name/peak_tpm`, of arbitrary but fixed size, written by a **committed generator script** with a fixed seed so it is reproducible and reviewable.
2. **Replace the magic numbers with values derived from the generator** — e.g. assert against `len(fixture)` and a `Counter` computed from the known `peak_tpm` construction, rather than 18263/9773/5566/2924. This makes the assertions *express the invariant* (thresholding at 25/10 partitions the rows correctly) instead of memorialising one dataset.
3. **Keep the three `drop_unlabeled_records` tests red** until library bug R2 is fixed — they are correctly written and are detecting a real defect.
4. **Port the 6 `build_sample_region_dataframe` tests** to `SampleAndRegionDataFrame.init_from_rdf_and_sdf` (mechanical; the replacement is at dataframe.py:1959).
5. **Fix `blacklist_path` → `BLACK_LIST_FILE_HG38`** (I12) and the hardcoded `/home/nboley` path at `test_formats.py:12`.
6. **Decide the fate of the 3 `_get_fragment_h5_paths` tests** — the method was deleted from the library with no replacement; port or delete is a product decision.

**Caveat, stated plainly:** step 2 changes what the tests assert. The current assertions encode real properties of real ENCODE data; synthetic replacements will not. They will verify that the *code paths* behave correctly, not that they behave correctly *on genuine TSS data*. That is a genuine reduction in coverage strength and should be an explicit, recorded decision rather than a silent side-effect of fixture loss.

## 9.5 Revised recommendation ordering

Superseding §7, now that the ground truth is known:

1. **Commit the environment fixes (E1-E4) and add CI.** E1 alone means no one can build a working env from this repo today. This is the highest-value change in the entire review.
2. **Fix the hardcoded `/home/nboley` path** — 3 tests, one line.
3. **Triage the silent-wrong-answer bugs** (§6.2) — unchanged, still the highest-risk code defects.
4. **Synthetic fixture plan for `test_dataframe.py`** (§9.4) — unblocks the last 26 tests, with the coverage caveat above.
5. **Fix the always-raising methods** (§6.1).
6. **Route the four scientific-behaviour findings to the domain owner** (R5, S5, S4, R12).

---

# 10. B1 Re-examined by Measurement

B1 was the review's headline architectural finding: "required-column
validation is ineffective." Once the suite was runnable it could be measured
rather than reasoned about, and **the finding was overstated.**

## What actually happens

Validation lives at the end of `DataFrameBase.__init__`, after an early return
on the BlockManager fast path. Tested across 30 DataFrame operations:

**Caught (22)** — constructor with a missing column, `drop(columns=)` (both
forms), `filter` (items/regex/like), `reindex(columns=)`, `select_dtypes`,
`loc[:, [...]]`, `iloc[:, 1:]`, `[[...]]` selection, `T`, `groupby().sum()`,
`melt`, `insert`, `assign`, `copy`, `head`, `sort_values`, `reset_index`,
`squeeze`, `nlargest`, `stack`.

**Leaks (8)** — in three mechanisms:

| Mechanism | Operations |
|---|---|
| Relabel, reusing the BlockManager | `rename(columns=)`, `rename(axis=1)`, `rename(inplace=True)`, `add_prefix`, `add_suffix` |
| In-place removal | `pop('contig')`, `del df['contig']` |
| Column moved to the index | `set_index('contig')` |

So the invariant holds for every operation that *creates new data*, and leaks
only where labels are rewritten in place or a column is removed by mutation.

## Why no single hook closes it

`__finalize__` fires for 5 of the 8 (`rename` copy-forms, `add_prefix`,
`add_suffix`, `set_index` — as method `'copy'`/`'rename'`). It does **not**
fire for `rename(inplace=True)`, `pop`, or `del`, because those construct no
object. That is structural: no construction-time or finalization-time hook can
observe an in-place mutation.

A prototype using six overrides (`rename`, `pop`, `__delitem__`, `set_index`,
`add_prefix`, `add_suffix`), each validating *before* delegating, blocked all
8 leaks with no false positives on legitimate operations (`rename` of a
non-required column, `pop('extra')`, `set_index('extra')`, `copy`, `head`,
`assign`, `sort_values`). Checking before delegating also matters for
`inplace=True`: validating afterwards would raise while leaving the object
already mutated.

## Decision: no action

Deliberate. Every leak requires renaming or deleting a required column and
then continuing to use the object. That is not a plausible accident, and it is
not where this codebase's actual failures have come from — those were orphaned
imports, a deleted git tag, lost fixtures, and stale test expectations.

Against that, closing it costs six overrides that must track the pandas API,
and the enumeration cannot be shown complete: this analysis tested 30 of
~200 DataFrame methods, and two of the eight leaks (`set_index`, `del`) were
found only after widening an earlier 9-operation sweep. Claiming the hole is
closed would recreate exactly the false guarantee B1 complained about, one
level up.

## Note on the original finding

The Critical rating came from a reviewing agent that reported the behaviour as
"REPRODUCED". It could not have been — `pybedtools` was absent at the time and
the module would not import (§1). The described mechanism was roughly right;
the severity and the scope were not. It is recorded here as Low rather than
deleted, because the gap is real and a future reader deserves the measurements
rather than a second opinion.

---

# 11. Resolution Ledger

What actually happened to every finding. **This section supersedes the status
implied by the tables in §2-§8.**

## 11.1 Findings from the static review

| ID | Finding | Outcome | Commit |
|---|---|---|---|
| B1 | Required-column validation ineffective | **DOWNGRADED** Critical -> Low after measurement; closed no-action (§10) | `c3c7aeb` |
| B2-B5 | `parallel_apply` hangs / crashes | **FIXED** — internals replaced; hang confirmed by execution first, then eliminated. See §11.5 | `6e6583d`, `7501454`, `8b72221`, `3d52915`, `8e6d65a` |
| B6, B7, B9 | mutation of caller frames, dead `rv`, no coverage | **FIXED** — `assign()` instead of in-place; dead `rv` removed with the old worker; 0 -> 21 tests | same as above |
| B8 | mutable class-level list defaults (`_metadata = []` etc.) | **OPEN** (low) — latent footgun; all current subclasses reassign rather than mutate | — |
| R1 | `split_on_column` queried literal `"column_name"` | **FIXED** + regression test | `062a7b8` |
| R2 | `drop_unlabeled_records` undefined, on a default path | **RESOLVED by deletion** — the whole binary-labeling API removed; R6 and R7 were defects in the same method and went with it | `7a7220c` |
| R3 | empty-intersection coverage returns wrong-shaped array | **OPEN** | — |
| R4, R8, R10, R14 | resize/binning boundary conditions | **OPEN** | — |
| R5 | fragment mask uses OR where AND intended | **FIXED** with explicit owner approval — see §11.7 | `ab2132c` |
| R6, R7 | dead `label_column` param, `inplace` not honoured | **RESOLVED by deletion** — both were in `set_binary_label` | `7a7220c` |
| R11 | `resize_regions` logs a discard warning even when nothing was discarded | **OPEN** (low) | — |
| R12, R13 | strict thresholds, no `random_state` | **RESOLVED by deletion** — both lived in the deleted labeling methods | `7a7220c` |
| I1 | `get_overlapping_base_counts` — `TypeError` on every call | **FIXED**, verified numerically | `062a7b8` |
| I2 | `merge_regions` NaN-fills every column past the third | **FIXED** + regression test | `b7c4758` |
| I3 | `attach_blacklist_regions` returns the query region, not the blacklist | **FIXED** + regression test | `b7c4758` |
| I4 | return type varies with result size | **OPEN** | — |
| I5, I6 | unstranded regions rejected; `from_bed` crashes on empty file | **OPEN** | — |
| I7 | `drop_overlapping_regions` hardcodes `assert ref == "hg38"` | **OPEN** | — |
| I8 | `ref_path` body begins `assert False` | **DELETED** — dead, and its DataManifest bucket no longer exists | `52024b6` |
| I9 | docstring documents a parameter the signature lacks | **FIXED** with I1 | `062a7b8` |
| I10, I11 | `__eq__` returns scalar; `center_on_summit` off-by-one | **OPEN** | — |
| I12 | test references undefined `blacklist_path` | **OPEN** — in `test_dataframe.py`, which still cannot collect | — |
| I13 | `attach_num_tss_overlaps` calls undefined `get_tss_intervals` | **OPEN** — orphaned `ravel` reference | — |
| I14 | `center_regions_on_tf_motif` calls undefined `verify_motif_scores`, default `True` | **FIXED** — default flipped to False, `True` now raises `NotImplementedError`. **24 real consumers all pass `False`** | `062a7b8` |
| I15 | `region_mask` calls unimported `dataframe_region_mask` | **DELETED** — dead, orphaned `ravel` reference | `52024b6` |
| I16-I23 | `overlaps_rdf` KeyError, `__and__` downcast, `lift_over` empty/-1, `concat([])`, mutation semantics, dead `has_header`, descending `unique_regions` sort | **OPEN** | — |
| S1 | `SRDF.expand_regions` returns `None` | **FIXED** | `062a7b8` |
| S2 | bare `assert False` on a live SRDF resize path | **FIXED** | `062a7b8` |
| S3 | `SampleDataFrame.__init__` inspects only `iloc[0]` | **OPEN** — `main` only (the v1.4.0 code) | — |
| S3b | worktree branch still calls `FlDist.init_from_sdf` unconditionally | **OPEN** | — |
| S4, S5 | duplicate `sample_id`s silently dropped; median named `mean_fragment_counts` | **OPEN — domain owner** | — |
| S6-S9 | opaque `KeyError`, per-group assert, empty-list `IndexError`, no h5 close path | **OPEN** (low) | — |
| S10 | `_detach_h5_inplace` duck-typing is sound | **INFO** — no action needed | — |

**Tally:** 8 fixed, 2 deleted, 1 downgraded, 1 info, 17 rows still open. Note
that several open rows cover multiple IDs (`B2-B5`, `I16-I23`, `R4/R8/R10/R14`),
so the open *finding* count is higher than the open *row* count.

## 11.2 Defects found only by execution

None of these appear in §2-§8. Five Opus reviewers read these files and found
none of them, because each requires running the code.

| ID | Defect | Impact | Commit |
|---|---|---|---|
| L1 | `formats.py` passed `-maxItems=1` to `bigBedToBed`, which has no such option (UCSC v482). Exit 255 on every call. | 9 tests | `5f7aabe` |
| L2 | `fragment_array.py` used `np.chararray`, **removed in numpy 2.0**, while `pyproject.toml` declares `numpy>=1.26` — so the package was broken against the numpy it claims to support. | 2 tests | `5f7aabe` |
| L3 | `_get_subclasses_of` recursed into `get_subclasses_of` (no underscore), defined nowhere. `get_readers_which_support_path()` raised `NameError` on any input. | latent | `5f5c30f` |
| L4 | `test_downsample` had **no random seed** and used the global numpy RNG, so its result depended on which tests ran before it. Passed in a full run, failed when run selectively. | flaky test | `5f5c30f` |
| L5 | `environment.yml` unbuildable: dead `fragments_h5@v2.10.1` pin (tag deleted from the remote), 7 undeclared binaries, no `pytest`, `name: base`. Also broke `docker build`. | whole suite | `d52172c`, `a731fc3` |
| L6 | `test_formats.py` hardcoded `/home/nboley/...` — a different user's home directory. | 13 tests | `a731fc3` |
| L7 | `test_fragment_array.py` called `build_fragments_h5(..., fasta_file=)` and `from_fragments_h5(..., include_fragment_strand=)`; neither parameter exists. | 4 errors | `2bf5018` |
| L8 | Two fragment expectations were stale — they asserted the absence of a fragment that is present in the BAM (mapq 60, proper pair, not duplicate) *and* in the h5. The tests were asserting an old library bug. | 2 tests | `f672a7c` |
| L9 | `test_jitter` asserted exact equality against a hand-computed dense slice — not an invariant, it held only when no fragment straddled the boundary. | 1 test | `5f5c30f` |

## 11.3 Dead code removed

| What | Why | Commit |
|---|---|---|
| `RegionFragmentArray.from_frag_bed` + `FragmentBedReader`, `MethylFragmentBedReader`, `FragmentBigBedReader`, `FragmentBedWriter` | Zero callers in the library, the tests, or any of the four dependent repos; also unreachable via `get_default_reader_class`, which is a hardcoded chain naming six other classes. Their 16 tests were **ported** to `from_fragments_h5`, not deleted. | `5f5c30f` |
| `ref_path`, `region_mask` | Both dead, both broken; orphaned `ravel` references | `52024b6` |
| Two commented-out `ravel` imports | Last consumers removed | `52024b6` |
| `set_binary_label`, `set_binary_label_by_thresholds`, `downsample_stratified_by_label` + 3 tests | The binary-labeling API (R2/R6/R7). Broken by default and unused — see below. | `7a7220c` |

**528 lines deleted, no behaviour change.**

### The binary-labeling API (R2, R6, R7)

`set_binary_label` called `self.drop_unlabeled_records`, which is defined
nowhere in the codebase. The *parameter* of the same name defaults to `True`,
so the **default path raised `AttributeError`**, and
`set_binary_label_by_thresholds` reached it by delegation — both public entry
points were broken unless the caller explicitly passed `False`. Two further
defects in the same method were silent: `label_column` was accepted and then
ignored (the body hardcodes `self["label"]`), and `inplace` was ignored for the
labeling itself, which always mutated `self`.

Zero call sites across `biomarker`, `biomarker-pipeline` and
`biomarker-projects`, and none in the library beyond the internal delegation.
It survived because the only tests exercising it live in `test_dataframe.py`,
which cannot collect (§9.3) — **the broken default had never run.**

Implementing was the alternative, and the test contract is explicit enough to
code against (drop rows where `label == -1`, honour `inplace`). It was rejected
because that contract **cannot be executed**: the assertions pin exact counts
from the unrecoverable fixture, so the code would have been written against a
specification nothing can check — the same shape as the false "17/17 passing"
claim in §8.

`label_balanced` and `get_indices_of_balanced_labels` were **kept**: they take
an arbitrary column name, work correctly, and have doctests. They are
label-adjacent by name only.

## 11.4 Current state

```
pytest test/ --ignore=test/test_dataframe.py   ->  2 failed, 240 passed
```

The 2 failures are missing data only: `test_slice_encode_big_wig` (encode
bigwig) and `test_get_one_hot_encoded_sequence` (the in-package GRCh38
reference). `test_dataframe.py` (26 tests) still cannot collect — its primary
fixture is unrecoverable (§9.3), and its assertions pin values derived from
that exact file (§9.4).

Highest-value remaining work, in order:

1. **The two remaining domain-owner findings** — S5 (median returned under the name `mean_fragment_counts`) and S4 (duplicate sample_ids silently dropped). Both alter scientific output and remain unchanged. R5 was a third until it was fixed (§11.7); R12 was a fourth until the method containing it was deleted.
2. **`test_dataframe.py`** — 26 tests, blocked on the synthetic-fixture decision in §9.4.
3. **The remaining always-raising methods** — I13 in particular is another orphaned `ravel` reference and may simply be deletable, as the BED layer and the labeling API were.

One latent item found while deleting the labeling API and left alone: the
doctest on `get_indices_of_balanced_labels` fails under numpy 2.x, which
renders `np.int64(1)` where the docstring expects `1`. It is pre-existing
(reproduces identically against the pre-deletion file) and invisible to the
suite, which does not run `--doctest-modules`.

## 11.5 `parallel_apply` — rewritten under a three-round review loop

B2–B5 were the largest untouched risk in §11.4 of the previous revision. The
hang was confirmed by execution first: killing a worker mid-run left the parent
waiting forever (`timeout` exit 124), because the dead worker had already
claimed its row from the shared counter, so no other worker would process it
and the loop could never be satisfied.

Replaced with `ProcessPoolExecutor`, which supervises its workers and raises
`BrokenProcessPool`. This **deleted** the hand-rolled counter, pipe, lock and
polling loop rather than adding guards to them. The one good property of the
old design is preserved — a `fork` context means children inherit the frame
copy-on-write and it is never serialized, so `fn` may be a lambda.

| Round | Grade | Outcome |
|---|---|---|
| 1 | **C+** | Found a **thread-safety regression I introduced**: passing state via a module global replaced the old per-`Process` arguments, so concurrent calls from threads silently returned each other's data. Fixed by moving state into `initializer`/`initargs`, which under `fork` is still not pickled. |
| 2 | **A-** | Independently re-verified round 1's fixes hold. Found mixed return types silently corrupting output and `None` returns producing a cryptic pandas error. Both now rejected with named errors. |
| 3 | **A-** | Found the round-2 guard did not cover its own stated contract — mixed Series/dict slipped through. Guard generalized to compare return *kinds*. |

Tests went from **zero to 21**. The new suite hangs against the old
implementation (exit 124), so it genuinely exercises the defect it was written
for.

The round-1 finding is the important one. A rewrite motivated by a hang
introduced a *silent corruption* — strictly worse — and it was caught only
because a reviewer with no stake in the design went looking. The same
verify-don't-assume discipline this document argues for applies to the
document's own remediation work.

## 11.6 What this exercise demonstrated

The review's own §8 note argued that acknowledged uncertainty beat unverifiable
confidence among the reviewing agents. Execution made that concrete:

- **Two static Criticals were wrong.** B1 was overstated (measured: 22 of 30 operations *are* caught). The B1 agent reported it "REPRODUCED" when the module could not even be imported.
- **Nine real defects were invisible to static review**, including one where the package was broken against its own declared numpy range.
- **The single highest-value fix was not a code change at all** — four undeclared conda packages took the suite from 63 failures to 34, with no code and no fixtures touched.
- **One "library regression" was the opposite.** The fragment-count failures looked like 2.13.3 breaking inclusion; the fragment turned out to be present in the BAM *and* the h5, clean by every criterion, and absent only from a hardcoded array. The tests were asserting a bug that had since been fixed.

The general lesson, and the reason §1 outranked everything: **a codebase with no feedback loop accumulates defects that no amount of reading will find.**

## 11.7 R5 — the fragment weight mask (fixed with owner approval)

Each `pred_dist.*` column is a weight track for one (strand, fl_band,
coverage_type) combination. A fragment should receive that track's weights only
if it is in the band **and** on the strand. The mask used OR.

The length line turned out to be **accidentally correct**: `&` binds tighter
than `|`, so `mask | (lengths >= lb) & (lengths <= ub)` parses as
`False | in_band`, and the leading `mask |` is a no-op that merely *reads* as
accumulation. That no-op is most likely what disguised the real defect on the
next line, by making the two conditions look symmetric and deliberate.

Measured, band 40-65, strand `+`:

```
lengths [50, 50, 200, 200], strands [+, -, +, -]
after length line : [ T  T  F  F]
after strand line : [ T  T  T  F]   <- OR
intended          : [ T  F  F  F]   <- AND
```

A 200 bp fragment, far outside the band, was selected for the `(+, 40-65)`
track purely because it was on the plus strand. The loop overwrites
`attr[mask]` once per combination, so the weight a fragment ended up with
depended on **iteration order**.

The strandless (`.`) branch skips the strand clause entirely, leaving the band
as the only filter. That is correct, and is now pinned by a test so the fix
cannot silently change it.

**Blast radius was nil.** `set_fragment_array_weights` has zero callers in this
library and none in `biomarker`, `biomarker-pipeline` or `biomarker-projects`;
the sibling `_set_fragment_array_weights_from_pred_record` is dead behind an
`assert False`; and both import from `bias_correction/`, which `CLAUDE.md`
marks superseded. No published result can have been affected. An earlier
revision of this document implied otherwise — that was speculation stated as
fact, and it is corrected here.

Both copies were fixed, including the dead one, so the defect cannot return
with it. `test/test_fragment_weight_mask.py` adds 5 tests; against the pre-fix
code **4 fail and the strandless one passes**, which is the expected signature
of a fix that touches only the strand clause.
