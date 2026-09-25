# `dataframe.py` — open findings and load-bearing invariants

**Scope:** `fragmentomics_tools/dataframe.py` and its subclasses
(`DataFrameBase` -> `RegionDataFrame` -> `SampleAndRegionDataFrame`, and
`SampleDataFrame`).
**Last reconciled:** 2026-09-25, against `cf5d3f9`.

This file used to be a 1000-line review ledger. Two remediation passes closed
almost all of it, so what remains here is the part that is still actionable:
what is **open**, and what is **true and easy to break again**. The full
history — 57 original findings, every commit that closed one, and the
investigations behind them — is in git. Start with
`git log --oneline a73ea74..cf5d3f9` and the review commits before it.

---

## Current state

```
make test     ->  2 failed / 331 passed  (~40s)
```

Both failures are missing data, not defects:
`test_slice_encode_big_wig` needs an ENCODE bigwig, and
`test_get_one_hot_encoded_sequence` needs the in-package GRCh38 reference.

Always run via `make test`, never bare `pytest` — see CLAUDE.md for why (this
suite can hang rather than fail, and once did so for 12 hours).

---

## Still open

Everything below alters behaviour or scientific output, which is why none of
it was changed unilaterally. The first three need a domain-owner decision.

| ID | Where | Finding |
|---|---|---|
| **S5** | `dataframe.py:2160` | `get_sample_count_bounds` computes a **median** and names it `mean_fragment_counts`. That value drives `min_fragments`/`max_fragments`, which `filter_outlier_counts` uses to exclude samples. The computation may well be the intended robust choice; the *name* misleads every reader and every downstream consumer of the column. Decide explicitly: rename to `median_*`, or switch the computation. |
| **S4** | `FlDist.init_from_sdf` | Duplicate `sample_id`s are silently dropped — last write wins, earlier fragment-length distributions discarded without warning. Raise, or aggregate explicitly. |
| **S3** | `dataframe.py:2321` | `SampleDataFrame.__init__` inspects only `iloc[0]` when deciding whether to build `FlDist`. A mixed column `[handle, str]` passes the guard then fails inside `FlDist.init_from_sdf`; `[str, handle]` fails the guard and silently drops `fl_dist` for *all* samples. (This was previously recorded as "`main` only". That distinction is gone — the branches are merged and it is in the working tree.) |
| S6-S9 | `SampleDataFrame` | Low severity: opaque `KeyError` on a missing sample, a per-group `assert`, `IndexError` on an empty list, and no close path for h5 handles. Carried forward from the original review and **not re-verified** in the latest pass. |
| — | `parallel_apply` | **Residual fork exposure.** The guard stops *us* forking from a non-main thread; it cannot stop a lock being held by a thread we did not start. A caller running its own `ProcessPoolExecutor` concurrently can still deadlock a worker. Demonstrated, not theoretical. Closing it means abandoning `fork`, and with it lambda support and copy-on-write frames. Documented in the `parallel_apply` docstring. |
| — | `get_indices_of_balanced_labels` | Its doctest fails under numpy 2.x, which renders `np.int64(1)` where the docstring expects `1`. Pre-existing and invisible to the suite, which does not run `--doctest-modules`. |

**Closed without action:** required-column validation (`B1`) was measured
rather than argued — across 30 operations, 22 are caught and 8 leak, every
leak requiring a deliberate rename or in-place deletion of a required column.
Downgraded Critical -> Low and closed.

---

## Invariants that will be re-broken if nobody writes them down

CLAUDE.md carries the repo-wide traps. These are specific to this module and
each one has already cost something.

**Class-level defaults are tuples on the base and lists on subclasses, and
that asymmetry is deliberate.** `DataFrameBase` uses `_metadata = ()` so a
mutable default cannot be shared; subclasses use `_metadata = ["ref"]`
because pandas concatenates these during propagation and `tuple + list`
raises `TypeError`. A review once flagged the inconsistency as cosmetic and
recommended unifying it — applying that broke 69 library and 25
`background_model` tests.

**`join_on_overlap` returns whole-A intervals, not geometric
intersections.** It is bedtools `-wa -wb`, and always has been. The old name
`intersect_with_rdf` claimed otherwise and now raises, pointing at the new
one. Notebooks were deliberately not updated so they break loudly.

**Valid-mode binning is centred, not start-anchored.** `bin_regions_into_windows`
must distribute the leftover remainder equally at both ends. Callers centre
regions on a feature (`center_on_summit().resize_regions(...)`) and then bin;
start-anchoring drops the whole remainder off the right and silently
decentres every meta-profile. This was broken once and caught only because a
guard test pins exact coordinates — window *counts* are identical under both
anchorings, so any count- or bounds-based assertion passes either way.

**Flip must precede concat in the `gc` path.** The per-member reversal is
handled by construction only while that order holds; reordering them
reintroduces the misalignment silently. There is a test pinning it.

**`parallel_apply` must be called from the main thread** and raises if not.
It also must not leave a `tqdm` monitor thread behind — and note that tqdm
stores its monitor on the class that built the bar, so `tqdm.auto` and
`tqdm.notebook` keep their own. Checking only `tqdm.std.tqdm.monitor` misses
the notebook case, which is where essentially every caller runs.

---

## Two results worth carrying forward

**Static review has a blind spot that execution does not.** Nine defects were
found by running the code that five Opus reviewers reading the same files did
not find: a CLI flag absent from the installed tool, a numpy API removed in
2.0, a `NameError` in a helper, a test whose outcome depended on execution
order, and an API drift in `fragments_h5`. None are visible by reading; all
are obvious on execution.

**Reproducing a plausible failure is not diagnosing the actual one.** The
`parallel_apply` deadlock had four proposed mechanisms. Three were wrong —
including one that was reproduced *deterministically in a purpose-built
harness*, which felt like confirmation and was not; the live capture later
showed that thread sitting idle while the process deadlocked elsewhere. What
worked was catching a real occurrence alive and running `py-spy` on it. For a
bug that is rare but reproducible, instrument a real occurrence before
building a harness for a guess.
