# `dataframe.py` — load-bearing invariants and open findings

**Scope:** `fragmentomics_tools/dataframe.py` and its subclasses
(`DataFrameBase` -> `RegionDataFrame` -> `SampleAndRegionDataFrame`, and
`SampleDataFrame`).

This file holds two things: invariants that are true and easy to break again,
and findings that are still open. It deliberately carries **no status**
— no test counts, no "last reconciled" date, no closed-findings ledger.
Status inside a design document goes stale by construction: this file was
reconciled twice and went stale both times, the second time reporting four
fixed findings as open. A reader who trusts a stale ledger re-investigates
closed work.

Current test counts live in `CLAUDE.md`, which also warns that they move with
almost every commit. The history of the original 57 findings is in git.

---

## Invariants

CLAUDE.md carries the repo-wide traps. These are specific to this module and
each one has already cost something.

**Class-level defaults are tuples on the base and lists on subclasses, and
that asymmetry is deliberate.** `DataFrameBase` uses `_metadata = ()` so a
mutable default cannot be shared; subclasses use `_metadata = ["ref"]` because
pandas concatenates these during propagation and `tuple + list` raises
`TypeError`. A review once called the inconsistency cosmetic and recommended
unifying it; applying that broke 69 library and 25 `background_model` tests.

**Overlap joins return whole-A intervals, not geometric intersections.**
`intervals.overlap_indices` reports which A rows overlap which B rows — it is
bedtools `-wa -wb`, not a clip. Nothing in the library returns the geometric
intersection, and the overlap *length* (`overlap_bases`) is provided instead
because no caller has needed the clipped coordinates.

**Valid-mode binning is centred, not start-anchored.**
`bin_regions_into_windows` must distribute the leftover remainder equally at
both ends. Callers centre regions on a feature
(`center_on_summit().resize_regions(...)`) and then bin; start-anchoring drops
the whole remainder off the right and silently decentres every meta-profile.
This broke once and was caught only because a guard test pins exact
coordinates — window *counts* are identical under both anchorings, so any
count-based or bounds-based assertion passes either way.

**`parallel_apply` must be called from the main thread** and raises if not. It
also must not leave a `tqdm` monitor thread behind. tqdm stores its monitor on
the class that built the bar, so `tqdm.auto` and `tqdm.notebook` keep their
own; checking only `tqdm.std.tqdm.monitor` misses the notebook case, which is
where essentially every caller runs.

---

## Open

**`parallel_apply` residual fork exposure.** The guard stops *us* forking from
a non-main thread. It cannot stop a lock being held by a thread we did not
start, so a caller running its own `ProcessPoolExecutor` concurrently can
still deadlock a worker. Demonstrated, not theoretical. Closing it means
abandoning `fork`, and with it lambda support and copy-on-write frames.
Recorded in the `parallel_apply` docstring.

**`groupby().first()`, `describe()` and `.T` raise.** pandas reconstructs the
subclass with a reduced column set, and the required-columns assert in
`DataFrameBase.__init__` fires. `groupby().size()` and `groupby().agg()` are
unaffected. This predates the layering work and is present on `main`. A
candidate fix is the geopandas constructor-fallback shape — `_constructor`
returns a plain `DataFrame` when required columns are absent rather than
asserting — but that pattern is **unverified here**, because geopandas is not
installed.

---

## Not in this file

**Fragment orientation.** `from_fragments_h5` reverses coordinates and swaps
strands for minus-strand regions, and flip must precede concat in the `gc`
path. That invariant is real and expensive, but it belongs to
`fragment_array/`, not here; its guard is
`test/fragment_array/test_gc_alignment.py`. It matters for the annotation
protocol work — see the handoff.

**Previously tracked findings S3-S9.** S3, S4 and S5 are fixed in the code.
S6-S9 were low severity, were never re-verified, and named a `SampleDataFrame`
that has since shrunk to three methods, so they cannot be carried forward as
findings. If any of them still matters it needs re-finding against the current
class, not reviving from this list.
