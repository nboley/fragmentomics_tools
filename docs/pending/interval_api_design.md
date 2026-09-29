# The interval API

Design for the interval-algebra layer of `fragmentomics_tools`: the set
operations over region frames, the backend they call, and the API that replaces
the thirteen `bedtools`-backed methods on `RegionDataFrame`.

Self-contained and implementable on its own. It is a prerequisite for
[`dataframe_layering_design.md`](dataframe_layering_design.md), which covers the
layering, the annotation protocol and the later phases.

## Requirements

Binding constraints. These are owner decisions, recorded here because a design
that drifts from them is wrong even if it is internally coherent.

1. **This is a library, and it should be generally useful** — shaped around
   interval work in general, not around the current project's immediate needs.
2. **Correctness before restructuring.** A new test that fails against
   production code is a finding to report, not something to patch away.
3. **The useful interactions are two.** Joining region sets on overlap — with
   wiggle, join types, and transitive grouping — and statistics over those
   overlaps.
4. **Statistics are `groupby` at the call site, not API surface.** The API's job
   is to return something a `groupby` can reduce, not to enumerate reductions.
5. **Backend arguments never appear in our signatures.** Our schema is an
   invariant, not a parameter.
6. **Deletions must not withdraw capability.** Every method removed has to
   survive as an expression over what replaces it.
7. **Differential fixtures cover code we implement, not code we delegate.**
   Pinning a third-party library against another one tests neither of ours.

Explicitly **out of scope**, each considered and declined:

| Out | Why |
|---|---|
| enrichment and null models — `shuffle`, `fisher`, `jaccard`, `reldist` | a coherent module of its own; not wanted at this point |
| multi-file `annotate`-style operations | easily composable at the call site |
| aggregating B's data columns onto A (`bedtools map`) | `groupby` already does this |
| per-base coverage vectors (`genomecov`) | the library has domain-specific fragment coverage |
| genome-end clamping (`slop`/`flank`) | not needed |
| geometric intersection | nothing needs the clipped coordinates; overlap *length* is enough |

## `pybedtools` — a decision, not an open question

**Thirteen methods reach `pybedtools`**, which shells out to the `bedtools`
binary. The dependency crosses three buckets — interval algebra, the two
cross-layer methods, and the fragment-coverage pair. Derived by AST call-graph
closure over the class:

```
merge_regions ────────────────────► BedTool.from_dataframe().sort().merge()

join_on_overlap ──────────────────► BedTool.from_dataframe().intersect()
      ▲
      ├── attach_blacklist_regions            [cross-layer]
      ├── drop_overlapping_regions            [cross-layer]
      └── intersect_with_bed
                ▲
                └── get_overlapping_base_counts
                          ▲
                          ├── overlaps_with_bed ──────► overlaps_with_beds
                          └── bases_overlap_with_bed ─► bases_overlap_with_beds

from_beds_merged ─────────────────► BedTool(path).filter().cat()   [construction/IO]

_get_fragment_coverage_sum ───────► BedTool(path).intersect()      [fragment coverage]
      ▲
      └── get_fragment_coverage_sum
```

**Four** methods touch `pybedtools` directly; the other nine inherit it.
Swapping four function bodies therefore migrates the whole surface, and the
differential test only has to pin those four.

Three methods look like they belong here and do not:

| Method | Reality |
|---|---|
| `sort` | pure `self.sort_values([...])`. The only `pybedtools` token is the docstring phrase "same as pybedtools", so text scans flag it falsely. |
| `overlaps_rdf` | genuine second implementation of overlap, via `get_interval_dict` + `IntervalTree`. The "two implementations of one rule, free to drift" hazard CLAUDE.md names — and proof a non-`bedtools` path is viable. |
| `intersect_with_rdf` | a stub raising `AttributeError`. Dead. |

The binary is "frequently *not* on `PATH` in sandboxes and AWS Batch
containers, and this has caused real failures" (CLAUDE.md). At 13 methods the
dependency is effectively total, contradicting the central promise that layer 2
imports cleanly.

**Measured, 2026-09-26** (pybedtools 0.12.1, bedtools 2.31.1):

- `bedtools` is in the conda env's `bin/` but **not on `PATH`** by default —
  which is why the suite must prepend it.
- **Every set operation spawns the binary.** Observed via a `subprocess.Popen`
  spy: `sort` 1, `merge` 2 (it sorts first), `intersect -wa -wb` 1,
  `intersect -wao` 1. The compiled `cbedtools` extension is real but only
  covers interval parsing and iteration — 0 subprocesses — so it does not
  cover anything in layer 2.
- **The failure is latent and the message is misleading.** Import and
  `BedTool` construction both succeed without the binary; the error arrives at
  call time as `NotImplementedError: "sortBed" does not appear to be installed
  or on the path`. It names the *legacy* binary (`sortBed`, absent from modern
  bedtools and from this codebase), claims "not installed" when the binary is
  merely off `PATH`, and is an error type that reads as unfinished library
  code rather than a broken environment.
- **Worst property: the disabling is decided at `import` time.** Setting
  `PATH` after `import pybedtools` does not re-enable the method — verified by
  fixing `PATH` post-import and observing the same raise. So the obvious
  container fix (export `PATH` before the call) silently fails whenever
  anything imported pybedtools earlier. That is an ordering-dependent failure
  with an error that points at the wrong cause.

This is a stronger case for replacement than "it needs a binary": it needs the
binary *on `PATH` before first import*, and misreports when it isn't.

**DECIDED: replace `pybedtools` with `bioframe`.**

Rejected alternatives: keeping `pybedtools` (leaves the binary dependency and
the import-ordering hazard below); `pyranges` (ruled out on shape — see the
table); and implementing the algebra directly (most work, most divergence
risk against established semantics).

`RegionDataFrame` stays a `pandas.DataFrame` subclass. `bioframe` is a
function library, not a container, so nothing about the class hierarchy
changes — only what the interval functions call underneath. The duplicate
overlap implementation collapses into it too: the IntervalTree path goes.

The decision rests on three independent measurements below — semantics,
dependency behaviour, and speed. Any one of them alone would be a weak case;
together they point the same way.

**Measured 2026-09-26** (bioframe 0.8.0, pyranges 0.1.4, pandas 3.0.6, in a
throwaway `/tmp` venv).

*Can either be `RegionDataFrame`'s base?*

| check | bioframe | pyranges |
|---|---|---|
| API shape | plain **functions** over DataFrames | its own container class |
| subclasses `pandas.DataFrame`? | n/a | **No** |
| DataFrame subclass survives an op? | **No** — downcast to `DataFrame` | n/a |
| accepts our column names? | **Yes**, `cols=('contig','start','stop')` | imposes its own |

**pyranges is ruled out as a base**: `PyRanges` is not a `DataFrame` subclass,
so adopting it would reverse the "stay a subclass" decision above. `bioframe`
is not a base either — it is a function library — which is the outcome we
want: `RegionDataFrame` stays a `pandas.DataFrame` subclass and the interval
methods call into it. Its downcast means re-wrapping results as
`type(self)(...)`, which is exactly what the current code already does with
`pybedtools`' `to_dataframe()`, so it is not a regression. That `bioframe`
takes `cols=('contig','start','stop')` natively removes any renaming layer.

*Do the boundary semantics match?* Differential-tested against
`bedtools 2.31.1`. **No disagreements found.**

| case | merge: bedtools vs bioframe |
|---|---|
| book-ended `[0,10)+[10,20)` | both merge to `[0,20)` |
| 1 bp gap `[0,10)+[11,20)` | both keep separate |
| share 1 bp `[0,10)+[9,20)` | both merge |
| nested `[0,100)+[10,20)` | both `[0,100)` |
| duplicates, single-base, zero-length, two contigs | identical |

`subtract` also matches exactly. The critical case — `intersect -wao`, which
backs `get_overlapping_base_counts` — reproduces bedtools' per-region base
counts exactly **including zeros for non-overlapping regions**, via
`bioframe.overlap(how="left")` plus a clipped `min(end)-max(start)`.

*Under the hood:* both libraries are **pure Python** — no `.so` files in
either — and `merge`/`overlap`/`subtract` spawn **zero** subprocesses. That
removes the binary, the `PATH` ordering hazard, and the import-time method
disabling in one step.

*Performance.* Benchmarked DataFrame-in / DataFrame-out, which is how this
codebase calls it. Random intervals across chr1-22, best of 3:

| n | merge (pybedtools -> bioframe) | intersect `-wao` |
|---:|---|---|
| 1,000 | 0.014s -> 0.030s (**0.5x**) | 0.016s -> 0.018s |
| 10,000 | 0.045s -> 0.034s (1.4x) | 0.061s -> 0.028s (2.2x) |
| 100,000 | 0.316s -> 0.094s (**3.4x**) | 0.506s -> 0.137s (**3.7x**) |
| 500,000 | 1.511s -> 0.541s (2.8x) | 2.630s -> 0.882s (3.0x) |

bioframe is roughly **3x faster** at every realistic scale, losing only below
~5k regions where the absolute gap is 14 ms vs 30 ms. The cause is not that
bedtools' C is slow: `pybedtools` serializes the frame to a temp BED file,
spawns a process, and parses the output back on *every* call, and that
round-trip dominates. Read this as specific to our usage — for data that
already lives in BED files and stays there, raw bedtools compares far better.

**Strand-aware behaviour — RESOLVED, and it diverges.** Measured 2026-09-26,
all nine `(A strand, B strand)` combinations for `intersect -u`:

| A | B | bedtools `-s` | bioframe `on=['strand']` | |
|---|---|---|---|---|
| `+` | `+` | True | True | agree |
| `+` | `-` | False | False | agree |
| `+` | `.` | False | False | agree |
| `-` | `+` | False | False | agree |
| `-` | `-` | True | True | agree |
| `-` | `.` | False | False | agree |
| `.` | `+` | False | False | agree |
| `.` | `-` | False | False | agree |
| **`.`** | **`.`** | **False** | **True** | **DIVERGE** |

Eight of nine agree. The exception is strandless-vs-strandless: bedtools `-s`
treats `.` as *no strand*, so two strandless features never satisfy "same
strandedness". bioframe's `on=['strand']` is an equality join and `"." == "."`
is true. The result inverts — no matches becomes all matches — on the input
shape this codebase uses by default, since `Region(strand=".")` normalises to
`None` and the ordinary path is strandless.

**Not a live bug.** No library code passes a strand flag to bedtools; the only
`same_strand` references are in `plot/tracks.py`, which compares in pandas.
The migration is safe as things stand.

**But a landmine.** `merge_regions(self, **kwargs)` forwards arbitrary kwargs
straight to `bedtool.merge(**kwargs)`, so `s=True` is already reachable by a
caller. The first person to add strand-aware overlap after the migration would
get an inverted result on the common case, with no error. So: **do not
implement `-s` as `on=['strand']`.** Same-strand must exclude `.` explicitly,
and that rule needs a test pinning `.` vs `.` to *no match*.

`-S` (opposite strand) has no bioframe equivalent and was not tested. Also
unused.

### The two non-algebra `pybedtools` sites — both in Phase 2 scope

Interval algebra is not the whole dependency. Two further call sites use
`pybedtools` for I/O-shaped work, and while the migration removes the `import`
they must go too — otherwise Phase 2 delivers **no import-weight reduction and
no escape from the `bedtools`-on-`PATH` hazard**, which are its two stated
benefits. Both are in scope.

**1. `from_beds_merged` — migrate, and drop `bed_filter_callback`.**
It calls `BedTool(f).filter(cb)` per file, then
`.cat(*rest, postmerge=True, force_truncate=True)`. The `cat`/`postmerge` half
is plain concat-then-merge and maps onto `bioframe.merge` directly.

The blocker is supposedly `bed_filter_callback`, documented as taking a
pybedtools filter function, which makes the dependency contractual rather than
implementational. **Measured: it is contractual on paper only.** Zero callers
pass it — across four repos, 0 `.py` call sites and 0 notebook *source* cells
(the handful of notebook hits are output cells, the known false-positive class
in this repo). `from_beds_merged` itself has no external `.py` caller either.

It is worse than unused. The docstring's own example calls
`from_beds_merged(in_bed_files, bed_filter=bed_filter)` while the parameter is
named `bed_filter_callback`, so anyone who copied the documented usage got
`TypeError`. **A contract nobody has ever successfully invoked does not
constrain the migration.** Delete the parameter; if a filter is wanted later,
re-add it as a predicate over DataFrame rows.

**2. `_get_fragment_coverage_sum` — migrate, but this one has a real
constraint.** It runs `BedTool(in_fname).intersect(BedTool.from_dataframe(self),
wa=True, wb=True)` where `in_fname` is a *fragment-level* BED. bedtools
**streams** that file; bioframe needs it as an in-memory DataFrame, and the
whole cfDNA fragment set will not fit. So this is a genuine memory tradeoff,
not a semantic one, and the swap is not a one-liner: it needs a chunked read
with a per-chunk overlap and accumulation into `counts_vect`. The output is
only a per-region count, so chunking is exact — no cross-chunk state beyond
the running sum. Callers are `get_fragment_coverage_sum` and one test, so the
blast radius is small.

**Prerequisite for Phase 2, and it is a small one.** `bioframe` is declared in
none of `environment.yml`, `pyproject.toml`, `recipe/recipe.yaml` or
`requires.txt` — all four of which list `pybedtools` — and is installed in no
env on this host. Adding it is a one-line change per manifest:
**`bioframe 0.8.0` is packaged on bioconda** (`pyhdfd78af_0`, noarch), and
`bioconda` is already a declared channel in `environment.yml`, immediately
above where `pybedtools` is pulled from. No pip section, no vendoring, no
index pinning. It is *not* on conda-forge, so the channel matters.

The benchmark and strand numbers above were measured in a throwaway venv;
re-run them in `biomarker_env` once the dependency lands, before relying on
them.

## The interval API — DECIDED

Layer 2 exposes **five free functions**. Everything the current thirteen
methods do is either one of these or a pandas expression over one of them.

```python
# One primitive. Returns index pairs, never joined columns.
overlap_indices(a, b, *, how="inner", wiggle=0,
                min_frac_a=0.0, min_frac_b=0.0, reciprocal=False,
                same_strand=False) -> DataFrame[a_index, b_index, overlap_bases]

nearest(a, b, *, k=1, ignore_overlaps=False, direction=None,
        same_strand=False)        -> DataFrame[a_index, b_index, distance]

cluster(a, b=None, *, wiggle=0, same_strand=False) -> Series[int]
merge(a,   *, wiggle=0, same_strand=False)         -> RegionDataFrame

overlaps(a, b, *, wiggle=0, same_strand=False)     -> Series[bool]
```

**`overlaps` is deliberately redundant.** It is
`a.index.isin(overlap_indices(a, b).a_index)` and nothing more. It exists
because it is the most-used operation in the family — 22 call sites across 11
notebooks — and that composition is too noisy to write at each of them. This is
an ergonomics exception to the "one way to do it" rule, recorded as an
exception rather than dressed up as a principle. No other reduction gets one.

**Why index pairs are the primitive.** Today `join_on_overlap` stuffs
`self.index` into a column, serialises to BED text, and reconstructs the index
on the way back — along with `reordered_columns`, suffix collision handling and
column renaming. All of that machinery exists *only* to carry columns through a
backend that cannot hold them. A primitive returning
`(a_index, b_index, overlap_bases)` deletes it: column carrying becomes one
pandas `.join`, performed where we control the semantics. It also collapses the
differential-test surface to a single function returning integers, which is far
less brittle to pin than a DataFrame with backend-dependent column order.

**Everything else is pandas.** These are call-site expressions, not API:

| Need | Expression |
|---|---|
| B's column values for each hit | `b.loc[idx.b_index]`, reindexed onto `idx.a_index` |
| boolean mask | `overlaps(a, b)` |
| blacklist / non-overlapping | `overlap_indices(a, b, how="anti")` |
| overlapping bases per region | `.groupby("a_index").overlap_bases.sum()` |
| widest single overlap | `.groupby("a_index").overlap_bases.max()` |
| count of hits per region | `.groupby("a_index").size()` |

**Joins preserve provenance; set algebra does not.** This is the boundary that
decides what may return an index-aligned result:

| | Produces | Index-aligned result possible? |
|---|---|---|
| `overlap_indices`, `nearest`, `overlaps`, `cluster` | existing rows, or labels for them | **yes** |
| `merge` | new intervals | **no** |

`cluster` is the index-preserving sibling of `merge`: same grouping, but it
returns a label per input row instead of collapsing rows, so the caller keeps
the ability to say *which* originals formed a group.

**`cluster` is transitive, and that is why it is not a join type.** Given
`A=[100,200)`, `B=[150,300)`, `C=[250,400)`, A and C do not overlap but both
reach B. A self-join yields the two edges `(A,B)`, `(B,C)`; `cluster` puts all
three in one group. Overlap is a *relation*, compactly encoded as pairs;
transitive overlap is an *equivalence*, compactly encoded as labels. Forcing
the latter into the pair contract would emit k² pairs for a component of size k
and leave `overlap_bases` undefined for indirectly connected pairs. Passing `b`
gives the transitive grouping across two frames.

**Scope rules.**

- **Inputs are frames, never paths.** Callers load with `from_bed` explicitly.
  This is what dissolves `intersect_with_bed`, which combined I/O with algebra
  and belonged to neither layer.
- **Returns are index-aligned.** Never a bare ndarray, dict, or list.
- **No backend arguments in any signature.** `cols1`, `cols2`, `on`,
  `suffixes`, `return_input`, `keep_order`, `ensure_int` are implementation
  detail. Our schema is an invariant, not a parameter. This also removes the
  `**kwargs` passthrough that makes `s=True` reachable today.
- **One word per concept.** `wiggle` means the same thing in `overlap_indices`,
  `overlaps`, `cluster` and `merge`.

**`nearest` is not `overlap_indices(wiggle=n)`.** Wiggle answers *whether*
something is within range; `nearest` answers *which* and *how far*, with a
signed strand-aware distance. Neither substitutes for the other.

### Net change to the thirteen

Every method is named rather than paraphrased, so the table can be checked
against the thirteen by name instead of by counting.

| | Methods | of the 13 |
|---|---|---:|
| **Deleted** | `get_overlapping_base_counts`, `overlaps_with_bed`, `bases_overlap_with_bed`, `overlaps_with_beds`, `bases_overlap_with_beds`, `intersect_with_bed` | 6 |
| **Renamed / moved** | `join_on_overlap`→`overlap_indices`, `merge_regions`→`merge`, `drop_overlapping_regions`→ sugar for `how="anti"` | 3 |
| **Rewritten** | `attach_blacklist_regions` — see below | 1 |
| **Migrated in Phase 2, signature unchanged** | `from_beds_merged` (drops `bed_filter_callback`), `_get_fragment_coverage_sum` (chunked read), `get_fragment_coverage_sum` | 3 |
| | | **13** |

Two more methods change without being among the thirteen, because neither
reaches `bedtools` today:

| | Methods |
|---|---|
| **Renamed** | `overlaps_rdf`→`overlaps` — the `IntervalTree` implementation |
| **Deleted** | `intersect_with_rdf` — already a stub that raises |

And **`nearest` and `cluster` are added**, replacing nothing.

`overlaps_rdf`'s rename carries a signature change: its `max_distance`
parameter becomes `wiggle`, for the one-word-per-concept rule. All 22 call
sites pass it positionally or not at all, so no caller breaks on the parameter
name — but the rename is real and belongs in the release notes alongside the
method rename.

**`attach_blacklist_regions` is the one method the primitive does not serve for
free, and it is worth being precise about why.** It does not merely test for
overlap: it reads B's *coordinate values* out of the join result
(`contig_{rsuff}`, `start_{rsuff}`, `stop_{rsuff}`) to build a `Region` per
overlapping blacklist interval. `overlap_indices` returns no B columns, so its
~30-line body must be adapted — not deleted, and not carried over unchanged:

```python
idx = overlap_indices(rdf, blacklist)
hits = blacklist.loc[idx.b_index]           # B's columns, via the returned index
hits.index = idx.a_index                     # realign onto A
regions = hits.groupby(level=0).apply(to_region_list)
```

This is the general pattern for any consumer that needs B's *values* rather
than a count or a mask, and it is the honest test of the primitive: one real
call site needs B's columns, and an index plus a `.loc` is enough to serve it.
It also fixes an existing inconsistency — the method currently mutates the
caller's frame on the empty-overlap path (`self["blacklist_regions"] = ""`)
while returning a new frame otherwise, so one call has two aliasing contracts
selected by the data. The rewrite returns a new frame on both paths.

All seven deletions are dead. Measured over `fragmentomics_tools`,
`biomarker-pipeline`, `biomarker-projects` and `flgc` — 1022 notebooks, source
cells only, with the four `.claude/worktrees/` copies of `dataframe.py` and all
`.ipynb_checkpoints` excluded, because both otherwise report this file's own
internal wrappers as external consumers. `overlaps_with_bed`,
`bases_overlap_with_bed` and both `*_beds` variants have **zero** callers
anywhere; `get_overlapping_base_counts`' only two callers are those dead
wrappers. Of the seven, only `intersect_with_rdf` is referenced externally, and
it already raises `AttributeError`, so those callers are broken today.

`overlaps_rdf` is the one heavily used member — **22 call sites across 11
notebooks**, all in `biomarker-projects`, at exactly 2 per notebook (they are
template-derived). That is why it survives as `overlaps` rather than being
replaced by the bedtools boolean.

**The deletion set is closed under its own call graph**, which is what makes it
safe to remove in one step: `overlaps_with_bed` and `bases_overlap_with_bed`
call `get_overlapping_base_counts`, which calls `intersect_with_bed`, which
calls `join_on_overlap`; the `*_beds` pair calls the singular pair. Nothing
outside the set calls into it. `_get_fragment_coverage_sum` is *not* part of
it — it invokes `pybedtools` directly and never routes through
`intersect_with_bed`.

The deletions also remove a latent defect without needing a fix.
`get_overlapping_base_counts` keys aggregation on `(contig, start, stop)`,
strand excluded, so any two rows sharing coordinates collide: the later
overwrites the earlier's entry and receives both rows' bases. A locus annotated
on both strands returns `overlaps = [False, True]` where both are `True`, and
counts `[0, 200]` where both are `100`. Index-aligned aggregation cannot express
this bug, so it disappears with the code rather than needing a patch.

### Mapping onto `bioframe`

Verified against bioframe 0.8.0 by introspection, not from documentation.
`bioframe` is called in three places and appears in no signature of ours.

| Ours | `bioframe` | We supply |
|---|---|---|
| `overlap_indices` | `overlap(..., return_index=True)`, `how ∈ {left,right,outer,inner}` | `contig`→`chrom` mapping; **column renaming — see below**; `how="anti"` as outer + null filter; fraction thresholds; `same_strand` |
| `nearest` | `closest(k=, ignore_overlaps=, ignore_upstream=, ignore_downstream=, return_distance=)` | signed-distance and direction convention |
| `cluster`, `merge` | `cluster(min_dist=, return_cluster_ids=)`, `merge(min_dist=)` | `b=None` handling, label alignment |

**The returned column names are ours to produce, not `bioframe`'s.** Verified
against 0.8.0: `return_index=True` emits `index` and `index_` — a bare name and
the same name with the default suffix — not `a_index`/`b_index`. And
`return_overlap=True` emits `overlap_start`/`overlap_end`, the clipped
coordinates, not a length. So `overlap_bases` is **computed by us** as
`overlap_end - overlap_start`, not read from the backend. This is small but it
is the kind of detail that silently produces a column of the wrong meaning:
`overlap_end` alone is a coordinate, and treating it as a count would be
wrong everywhere it is summed.

Two gaps confirmed by reading the source rather than assumed:

- **Fraction thresholds do not exist in `bioframe`.** No `min_frac`,
  `reciprocal` or equivalent token appears anywhere in the package, so
  bedtools' `-f/-F/-r` must be implemented on our side by filtering on the
  returned overlap length. This is straightforward but must not be forgotten —
  it is a silent behaviour gap, not an error.
- **`how="anti"` does not exist, and `bioframe` will not tell you so.**
  Measured: `bioframe.overlap(df1, df2, how="anti")` raises nothing. It
  validates `how` not at all, and an unrecognised value falls through to the
  non-inner branch, returning a **left join** — that is, every matching row,
  which is the exact complement of what `anti` means. A typo therefore returns
  the opposite result silently.

  **Therefore `overlap_indices` validates `how` itself, before the backend is
  called**, rejecting anything outside `{inner, left, right, outer, anti}` and
  handling `anti` by composition (outer, then filter to null `b_index`) rather
  than forwarding it. This is a one-line guard and it is not optional: it is
  the only thing standing between a typo and an inverted blacklist filter.

`same_strand` must **not** be implemented as `on=['strand']` — that is the
`.`-vs-`.` divergence measured above, which inverts the result on this
codebase's default strandless input. Implement it as equality excluding `.`,
with a test pinning `.` vs `.` to no match.

**The backend is replaceable.** Because the adapter is three functions wide,
the same API sits on `pybedtools` unchanged if `bioframe` proves unsuitable or
cannot be packaged. The `bioframe` decision carries a fallback rather than
being one-way.

## Phasing

All of this happens in a worktree, not on `main`.

**Phase 0 — differential fixtures, before anything changes.** Capture current
output on large, real region sets plus deliberate edge cases, so that
everything after this validates against a recorded baseline. Capturing them
first rather than just before the `bioframe` swap costs nothing and means
Phase 1 is covered too.

**Scope them to what we implement, not to what we delegate.** A fixture that
pins `bedtools merge` against `bioframe.merge` is testing someone else's
library, and it will fail on their next release for reasons that are not our
bug. The fixtures exist to protect the code *we* write. That means:

| Fixture-worthy — we implement it | Not fixture-worthy — delegated |
|---|---|
| `how` handling, especially `anti` by composition | the underlying inner/left/right/outer join |
| fraction thresholds (`min_frac_a/b`, `reciprocal`) — absent from `bioframe` | interval containment arithmetic |
| `same_strand` as equality **excluding** `.` — the divergence | plain strand equality |
| the `contig`/`chrom` schema mapping, index preservation and realignment | — |
| `overlap_bases` arithmetic where we compute rather than read it | — |
| `wiggle` semantics being identical across all four functions | — |
| `from_beds_merged` — concat-then-merge, and that dropping `bed_filter_callback` changes nothing | — |
| `_get_fragment_coverage_sum` — that chunked accumulation equals the unchunked result | the streaming read itself |

The last two are easy to forget because they are not interval algebra, but they
are migrated in Phase 2 and so need a Phase 0 baseline like everything else.
`_get_fragment_coverage_sum` needs its fixture most of all: chunking is the one
migration in this design that changes the *shape* of the computation rather
than its backend, so "same answer as before" is the only thing that will catch
a boundary error.

The corner cases must be written deliberately, not sampled — real data will
not contain a zero-length interval or a book-ended pair often enough to catch a
regression. At minimum the eight already differential-tested here, plus their
strand-aware variants, plus `.` vs `.` pinned to no-match.

`nearest` and `cluster` have no current implementation to capture a baseline
from, so they get **specification tests rather than differential fixtures** —
hand-written cases asserting the documented semantics, including the transitive
chain (`A-B` overlap, `B-C` overlap, `A-C` not, all one cluster) and signed
strand-aware distance. This is a weaker safety net than the other functions
get, and it is the price of adding them without a prior implementation.

**Phase 1 — the interval API.** Build the five functions *before* layering.
Every method collapsed is one that does not have to be assigned a layer,
migrated, and reviewed — and the layer boundaries get easier to see once the
near-duplicates are gone. Validated against Phase 0.

The work: create the `intervals` module, implement the five functions there
over the existing `pybedtools` backend, make the surviving `RegionDataFrame`
methods delegate to it, and delete the seven the API makes redundant.

**The module is created here, not in Phase 3** (Phases 3-4 are defined in
[`dataframe_layering_design.md`](dataframe_layering_design.md); this document
covers 0-2). Phase 1 has to put the five
functions somewhere, and putting them on the class only to move them two
phases later would mean migrating every call site twice. Phase 3 is then the
*remaining* extraction — geometry, binning, `FlDist` — not the creation of
`intervals`. Phase 1 deliberately does **not** swap the backend: the functions
wrap `pybedtools` at first, so that Phase 0's fixtures are validating a pure
restructuring, with the backend change isolated in Phase 2. Two changes that
would otherwise be tangled stay independently reviewable and independently
revertible.

**The signature is chosen for the destination backend, deliberately.**
`(a_index, b_index, overlap_bases)` maps directly onto
`bioframe.overlap(return_index=True)`, whereas over `pybedtools` it requires
stuffing the index through a BED round-trip — the technique `join_on_overlap`
already uses today, so it is proven, but it is not the shape one would pick for
`pybedtools` alone. That is an accepted coupling, not an oversight: designing
Phase 1 around the backend it is about to discard would mean changing the
signature twice. If Phase 2 were abandoned, `overlap_indices` still works over
`pybedtools` — it is merely less natural there.

**This is replacement, not removal — and the distinction matters**, because
this document's standing rule is that a method is removed only when it *cannot
execute*, never because callers appear absent. That rule exists for a good
reason: this library was extracted from a larger project that is no longer
accessible, so an absent caller measures our visibility rather than the code's
deadness.

The seven deletions do not rely on that rule being relaxed. **No capability is
withdrawn** — every one of them survives as an expression over
`overlap_indices`, listed in the table above. `bases_overlap_with_bed` becomes
a `groupby().sum()`; `overlaps_with_bed` becomes `overlaps`. A caller we cannot
see loses a spelling, not a capability, and loses it loudly at import rather
than silently at runtime.

Two places where that reconciliation is genuinely imperfect, stated rather than
glossed:

- **`overlaps_with_bed(min_size=n)` compares with `>`, so `min_size=10` means
  *at least 11* bases.** The replacement uses `min_bases` with `>=`. Any unseen
  caller passing a non-default `min_size` shifts by one. The parameter is
  off-by-one against its own name, so preserving it would mean preserving a
  defect.
- **The `*_beds` plural variants parallelised over BED files with joblib.** The
  replacement is a list comprehension, which is serial. Parallelism becomes the
  caller's decision. This is a deliberate loss of implicit concurrency, and it
  removes two of the four joblib call sites ahead of the later consolidation.
  The loss is likely notional: that parallelism was hiding per-call subprocess
  spawn and BED serialisation, and after Phase 2 there is no subprocess to
  hide, so the serial version may well be faster in wall time than the parallel
  one it replaces. Not measured, and not claimed as a benefit — only a reason
  not to treat the loss as a regression without measuring it first.

Remaining consolidation candidates, to be assessed rather than assumed: the six
constructors (`from_bed`, `from_beds_merged`, `rdf_from_bed3`, `from_regions`,
`from_random_regions`, `from_fname_s3_or_local`), the three splitters
(`split_on_query`, `split_on_column`, `split_on_contig`), and the
`get_*`/`attach_*` pairs already covered by the `inplace=` decision.

The cannot-execute rule still governs everything else.
`annotate_regions_with_max_tf_scores` (imported the removed `ravel` package)
and `get_chromosome_q_arm_starts` (referenced undefined names) qualified and
are gone. Anything that merely looks unused stays.

**Phase 2 — `bioframe` swap.** Replace `pybedtools` behind the five functions
Phase 1 established, validated against Phase 0. Because Phase 1 already routed
every caller through them, this phase touches three function bodies and no call
sites. Scope includes the two non-algebra sites (`from_beds_merged`,
`_get_fragment_coverage_sum`) — without them the `import pybedtools` survives
and the phase delivers neither of its stated benefits. Collapse the
IntervalTree path in `overlaps` into the backend. Implement `same_strand` as
strand-equality **excluding `.`**, not `on=['strand']`, and pin `.` vs `.` to
no-match with a test; see the strand table above. Implement the fraction
thresholds on our side — `bioframe` has no equivalent.

Ordered, because the last step is the one that delivers the benefit:

1. Add `bioframe` to all four dependency manifests and re-measure the
   benchmark and the strand table in `biomarker_env`. Nothing below can start
   until this is done.
2. Swap the bodies of `overlap_indices`, `merge` and — if Phase 1 built them —
   `nearest` and `cluster`. Nothing else in layer 2 changes: Phase 1 already
   routed every caller through these.
3. Migrate `from_beds_merged` (concat + `bioframe.merge`) and **delete**
   `bed_filter_callback` — zero callers, and its documented example never
   worked.
4. Migrate `_get_fragment_coverage_sum` to a chunked read. This is the only
   step with a real design question (streaming vs in-memory), so it is last
   and can slip to its own phase without blocking the rest.
5. **Remove `import pybedtools` and drop it from the manifests.** If this step
   cannot be completed, Phase 2 has not delivered — the import weight and the
   `bedtools`-on-`PATH` hazard are the point, and both survive a partial
   migration. Treat a surviving import as a failed phase, not a follow-up.

## Still open

**`nearest` and `cluster` ship without production use.** Both are new code
rather than migrations — nothing calls them today, and design review round 2
recommended deferring them on exactly that basis. They are kept by owner
decision: this is a general-purpose library, and a missing "nearest feature and
distance" sends the next caller to hand-roll interval logic, which is the
failure mode `CLAUDE.md` names first.

The cost is accepted and named rather than waved away. They have no Phase 0
baseline to capture, so they are covered by specification tests instead of
differential fixtures — a weaker safety net than the other functions get. Both
are thin adapters over `bioframe.closest` and `bioframe.cluster`, so the code we
own for them is the mapping, not the algorithm.
