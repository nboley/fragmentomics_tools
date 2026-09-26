# Layering `RegionDataFrame`

Design for restructuring `dataframe.py` in the fork. Clean break — no
compatibility shim, no deprecation period.

## Problem

`RegionDataFrame` has 61 methods spanning concerns that share nothing but the
rows they sit on:

| Concern | Methods | Needs |
|---|---:|---|
| construction / IO | 11 | BED, TSV, S3, fasta paths |
| interval algebra | 10 | a set-operations backend |
| geometry / resizing | 10 | nothing but the data model |
| plumbing | 8 | — |
| sequence | 5 | a fasta |
| splitting / labels | 5 | an annotated table |
| **cross-layer** | **5** | see below |
| fragment coverage | 4 | h5 handles |
| reduction | 2 | annotated sequence |
| TF motifs | 1 | JASPAR + a scoring model + GPU |

Sums to 61. `SampleAndRegionDataFrame` adds 15, `DataFrameBase` 9,
`SampleDataFrame` 6, `FlDist` 4.

The five cross-layer methods are named deliberately, because they are where a
naive layering breaks: `sort` (plumbing + ordering semantics), `lift_over`
(coordinate transformation, neither algebra nor geometry), `unique_regions`
(algebra + selection policy), `drop_overlapping_regions` and
`attach_blacklist_regions` (both perform an overlap join to produce an
annotation).

Two consequences. Interval arithmetic — the most reusable and most testable
code here — cannot be imported without pulling in pysam, the motif stack, and
`pybedtools` (which additionally needs the `bedtools` *binary* on `PATH`; see
below). And every new data source means another method on the class, so the
class only ever grows.

## Layers

1. **Data model** — contig/start/stop/strand plus `ref`.
2. **Interval algebra** — merge, overlap, subtract, resize, bin.
3. **Annotation** — adds columns derived from an external resource.
4. **Analysis** — splitting, balancing, reductions over annotated tables.

Reductions need no machinery of their own. `get_pfm` returns a `Pfm` domain
object that computes `pwm = freqs.mean(axis=0)`, entropy, information content,
logo heights, and `plot()`. That object is *constructed from* annotated
Series; it is not a layer. A chained `apply(...).agg(...)` does not reproduce
it and is not proposed as a replacement.

## Decisions

**Stay a `pandas.DataFrame` subclass.** `rdf.start` beats `rdf.df.start`, and
`_metadata` carries `ref`. The subclassing has been robust.

**Drop required-column validation.** `_required_columns` means three different
things: a tuple attribute on `DataFrameBase`, a computed `@property` on
`RegionDataFrame` concatenating `_additional_required_columns` with
`_critical_bed_columns`, and a plain list attribute on `SampleDataFrame`. It
is also unenforced: `DataFrameBase.__init__` skips the check entirely when
`data` is a `BlockManager`, which is every pandas-internal construction —
slicing, copy, join, groupby — and the check is an `assert`, so `python -O`
removes it too. The mechanism is where the `tuple + list` trap lives.
`_metadata` stays.

Note this does not fix the root cause: the `_constructor` hack that bypasses
`__init__` will defeat *any* future invariant enforced there. Known limitation.

**Keep h5 handles in the column.** Storing `sample_id` with a
`sample_id -> handle` map elsewhere was rejected: it threads two objects
through every call site, and an explicit detach-for-serialization method is
needed regardless. The shared-handle close hazard stays documented on
`close_handles()`.

**`inplace=` to collapse `get_*`/`attach_*` pairs.** Consistent with
`center_on_summit`, which already takes it. Verified against pandas 3.0.6:
`inplace` is not deprecated and emits no warning; Copy-on-Write is permanently
enabled, so it is an API convenience, not a performance one. The only safe
implementation is the one already used at `dataframe.py:700` —
`rdf = self if inplace else self.copy()` — not `return self.join(...)`, which
always produces a new object regardless of the flag.

## Annotation: composition, not a value protocol

**An annotation source takes an RDF and returns an RDF with columns added.**

```python
rdf = source.annotate(rdf, inplace=False)
```

Not `values_for(regions) -> Series`. The value-returning form cannot express
what the real annotations do:

- TF scoring emits **three** columns per TF (score, strand, offset)
- coverage tracks are 2-D (regions x positions), not a Series
- orientation differs per source, so no single return convention fits them all

Returning an RDF dissolves all three. The source owns its columns, its naming,
its orientation, and its parallelism.

**What the framework enforces, and how.** "Layer order" is not a runtime
system. It is: layer 2 modules may not import layer 3, and layer 3 may not
import layer 4. That is checkable by a lint rule (an import-linter contract in
CI) and by the fact that layer 2 has no external dependencies to reach for.
There is no scheduler, no dependency graph, no registry. If enforcement needs
more machinery than an import rule, the layering is wrong.

**Correction is not an annotation source, and does not pretend to be.**
`apply_fragment_weights` operates on a `RegionFragmentArray` pulled out of an
SRDF, and mutates that array's weights rather than adding columns. It does not
fit `annotate(rdf) -> rdf`, and forcing it to would either make the source
SRDF-aware — breaking layer 3's RDF-level abstraction — or hide its real
signature behind a shim. It stays a fragment-array operation invoked
explicitly. The design's claim is only that the *annotation* sources fit;
correction is named here so nobody discovers the mismatch mid-implementation
and hacks around it.

**Shared helpers, because the boilerplate is real.** Sources must not each
reinvent column attachment. The library provides the common cases:

- `attach_columns(rdf, mapping, inplace)` — the column-concat path every source
  ends in, including index alignment and collision detection
- `attach_prefixed(rdf, prefix, mapping, inplace)` — the multi-column case,
  e.g. `{tf}_max_motif_{score,strand,offset}`

A source that needs neither is free to build its frame directly. The helpers
are a library, not a base class.

**No new map helper. DECIDED.** An earlier draft proposed
`map_over_regions`. Dropped: `dataframe.py` already has *two* parallel-map
mechanisms — `DataFrameBase.parallel_apply` (fork-based, main-thread only,
supports lambdas because nothing is pickled) and `joblib.Parallel`, used by
`overlaps_with_beds`, `bases_overlap_with_beds`,
`get_fragment_coverage_track`, and `get_fragment_coverage_sum`. Their
semantics differ: joblib's default loky backend spawns, `parallel_apply`
forks, and only the fork path carries the main-thread restriction. A third
would leave callers picking whichever is nearest.

The intended direction is the opposite: **consolidate joblib onto
`parallel_apply`**, leaving one map. Scope TBD — it touches the fork-safety
rules, so it is not free.

**What is actually missing is a reduction helper, not a map.** `get_pfm`
hand-rolls the reduce: it requires uniform region lengths, then stacks
per-region arrays and aggregates along the region axis. That pattern — stack
a Series of equal-shaped arrays, reduce across regions — is the one piece of
plumbing every aggregation needs and none of them share. The naive
`Series.agg(sum)` over object dtype is not it; the real body is
`np.stack(...).sum(axis=0)`, and the uniform-length precondition `get_pfm`
checks by hand is exactly what `np.stack` enforces. One tested helper for
that, and `Pfm` becomes a thin domain object constructed from its output.

## Orientation: three patterns, not one

An earlier draft proposed a single `positional` flag plus `reverse(values)`.
That is wrong — the three existing implementations are genuinely different
problems:

| Case | Where | Pattern |
|---|---|---|
| sequence / one-hot | `region.py` — `one_hot[::-1, ::-1].T` | caller-decided flag, applied on output |
| fragment arrays | `from_fragments_h5` sets `is_flipped` | applied at **construction**, not query |
| correction | `correction.py` asserts `not rfa.is_flipped` | **refuses** oriented input; consumer orients afterwards |

The third cannot be expressed as "reverse the output." Correction operates in
the forward genomic frame by design and requires the caller to query
strandless, correct, then orient at the aggregation layer. A framework that
auto-flipped for minus-strand regions would feed correction exactly the input
it refuses. CLAUDE.md records that getting this wrong silently destroys strand
asymmetry.

Under composition this stops needing a framework mechanism at all: each source
orients its own output, and correction simply does not. What the library owes
is one shared, tested implementation of *how* — reverse-complement for
sequence, axis reversal plus track permutation for profiles — not a policy
about *when*.

## `SampleAndRegionDataFrame` — the hardest problem

SRDF overrides four layer-2 geometry methods (`expand_regions`,
`resize_regions`, `_resize_region_boundaries`, `bin_regions_into_windows`) so
that resizing a region also resizes its fragment arrays. That is layer 2 and
layer 3 fused by necessity: the geometry is meaningless if the attached
fragment data does not follow it.

**DECIDED: geometry callbacks.** Fragment arrays become an annotation that
declares a geometry response — the source supplies `on_resize(...)`, and
layer 2 invokes it for any attached annotation that declares one. Layer 2
never learns what a fragment array is; it only knows that some annotations
care when coordinates move.

Rationale: the overhead is small, and the resize logic belongs *with* the
fragment array rather than inside the geometry methods. It also generalises
— any future annotation whose values are positional (a coverage track, a
per-base score) needs the same hook, and today would need the same copy-paste
into the same four methods.

The rejected alternatives, recorded so they are not re-proposed:

- **Keep the overrides.** Honest about the coupling, but leaves layer 2
  importing layer 3 concepts, and the next positional annotation repeats it.
- **Forbid resizing after attachment.** Closer to current reality than it
  sounds — `expand_regions` on an SRDF with fragments already raises
  `ValueError` — but shrinking via `resize_regions` is legitimate, so a
  blanket ban is too strict.
- **Load fragment arrays lazily.** Removes the coupling by removing the
  stored state, but pays repeated h5 I/O on every access.

**`on_resize` receives the new coordinates, nothing more.** An earlier draft
claimed it needed both old and new frames because
`bin_regions_into_windows` is one-to-many. That was wrong. The row
multiplication happens in the frame operation itself — the join duplicates
each parent's annotation onto every output row — so by the time the hook
runs it is a plain per-row call needing only that row's new region. The
existing SRDF override is already written this way:
`record.fragment_array.subset_by_region(region)`, with `region` being the new
one.

The real constraint is different: **the hook must be able to refuse.**
`subset_by_region` can only narrow a fragment array; widening requires going
back to the h5. The current override encodes this by rejecting `full` mode
up front when fragment arrays are attached. So the protocol needs a way for
an annotation to say "I cannot follow this transform" and have layer 2 raise
with a useful message, rather than silently producing wrong data. A
`shrink_only` declaration plus a clear error is enough; nothing richer is
needed.

Call sites for the hook are the four methods SRDF currently overrides:
`expand_regions`, `resize_regions`, `_resize_region_boundaries`,
`bin_regions_into_windows`.

**`lift_over` is a third category, and is silently broken today.** Verified:
SRDF does **not** override `lift_over`, and the base implementation makes no
reference to `fragment_array`. Lifting an SRDF with attached fragment arrays
therefore yields new-assembly coordinates with old-assembly fragments still
attached — no error, no warning.

It fits neither `on_resize` nor a positional `on_move`, because both assume
the annotation survives and needs adjusting. Liftover changes the coordinate
*system*: fragments were aligned to the old reference, sequence came from the
old reference, and anything derived from those is invalid rather than
shifted. It also fails per region (`pd.NA` since I20), can flip strand, and
can split one region into several — none of which a transform hook models.

The safe semantics are **invalidation**: `lift_over` drops every derived
annotation and the caller re-attaches against the new assembly. That converts
today's silent corruption into an explicit step.

## `pybedtools` — a decision, not an open question

**Nine of the ten** interval-algebra methods reach `pybedtools`, which shells
out to the `bedtools` binary. Three do so directly (`merge_regions`,
`join_on_overlap`, `get_overlapping_base_counts`); six more transitively, all
funnelling through `get_overlapping_base_counts`. Excluding the dead
`intersect_with_rdf` stub, that is 8 live methods out of 9.

The single exception is `overlaps_rdf`, which uses `get_interval_dict` and an
**IntervalTree** — a pure-Python overlap implementation already in the tree.
That is a second implementation of overlap living alongside the bedtools one,
which is precisely the "two implementations of one rule, free to drift" hazard
CLAUDE.md warns about for genomic logic. It is also proof that a
non-`bedtools` path is viable here.

CLAUDE.md records that the binary is "frequently *not* on `PATH` in sandboxes
and AWS Batch containers, and this has caused real failures." At 9 of 10, the
dependency is effectively total, so this contradicts the central promise that
layer 2 imports cleanly.

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
changes — only what the interval methods call underneath. The duplicate
overlap implementation collapses into it too: the IntervalTree path in
`overlaps_rdf` goes.

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

## What a call site looks like

Concretely, for one annotation, before and after:

```python
# now
rdf.attach_sequence()                       # fasta resolved internally from ref
rdf.attach_one_hot_encoded_sequence()

# after
seq = SequenceSource(rdf.get_fasta_path())
rdf = seq.annotate(rdf)                     # adds "sequence"
rdf = OneHotSource(seq).annotate(rdf)       # adds "one_hot", reuses the fasta handle
```

The resource becomes a constructor argument instead of being resolved inside
the method, which is what makes a fake source possible in tests. Convenience
wrappers on `RegionDataFrame` may keep the one-line form where it is worth it;
they delegate rather than implement.

## Testability

A fake source with no external resource makes orientation, batching, and
column-attachment testable without a fasta, an h5, or JASPAR. None of that is
reachable today.

## Module layout — DECIDED

**Layer 2 lives in an `intervals` module of free functions**, with
`RegionDataFrame` methods delegating to it. This is the only arrangement
that makes the algebra importable without the class, which is the point of
the layering. The methods stay for ergonomics. `bioframe` has the same
shape, so the delegation is thin rather than an adapter.

**`FlDist` moves out of `dataframe.py`.** It is the only non-DataFrame class
in the file, is four methods, is constructed *from* a `SampleDataFrame`
without being one, and is the only thing here that has never produced a
propagation bug. The move is import-path churn for `biomarker/flgc`, nothing
semantic.

**`SampleDataFrame` keeps its identity, and the duplication question
resolves itself.** Sample-ness was declared twice — SRDF's
`_additional_required_columns` and SDF's `_required_columns`, both
`["sample_id", "frag_h5"]`. Dropping required-column validation removes both
declarations, so the duplication disappears without a mixin. What remains is
a thin six-method class representing a samples table with no regions, which
is a genuinely different shape from SRDF. Not restructured on this pass.

**The `joblib` -> `parallel_apply` consolidation is a later phase**, not part
of this work. It touches the fork-safety rules just landed (main-thread
guard, tqdm monitor handling), and joblib's loky backend spawns where
`parallel_apply` forks — so each of the four call sites changes semantics,
not just syntax. It is not a precondition for anything here.

## Phasing — DECIDED

All of this happens in a worktree, not on `main`.

**Phase 0 — differential fixtures, before anything changes.** Capture current
output for every interval operation on large, real region sets, plus the
edge and corner cases. Everything after this validates against those
fixtures. Capturing them *first* rather than just before the `bioframe` swap
costs nothing and means Phase 1 is covered too.

The corner cases must be written deliberately, not sampled — real data will
not contain a zero-length interval or a book-ended pair often enough to
catch a regression. At minimum the eight already differential-tested here,
plus their strand-aware variants.

**Phase 1 — consolidation.** Collapse near-duplicate methods *before*
layering. Every method collapsed is one that does not have to be assigned a
layer, migrated, and reviewed — and the layer boundaries get easier to see
once near-duplicates are gone. Behaviour-preserving, independently
reviewable, and validated by Phase 0.

Confirmed candidate, read and verified:

```
overlaps_with_bed        -> get_overlapping_base_counts(bed)["max_counts"] > min_size
bases_overlap_with_bed   -> get_overlapping_base_counts(bed)["counts"]
overlaps_with_beds       -> joblib loop over overlaps_with_bed
bases_overlap_with_beds  -> joblib loop over bases_overlap_with_bed
```

Four methods that are thin wrappers over one, differing only in which column
they select and whether they loop. That is one method with two parameters.
It is also four of the joblib call sites, so this reduces the later
consolidation surface.

Unverified candidates, to be assessed rather than assumed: the six
constructors (`from_bed`, `from_beds_merged`, `rdf_from_bed3`,
`from_regions`, `from_random_regions`, `from_fname_s3_or_local`), the three
splitters (`split_on_query`, `split_on_column`, `split_on_contig`), and the
`get_*`/`attach_*` pairs already covered by the `inplace=` decision.

**Removal uses one criterion only: cannot execute.** Caller counts are not
evidence here — this library was extracted from a larger project that is no
longer accessible, so nearly every method had callers and their absence
measures our visibility, not the code's deadness. The defensible standard is
code that raises on every path: `annotate_regions_with_max_tf_scores`
(imported the removed `ravel` package) and `get_chromosome_q_arm_starts`
(referenced undefined names) both qualified and are gone. Anything that
merely looks unused stays.

**Phase 2 — `bioframe` swap.** Replace `pybedtools` behind the existing
method signatures, validated against Phase 0. Collapse the IntervalTree path
in `overlaps_rdf` into it. If any same-strand behaviour is added, implement it
as strand-equality **excluding `.`** — not `on=['strand']` — and pin `.` vs
`.` to no-match with a test. See the strand table above for why.

**Phase 3 — module extraction.** `intervals` module of free functions,
`RegionDataFrame` delegating; `FlDist` moves out.

**Phase 4 — annotation protocol.** Sources, `on_resize` with `shrink_only`,
`lift_over` invalidation.

The `joblib` -> `parallel_apply` consolidation stays a later phase, after all
of the above.

## Still open

Nothing. The strand question — the last open item — was measured and is
recorded above: bedtools and bioframe agree on eight of nine strand
combinations and diverge on `.` vs `.`, which is not currently reachable but
is a landmine for the first strand-aware caller.

## Not in scope

Name-based source registration and config-driven pipelines.
