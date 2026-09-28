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

Sums to 61. **Counting rule** (stated so the table is checkable rather than
asserted): every `def` in the class body, including `__init__`, dunders,
`_private` helpers, `@property` and `@classmethod`. Recount with
`ast.parse` over the class body, not `dir()` — inherited pandas methods
would swamp it.

`SampleAndRegionDataFrame` adds 16, `DataFrameBase` 9, `SampleDataFrame` 4,
`FlDist` 4 — all re-measured 2026-09-26.

`region_lengths` is the one genuinely arguable placement. It is counted under
geometry, and it has to be: every other bucket is exactly accounted for
without it (plumbing is `_required_columns`, `__and__`, `concat`,
`equals_rdf`, `nrow`, `get_interval_dict`, `iter_regions`, `iter_region_row` —
exactly 8), so moving it would break the sum. Noted because a reviewer reading
"resize-related methods" naturally counts 9 and concludes the table is wrong.

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
removes it too. `_metadata` stays.

**This does not retire the `tuple + list` trap, and the document should not be
read as claiming it does.** The trap lives in `_metadata` itself: it is a tuple
on `DataFrameBase` and a list on `SampleAndRegionDataFrame`, and pandas
concatenates the two during `concat`, where `tuple + list` raises. Dropping
`_required_columns` removes one site that exhibited the same shape; it leaves
`_metadata` exactly as it is. Normalising `_metadata` to a single type across
the hierarchy is the actual fix and is **not** in scope here — it is a
behaviour change to pickling and propagation that deserves its own pass. Until
then the invariant stands as documented in `CLAUDE.md`: base tuple, subclass
list, do not "tidy" either one.

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

**No new map helper. DECIDED.** `dataframe.py` already has *two* parallel-map
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

A single `positional` flag plus `reverse(values)` cannot express this — the
three existing implementations are genuinely different problems:

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

**`on_resize` receives the new coordinates, nothing more.** One-to-many
operations like `bin_regions_into_windows` do not require the old frame: the
row multiplication happens in the frame operation itself — the join duplicates
each parent's annotation onto every output row — so by the time the hook runs
it is a plain per-row call needing only that row's new region. The existing
SRDF override is already written this way:
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

**The API — DECIDED.** A runtime-checkable `Protocol`, not an ABC and not a
registry. Annotations are plain objects stored in a column; a `Protocol` lets
layer 2 ask `isinstance(value, PositionalAnnotation)` without the annotation
inheriting from anything layer 2 owns, which is the whole point of the
decoupling.

```python
@runtime_checkable
class PositionalAnnotation(Protocol):
    # Declared as a class attribute, read BEFORE any row is touched.
    shrink_only: ClassVar[bool]

    def on_resize(self, region: Region) -> "PositionalAnnotation":
        """Return this annotation transformed onto `region`.

        `region` is the NEW region for this row. Must not mutate self;
        returns a new annotation (RegionFragmentArray.subset_by_region
        already has exactly this shape and signature).
        """
```

Four rules govern how layer 2 applies it:

1. **Discovery.** Layer 2 scans object-dtype columns once per operation and
   collects those whose values satisfy the protocol. Columns that do not are
   carried through untouched.
2. **Refusal happens up front, not per row.** Before transforming anything,
   layer 2 compares the requested transform against each annotation's
   `shrink_only`. If any annotation declares `shrink_only` and the transform
   can widen, it raises immediately, naming the column and the method. This
   matches today's behaviour — `expand_regions` on an SRDF with fragments
   raises before doing work — and it matters because a per-row check would
   leave the frame half-transformed when row 5000 refuses.
3. **Ordering is not a concern, because hooks may not observe each other.**
   Each `on_resize` sees only its own value and the new region. Annotations
   are therefore independent by construction and may be applied in any order,
   including in parallel. This is the cheapest available answer to the
   reviewer's hook-ordering risk: forbid the interdependence rather than
   specify a resolution order for it.
4. **One-to-many is already handled by the frame op.** `bin_regions_into_windows`
   duplicates the parent row onto every output row first; the hook then runs
   per output row. No fan-out logic lives in the protocol.

`shrink_only` is a `ClassVar` rather than a method so that rule 2 can be
evaluated without instantiating or touching data. `RegionFragmentArray` sets
it `True`; a future per-base score track that can pad with zeros would set it
`False` and need no other change.

**`lift_over` is a third category.** SRDF now **refuses** to lift a frame with
fragment arrays attached. Before that guard, the base implementation made no
reference to `fragment_array`, so lifting produced new-assembly coordinates
with old-assembly fragments still attached — no error, no warning. The guard
stops the corruption but does not answer what lifting an annotated frame
*should* do, which is what this section decides.

It fits neither `on_resize` nor a positional `on_move`, because both assume
the annotation survives and needs adjusting. Liftover changes the coordinate
*system*: fragments were aligned to the old reference, sequence came from the
old reference, and anything derived from those is invalid rather than
shifted. It also fails per region (`pd.NA` since I20), can flip strand, and
can split one region into several — none of which a transform hook models.

The safe semantics are **invalidation**: `lift_over` drops every derived
annotation and the caller re-attaches against the new assembly. That
generalises the current refusal — which covers fragment arrays only — to every
derived annotation, and turns a hard error into a defined operation.

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

| | Methods |
|---|---|
| **Deleted** | `get_overlapping_base_counts`, `overlaps_with_bed`, `bases_overlap_with_bed`, `overlaps_with_beds`, `bases_overlap_with_beds`, `intersect_with_bed`, `intersect_with_rdf` |
| **Renamed / moved** | `join_on_overlap`→`overlap_indices`, `overlaps_rdf`→`overlaps`, `merge_regions`→`merge`, `drop_overlapping_regions`→ sugar for `how="anti"` |
| **Added** | `nearest`, `cluster` |
| **Rewritten** | `attach_blacklist_regions` — see below |
| **Unchanged** | the fragment-coverage pair |

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
| `overlap_indices` | `overlap(..., return_index=True)`, `how ∈ {left,right,outer,inner}` | `contig`→`chrom` mapping; `how="anti"` as outer + null filter; fraction thresholds; `same_strand` |
| `nearest` | `closest(k=, ignore_overlaps=, ignore_upstream=, ignore_downstream=, return_distance=)` | signed-distance and direction convention |
| `cluster`, `merge` | `cluster(min_dist=, return_cluster_ids=)`, `merge(min_dist=)` | `b=None` handling, label alignment |

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

**The module is created here, not in Phase 3.** Phase 1 has to put the five
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

**Phase 3 — remaining module extraction.** The rest of layer 2 — geometry,
resizing, binning — moves out alongside the `intervals` module Phase 1
created, with `RegionDataFrame` delegating; `FlDist` moves out.

**Phase 4 — annotation protocol.** Sources, `on_resize` with `shrink_only`,
`lift_over` invalidation.

The `joblib` -> `parallel_apply` consolidation stays a later phase, after all
of the above.

## Still open

Genuine design questions only. The `bioframe` dependency addition is a
prerequisite, not an open question, and is stated under Phase 2 above.

**1. Where `center_regions_on_tf_motif` goes.** The largest method on the class
(~150 LOC), it needs JASPAR, a scoring model and a GPU, and defers its torch
and motif imports to call time precisely to keep them off the import path. It
is nominally an annotation source, but it does not fit the
`PositionalAnnotation` shape above: it *relocates* regions rather than
following them, so it is a producer of new geometry, not a responder to it.
Candidates: its own module outside the layer stack, invoked explicitly; or a
"region transform" concept distinct from the annotation protocol. Deferring
the choice is safe — nothing in Phases 0-3 touches it — but Phase 4 should not
try to force it into `on_resize`.

**2. ~~Whether `nearest` and `cluster` belong in Phase 1.~~ DECIDED: they
stay.** Both are new code rather than migrations — nothing calls them today,
and design review r2 recommended deferring them on exactly that basis. Kept
anyway, by owner decision: this is a general-purpose library, and "nearest
feature and distance" is a core interval operation whose absence would send the
next caller to hand-roll it, which is the failure mode `CLAUDE.md` names
first. `nearest` is also the one operation not derivable from the overlap
primitive, and `cluster` is what keeps `merge` from being lossy.

The cost is accepted and named rather than waved away: they ship without
production use, and Phase 0 has no baseline to capture for them, so they are
covered by specification tests rather than differential fixtures. Both are
thin wrappers over `bioframe.closest` and `bioframe.cluster`, so the code we
own for them is the adapter, not the algorithm.

## Not in scope

Name-based source registration and config-driven pipelines.
