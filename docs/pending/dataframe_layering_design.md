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
`pybedtools` (which additionally needs the `bedtools` *binary* on `PATH` —
see [`interval_api_design.md`](interval_api_design.md), which replaces it).
And every new data source means another method on the class, so the class only
ever grows.

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

## Interval algebra — specified separately

The interval-algebra surface of layer 2 is large enough to stand alone and is
specified in [`interval_api_design.md`](interval_api_design.md): the five
functions that replace the thirteen `bedtools`-backed methods, the
`pybedtools` -> `bioframe` decision, and Phases 0-2 which deliver them.

That document is a prerequisite for this one. What remains here is the layering
itself — the annotation protocol, orientation, the
`SampleAndRegionDataFrame` geometry problem, module layout, and Phases 3-4.

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
`RegionDataFrame` methods delegating to it. This is the only arrangement that
makes the algebra importable without the class, which is the point of the
layering. The methods stay for ergonomics.

The module is created by Phase 1 in
[`interval_api_design.md`](interval_api_design.md), which puts the five
interval functions there. Phase 3 moves the *rest* of layer 2 — geometry,
resizing, binning — in alongside them.

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

**Phases 0-2 are specified in [`interval_api_design.md`](interval_api_design.md)**
— differential fixtures, the interval API, and the `bioframe` swap. They are
prerequisites for what follows, and nothing below can start until the
`intervals` module exists.

**Phase 3 — remaining module extraction.** The rest of layer 2 — geometry,
resizing, binning — moves out alongside the `intervals` module Phase 1
created, with `RegionDataFrame` delegating; `FlDist` moves out.

**Phase 4 — annotation protocol.** Sources, `on_resize` with `shrink_only`,
`lift_over` invalidation.

The `joblib` -> `parallel_apply` consolidation stays a later phase, after all
of the above.

## Still open

**Where `center_regions_on_tf_motif` goes.** The largest method on the class
(~150 LOC), it needs JASPAR, a scoring model and a GPU, and defers its torch
and motif imports to call time precisely to keep them off the import path. It
is nominally an annotation source, but it does not fit the
`PositionalAnnotation` shape above: it *relocates* regions rather than
following them, so it is a producer of new geometry, not a responder to it.
Candidates: its own module outside the layer stack, invoked explicitly; or a
"region transform" concept distinct from the annotation protocol. Deferring
the choice is safe — nothing in Phases 0-3 touches it — but Phase 4 should not
try to force it into `on_resize`.

## Not in scope

Name-based source registration and config-driven pipelines.
