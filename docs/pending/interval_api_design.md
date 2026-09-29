# The interval API

Design for the interval-algebra layer of `fragmentomics_tools`: five free
functions over region frames, backed by `bioframe`, replacing the thirteen
`bedtools`-backed methods on `RegionDataFrame` and removing the `pybedtools`
dependency and the `bedtools` binary along with them.

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
8. **Load, operate, write are three separate layers, and layer 2 never takes a
   file path.** Callers load with a layer-1 constructor and pass frames. This
   is what dissolves every `*_with_bed` method rather than migrating it.
9. **Count consumers before designing a migration.** Three separate non-trivial
   migrations in this design were planned for code with zero live callers. The
   consumer count is what bounds a claim about impact, and it comes first.

Explicitly **out of scope**, each considered and declined:

| Out | Why |
|---|---|
| enrichment and null models — `shuffle`, `fisher`, `jaccard`, `reldist` | a coherent module of its own; not wanted at this point |
| multi-file `annotate`-style operations | easily composable at the call site |
| aggregating B's data columns onto A (`bedtools map`) | `groupby` already does this |
| per-base coverage vectors (`genomecov`) | not an interval-algebra concern; per-region fragment data comes from `attach_fragment_arrays` / `RegionFragmentArray.from_fragments_h5` |
| genome-end clamping (`slop`/`flank`) | not needed |
| geometric intersection | nothing needs the clipped coordinates; overlap *length* is enough |

## No `bedtools` — the whole dependency goes

**DECIDED (owner, 2026-09-29). `pybedtools` is removed, not abstracted**, and
the API is built directly on `bioframe` in a single phase. Staging it — an
implementation over `pybedtools` first, a backend swap second — is explicitly
not the plan: it would build a layer already marked for deletion.

The removal is contained, which is why it is affordable now. Measured:
`pybedtools` is imported in **exactly one file**, `dataframe.py`. It is declared
in three manifests — `environment.yml`, `pyproject.toml`, `recipe/recipe.yaml` —
and supplies the `bedtools` binary. Nothing else needs that binary: the other
declared externals (`htslib`, the `ucsc-*` tools) serve `formats.py`. So the
last `import pybedtools` and all three declarations go in the same change.

This is the point of the exercise. The `bedtools`-on-`PATH` hazard that
CLAUDE.md documents — present in the conda env's `bin/` but absent in sandboxes
and AWS Batch containers, and the cause of real failures — does not get
mitigated. It stops existing.

## Three layers

**DECIDED (owner, 2026-09-29).** The library is organised as three layers, and
the boundary between them is what makes the interval API small:

1. **Load** — classmethods that build a `RegionDataFrame` from a source: BED,
   TSV, S3, a list of `Region`s, random regions.
2. **Operate** — the interval algebra. **This document covers only this layer.**
3. **Write** — output to BED, TSV, and other formats.

The rule that follows, and it is the one that deletes most of the thirteen:
**no layer-2 function takes a file path.** A caller loads with layer 1 and
passes frames. Every `*_with_bed` method existed only to fuse a load with an
operation, and each collapses into two explicit calls.

One place the layering legitimately bends, named here so it is not mistaken for
an accident at implementation time: **`from_beds_merged` is a layer-1
constructor whose job is "read several BEDs, then merge"**, so it calls the
layer-2 `merge`. A constructor depending on the algebra layer is correct and
unavoidable — the alternative is a second merge implementation, which is the
duplicated-logic failure mode CLAUDE.md warns about. Layer 1 may call layer 2;
layer 2 may not call layer 1.

### Dead code removed rather than migrated

Three separate removals, each justified by a measured consumer count rather
than by taste. The pattern repeated often enough to be worth stating as a rule:
**count consumers before designing a migration.** Three times in this design a
non-trivial migration was planned for code that nothing calls.

| Removed | Live callers | Note |
|---|---|---|
| `get_fragment_coverage_sum`, `_get_fragment_coverage_sum`, `get_fragment_coverage_track` | **0** `.py`, **0** across 1022 notebooks | one test, passing an empty BED |
| `Region.intersect_with_bed`, `Region.get_bed_coverage_array` | **0** external | `get_bed_coverage_array` has no caller at all |
| `bed_filter_callback` on `from_beds_merged` | **0** | its own docstring example names the parameter wrong, so nobody ever invoked it successfully |

The fragment-coverage trio deserves a specific note, because deleting it
removes the single riskiest item in the previous design. Its BED branch was to
be migrated with a **chunked** read — `bedtools` streams a fragment-level BED
while `bioframe` needs it in memory, and a cfDNA fragment set does not fit. That
chunking was the one migration that changed the *shape* of a computation rather
than its backend, and its accumulation loop (`counts_vect[idx] = len(group)`,
an assignment, not an accumulation) would have silently returned only the last
chunk's count. The entire hazard existed to preserve a method with no callers.

It was also already broken: the BED branch does `enumerate(self.id)`, but `id`
is in `_optional_bed_columns` and `from_bed` does not produce it, so the
canonical entry point raises `AttributeError`. Strong evidence nobody had run
it in a long time.

`Region.intersect_with_bed` is removed on deadness grounds, **not** as a
`bedtools` operation — it is a `TabixBedReader` fetch, no `pybedtools`
involved, and under the three-layer model it is layer 1. The distinction
matters: if a per-region tabix fetch is ever wanted back, it returns as a
loader, not as interval algebra.

### `bioframe` is the backend

`bioframe 0.8.0` is packaged on **bioconda** (`pyhdfd78af_0`, noarch), and
`bioconda` is already a declared channel in `environment.yml` — immediately
above where `pybedtools` is pulled from today. One line per manifest, swapped
for the line being removed. It is *not* on conda-forge, so the channel matters.

**Binding rule, carried forward from the measured strand comparison.**
`bedtools -s` and `bioframe`'s `on=['strand']` agree on eight of nine
`(A strand, B strand)` combinations and diverge on `.` vs `.`: bedtools treats
`.` as *no strand* so two strandless features never match, while bioframe's
equality join makes `"." == "."` true. That inverts the result — no matches
becomes all matches — on the shape this codebase uses by default, since
`Region(strand=".")` normalises to `None` and the ordinary path is strandless.
Therefore: **implement `same_strand` as equality excluding `.`, never as
`on=['strand']`, and pin `.` vs `.` to no-match with a test.**

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

### Net change

The thirteen `bedtools`-backed methods, plus the removals the three-layer rule
and the consumer counts pulled in with them.

| | Methods |
|---|---|
| **Deleted — dead** | `get_overlapping_base_counts`, `overlaps_with_bed`, `bases_overlap_with_bed`, `overlaps_with_beds`, `bases_overlap_with_beds`, `intersect_with_rdf` (already a raising stub) |
| **Deleted — path-taking, dissolved by the layer rule** | `intersect_with_bed` |
| **Deleted — dead, and removes the chunked-migration risk** | `get_fragment_coverage_sum`, `_get_fragment_coverage_sum`, `get_fragment_coverage_track` |
| **Deleted — dead, on `Region`** | `Region.intersect_with_bed`, `Region.get_bed_coverage_array` |
| **Renamed / moved to layer 2** | `join_on_overlap`→`overlap_indices`, `overlaps_rdf`→`overlaps`, `merge_regions`→`merge`, `drop_overlapping_regions`→ `how="anti"` |
| **Rewritten** | `attach_blacklist_regions` — needs B's coordinate values, which `overlap_indices` does not return; see below |
| **Reimplemented, stays layer 1** | `from_beds_merged` — read each BED, concat, call layer-2 `merge`; `bed_filter_callback` dropped |
| **Added** | `nearest`, `cluster` |

Every deletion survives as an expression over what replaces it, per Requirement
6. The replacements are one line each:

| Was | Now |
|---|---|
| `a.intersect_with_bed(p)` | `overlap_indices(a, from_bed(p, ref=a.ref))` |
| `a.get_overlapping_base_counts(p)["counts"]` | `overlap_indices(a, b).groupby("a_index").overlap_bases.sum()` |
| `a.get_overlapping_base_counts(p)["max_counts"]` | `...groupby("a_index").overlap_bases.max()` |
| `a.overlaps_with_bed(p)` | `overlaps(a, from_bed(p, ref=a.ref))` |
| `a.bases_overlap_with_bed(p)` | `overlap_indices(a, b).groupby("a_index").overlap_bases.sum()` |
| `a.overlaps_with_beds(ps)` | `[overlaps(a, from_bed(p, ref=a.ref)) for p in ps]` |
| `a.drop_overlapping_regions(b)` | `overlap_indices(a, b, how="anti")` |
| `a._get_fragment_coverage_sum(p)` | `overlap_indices(a, from_bed(p, ref=a.ref)).groupby("a_index").size()` |

The last row is worth noting: the method whose migration was going to require a
chunked reader reduces, at the call site, to a `groupby` over the primitive.
The complexity was in preserving an interface, not in the computation.

**`attach_blacklist_regions` is the one method the primitive does not serve for
free.** It does not merely test for overlap — it reads B's *coordinate values*
out of the join result (`contig_{rsuff}`, `start_{rsuff}`, `stop_{rsuff}`) to
build a `Region` per overlapping blacklist interval. `overlap_indices` returns
index pairs, not columns, so this is a genuine rewrite: take the pairs, use
`b_index` to look the coordinates up in `b`, then group. Roughly thirty lines,
and it must be called out because an earlier version of this table listed it as
unchanged.

It also has a latent defect to fix while it is open: it **mutates the caller's
frame** on the empty-overlap path (`self["blacklist_regions"] = ""; return
self`) but returns a new frame otherwise — two aliasing contracts from one
call, selected by the data.

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

**The backend is replaceable, and that is the point of the adapter being three
functions wide.** No `bioframe` name, argument or column convention escapes
into our signatures, so replacing it later is a change to three function bodies
rather than to the API or to any caller.

What this is *not* is a retained `pybedtools` fallback. That dependency is
deleted in the same phase, and the `bedtools` binary with it; falling back
would mean re-adding both. The insurance is the narrowness of the seam, not a
second implementation kept alive behind it.

## Phasing

**Phase 0 — differential fixtures. DONE** (`6aa9623`, `b462e3a`).

Captured the current behaviour of every interval operation on real data —
964,593 CTCF intervals and the hg38 blacklist — as row counts plus digests that
cover content *and row order*, with input digests recorded so a future mismatch
can be attributed to a code change rather than to a moved EFS file. Plus 52
synthetic corner-case tests.

**Their status changed when the backend decision changed, and this must be
understood before anyone reads a failure.** They were written to pin
`bedtools`. `bedtools` is being removed. So they are now a **change detector,
not a specification**: a fixture mismatch means "the answer moved, come and
look", not "there is a regression". Several will legitimately move — the
zero-length and book-ended cases are exactly where independent implementations
differ.

The failure mode to avoid is someone editing a fixture to match new output in
order to get a green suite. That converts the one artifact that can tell us
what changed into a rubber stamp. **A moved digest is a decision to record, not
a number to update.**

**Phase 1 — the interval API on `bioframe`.** One phase, collapsed from the
previous two.

1. Build the five functions in a new `intervals` module against `bioframe`.
   Validate the `how` argument ourselves — `bioframe.overlap` performs no
   validation at all and silently treats an unrecognised `how` as a left join,
   so `how="anti"` returns wrong results rather than raising. Implement the
   fraction thresholds ourselves; `bioframe` has no `-f`/`-F`/`-r` equivalent.
2. Delete the methods in the Net change table.
3. Rewrite `attach_blacklist_regions`; reimplement `from_beds_merged` on layer 1
   plus layer-2 `merge`.
4. Remove `pybedtools` from `environment.yml`, `pyproject.toml` and
   `recipe/recipe.yaml`; add `bioframe` from bioconda. Verify no `import
   pybedtools` remains and that the suite passes without the `bedtools` binary
   on `PATH` — that last check is the one that proves the exercise worked.
5. Update CLAUDE.md: its sanctioned-entry-point table names
   `join_on_overlap` and `drop_overlapping_regions`, and its "legitimate escape
   hatch" paragraph describes the `bedtools`-on-`PATH` hazard as a live
   constraint. Both become wrong in this phase.

Re-run the Phase 0 capture afterwards and diff. Every moved digest gets an
explicit decision recorded in this document.

**Phase 2 — the `bedtools` equivalence document, then its tests.**
Owner-requested, and deliberately *after* implementation rather than before:
write a document showing how each common `bedtools` command is expressed in
this interface, then write equivalence tests from it. Doing it in this order
means the document is written against an API that exists, and the tests derive
from the document rather than from anyone's recollection of `bedtools`.

This is also what retires the Phase 0 corner-case tests from their pinning
role: once equivalence is asserted deliberately, the incidental pins are
redundant.

## Release

Version bump plus git tag only — no conda publish, no container rebuild, no
CHANGELOG. There is no CI in this repo; it is manual.

**`2.0.0b1`** after Phase 1. Written without a hyphen deliberately: `2.0.0-b1`
is valid PEP 440 but normalises to `2.0.0b1`, while conda treats `-` as the
version/build separator and rejects it inside a version — so the hyphenated
spelling would make the wheel and the conda package disagree about their own
version.

Major, because this is a clean break: methods deleted, methods renamed, no
compatibility shim and no deprecation period. **Downstream breaks loudly and is
not updated** — the same call made for `join_on_overlap`. `overlaps_rdf` →
`overlaps` alone breaks 22 call sites across 11 `biomarker-projects` notebooks.

The bump also closes a pre-existing hazard unrelated to this work: `main` is
198 commits ahead of `v1.4.0` while `pyproject.toml` still declares `1.4.0`, so
anything installed from `main` in that window reports a version it is not.

## Still open

- **`wiggle` off-by-one.** `overlaps_rdf(max_distance=N)` bridges a gap of only
  `N-1` — measured: a 10 bp gap needs `max_distance=11`. The cause is
  `IntervalTree` half-open semantics against an interval expanded by `N` on
  each side. `wiggle` must either preserve this or fix it, and the choice
  changes computed results, so it needs an explicit owner decision before
  implementation.
- Whether a per-region tabix BED fetch should return as a layer-1 loader, now
  that `Region.intersect_with_bed` is deleted.
