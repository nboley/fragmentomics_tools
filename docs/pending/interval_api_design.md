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

**In the library, `pybedtools` is imported in exactly one file** —
`dataframe.py`. It is declared in three manifests — `environment.yml`,
`pyproject.toml`, `recipe/recipe.yaml` — and supplies the `bedtools` binary.
Nothing else in the library needs that binary: the other declared externals
(`htslib`, the `ucsc-*` tools) serve `formats.py`.

**But one script outside the library uses it heavily, and the naive removal
breaks it.** `scripts/build_inactive_regions.py` (916 lines, last touched
2026-09-09, and the generator of the committed `docs/qc/region_set_qc.md`) has
roughly 25 `pybedtools` call sites. It is not portable to this API even in
principle: it relies on `slop` — genome-end clamping, which this document
declares explicitly **out of scope** — plus `sort(g=GENOME_FILE)`, `count()`
and `set_tempdir`. It is also the code that already works around the PATH
hazard by hand, calling `pybedtools.helpers.set_bedtools_path`.

**Disposition: `pybedtools` becomes a script-only dependency.** Remove it from
`pyproject.toml` and `recipe/recipe.yaml`, so that *installing the library* no
longer requires `bedtools`. Keep it in `environment.yml`, the development and
test environment where scripts run. The script is left working and unmodified.

This keeps the benefit that motivates the whole exercise while paying for it
honestly. The `bedtools`-on-`PATH` hazard CLAUDE.md documents — present in the
conda env's `bin/` but absent in sandboxes and AWS Batch containers, and the
cause of real failures — is what breaks *library* code in *deployed*
environments. After this change, importing `fragmentomics_tools` cannot
encounter it. A developer running a one-off region-building script from the
dev env is a different situation, and one the script already handles.

What this is *not* is "bedtools is gone from the repository". Claiming that
would be false, and the difference is exactly the part that matters: the
library's dependency closure versus a script's.

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

### The import graph — DECIDED, and it is the point of extracting the module

**Goal: `intervals` must be importable without the heavy stack, and the
dependency chains must stay clean.**

This nearly did not happen. The design specified `merge(a, ...) -> RegionDataFrame`,
which forces `intervals` to import `dataframe.py`. Measured:

| import | time | modules | pulls in |
|---|---|---|---|
| `fragmentomics_tools.dataframe` | **6.62 s** | **2779** | pysam, pybedtools, matplotlib, sklearn, numba |
| `pandas` + `bioframe` alone | 2.01 s | 822 | matplotlib |

So the extracted module would have reproduced, exactly, the problem the
layering document opens by describing. Four review rounds missed it because
they checked the API surface rather than the import graph.

**The types do not change** — layer 2 keeps taking and returning
`RegionDataFrame`. The fix is that the heavy imports become *lazy*.

Per-dependency, measured, with what each is actually for:

| dep | cost | what for | disposition |
|---|---|---|---|
| `sklearn` | **2.22 s** | one `sk_shuffle` call in `label_balanced` | **gone** — the method is deleted |
| `numba` | 0.80 s | two `@numba.njit` kernels in `fragment_array.py`, reached via `dataframe.py` importing `RegionFragmentArray` | **lazy** — import inside the fragment-attachment methods |
| `matplotlib` | 0.45 s | `fragment_matrix.py`, `plot/` | **lazy** now; plotting moves into `plot/` in a later phase |
| `pybedtools` | 0.44 s | interval algebra | **removed from the library** |
| `pysam` | **0.09 s** | FASTA/tabix in `region.py`, `formats.py` | **stays eager** — it is not the problem |
| `intervaltree` | 0.03 s | `overlaps_rdf` | removable once that method migrates; a later phase |

**Correct a claim the layering document makes.** Its problem statement says
interval arithmetic "cannot be imported without pulling in pysam, the motif
stack, and `pybedtools`". `pysam` costs **0.09 s** — naming it first is
misleading, and the 2.22 s dependency goes unmentioned. Cost, not count, is
what matters here.

**`dataframe.py` imports `intervals` lazily too.** Eagerly, `bioframe`'s 1.80 s
lands on every `import fragmentomics_tools.dataframe` and consumes most of what
the other deferrals just bought. Imported inside the delegating methods,
`intervals` stays cheap standalone and `dataframe` pays only on first use.
Python caches the module, so the cost is paid once.

**The general rule, for anything not resolved above:** an import used by a
method that may later be removed or relocated goes *inside that method* for
now. That is deliberately a holding position rather than an architecture — it
buys the import-time win immediately without pre-judging which methods survive,
and it gets revisited in the later phases.

**Explicitly not in this phase:** `fragment_matrix` is heavily used — 17 live
notebooks, plus 8 library files including `plot/tracks.py` and `dataframe.py`.
Replacing or removing it is its own migration with its own design, not a line
item here.

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
| **Deleted — NOT dead; owner accepted the break** | `label_balanced` (both definitions) — see below |
| **Renamed / moved to layer 2** | `join_on_overlap`→`overlap_indices`, `overlaps_rdf`→`overlaps`, `merge_regions`→`merge`, `drop_overlapping_regions`→ `how="anti"` |
| **Rewritten** | `attach_blacklist_regions` — needs B's coordinate values, which `overlap_indices` does not return; see below |
| **Reimplemented, stays layer 1** | `from_beds_merged` — read each BED, concat, call layer-2 `merge`; `bed_filter_callback` dropped |
| **Added** | `nearest`, `cluster` |

**`label_balanced` is the one deletion that is not dead code, and the cost is
recorded rather than glossed.** Measured across live notebooks — excluding
`archive/` and `.notebook_backups/` — it has **30 calls across 17 notebooks**,
which is *more* than any other method in this change, including `overlaps_rdf`
at 22 across 11. Owner decision, taken with those numbers in hand: delete it.
It is downsampling-to-the-minority-class, not interval algebra, and it does not
belong in this library.

Note for whoever reads this later wondering why: deleting it was **not**
necessary to drop `sklearn`. `sklearn` costs 2.22s at import — a third of
`dataframe.py`'s total — for this single `sk_shuffle` call, and moving the
import inside the method would have saved exactly the same time with no
breakage. The deletion is a scope judgement, and the import saving is a
consequence, not the reason.

Every other deletion survives as an expression over what replaces it, per
Requirement
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
| `cluster`, `merge` | `cluster(min_dist=, return_cluster_ids=)`, `merge(min_dist=)` | two-frame `cluster` — see below — and label alignment |

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

- **The two-frame `cluster(a, b)` is new code, not a thin wrapper.**
  `bioframe.cluster` accepts a single frame only. The single-frame case
  delegates; the two-frame case — label the connected components across `a` and
  `b` together — must be built here: concatenate with provenance, cluster, then
  split the labels back onto each input. Budget for it as an implementation
  task rather than assuming the backend supplies it.
- **Fraction thresholds do not exist in `bioframe`.** No `min_frac`,
  `reciprocal` or equivalent token appears anywhere in the package, so
  bedtools' `-f/-F/-r` must be implemented on our side by filtering on the
  returned overlap length. This is straightforward but must not be forgotten —
  it is a silent behaviour gap, not an error.
- **`how="anti"` does not exist, and `bioframe` will not tell you so.**
  Measured: `bioframe.overlap(df1, df2, how="anti")` raises nothing, and
  `how="completely_bogus_value"` does not either — `how` is not validated at
  all. An unrecognised value produces **exactly the inner join**. Internally
  `_minus` recognises only `left`/`right`/`outer`, so no unpaired rows are
  generated, and the later non-inner masking finds no sentinels to act on.

  Measured on an `A` of two rows, one overlapping `B` and one not:

  | `how` | rows | `A.start` |
  |---|---|---|
  | `inner` | 1 | `[100]` |
  | `left` | 2 | `[100, 5000]` |
  | **`anti`** | **1** | **`[100]`** — identical to `inner` |
  | what `anti` should give | 1 | `[5000]` |

  So it returns **the exact complement of its meaning**: every row that *does*
  overlap, when asked for the rows that do not. For the operation this replaces
  — `drop_overlapping_regions`, the sanctioned blacklist path — that means
  keeping precisely the regions intended to be dropped, silently.

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

**The old interface is not being preserved, and the fixtures do not constrain
the new one.** Owner position: this is a breaking change, and the goal is a
clean interface going forward rather than bit-compatibility with `bedtools`.
So a moved digest is *expected*, and nobody needs to justify it against the old
answer.

What the fixtures are still good for is the narrower and more useful question:
**did something move that nobody intended to move?** Record movements in the
section below as they are found, so the diff is visible rather than silently
absorbed. That costs a line per movement and is the only thing standing between
"we changed the semantics deliberately" and "we changed them by accident".

### Fixture movements — append during implementation

| fixture | old | new | why it moved |
|---|---|---|---|
| `overlaps_rdf_d10` | (Phase 0 digest) | — | expected: `wiggle` corrects the off-by-one, so a 10 bp gap now matches at 10 rather than 11 |
| `merge_book_ended` | 1 merged row | 2 separate rows | `merge(wiggle=0)` does not merge book-ended; this is the decided semantics (wiggle=0 = strict overlap only) |
| `from_beds_merged_book_ended` | 1 merged row | 2 separate rows | `from_beds_merged` now delegates to `merge(wiggle=0)`; same reason as above |
| `merge_regions_c_o_collapse` | test deleted | — | `merge()` is a free function; bedtools `-c/-o` column aggregation is not part of the new API |
| `get_overlapping_base_counts` | test deleted | — | method deleted (0 live callers); expressible as `overlap_indices(...).groupby("a_index").overlap_bases.sum()` |
| `_get_fragment_coverage_sum` | test deleted | — | method deleted (0 live callers) |
| `join_on_overlap` return type | test deleted | — | method deleted; `overlap_indices` returns a plain DataFrame by design |
| `intersect_with_rdf` raises | test deleted | — | method deleted along with `join_on_overlap` |
| `overlaps_rdf` missing-contig | test deleted | — | replaced by `TestOverlapsWithMissingContig` using `intervals.overlaps` |
| `get_interval_dict` | test deleted | — | method deleted (was internal to `overlaps_rdf`) |
| `drop_overlapping_regions` | test deleted | — | replaced by `TestAntiJoinReplacement` using `overlap_indices(how="anti")` |
| `label_balanced` | test deleted | — | method deleted (scope decision, not interval algebra) |

**Phase 1 — the interval API on `bioframe`.** One phase, collapsed from the
previous two.

0. **Preserve the `ref` equality check.** `join_on_overlap` currently asserts
   `self.ref == other.ref`; that is what stops an hg19 frame being joined
   against an hg38 one. Requirement 6 forbids withdrawing capability, and this
   is a live guard against a silent wrong answer, so every two-frame function
   keeps it. This preserves existing behaviour rather than changing it.
1. Build the five functions in a new `intervals` module against `bioframe`.
   Validate `how` ourselves against an explicit allowed set, raising on anything
   else — `bioframe.overlap` does not validate it at all, and an unrecognised
   value yields the inner join, so `how="anti"` returns the exact complement of
   what it means rather than raising. Implement `anti` ourselves as an outer
   join filtered to null right-hand rows. Implement the fraction thresholds
   ourselves; `bioframe` has no `-f`/`-F`/`-r` equivalent. Build the two-frame
   `cluster` as real code, not a delegation.
2. Delete the methods in the Net change table.
3. Rewrite `attach_blacklist_regions`; reimplement `from_beds_merged` on layer 1
   plus layer-2 `merge`.
4. Make `pybedtools` a script-only dependency: remove it from `pyproject.toml`
   and `recipe/recipe.yaml`, **keep** it in `environment.yml` for
   `scripts/build_inactive_regions.py`, and add `bioframe` from bioconda.
   Verify no `import pybedtools` remains **in `fragmentomics_tools/`**, and that
   the library's own suite passes with the `bedtools` binary absent from `PATH`
   — that last check is the one that proves the exercise worked. Do not assert
   the repository is free of `bedtools`; it is not, and the script is why.
5. Delete `label_balanced` (both definitions), which removes the last
   `sklearn` import from `dataframe.py`. Make the remaining heavy imports lazy
   per the import-graph section — `RegionFragmentArray` inside the
   fragment-attachment methods, `intervals` inside the delegating methods.
   **Measure `import fragmentomics_tools.dataframe` before and after and record
   both numbers**; the baseline is 6.62s / 2779 modules. An unmeasured claim
   that the import got lighter is worth nothing, and this is the phase's
   headline benefit.
6. Update CLAUDE.md: its sanctioned-entry-point table names
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

## `wiggle` semantics — DECIDED, and it corrects an off-by-one

**`wiggle` is defined as: a pair matches when the edge-to-edge gap between them
is `<= wiggle`.** `wiggle=0` is plain overlap, where book-ended intervals do
*not* match.

This **fixes** a defect in the method it replaces rather than carrying it
forward. `overlaps_rdf(max_distance=N)` reaches only `N-1`: it expands the
query by `N` on each side and then tests with half-open `IntervalTree`
semantics, so an interval expanded to start exactly where the query ends is
book-ended and does not count. Measured on a 10 bp gap — `max_distance=10`
returns `False`, `11` returns `True`. The docstring meanwhile promises
"maximum distance (edge to edge)", so the code and its stated contract
disagree. It is the half-open-plus-padding interaction, the same family as the
`[lo, hi)` fl-band trap in CLAUDE.md.

Fixing it is free, which is why it is being fixed now rather than preserved:
**0 of the 22 `overlaps_rdf` call sites pass `max_distance`** — all 22 are the
same copy-pasted line across 11 notebooks using the default — and at the
default of 0 there is no expansion and the behaviour is already correct. So no
result that exists today moves. The rename to `wiggle` in a major version is
also the one moment where a semantics change cannot be silently inherited:
the parameter name and the major version both change at once.

**Required test, at the boundary.** A gap of exactly `G` must match at
`wiggle == G` and must not at `wiggle == G-1`. Boundary-adjacent behaviour is
what a reimplementation gets wrong, and asserting only the interior would pass
against both the old and the new semantics. The Phase 0 digest for
`overlaps_rdf_d10` pins the *old* answer and is therefore expected to move —
that movement is this decision landing, not a regression.

## Still open

- Whether a per-region tabix BED fetch should return as a layer-1 loader, now
  that `Region.intersect_with_bed` is deleted.
- **Whether `subtract` belongs in the API after all.** It was declined on the
  grounds that "nothing needs the clipped coordinates". That justification is
  **wrong**, and `scripts/build_inactive_regions.py` is the counter-example: it
  builds the background model's training set as "the genome MINUS the union of
  five exclusion sets", then samples tiles that must fit *entirely inside the
  remainder*. That needs the remaining coordinates, not the surviving rows —
  geometric subtraction, which `bioframe` provides and this API declines.
  Padding with genome-end clamping (`slop`) is the same story.
  Deferred by owner decision: the script stays on `pybedtools` for now. Revisit
  when the script is rewritten, at which point the honest options are to expose
  `subtract` or to leave a second interval implementation alive in `scripts/`.
