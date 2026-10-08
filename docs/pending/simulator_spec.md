# Simulator spec

## 1. Inputs

| # | Input | Source |
|---|-------|--------|
| 1 | Fragment h5 | manifest key `ibd/frag_h5s/<SEQRUN>/<SAMPLE>-Lib1.hg38.fragments.h5` |
| 2 | Regions BED | `quiet_v2_pad1200_repeats_removed_tile1536.bed` |
| 3 | Reference FASTA | nominally `s3://karius-biomarker-data-assets/pipeline-assets/hg38/sequence.fa.bgz` — **but see below; no run has used it** |

Manifest: `~/src/biomarker-projects/tf_binding_site_classification/projects/ibd/manifests/ibd.data_manifest.tsv`
— 763 h5, mirror `s3://karius-biomarker-data-assets/projects`.

**The FASTA actually in use is GRCh38.p12**, not the asset named above
(owner, 2026-10-07, "accept p12 for now"):
`/efs/analytics/nathanboley/test_fragments_h5/GRCh38.p12.genome.fa.gz`. The
canonical `sequence.fa.bgz` is not present at any local path checked, and the
only local file of that name belongs to another user and has no `.fai`.
Coverage is fine — all 23 BED contigs are present — and GRCh38 patch releases do
not alter primary assembly sequence, though **that last point was not verified
here**.

**The consequence is provenance, not sequence.** The store's `config_hash`
hashes the reference `.fai`, so a store built against p12 will not match one
built against `sequence.fa.bgz` **even where the sequence is byte-identical**.
p12-derived artifacts are therefore not interchangeable with the rest of the
pipeline's. Record the reference path and its `.fai` md5 in any output, so a
later switch is detectable rather than silent, and re-raise this before
comparing a p12 artifact against a pipeline one.

### Regions BED options

In the main checkout's gitignored `data/region_sets/`.

| BED | Regions | Tile | Repeats |
|---|---:|---|---|
| `quiet_v2_pad1200_repeats_removed_tile1536.bed` | **66,649** | 1536 | removed |
| `quiet_v2_pad1200_repeats_removed_tile2560.bed` | 11,505 | 2560 | removed |
| `quiet_v2_pad1200_repeats_kept_tile1536.bed` | 904,975 | 1536 | kept |
| `quiet_v2_pad1200_repeats_kept_tile2560.bed` | 523,604 | 2560 | kept |

## 2. Build

Four stages. **Stages 1, 2 and 4 share one pass over the h5** — all three come
off the same attached frame, which is why `count_sample` returns that frame
rather than discarding it. Stage 3 consumes stage 2.

1. **Observed hexamer counts** `C(h)` — cut-site 6-mers of real fragments.
   Per sample.
2. **Fragment-length distribution** `f(L)` — `FragmentLengthDist`, built from
   the SRDF after §3 admission, so the same population `C(h)` counts.
   Per sample.
3. **Uniform hexamer counts** `N(h)` — the same tabulation as stage 1 with
   every start site equally likely. `N_start` is sequence-only, so per region
   set; `N_end` is `f(L)`-weighted.
4. **Region counts** — fragments per region.

## 3. Admission

A fragment is admitted to a region when, in the order `filter_fragments`
applies them:

1. `min(mapq_read1, mapq_read2) >= 10` — at fetch, not in the callback
2. it survives dedup on `(start, stop)`
3. its **length** lies in `[L_MIN, L_MAX]` = `[25, 180]`
4. its **start** lies in `[gstart, gstop)`

**Only 1-before-2 is load-bearing.** Two fragments sharing `(start, stop)` with
mapq 5 and 30 survive differently depending which runs first, because the dedup
keeps the *first occurrence*, not the best-mapped one. 3 and 4 commute with each
other and with 2 — they are independent per-fragment predicates, and any two
fragments sharing `(start, stop)` necessarily share a length and a start, so
dedup cannot reorder them. The list is in code order anyway, so the two cannot
drift.

### Sequence frame

    left_pad  = HEX_HALF = 3
    right_pad = max_fl + HEX_HALF = 183

A hexamer index then equals its region-local coordinate. `max_fl` is `L_MAX` for
the fragment pass and `fl.max_fl` for the uniform pass. After §3 admission
`fl.max_fl <= L_MAX`, with **equality only if a fragment of exactly `L_MAX` was
observed** — `FragmentLengthDist.from_srdf` takes the maximum *observed* length.
At real depth the two coincide; they are not equal by construction.

## 4. Sample

`n` comes from the region counts. Per fragment:

**0. Strand** — Bernoulli. It selects which tables the next two steps use,
because a minus-strand fragment's genomic start is its 3′ end:

| | genomic start `i` | genomic stop `i + L` |
|---|---|---|
| plus | `r_start_fwd(hex_fwd(i))` | `r_end_fwd(hex_fwd(i+L))` |
| minus | `r_end_rev(hex_rc(i))` | `r_start_rev(hex_rc(i+L))` |

**1. Start** — genomic start `i` over `[gstart, gstop)`, weighted by the left
column and normalised **within the region**:

    p[i] = r_start(hex(i))        for i in [gstart, gstop), 0 if hex(i) invalid
    P(i) = p[i] / p.sum()

**2. Stop** — given `i`, build the weight vector over lengths and normalise it
over the **valid** entries only:

    w[l] = f(l) · r_end(hex(i + l))     for every l in f's support,
                                        0 unless the fragment is VALID
    P(l | i) = w[l] / w.sum()

The support is `f`'s own, `[fl.min_fl, fl.max_fl]`, and it is bounded by
`[L_MIN, L_MAX]` **because §3 admission produced it** — not because the sampler
re-checks the constants. It does not know them. See Settled, "the length bound
is enforced once".

### When a length weight is zero

`w[l]` is a product of three factors, and any one of them zeroes a length. This
is the canonical statement; everything else in this document refers back here.

| factor | zero when | property of |
|---|---|---|
| `valid` | the end hexamer contains a non-ACGT base | the **reference** |
| `r_end` | `C(h) = 0` — the hexamer was never observed as a cut site in this sample | the **sample**, and its depth |
| `f(l)` | the length is absent from the distribution and densified to zero | the **sample** |

Two consequences:

- The denominator is **per start**. Two starts in the same region normalise over
  different length sets, so `w.sum()` is not a region constant. And because two
  of the three factors are sample properties, *the same region normalises over
  different length sets for different samples.*
- The code **cannot distinguish the three causes** — it tests `w.sum() > 0`. So
  "a length is valid when both cut-site hexamers are ACGT-only" is necessary but
  not sufficient, and a zero cannot be attributed to a cause after the fact.

## 5. Propensity

A hexamer's weight is observed over expected-under-uniform:

    r(h) = C(h) / N(h)

§4 normalises it per region for starts and per start for stops.

`C` and `N` are one tabulation counted two ways, so both cover the same
candidate positions. **They do NOT use the same validity gate** — `C` applies a
JOINT gate (both of a fragment's cut-site windows must be ACGT-only) while
`N_start` and `N_end` each gate MARGINALLY, on their own window only. An earlier
version of this section claimed they matched, which was wrong.

**This is settled and is NOT a defect: `r` is definitional, not an estimate.**
See Settled, "`C` and `N` gate validity differently". The simulator draws from
`r` and the oracle is computed from the same `r`, so the gate choice determines
what the model *is* rather than whether it is right. Zero effect on the current
region set in any case. Do not re-open it.

### There are TWO expectations, and they differ in domain as well as weight

`uniform_hexamer_counts` returns `{"start": N_start, "end": N_end}`. Both are
used; neither is "the" normalisation.

| | weight per position | domain |
|---|---|---|
| `N_start` | 1 — under a uniform start model every position is an equally likely start | `[0, region_len)` only, because only those positions can be starts |
| `N_end` | `f(L)`-weighted, see below | the **whole padded track**, because a fragment starting near the right edge has its end out in the flank |

`N_start` is sequence-only, so it is a property of `(region set, reference)`.
`N_end` is keyed to `(region set, reference, f(L))`.

The end weight per position:

    w(i) = F(min(i, max_fl)) − F(max(min_fl − 1, i − region_len))

for the `FragmentLengthDist` CDF `F` over its own support. It is the **null's own
prediction** of how often each position is an end, not a correction applied to
ends. An end at `i` is reached from start `s = i − L` for each `L`, so its weight
is the `f(L)` mass over routes whose start is admissible:

    L ∈ [ max(min_fl, i − region_len + 1) , min(max_fl, i) ]

### Why the end weight ramps, and why that is correct rather than a loss

Verified against brute-force enumeration (closed form matches exactly). Two
*different* truncations, which are easy to conflate:

| | when | the excluded routes would need | span, for `region_len = 500` |
|---|---|---|
| left | `i < max_fl` | `L > i`, i.e. a start **left of** the region | `w = 0` on `[0, 24]`, ramping on `[25, 179]` |
| right | `i ≥ region_len + min_fl − 1` | `L < i − region_len + 1`, i.e. a start **at or right of** `region_len` | ramping on `[524, 679]` — **in the FLANK, past `region_len`** |

`w == 1` exactly on `[max_fl, region_len + min_fl − 2]`. Both ramps are the same
length and are mirror images of `F`, but they sit in different places: the left
one occupies the start of the region, the right one lies outside it.

**The flank does not fix this, and is not meant to.** It is easy to think the
right ramp is a sequence problem — we pad by `max_fl + HEX_HALF`, so surely the
hexamer is readable? It is. **Sequence availability and admissibility are
separate axes.** The flank exists so a fragment whose END lands past the region
edge still has a readable hexamer; it deliberately does NOT extend the start
domain, because start-in-region (§3) is what makes each fragment belong to
exactly one tile.

**So a route needing `s ≥ region_len` is not lost — it is counted in the NEXT
tile.** Measured over three contiguous tiles, summing `w` for each genomic end
position across all of them:

    interior of the run: min = 1.000000, max = 1.000000  (exactly complete)
    ends of the run:     0.4872 at the left, 0.5128 at the right

The per-tile ramp is the **partition boundary**, not a deficiency. And it is what
makes `N` *correct*: `C` is tabulated under the same partition — each real
fragment is counted in the tile holding its start — so if `N_end` did not ramp it
would credit a tile with candidate routes `C` can never claim there, and `r`
would be depressed at every tile edge.

Note the contrast with the validity gate above: **`C` and `N` agree on the
partition and disagree on the validity gate.** The first is right by
construction; the second is the open item.

**Where the truncation IS genuine: isolated tiles.** The ramps complement only
where a neighbouring tile exists. The 66,649-tile set is not uniformly
contiguous — its first six lines are isolated, lines 8-10 contiguous — so an
isolated tile has no neighbour to claim its edge routes. Nothing in the code
knows which case a tile is in, and it does not need to: ramping identically is
correct per-tile either way. But **coverage near the edges of an isolated tile is
genuinely depleted**, which matters to whoever computes the oracle and picks the
model's crop width — they must use the same truncated domain or the ceiling is
wrong.

### Which expectation each table divides by

**This is the pairing, stated explicitly, because leaving it implicit is what
allowed a live defect to sit here undetected.** The previous wording said only
that "the `_rev` tables divide by the permuted expectation,
`N_rc == N_fwd[rc_permutation()]`" — true but silent on *which* of
`N_start`/`N_end`, which is precisely where the bug was.

| table | tallies | divides by |
|---|---|---|
| `start_fwd` | genomic starts | `N_start` |
| `end_fwd` | genomic stops | `N_end` |
| `start_rev` | genomic **STOPS** | `N_end[perm]` |
| `end_rev` | genomic **STARTS** | `N_start[perm]` |

The `_rev` rows are **crossed relative to their names, on purpose.** Table names
are MOLECULE-relative — `start` means the molecule's 5′ cut site — but a
minus-strand fragment's 5′ cut site sits at its genomic STOP. Denominators must
be POSITION-relative, so the table applied at genomic starts needs the start
expectation whatever it is called.

This is not a relabelling: `N_start` weights every position 1 while `N_end`
carries the ramp above, so the two differ over the **180 in-region positions of
the left ramp — 11.7% of a 1536 bp tile** (25 where `w = 0`, 155 ramping). That
is the left ramp only, because it is the part that overlaps the start domain;
the right ramp lies in the flank, where `N_start` is not defined at all.
Getting the pairing backwards cost a measured median 1.2% and max 16% error per
`r` cell on real regions, silently, with every total intact. Fixed 2026-10-07;
`[perm]` applies because the `_rev` tables index reverse-complement hexamers,
which is a relabelling of the same position.

## 6. Code

`background_model/simulator/count_hexamers_rdf.py`. Self-contained: stdlib,
numpy, pandas, `fragmentomics_tools.dataframe`.

| Function | Does |
|---|---|
| `filter_fragments` | dedup, length filter, admission. All filtering |
| `cut_site_hexamers` | per region → `start_hex`, `stop_hex`, `strand` |
| `counts_from_hexamers` | genomic start/stop + strand → the four tables |
| `count_srdf` | an attached frame → `C(h)`, per-row admitted counts, stats. Asserts only that BOTH strand tables are non-empty — there is no balance check, see Settled |
| `count_sample` | stages 1, 2 and 4 end to end → `(C(h), region_counts, stats, srdf)`. **Returns the frame**, which `f(L)` and the sampler both need |
| `simulate_fragments_to_bed` | draws for every region → the 8-column BED that `build_fragments_h5` consumes. Sorts its output; `n_drawn < n_requested` is reported, not raised |
| `FragmentLengthDist` | `counts`, `densities`, `min_fl`, `max_fl`, cached CDF |
| `uniform_hexamer_counts` | → `N(h)` |
| `fl_end_weight` | `w(i)` above |
| `propensities` | `C / N` |
| `sample_region` | draw `n` fragments → `starts_0`, `lengths`, `is_plus` |
| `hexamer_indices` | sliding 6-mer encode; `str`/`bytes`/`uint8`, case-folded |
| `rc_permutation` | RC as a 4096 permutation, derived from the encoder |

`C(h)`: `attach_fragment_arrays(callback=filter_fragments)` →
`attach_sequence` → one `parallel_apply` of `cut_site_hexamers` → four
`bincount`s.

`N(h)`: serial sequence walk.

`FragmentLengthDist` is built by `from_dataframe` (columns
`fragment_length`, `count`; absent lengths densify to zero) or `from_srdf`,
which delegates to it. Its three arrays are read-only, since an in-place edit
would leave the cached CDF stale.

`sample_region` is vectorised over fragments: all `n` in a region share its
arrays, so the length draw is one `(n, n_lengths)` block.

## Open

Needs action. Nothing here has been decided.

- **18 of 42 mutation rows are named-but-UNVERIFIED.** Every row in the test
  design's matrix now has a test, and 24 have been shown RED under their own
  mutation (18 from the review sweep, 6 closed in `02f7f39`). The rest are
  unverified: a test exists and passes, but nothing has shown it would fail
  against the defect it targets. **Treat those as unknown, not covered** — M7
  passed for weeks while catching nothing, because its fixture planted no
  N-window fragment at all.

- **Second copy of the encoder** in `simulator/precompute.py`, kept until the
  old simulator is deleted (deferred: ~8 files still import it, some belonging
  to another stream). The drift is now actually guarded —
  `test_encoder_matches_oracle_all_4096` checks all 4096 against an INDEPENDENT
  oracle, which is stronger than checking the two copies against each other
  since those could drift in step. Measured 2026-10-07: they agree exactly.
  **What remains is a naming lie**: `count_hexamers_rdf.py` still cites
  `test_encoder_matches_precompute`, which does not exist. Delete that citation.

### Closed since this section was last accurate

Kept briefly because the Open list claimed all three for longer than they were
true, and a reader who saw it mid-day would have acted on stale information.

- ~~**No tests.**~~ 47 now reference `count_hexamers_rdf` (`844f227`, `45b32ec`,
  `d9c6e90`) — 43 plus an independent oracle, plus 4 propensity tests. Mutation
  tested: 18 mutations applied programmatically, all now caught.
- ~~**`attach_sequence` near a contig end is UNVERIFIED.**~~
  `test_contig_ends_raise` and `test_frame_through_attach_sequence` pin it: it
  raises, and the padded length is asserted exactly, which pins both pads.
- ~~**4 mutations have no test** (M12, M15, M24, M25).~~ Implemented and each
  verified RED under its own mutation, `02f7f39`. Settling M12 first required
  resolving the h5 ordering for equal `(start, stop)`, which the design had
  flagged as "one case is not a law": `_build_h5` uses a STABLE sort, so ties
  keep fixture order and the ordering is fixture-controlled. This item was
  closed and sat in Open anyway until the same day.
- ~~**`tests/conftest.py` cannot exist.**~~ It can; `2eb09ac`. The collision was
  real but narrow: `--doctest-modules` makes pytest COLLECT conftest files, and
  with no `__init__.py` each is imported under the bare name `conftest`.
  `--ignore=tests/conftest.py` in `PYTEST_ARGS` stops the collection while
  pytest still loads the file as a plugin — verified by a `pytest_report_header`
  hook firing. `importmode = "importlib"` was NOT needed, and would have broken
  `import cut_site_oracle`, which relies on prepend mode putting `tests/` on
  `sys.path`. This item sat here stale for an hour after being fixed, which is
  the same drift the rest of this section exists to catch.

---

## Settled

Decided. Recorded so none of it is re-opened as an oversight — if something here
looks like a defect, read the reason before changing it.

- **Bulk GC is not modelled.** The length marginal matches by construction; the
  `(length, GC)` joint does not. Accepted consequence of dropping the capture
  model, not a defect.

- **`C` and `N` gate validity differently, and that is NOT a defect here**
  (owner, 2026-10-08: "the simulator gets to define the probability model…
  not interested in further pursuing this"). **Do not re-open it.**

  The mechanism, so nobody has to re-derive it: `C` applies a JOINT gate
  (`ok = s_ok & e_ok` — a fragment needs both cut-site windows ACGT-only),
  while `N_start` and `N_end` each gate MARGINALLY, on their own window only.
  So a fragment with one bad window is dropped from `C` entirely while `N`
  keeps its valid partner as a candidate, depressing `r` at that partner's
  hexamer.

  **Why it does not matter: `r` is DEFINITIONAL, not an estimate.** The
  simulator draws from `r`, and the oracle is computed from the *same* `r`, so
  parameter recovery and the ceiling are self-consistent whichever gate `N`
  uses. There is no true `r` being approximated — the gate choice changes what
  the model *is*, not whether it is right. The simulator defines its own
  probability model; this is one of the things it gets to define.

  Scale, for the record: zero today. 0 invalid cut windows in 1,373,600
  positions over 800 regions; 4 of 66,649 regions contain any non-ACGT base.

  **The one circumstance that would change this**: if `r(h)` were ever read as
  a *measurement* of physical cut-site preference rather than as a generative
  parameter — a claim about biology rather than a knob — then the marginal gate
  biases that estimate and the joint gate would be the correct one. Nothing
  does that today.

  Note `uniform_hexamer_counts`' own comment still says "the validity gate must
  match the fragment pass", which reads as an unmet requirement. It is a
  reasonable thing to have wanted; it is simply not required for this purpose.
- **The length bound is enforced ONCE, at fragment-array construction** — not
  downstream (owner, 2026-10-07). `filter_fragments` runs as the
  `fragment_array_callback`, so `subset_fragment_lengths(l_min, l_max + 1)` has
  already applied `[L_MIN, L_MAX]` before anything else sees the data. `f(L)`
  is then built from that frame, so its support is bounded by construction.

  Consequence, intended: `sample_region` draws over `[fl.min_fl, fl.max_fl]`,
  **`fl`'s own support, and does not know `L_MIN`/`L_MAX` at all.** §4's
  `[L_MIN, L_MAX]` describes where the bound comes from, not a second check the
  sampler performs. Do not add one — a downstream re-check is the two-
  implementations-of-one-rule failure this repo keeps hitting.

  The contract that makes this safe: **`fl` must come from a filtered frame.**
  An `fl` built from an unfiltered one is out of contract. The frame has one
  index of slack — the hexamer track is `region_len + 181` long and the furthest
  end index is `region_len - 1 + fl.max_fl` — so `fl.max_fl = 181` reads one
  past the intended frame silently, and `>= 182` raises `IndexError`.

- **`p_plus = 0.5`, by construction rather than by fitting** (owner,
  2026-10-07). A fragment is double-stranded and has no intrinsic orientation;
  the strand label records which of its two ends became read 1, and adapter
  ligation is symmetric, so the label is a fair coin independent of sequence.
  There is nothing to measure and no producer is needed.
  `count_srdf` **asserts only that BOTH strand tables are non-empty**, and
  reports `n_plus` / `n_minus` / `plus_frac` without asserting them. That one
  check needs no threshold and catches the silent failure that matters:
  `fragment_strands` is `<U1`, so comparing it against `b'+'` yields an
  all-False mask and an *entirely empty* strand table, after which
  `sample_region` skips that strand without error and the simulator emits
  strand-pure data.

  **There is no balance check. Removed by owner decision 2026-10-07; do not
  reintroduce one without naming a failure mode the non-empty check misses.**
  Three measured reasons: a wholesale plus/minus swap maps `plus_frac` to
  `1 - plus_frac` and preserves every total, so it was blind to the error it
  appeared to guard; the real-data fraction is **overdispersed** relative to
  independent fragment draws (0.3758 / 0.5296 / 0.5041 at 10 / 200 / 2,000
  regions, the first two deviating in *opposite* directions, mechanism never
  measured), so it false-positived on a legitimate 10-region run; and rescaling
  it to `n_regions` to fix that made it vacuous below 10 regions, since
  `|plus_frac - 0.5| <= 0.5` by construction.

  Also note `start_fwd.sum() == end_fwd.sum()` and the `rev` pair are
  **tautologies**, not routing protection — measured: feeding
  `counts_from_hexamers` a start/stop swap leaves both true, because each sum is
  just that strand's row count however the hexamers are routed. They stay only
  to catch a malformed `counts` dict from outside this module.

  Scope of that guard, so it is not mistaken for more than it is.
  `sample_region` skips a whole strand block on `if tot <= 0: continue`, with no
  error. That cannot be *sequence*-driven: `valid` is shared across strands —
  one array, with only `track` switching between `fwd` and `rc` — so an N-driven
  zero hits both strands alike and is just the deferred shortfall below. A
  strand-*specific* zero needs the table itself dead, all 4096 entries, meaning
  no fragments of that strand were counted, which is precisely what the
  assertion raises on. The only remaining path is an `r` that did not come from
  `propensities` on counts measured in the same run — hand-built, or loaded from
  a stored artifact. Not reachable from this pipeline; worth knowing if the model
  agent ever loads stored tables and calls `sample_region` directly.

- **Starts are drawn WITH replacement, and the resulting dedup is correct**
  (owner, 2026-10-07). The real assay has no UMIs, so a genuine duplicate
  coordinate pair is indistinguishable from a resampled one there too. The
  simulator reproducing that is faithful, not a defect. Measured cost of the
  collisions is ~0.0075% at the 1536 tile; see the dedup note in the
  coordination file.

- **Draw order is not part of the contract.** `sample_region` returns the plus
  block then the minus block, unsorted within each. Output goes straight into
  fragment-h5 construction, which sorts before `bgzip`/`tabix` regardless.

- **A start with no valid length is dropped**, so that region yields fewer than
  `n` fragments. `P(start)` does not condition on a valid fragment existing.

  **OBSERVED, on the first real run — this said "not observed" until
  2026-10-07.** A 10-region run requested 322 and drew 320, one region short.
  The earlier 38,637-for-38,637 over ~1000 regions also stands; both are true,
  and the difference is the useful part:

  **The shortfall rate tracks how sparsely `r(h)` is estimated, not the
  sequence.** At 10 regions only 108-151 of 4096 cells were nonzero, so most end
  hexamers carried `r = 0` and whole length sets vanished — the `r_end` factor
  in §4, not the `valid` factor. It therefore shrinks as the region set grows,
  and **a shortfall on a small run is expected rather than a defect.** Do not
  chase one without first checking how many `r` cells are populated.

  **DEFERRED by owner, 2026-10-06: documented only. No counter, no test.**
  Recorded so the next reader does not re-open it as an oversight.

  The region frame is **not** the cause, which is the first thing everyone
  asks. `right_pad = L_MAX + HEX_HALF = 183` is exactly sufficient: the
  furthest end hexamer a drawn start needs sits at region-local
  `region_len - 1 + L_MAX` and reads through `region_len + 182`, which the
  flank provides. There is no truncation and no out-of-range read. The start
  side is independently safe — starts are gated on `valid`, so a start whose
  own hexamer is invalid has weight 0 and is never drawn.

  A drop therefore needs **every length in `f`'s support zeroed at once** — see
  §4, "When a length weight is zero", for the three factors. On repeats-removed
  quiet tiles that needs a ≥183 bp N-run immediately downstream, or that many
  consecutive hexamers all unobserved in the sample. Hence the clean 38,637.

  **The partial case is the larger exposure, and it is not a defect.** A start
  that loses *most* of its lengths does not drop; it reshapes `P(l | i)`, which
  §4 licenses. Nothing reports it. A shallow sample with many `C(h) = 0` cells
  therefore samples lengths from a quietly narrowed support, and the only
  symptom is that the length marginal drifts from `f(L)`. If a future
  length-marginal check fails without an obvious cause, measure the reachable
  fraction of `f(L)` per start before looking anywhere else.
