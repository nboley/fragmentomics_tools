# h5-derived per-region counts and LUT-deconvolved `marginal_fl`

> # SUPERSEDED — DO NOT IMPLEMENT FROM THIS DOCUMENT
>
> **Superseded 2026-10-05 by `simulator_basic_inputs.md`.** Read that instead.
>
> This draft predates the autosome artifacts and decisions 130-134. It is wrong
> or silent on: autosome-only scope (133), the L-restricted count population
> (134), the resolved duphist provenance, and the measured real per-region count
> of ~243 — against which this document treated the fabricated constant as
> merely *a* wrong number rather than ~4.7x wrong.
>
> Kept only as a record of the reasoning that preceded the measurements. Two
> live designs for one change is the same drift hazard `CLAUDE.md` warns about,
> in document form; this banner exists so the winner is unambiguous.

Design for owner decisions 126-129 (2026-10-04). Supersedes parts of
`simulator_and_fragment_nll.md` — see §7 for the reconciliation list.

**Status:** SUPERSEDED. Written in-session after two spawned design agents died
on API rate limits having produced nothing.

---

## 1. Problem

The simulator must run from **basic inputs**, with counting integrated into the
script. Two inputs are currently wrong in kind, and one of them is a live
statistical bug.

### 1a. Per-region counts are hardcoded

`background_model/simulator/sampler.py` carries

```python
_REGION_COUNTS = {2560: 54, 1536: 37}
```

behind `target_count_for_region(region_len)`. Every region of a given size gets
an identical count. The replacement must be the **exact per-region count of a
real sample**, derived from that sample's fragment h5 inside the script.

The earlier plan (decision 121) had the script read pre-made
`*.region_counts.tsv.gz` files. **Decision 126 removes those as an input.**

Settled semantics, not open questions:
- The target is the **FULL-region** count. Realised scored counts still vary
  through jitter and the FL filter; that variation is **expected**.
- **Zero-count regions are KEPT** and emit nothing.
- **Absolute** real depth is matched, not a scaled shape.

### 1b. `marginal_fl` applies capture twice — a live bug

`capture.py::build_marginal_fl` sums the duphist's deduplicated
`molecule_keys` per length and normalises. That is an **observed** distinct-molecule
marginal, i.e. already `N_true x P(seen)`.

But the generative weight in `weights.py::build_region_weights` is

```
E_s(c5, L) = end_s[hex(c3)] * marginal_fl(L) / predict(L, gc)
```

and `predict = min(1/P(seen), max_weight)`. Dividing by `predict` **multiplies by
`P(seen)`**. So `marginal_fl` must be the **latent** molecule marginal, and the
model applies capture itself. Supplying an already-capture-attenuated marginal
applies it a second time.

This is the double-counting objection raised under decision 60 and overruled
under 67 ("nothing is deconvolved out of it... do not reopen it"). Decision 127
reverses that.

---

## 2. Owner constraints (binding)

| # | Constraint |
|---|---|
| 126 | Counts **and** `marginal_fl` from the h5. `predict_lut` **stays** on the duphist — the ZTNB fit needs duplicate multiplicities a fragment h5 does not express as a histogram. Two inputs, by design. |
| 127 | `marginal_fl(L) ∝ Σ_gc n_distinct(L, gc) × predict(L, gc)`, then normalise. `n_distinct` is the **deduplicated** h5 population. No fixed point, no `capture_marginal` ratio — only this single LUT weighting is revived from the retired forms. |
| 128 | Use the **capped** `predict`. The weight formula divides by the capped value, so weighting the FL estimate by that same capped value makes the two cancel exactly; a cap-induced distortion then cannot bend the realised FL away from target. Do **not** substitute uncapped `1/P(seen)`. |
| 129 | Sample **RD-56804**, h5 `/efs/analytics/nathanboley/ibd/frag_h5s/RD-56804.fragments.h5`. `DEFAULT_SAMPLE` is `RD-56670` in **both** `run_simulator.py` and `validate_parameter_recovery.py` and must change. |

Decision 127 changes computed results, so **every anchor moves** and the
parameter-recovery gate must be re-run (§6).

---

## 3. What the research turned up

Four findings that shape the design. Two are pre-existing defects.

### 3a. DEFECT — `run_simulator.py`'s self-verification is broken by decision 122

The readback block builds its containment mask as

```python
inside = (g_starts >= gstart) & (g_stops <= gstop)
```

with the comment "every emitted fragment lies wholly inside its own region, so
containment counts each fragment exactly once." That was true under the
**retired containment** edge rule. Under the midpoint admission rule
(decision 122) a fragment's endpoints may extend up to `MAX_FL_HALF = 90` bp
beyond the region, so containment **undercounts**.

Two checks in the failure list are therefore now wrong:

```python
if n_emitted != target * n_regions:   # assumes a CONSTANT per-region target
if n_contained != n_emitted:          # assumes containment == emission
```

The first is also killed by decision 126 (counts stop being constant). This has
not surfaced because the driver has not been run since the geometry change —
the simulator package was unchanged from `9ab0721` until `a352c34`.

**This is the same defect class as the one repaired in `77823c2`**: a consumer
of the geometry that was not updated when the geometry changed. It was missed
because it lives in a *verification* path, where a wrong count reads as a
failing check rather than a wrong answer.

### 3b. TRAP — h5 GC is a quantised fraction with NaNs, the LUT wants floor-binned percent

`FragmentsH5.fetch_array(return_gc=True)` returns, per its own source:

- GC stored as **uint8**, decoded `uint8 / 254.0` -> **fraction in [0, 1]**, not percent
- **unknown GC is sentinel 255 -> NaN**
- "GC is stored in a uint8 so there are only two significant digits"

Meanwhile `build_region_weights` indexes the LUT via a **floor** rule on
**percent**: `gc_bins = floor(100 * gc_count / L / GC_BIN_WIDTH)`, clipped to
`[0, N_GC_BINS - 1]`.

Three consequences the implementation must handle explicitly:

1. **Unit mismatch is silent and total.** Feeding the fraction straight into the
   floor rule gives `floor(0.45 / 5) == 0` for every fragment — the entire
   population lands in GC bin 0 and the deconvolution multiplies everything by
   `predict(L, 2.5)`. It would run, produce a plausible normalised FL, and be
   wrong. Must be `100 * gc_fraction`.
2. **NaN must be handled deliberately.** `floor(nan).astype(intp)` is undefined.
   Dropping NaN fragments biases the FL if GC-unknown correlates with sequence
   content; assigning them a bin invents data. **Recommendation: drop, and
   report the dropped fraction as a gate number** so a large loss is visible
   rather than absorbed.
3. **A third GC convention now exists.** The duphist's `gc` column is
   whole-number **percent** binned *inclusively* via `SIM_GC_BINS`
   (`[(0,4),(5,9),...,(95,100)]`); `build_region_weights` uses **floor** on
   exact count-derived percent; the h5 gives a **quantised** fraction. The
   deconvolution indexes the *LUT*, so it must use the **floor rule**, matching
   `build_region_weights` and not the duphist's inclusive bins.

### 3c. The counting pass largely exists already

`run_simulator.py` already iterates regions and reads fragments back through
production:

```python
with FragmentsH5(h5_path, cache_pointers=False) as fh5:
    for contig, gstart, gstop in regions:
        region = Region(contig, gstart, gstop, strand=None)
        rfa = RegionFragmentArray.from_fragments_h5(fh5, region, min_mapq=10)
```

This is the sanctioned entry point per `CLAUDE.md` (and `from_fname` is the
broken one — do not use it). The new counting pass is this loop with a
**midpoint** membership test instead of containment. It should not be
hand-rolled from scratch.

`min_mapq=10` matches the retired count TSVs' `ibd_region_counts_mapq10`
provenance, and the `-1 >= 10` empty-readback trap is already guarded.

### 3d. Tiles are contiguous, so membership must be exact

Region tiles abut, and `RegionFragmentArray.from_fragments_h5` fetches by
**overlap**, so a neighbour's fragments come back in the window. Counting
overlap would double-count across tile boundaries. The midpoint rule resolves
this exactly: each fragment has exactly one midpoint, so it is counted by
exactly one tile. This is also the rule the retired count files used
("in_region: midpoint in [start, stop)"), so counting and generation stop being
divergent — which was the point of decision 122.

---

## 4. Design

### 4a. Counting (`region_counts`)

One pass over the region set against the real h5, producing an
`int64` array aligned to the region-set row order:

- membership: `midpoint = start + L // 2` (floor, matching the count files and
  `build_region_weights`), kept when `gstart <= midpoint < gstop`
- `min_mapq=10`, `>=` semantics
- zero-count regions retained as `0`

`target_count_for_region(region_len)` and `_REGION_COUNTS` are **deleted**, not
kept as a fallback. A fallback would silently resurrect constant counts if the
h5 path were misconfigured, which is precisely the failure the provenance guard
exists to prevent.

### 4b. FL deconvolution (`marginal_fl`)

From the **deduplicated** h5 population:

```
n_distinct[li, gi]  <- 2-D histogram over (length, floor-binned GC percent)
marginal_fl[li]     <- Σ_gi n_distinct[li, gi] * predict_lut[li, gi]
marginal_fl         <- marginal_fl / marginal_fl.sum()
```

- dedup via the library's `drop_duplicate_fragments()`; duplicate multiplicity
  tracks depth and PCR, not length-specific capture
- GC: `floor(100 * gc_fraction / GC_BIN_WIDTH)`, clipped — identical to
  `build_region_weights`
- NaN GC dropped, fraction reported
- restricted to `L in [L_MIN, L_MAX] = [25, 180]`
- accumulate in **float64** (project standing rule)

`predict_lut` is unchanged and still built from the duphist, so `fit_and_build`
splits: the capture surface keeps its duphist input, the FL moves to the h5.

### 4c. Provenance guard against circular input

`run_simulator.py` writes `{sample}.fragments.h5` into `--out-dir`, default
`/efs/analytics/nathanboley/background_model/sim_smoke`. So these are
**simulated** and are indistinguishable by filename from a real sample:

```
background_model/sim_smoke/RD-56670.fragments.h5
background_model/sim_run_tile1536_20260930/RD-56670.fragments.h5
```

Feeding one back in as the real sample would be circular and self-confirming.
Requirements:

- the h5 is an **explicit required path**, never glob-discovered or derived from
  `--sample`
- a guard rejects an h5 that carries the simulator's own manifest provenance
  (the emitted h5 sits beside a `manifest.json` the simulator wrote; its
  presence alongside, or any simulator provenance recorded inside the h5, is
  disqualifying)
- the resolved h5 path, its size and its mtime are recorded in the run manifest

**Honest limit of this guard:** it detects *our* simulator's output by its
provenance artifacts. It does not prove an arbitrary h5 is real data. That is
acceptable because the realistic failure is re-reading our own output from a
default directory, not an adversarial substitution — but it should not be
described as a proof of authenticity.

---

## 4d. THE POPULATION CONTRACT — added after self-review

Self-review found three gaps, and they are all one gap: **a derived quantity
silently computed over a different population than the thing it is combined
with.** The GC-convention trap in §3b is the same shape. That is the central
risk of this design, so it gets an explicit, asserted contract rather than
prose.

For every derived quantity, the following must be *decided, asserted at
runtime, and recorded in the manifest*:

Established from the build path (`flgc/dup_counting.py` in the biomarker repo —
see §4f for the evidence):

| Quantity | Source artifact | Genomic scope | mapq filter | GC-unknown | Dedup |
|---|---|---|---|---|---|
| `predict_lut` | `{sample}__duphist_wg.tsv.gz` | **AUTOSOMES 1-22 ONLY** (despite the `_wg` filename) | **none** (`min_mapqs=(0,)`, `config.MIN_MAPQ = 0`) | **dropped** before thresholding | duplicate-aware by construction (ZTNB needs multiplicities) |
| `marginal_fl` | fragment h5 | **autosomes 1-22**, to match | **none**, to match — see item 3 | **dropped**, to match | deduplicated |
| `region_counts` | fragment h5 | region set, midpoint rule | `min_mapq=10`, `>=` on `min(mate1, mate2)` | n/a | **NOT** deduplicated (real depth is the target) |

What the three gaps resolved to:

1. **Scope: autosomes, not "whole genome".** Decision 130 confirmed whole-genome
   scope, but the build path shows `contigs` **defaults to autosomes 1-22 only** —
   chrX, chrY, chrM and all alt/random/decoy contigs excluded. **The `_wg` filename
   is misleading.** So "whole-genome `marginal_fl`" must be implemented as
   *autosomes 1-22*. Computing FL over all contigs would let chrX/Y/M fragments into
   the length marginal that never entered the capture surface — reintroducing the
   exact mismatch decision 130 was meant to close.
2. **Dedup scope: resolved by decision 130.** A whole-genome (autosomal) FL pass
   never fetches a fragment twice, so the overlap/adjacent-tile double-counting
   hazard disappears. Had FL been region-restricted, each fragment would have needed
   midpoint assignment to a single tile *before* dedup.
3. **mapq: the duphist is UNFILTERED, which conflicts with the counts.** `predict_lut`
   sits on a mapq-unfiltered population (and `min_mapq <= 0` *deliberately retains*
   fragments whose MAPQ is missing, since fragments_h5 remaps the 255 sentinel to -1).
   `region_counts` uses `min_mapq=10`. These cannot both align to `marginal_fl`.
   **Recommendation: FL takes `min_mapq=0` to match `predict_lut`**, because decision
   128's exact-cancellation argument is between `marginal_fl` and `predict` — those two
   must share a population. Counts then legitimately sit elsewhere: they set *depth*,
   not *length shape*, and mapq≥10 is what the driver's own readback verifies against.
   **This split is deliberate and must be recorded, not silently inherited.**

**Why assert rather than document:** §3b demonstrated that a population or
convention mismatch here still yields a plausible, normalised, non-crashing
result. These failures are invisible in the output, so the contract has to be
checked by the code, not trusted to a reader.

---

## 4f. Duphist build path — evidence, and the limit of it

Builder: `/home/nathanboley/src/biomarker/flgc/dup_counting.py`,
`count_duplicates_from_h5_mapq_sweep`. Nothing in *this* repo builds the duphist;
`capture.py` and `ztnb_from_duphist.py` both consume it as precomputed.

Read from that source:

- **`contigs` defaults to AUTOSOMES ONLY (1-22)** via `select_autosomes`, which
  excludes chrX, chrY, chrM and every alt/random/decoy contig. Its own docstring
  notes it tolerates both `chr1` and bare `1` naming because a hardcoded `chr`
  prefix against an NCBI-style h5 "would silently select NOTHING".
- **`min_mapqs` defaults to `(0,)`** and `flgc/config.py` has `MIN_MAPQ = 0`, i.e.
  **no mapq filtering**. `_mapq_pass` requires BOTH mates to clear the threshold
  (`min(mate1, mate2) >= min_mapq`), and documents that `min_mapq <= 0` deliberately
  **retains** fragments with missing MAPQ, because fragments_h5 remaps the 255
  "empty mapq" sentinel to -1 and a plain `>=` would drop them.
- **Fragments with unknown GC (NaN) are dropped before any thresholding**, so every
  mapq arm sees the same GC-known population.
- Requires an h5 built with duplicates retained (not collapsed) and with `--fasta`
  so the `gc` dataset exists (`_h5_has_gc`).

**The limit of this evidence, stated plainly: the duphist file itself carries NO
provenance.** Its header is exactly `length  gc  multiplicity  molecule_keys` — no
comment lines, no mapq record, no contig list, no build command. So the filter
above is read from the **defaults in the builder**, not from the artifact. The
actual invocation that produced `duphist_merged/` was not located.

Supporting but not conclusive: the filename carries no mapq token, while this
project does tag mapq when it is used (`ibd_region_counts_mapq10`). That is
consistent with `min_mapq=0` and is the basis for the §4d recommendation, but it
is **inference from a naming convention, not proof.**

**Therefore Phase 0 must verify rather than trust this.** The check is cheap and
direct: recount the duphist from the sample's h5 at `min_mapq=0`, autosomes only,
GC-known only, and compare against the stored `duphist_wg` file. If they match,
the population is confirmed from data instead of from a default. If they do not,
the stored file was built with different settings and the table in §4d is wrong —
which is exactly the kind of silent mismatch this design exists to prevent.

---

## 4e. Verified facts (measured, not inferred)

- **`drop_duplicate_fragments()` dedups on `(starts_0, stops_0)` ONLY** —
  `np.unique` over those two rows. **Strand is not part of the key**, so two
  fragments with identical span and opposite orientation collapse to one. For
  molecule-level dedup that is defensible (same physical span = same molecule),
  but it is a semantic choice and is now verified rather than assumed.
- **`from_fragments_h5` supports GC** via `return_gc`, defaulting to
  `fragments_h5.has_gc`. So there is a **capability flag** to use as the
  precondition, which is cleaner than any check this design originally proposed.
- **Measured on `ibd/frag_h5s/RD-56804.fragments.h5`**, `chr1:1,000,000-1,200,000`,
  24,020 fragments: `has_gc` is `True`; GC is `float32` in **fraction** units
  (observed range 0.173 to 1.000), confirming §3b's unit trap; **NaN fraction is
  0.0**. So the NaN handling in §3b is cheap insurance rather than a live
  problem — but keep it, since the sentinel exists and a different region may hit it.
- **GC reaches exactly 1.0 in real data**, and `floor(100 * 1.0 / 5) == 20` while
  `N_GC_BINS == 20` (valid indices 0-19). The `np.clip` in `build_region_weights`
  is therefore **load-bearing on real data, not defensive**. The deconvolution
  must clip identically or it will index out of bounds on GC-1.0 fragments.

---

## 5. Phased plan

Each phase is independently reviewable and names **what must FAIL** if it is
wrong — not merely the invariant that should hold.

**Phase 0 (new, blocking): settle the population contract in §4d.** Establish the
duphist's mapq filter, fix `marginal_fl`'s genomic scope, and write the contract
table into the manifest with runtime assertions. Nothing downstream is
meaningful until the three quantities are known to sit on the same population.

**Must FAIL if wrong:** construct a case where the FL population and the
`predict_lut` population deliberately disagree (e.g. region-restricted FL
against the whole-genome duphist) and assert the contract check rejects it. A
check that only validates shapes passes this and is worthless.

### Phase 1 — fix the stale verification in `run_simulator.py` (§3a)

Pure repair of a pre-existing break, landed first so later phases are not
debugged against a driver that cannot pass its own checks.

- midpoint membership replaces containment
- drop the `n_emitted != target * n_regions` check (constant-count assumption)

**Must FAIL if wrong:** a fragment whose midpoint is in-region but whose
endpoint lies outside must be COUNTED; construct one at a region edge with
`L = L_MAX` and assert the count includes it. A containment implementation
passes a midpoint test only if no fragment straddles a boundary, so the test
must place one there deliberately.

### Phase 2 — the counting pass (§4a)

**Must FAIL if wrong:**
- two adjacent tiles must not both count the same fragment — assert the sum over
  tiles equals the number of distinct in-window fragments, with a fragment
  placed to straddle the shared boundary. An overlap-based implementation
  double-counts it and fails.
- a region with no fragments must yield `0` and be retained, not dropped; assert
  the output length equals the region-set length.

### Phase 3 — FL deconvolution (§4b)

**Must FAIL if wrong:**
- **the unit trap**: a synthetic population with a known GC spread must produce
  a different `marginal_fl` from the same population binned as a fraction. If
  the two agree, the percent conversion is missing and everything is in bin 0.
- **the deconvolution is actually applied**: with a deliberately non-flat
  `predict_lut`, the result must differ from the raw normalised length
  histogram. A no-op implementation passes any normalisation check, so this
  comparison is the guard.
- **NaN GC** fragments must not silently become bin 0.
- round trip: with a **flat** `predict_lut`, the result must equal the raw
  normalised dedup length marginal exactly.

### Phase 4 — wiring, defaults, manifest

- `DEFAULT_SAMPLE` -> RD-56804 in both scripts (decision 129)
- required `--fragments-h5`; provenance guard; manifest fields
- delete `_REGION_COUNTS` / `target_count_for_region`

**Must FAIL if wrong:** pointing `--fragments-h5` at a simulator output must
raise. Assert on an actual emitted h5, not a mock.

### Phase 5 — anchors and the gate (§6)

---

## 6. Anchors move

`marginal_fl` changes, so the generative distribution changes, so **the oracle
changes**. Required:

1. recompute the store's own anchors (oracle, uniform) — anchors are per-store
   and recomputed per store, never quoted from a previous store
2. re-run `scripts/validate_parameter_recovery.py` at its **default scale** and
   re-commit its artifacts
3. retire the pre-change anchor figures rather than carrying them forward

The gate's interpretation does **not** change: it tests whether the sampler
draws from the model the weights define, which is a self-consistency property.
It will pass both before and after a `marginal_fl` change, and so it **cannot**
confirm the deconvolution is correct — Phase 3's own tests do that. Recording
this explicitly because "the recovery gate passes" is exactly the evidence a
reviewer would wrongly accept here.

---

## 7. Design reconciliation

`simulator_and_fragment_nll.md` currently contradicts this design:

| Existing text | Status |
|---|---|
| Per-region counts constant in Layer 1 (54 / 37) | **Superseded** by 126/121 |
| "matching each region's depth to a real sample's realised count is out of scope" | **Superseded** by 126 |
| `marginal_fl` is the empirical unweighted marginal; "nothing is deconvolved out of it" (decision 67) | **Reversed** by 127 |
| `observed_len_p` / `capture_marginal` / fixed point removed | **Stays removed** — 127 revives only the single LUT weighting |
| Count TSVs as the count source (121) | **Superseded** by 126 |

Also outstanding from the prior round: the decision-122 write-up (midpoint rule,
`|Ω|` 447,564 -> 479,232, gate result, embedding `parameter_recovery.png`),
deferred by the owner to the end of Layer 2.

---

## 8. Self-assessment

**Grade: B (revised down from B+ after self-review).** The owner constraints are
settled, the two defects in §3a/§3b are verified against source, and §4e now
rests on measurement rather than inference. It is downgraded because self-review
found **three population-alignment gaps (§4d) in a design whose central risk is
population alignment** — `marginal_fl`'s genomic scope, dedup scope against
overlap-based per-region fetch, and an unknown mapq filter on the duphist. Those
are not nits; a wrong choice on the first silently mixes a whole-genome capture
surface with a region-restricted length marginal.

The self-review is also the weakest evidence here: I reviewed my own design, and
the only reason it caught anything is that I re-derived the claims against source
and the real h5 instead of re-reading my own prose. Treat §4d as the list a
genuine external reviewer should attack first.

### Not verified

- **Cost and memory at full scale are unknown.** The research brief asked for this
  and it is still absent. The region set is ~1.5M rows across four sets; a
  whole-genome FL pass over a real cohort h5 may want caching rather than running
  inline. Deferred to Phase 2/3 **with a required measurement**, not a guess.
- **The duphist's mapq filter is unknown** — §4d item 3. This blocks Phase 0.
- **RD-56804 depth is quoted, not measured** — decision 123 records 2,551,319
  fragments / 38.3 per region. The 24,020 fragments measured in §4e are a
  200 kb chr1 slice and say nothing about total depth.
- **The GC measurement is one 200 kb window on chr1.** NaN fraction 0.0 there does
  not prove 0.0 genome-wide; unmapped or low-complexity regions are the likely
  place a 255 sentinel appears.
- **The `max_weight` cap's binding frequency is unmeasured.** Decision 128 settles
  which value to use and the cancellation argument holds regardless, so this does
  not block — but if the cap binds on substantial mass the realised FL will
  visibly differ from the target, and that should not arrive as a surprise.
- **No code has been run against the design.** Nothing is implemented.
