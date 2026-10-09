# Simulator from basic inputs: real per-region counts and a latent `marginal_fl`

Design, 2026-10-05. Written sequentially in one session; no sub-agents.

Supersedes `h5_derived_counts_and_fl.md` (2026-10-04), which was written before
the autosome artifacts existed and before the autosome scope was a decision
rather than a recommendation. Also contradicts specific text in
`simulator_and_fragment_nll.md` — the line-by-line reconciliation is §9.

**Status: design. Nothing implemented. Nothing in this document has been run
end to end.** Every number marked *(measured)* was produced in this session and
is reproducible from the commands in §7; every other number is attributed.

---

## 1. Problem

The simulator must run from **basic inputs**, with counting integrated into the
pipeline. Two inputs are wrong in kind.

### 1a. Per-region counts are fabricated

`background_model/simulator/sampler.py` carries

```python
_REGION_COUNTS = {2560: 54, 1536: 37}
```

behind `target_count_for_region(region_len)`. Every region of a given size gets
the same count. It must become the **real per-region fragment count of a real
sample**, derived from that sample's h5 inside the pipeline.

How far off the constant is, *(measured)*: over 500 regions drawn uniformly at
random from the 11,016-region autosome set, against
`ibd/frag_h5s/RD-56804.fragments.h5` at `min_mapq=10` under the midpoint rule,
the mean per-region count is **251.4** (median 247, min 0, max 1317). The
constant is 54. Real depth is ~4.7× the fabricated target, and — more to the
point — it varies over a 0-to-1317 range that a constant cannot express at all.

### 1b. `marginal_fl` applies capture twice — a live bug

`capture.py::build_marginal_fl` sums the duphist's deduplicated `molecule_keys`
per length and normalises. That is an **observed** distinct-molecule marginal,
i.e. already `N_true × P(seen)`.

But the generative weight in `weights.py::build_region_weights` is

```
E_s(c5, L) = end_s[hex(c3)] · marginal_fl(L) / predict(L, gc)
```

with `predict = min(1/P(seen), max_weight)`. Dividing by `predict` **multiplies
by `P(seen)`**. The model applies capture itself, so `marginal_fl` must be the
**latent** molecule marginal. Supplying an already-attenuated marginal applies
capture a second time.

This is not a refinement. It is a defect in the generative specification, and it
is invisible in the output: the result is still a normalised distribution, every
weight invariant still holds, and the realised FL still looks like a cfDNA length
profile.

---

## 2. Binding owner decisions

Not open questions. Listed so the implementation has one place to check against.

| # | Constraint |
|---|---|
| A | **Estimator, verbatim:** `marginal_fl(L) ∝ Σ_gc n_distinct(L,gc) · predict(L,gc)`, then normalise. `n_distinct` is the **deduplicated** h5 population. No fixed point, no `capture_marginal` ratio, no alternative estimator. |
| B | **Use the CAPPED `predict`** (`min(1/P(seen), max_weight)`). The weight formula divides by the capped value, so weighting the FL estimate by that same capped value makes the two cancel exactly. Not `1/P(seen)`. |
| C | `predict_lut` **stays on the duphist** via `GCFlDistModel` — the ZTNB fit needs duplicate multiplicities a plain h5 does not express as a histogram. Duphist for the capture surface, h5 for counts and FL. Two inputs by design. |
| D | **AUTOSOMES ONLY (chr1-22)** for all three quantities. The sample is male; chrX/chrY are single-copy and their duplicate structure differs from diploid autosomes. Matches the existing chrM exclusion in `build_duphist.py`. |
| E | **`marginal_fl` at `min_mapq=0`** to share a population with `predict_lut` (the duphist is mapq-unfiltered). **Per-region counts at `min_mapq=10`.** The split is deliberate. |
| F | **Sample RD-56804**, h5 `/efs/analytics/nathanboley/ibd/frag_h5s/RD-56804.fragments.h5`. `DEFAULT_SAMPLE` is `RD-56670` in both `run_simulator.py` and `validate_parameter_recovery.py` and must change. |
| G | Counts are the **FULL-region** target. Realised scored counts still vary via jitter and the FL filter; that variation is expected. **Zero-count regions are KEPT and emit nothing.** Absolute real depth is matched, not a rescaled shape. |

Decision A changes computed results. **Every anchor moves** (§6, Phase 5).

### 2.1 One question these decisions do not answer

**Does the per-region count include fragments with `L` outside `[25, 180]`?**

The simulator's support is `L ∈ [25, 180]`. The real h5 contains fragments
outside it. *(measured, same 500 regions)*: mean 251.4 with all lengths,
**243.3** restricted to `L ∈ [25, 180]` — a **3.3%** difference.

Decision G says "FULL-region", which settles the *spatial* question (whole region,
not the jittered crop) and names the FL filter as a scoring-time effect. It does
not say whether the count's own population is L-restricted. The two readings give
targets differing by 3.3%:

- **All-L count:** the simulator emits 251.4 fragments/region, all of them in
  `[25, 180]`, so the in-band depth is 3.3% *above* the sample's real in-band depth.
- **L-restricted count:** in-band depth matches exactly; total depth is 3.3% below.

Since the stated goal is matching **absolute real depth** and the simulator can
only produce in-band fragments, the L-restricted reading is the one that makes the
emitted population comparable to the real one. **But this is an algorithmic choice
that changes computed results, so it needs an explicit decision before Phase 3
lands.** It is flagged here rather than chosen.

### 2.1.1 RESOLVED — decision 134 (owner, 2026-10-05)

**L-RESTRICTED. The per-region target is the `L ∈ [25,180]` count: 243.3, not 251.4.**
Rationale as argued above — an all-L target asks for 251.4 fragments while every one
emitted must be in-band, inflating in-band density 3.3% over reality. Out-of-band
fragments are real biology the model does not represent. Phase 3 is unblocked.

**TODO, deliberately NOT part of this design — generate for ALL BANDS and let the
MODEL own the length filter.** The owner's accompanying note: the band is currently a
property of the *generator* (`L_MIN=25 / L_MAX=180`, `N_LENGTHS=156` in `weights.py`),
and it arguably belongs to the *model* — the simulator would emit the full length range
and the model would apply the restriction. That is a different architecture, not a
refinement of 134, and it would make 134 moot by construction, since an all-band
generator has no out-of-band population to exclude.

Do not begin it inside this work: it changes `|Ω|`, every anchor, and the `marginal_fl`
support, so starting it mid-flight would invalidate the gate this design exists to pass.
Resurface after the counts/FL work merges.

---

## 3. Inputs that already exist

Do not rebuild these.

| Artifact | Path | Status |
|---|---|---|
| Autosome duphist | `/efs/analytics/nathanboley/ctcf_fa_cache/duphist_merged/RD-56804__duphist_autosomes.tsv.gz` | exists, 3,848,469 bytes *(measured)* |
| Its provenance sidecar | `…/RD-56804__duphist_autosomes.json` | exists; records `total_molecules` 75,811,761, `n_rows` 1,077,130, 22 contigs, and two independent agreeing counts *(read directly)* |
| Autosome region set | `data/region_sets/quiet_v2_pad1200_repeats_removed_tile2560_autosomes.bed` | exists, **11,016 lines** *(measured, `wc -l`)*; source set is 11,505, so 489 chrX rows were dropped |
| Real sample h5 | `/efs/analytics/nathanboley/ibd/frag_h5s/RD-56804.fragments.h5` | exists, 589,166,072 bytes *(measured)* |

`predict_lut` must be **REFIT** on the autosome duphist. `load_duphist`
hardcodes `{sample}__duphist_wg.tsv.gz` and cannot reach the new file at all.

### 3.1 FINDING — the h5 named by the owner is not the h5 named by the duphist sidecar

The sidecar records

```
"h5_key": "ibd/frag_h5s/NC-14185/RD-56804-Lib1.hg38.fragments.h5"
```

while decision F names `/efs/analytics/nathanboley/ibd/frag_h5s/RD-56804.fragments.h5`.
Those are different paths and different filenames. The second exists as a flat
file; the `NC-14185/` subdirectory does not appear under `ibd/frag_h5s/` *(measured,
glob returned only the flat file)*.

This matters more than it looks. Decision B's exact-cancellation argument is
between `marginal_fl` and `predict`. It only holds if both sit on the **same
physical molecule population**. If the flat h5 is a different library prep, a
different merge, or a re-run of the same sample, `predict_lut` and `marginal_fl`
are estimated on different populations and the cancellation is void — silently,
because both quantities remain perfectly well-formed.

The sidecar does record `"independent_flgc_recount_mapq0": 75811761`, which
agrees with the per-contig sum. But it does not record **which h5 was recounted**.
If the recount read the flat file, the identity is already established and this
finding dissolves. If it read the `NC-14185/` key, nothing has been checked.

This is the blocking item for Phase 0. It is cheap to settle (§6, Phase 0).

---

## 4. The population contract

The central risk in this design is **a derived quantity silently computed over a
different population than the thing it is multiplied by**. §1b is that failure.
§3.1 is that failure. The GC unit trap in §5.2 is that failure. All three produce
plausible, normalised, non-crashing output.

So the contract is asserted at runtime and recorded in the manifest, not left to
prose.

| Quantity | Source | Genomic scope | mapq | GC-unknown | Dedup |
|---|---|---|---|---|---|
| `predict_lut` | `RD-56804__duphist_autosomes.tsv.gz` | chr1-22 | none (`min_mapq=0`) | dropped at build | duplicate-aware (ZTNB needs multiplicities) |
| `marginal_fl` | `RD-56804.fragments.h5` | chr1-22 | none (`min_mapq=0`) | **dropped**, count reported | **deduplicated** on `(start, stop)` |
| `region_counts` | same h5 | 11,016-region set, midpoint rule | `min_mapq=10` | n/a | **NOT** deduplicated — real depth is the target |

The `marginal_fl` / `region_counts` mapq split is deliberate (decision E) and must
be **recorded in the manifest, not silently inherited**. The justification is
asymmetric on purpose: `marginal_fl` must share a population with `predict` because
they multiply; `region_counts` sets *depth*, not *length shape*, and `mapq >= 10`
is what the driver's own readback verifies against.

---

## 5. Design

### 5.0 Call order changes

`fit_and_build(sample)` currently returns `(predict_lut, marginal_fl)` from one
duphist read. That cannot survive: the FL estimator **consumes** `predict_lut`, and
the two now read different files. The sequence becomes

```
predict_lut  = predict_lut_from_model(fit_capture_surface(duphist_path))
marginal_fl  = build_marginal_fl_from_h5(h5_path, predict_lut)
counts       = count_regions(h5_path, region_bed)
```

`fit_and_build` is deleted rather than kept as a wrapper. A wrapper that still
takes a bare `sample` string would re-admit the filename-derived path resolution
that §5.4 exists to remove.

### 5a. Refit `predict_lut` on the autosome duphist

`load_duphist(sample, duphist_dir)` → `load_duphist(path)`. The `{sample}__duphist_wg.tsv.gz`
pattern is deleted, not parameterised: a pattern with a `{scope}` placeholder is
one typo away from silently reading the whole-genome file.

Everything else in the fit path is unchanged — `prebin_gc_to_midpoints`,
`build_cell_map`, `SIM_LENGTH_BINS`, `SIM_GC_BINS`, `min_cell_size=200`,
`predict_lut_from_model`'s two guards. In particular, commit `a6dfc43`'s
pre-binning of duphist GC to bin midpoints via `weights.gc_bin_index` **must not
regress**; it is the thing that makes the fit path and the predict path agree by
construction. There is a test pinning it.

**Never join duphist tables on float `gc`.** Any verification that compares the
autosome duphist against the whole-genome one — a natural Phase 1 sanity check —
must join on the **integer bin code**, not on `gc`. Values like `30.31500053` are
text-parsed on one side and computed on the other, and the merge misses on
last-bit differences. This produced a false MISMATCH verdict in a prior session.

Expected and worth recording in the fit report, because a drop here is the same
silent-subsetting failure as the one `prebin_gc_to_midpoints` fixed: number of
`(length, gc)` cells offered to `fit`, number actually fitted after
`min_cell_size=200`, and the fraction of `predict_lut` entries sitting exactly at
`max_weight` (the cap-binding rate). Decision B's cancellation holds regardless of
the cap, but if the cap binds on substantial mass the realised FL will visibly
differ from target, and that should not arrive as a surprise.

### 5b. The latent FL estimator

Per autosome contig, read the whole contig's `starts`, `lengths`, `gc`, `mapq`:

```
drop gc == 255                       # unknown sentinel, BEFORE any decode
restrict L to [25, 180]
mapq: min(mate1, mate2) >= 0 under the retaining semantics (decision E)
dedup on (start, stop)
gi = clip(floor(100 * gc_u8 / 254 / 5), 0, 19)
li = L - 25
n_distinct[li, gi] += 1              # float64 accumulator
```

then, after all contigs:

```
marginal_fl[li] = Σ_gi n_distinct[li, gi] * predict_lut[li, gi]    # float64
marginal_fl    /= marginal_fl.sum()
```

Four things that are easy to get wrong and silent when wrong:

1. **GC is a fraction, not a percent.** The h5 stores `uint8`; `255` means unknown;
   otherwise the value decodes as `u8 / 254.0` into `[0, 1]`. Feeding the fraction
   straight into the floor rule gives `floor(0.45 / 5) == 0` for every fragment —
   the whole population lands in GC bin 0 and the deconvolution multiplies
   everything by `predict(L, 2.5)`. It runs, it normalises, it is wrong.
2. **Bin with `weights.gc_bin_index` (FLOOR on percent), not `flgc._bin_index`.**
   `flgc._bin_index` now ROUNDS, and the two disagree across the upper half of
   every bin (`4.9` → flgc 1, floor 0). The deconvolution indexes the *LUT*, so it
   must use the LUT's own rule — the same one `build_region_weights` uses.
3. **The clip is load-bearing, not defensive.** GC reaches exactly 1.0 in real
   data, and `floor(100 · 1.0 / 5) == 20` while valid indices are 0-19.
4. **Accumulate in float64.** Standing project rule. `np.add.at` / `+=` into a
   float32 buffer is the hazard class; `np.bincount` into int64 then one float64
   multiply avoids it entirely and is also faster.

**Dedup scope — stated explicitly, because this is the trap.** The dedup is
**contig-wide**, not per-region, and the FL population is **autosome-wide**, not
region-restricted.

`drop_duplicate_fragments()` is a method on a per-region array while
`from_fragments_h5` fetches by **overlap**. Tiles are contiguous, so a fragment
appears in adjacent tiles; dedup per region and then sum, and boundary fragments
are counted twice. A contig-wide pass visits every fragment exactly once, so the
hazard disappears — but only because the scope is contig-wide. If FL were ever
made region-restricted, each fragment would need midpoint assignment to a single
tile *before* dedup.

**Sanctioned-entry-point divergence, written down as CLAUDE.md requires.** The
contig-wide dedup does not go through `drop_duplicate_fragments()`, because that
method is scoped to a `RegionFragmentArray` and there is no region covering a whole
chromosome in this pipeline. The key is identical — `(starts_0, stops_0)` only,
**strand deliberately ignored**, matching the library's verified semantics — and is
implemented as a single `np.unique` over `start * 65536 + L`, which is bijective
with `(start, stop)` because `max_fragment_length` is 65535 *(measured, h5 root
attr)*. A test pins the equivalence: fetch one window as a `RegionFragmentArray`,
call `drop_duplicate_fragments()`, and assert the int64-key dedup over the same
arrays gives the same survivors. That test is what prevents the two
implementations from drifting.

The int64-key form is not just tidier, it is the difference between cheap and
not: *(measured, chr22, 4,130,133 length-filtered fragments)* `np.unique(key2d,
axis=1)` takes **6.18 s**, `np.unique(start*65536+L)` takes **0.19 s**, both
returning **1,025,848** distinct. Scaled to the autosomes that is ~8.5 min versus
~16 s.

### 5c. The per-region counting pass

One pass over the region set in BED order:

```python
with FragmentsH5(h5_path, cache_pointers=False) as fh5:
    for contig, gstart, gstop in regions:
        rfa = RegionFragmentArray.from_fragments_h5(
            fh5, Region(contig, gstart, gstop, strand=None), min_mapq=10)
        mid = gstart + rfa.starts_0 + rfa.lengths // 2
        counts[i] = int(((mid >= gstart) & (mid < gstop)).sum())
```

- `from_fragments_h5`, not `from_fname` (which is broken — it forwards kwargs the
  callee rejects).
- `strand=None`, so nothing arrives flipped.
- **Midpoint containment, not overlap.** `midpoint = start + L // 2` (floor),
  kept when `gstart <= midpoint < gstop`. This is the same rule
  `build_region_weights` admits fragments under, and the same rule the retired
  count TSVs used, so counting and generation stop being divergent.
- **Not deduplicated** — real depth is the target.
- Zero-count regions kept as `0`. *(measured)*: 6 of 500 random regions are zero,
  ≈1.2%, so ≈130 of 11,016 regions will emit nothing.

How much the overlap-vs-midpoint distinction is actually worth, *(measured)*:
overlap-fetched 261.8 fragments/region against 251.4 midpoint-contained — overlap
over-counts by **4.17%**. This is not a rounding detail; it is a systematic
inflation of every region's target.

`target_count_for_region` and `_REGION_COUNTS` are **deleted**, not kept behind a
fallback. A fallback silently resurrects constant counts whenever the h5 path is
misconfigured, which is exactly the failure §5.4 exists to prevent.

### 5d. Circularity guard on the real h5

`run_simulator.py` writes `{sample}.fragments.h5` into `--out-dir`, so simulated
output already sits at paths indistinguishable *by name* from a real sample h5:

```
/efs/analytics/nathanboley/background_model/sim_smoke/RD-56670.fragments.h5
/efs/analytics/nathanboley/background_model/sim_run_tile1536_20260930/RD-56670.fragments.h5
```

Feeding one back as the "real" sample would be self-confirming and would still
produce a clean, fully-normalised run.

**What the guard checks**, in order, all hard failures:

1. `--fragments-h5` is a **required, explicit path**. No default, never globbed,
   never derived from `--sample`. This is the whole guard's foundation; the rest
   is defence in depth.
2. **`_source_format` must be `"BAM"`.** *(measured)* The real h5 carries
   `_source_format = 'BAM'`; the simulator's h5 at
   `sim_smoke/RD-56670.fragments.h5` carries `_source_format = 'TSV'`.
3. **`_bam_header` must be non-empty.** *(measured)* The real h5's is a full
   `@HD`/`@SQ`/`@PG` chain naming `bwa-mem2 mem` and `samtools markdup`. The
   simulator's is the empty string.
4. **`_build_argv` must be absent.** *(measured)* The simulator's h5 carries
   `_build_argv` recording `build-fragments-h5 … RD-56670.bed.gz … --fasta …`,
   and `_build_code_revision = git:v2.14.0`. The real h5 carries neither attribute.
5. `realpath(dirname(h5)) != realpath(out_dir)`.

**Why this suffices, argued rather than asserted.** Checks 2-4 are *positive*
evidence of a BAM lineage, not merely absence of our fingerprint. A fragment h5
cannot acquire a `bwa-mem2`/`samtools markdup` `@PG` chain unless it was built
from an actual aligned, duplicate-marked BAM — and that is precisely the property
the duphist's ZTNB fit presupposes. So the guard is checking the thing that
actually matters (this came from a real sequencing run with real duplicate
structure), and it happens to exclude our own output as a corollary. Check 4 is
the sharpest: `_build_argv` literally records the BED the simulator emitted.

**What it does not do, stated plainly.** It does not prove the h5 is *the* sample
the duphist was built from — that is §3.1 and is a different check. It does not
resist forgery; attrs are writable. And its failure mode is conservative in the
safe direction: a legitimately-real h5 that happened to be rebuilt from a TSV
intermediate would be *rejected*, not accepted. Given the realistic failure is
re-reading our own output from a default directory, that trade is correct.

Recorded in the manifest for the h5: resolved `realpath`, size, mtime, and the
**per-contig fragment count vector**. The count vector is chosen over a sha256 of
589 MB deliberately — it is cheap (read from dataset shapes, no data read), it is
the population identity rather than a byte identity, and it is **directly
comparable with the duphist sidecar's `per_contig_molecules`**, which is what
§3.1 needs.

### 5e. Manifest changes

Manifest `version` bumps `1 → 2`. `load_manifest` must reject version 1 rather
than reading it with v2 semantics, because a v1 manifest's `marginal_fl` is the
double-counted one and nothing about its shape says so.

| Field | Change |
|---|---|
| `region_set_name` | → `quiet_v2_pad1200_repeats_removed_tile2560_autosomes` |
| `region_set_hash` | **recomputed.** The old hash must NOT be carried forward — the region set is a different file with 489 fewer rows, and a stale hash makes `load_manifest`'s mandatory `region_set` check pass against the wrong BED. |
| `per_region_counts` | same dict shape, keys `contig:start-stop`; now 11,016 real counts instead of a constant. Consumers keep working (`cut_site_oracle.py` sums it for `W_D`). |
| `counts_source` | **new.** `{h5_realpath, h5_size, h5_mtime, per_contig_counts, min_mapq: 10, membership: "midpoint in [start,stop)", dedup: false, l_range: <per §2.1>}` |
| `marginal_fl_source` | **new.** `{h5_realpath, scope: "autosomes_chr1_22", min_mapq: 0, gc_unknown: "dropped", n_gc_unknown_dropped, dedup: true, dedup_key: "(start, stop)", estimator: "sum_gc n_distinct * capped predict_lut", n_distinct_total}` |
| `predict_lut_source` | **new.** `{duphist_path, duphist_sha256, scope: "autosomes_chr1_22", min_mapq: 0, min_cell_size, max_weight, n_cells_offered, n_cells_fitted, cap_binding_fraction}` |

Manifest size grows: 11,016 count entries at ~45 bytes of indented JSON each is
~0.5 MB, on top of the ~0.84 MB the current smoke manifest already occupies
*(measured: `sim_smoke/RD-56670.manifest.json` is 838,760 bytes)*. ~1.4 MB is
acceptable; it is not worth a format change.

---

## 6. Phased plan

Each phase is independently reviewable and names **what must FAIL** if it is
wrong — a construction that a wrong implementation cannot pass, not an invariant
a wrong implementation also satisfies.

### Phase 0 (blocking) — establish that the h5 and the duphist are the same population

§3.1. Recount distinct molecules per autosome from
`ibd/frag_h5s/RD-56804.fragments.h5` at `min_mapq=0`, GC-known only, using the
duphist's own length range and dedup key, and compare against the sidecar's
`per_contig_molecules` (chr1 6,853,720 … chr22 1,104,152, total 75,811,761).

If they agree, the population contract rests on data. If they disagree, decision
B's cancellation is void and **everything downstream is meaningless** — that is
the finding, and the right response is to locate the h5 the duphist was built
from, not to proceed.

**Must FAIL if wrong:** compare per contig, not on the total. A total can match
while the per-contig vector does not (different merge, same depth). Assert the
full 22-element vector.

**Second must-FAIL, for the contract machinery itself:** construct a case where
the FL population and the `predict_lut` population deliberately disagree — e.g. a
region-restricted FL against the autosome-wide duphist — and assert the contract
check rejects it. A check that only validates array *shapes* passes this and is
worthless; the shapes are identical either way.

### Phase 1 — refit `predict_lut` on the autosome duphist

`load_duphist(path)`; fit; report cells offered / cells fitted / cap-binding rate.

**Must FAIL if wrong:** re-point `load_duphist` at the whole-genome file and
assert the resulting `predict_lut` **differs**. If the autosome and whole-genome
LUTs come out identical, the new path is not being read and the refit is a no-op —
which is the realistic failure, since both files live in the same directory with
nearly the same name. Separately, assert the GC pre-binning still holds: feed the
known fractional values (`4.3307, 4.7244, 9.0551, …`) and assert zero mass is
dropped. The ~18% silent-drop defect that `a6dfc43` fixed is one careless edit away.

**Do not** verify by joining the two duphist tables on `gc`.

### Phase 2 — the latent FL estimator

**Must FAIL if wrong:**
- **The unit trap.** A synthetic population with a known GC spread must produce a
  *different* `marginal_fl` when binned from percent than when binned from the raw
  fraction. If the two agree, the `×100` is missing and every fragment is in bin 0.
  Asserting "sums to 1" passes in both cases.
- **The deconvolution is actually applied.** With a deliberately non-flat
  `predict_lut`, the result must differ from the raw normalised dedup length
  histogram. A no-op implementation satisfies every normalisation check; this
  comparison is the only guard.
- **Flat-LUT round trip.** With a flat `predict_lut`, the result must equal the raw
  normalised dedup length marginal *exactly*. This is the complement of the
  previous test and catches an over-applied weighting.
- **Dedup key equivalence.** Fetch one window as a `RegionFragmentArray`, call
  `drop_duplicate_fragments()`, and assert the int64-key dedup selects the same
  survivors. Include a same-span opposite-strand pair, since the library key
  ignores strand and collapses them — if the fast path keeps both, the two
  implementations have already diverged.
- **GC-unknown.** A population containing `255` must not silently place those
  fragments in bin 0; the dropped count must be reported.
- **Dedup scope.** Compute FL region-wise-then-summed and contig-wide on the same
  data with tiles that abut, and assert they differ. If they agree, the test data
  has no boundary-straddling fragment and the test proves nothing — so the
  fixture must place one deliberately.

### Phase 3 — the per-region counting pass

Depends on §2.1 being decided first.

**Must FAIL if wrong:**
- **Boundary double-count.** Place a fragment straddling the shared boundary of
  two adjacent tiles and assert the sum over tiles equals the number of distinct
  in-window fragments. An overlap-based implementation counts it twice and fails.
  *(measured: the real overlap/midpoint inflation is 4.17%, so this is live, not
  hypothetical.)*
- **Midpoint-in / endpoints-out.** A fragment whose midpoint is in-region but whose
  endpoint lies up to `MAX_FL_HALF = 90` bp outside must be COUNTED. Construct one
  at a region edge with `L = L_MAX` and assert inclusion. A containment
  implementation passes a midpoint test *only* if no fragment straddles a boundary,
  so the fixture must place one there deliberately.
- **Zero-count retention.** A region with no fragments yields `0` and is retained.
  Assert `len(counts) == len(region_set)`, not merely that zeros are allowed.
- **Constant-count regression.** Assert the realised counts have non-zero variance.
  If a future refactor reintroduces a constant, every other test here still passes.

### Phase 4 — wiring, defaults, guard, manifest

`DEFAULT_SAMPLE` → RD-56804 in `run_simulator.py`, `validate_parameter_recovery.py`
and `profile_simulator.py`; `DEFAULT_REGION_SET` → the autosome BED; required
`--fragments-h5` and `--duphist`; the §5d guard; the §5e manifest fields; delete
`_REGION_COUNTS` and `target_count_for_region`.

**Must FAIL if wrong:** point `--fragments-h5` at
`sim_smoke/RD-56670.fragments.h5` — an **actual emitted h5**, not a mock — and
assert the guard raises. A mock cannot test this, because the whole question is
whether the attrs real `build-fragments-h5` writes are distinguishable from the
attrs real `build-fragments-h5-from-BAM` writes. Also assert the *converse*:
the real h5 passes. A guard that rejects everything also "passes" the first test.

**Must FAIL if wrong, manifest:** load a v1 manifest and assert it raises. A v1
manifest's `marginal_fl` is the double-counted one; reading it silently under v2
semantics reintroduces the bug this design removes.

### Phase 5 — anchors and the gate

1. Recompute the store's own anchors (oracle, uniform, gap) from
   `scripts/cut_site_oracle.py`. Anchors are **per store** and are recomputed per
   store, never quoted from a previous one.
2. Re-run `scripts/validate_parameter_recovery.py` at its default scale against
   RD-56804 and re-commit its artifacts.
3. **Retire** the pre-change anchor figures rather than carrying them forward.

**How to re-interpret the gate — and the trap in it.** The recovery gate's
*interpretation does not change*: it tests whether the sampler draws from the
model the weights define, which is a self-consistency property. It will pass both
before and after the `marginal_fl` change, so it **cannot** confirm the
deconvolution is correct. Phase 2's tests do that. This is recorded explicitly
because "the recovery gate passes" is exactly the evidence a reviewer would
wrongly accept here.

What *does* move and must be reported: the oracle NLL, the uniform NLL, and the
gap — all three, with `|Ω|` and the FL bands stated alongside, since the
denominator changes with the bands. And `W_D`, which now uses real per-region
counts; `cut_site_oracle.py` computes it as `n_total_store / n_total_emitted` from
`manifest["per_region_counts"]`, so it keeps working but takes a different value.

**Dependency on out-of-scope fixes.** Phases 3-5 assume the two latent breaks in
`run_simulator.py`'s self-check are repaired separately: the
`n_emitted != target * n_regions` assert (assumes a constant per-region target —
dead on arrival under Phase 3) and the `inside = (g_starts >= gstart) & (g_stops <= gstop)`
containment test (undercounts under the midpoint rule). **This design does not fix
them and must not be merged ahead of them**, or Phase 3 will be debugged against a
driver that cannot pass its own checks.

---

## 7. Cost and memory at full scale

All *(measured)* this session, single-threaded, on the real 589 MB h5 over NFS,
under `/home/nathanboley/miniconda3/envs/biomarker_env/bin/python`.

| Pass | Measurement | Extrapolation to full scale |
|---|---|---|
| Per-region counting, BED order | 200 regions in 2.29 s = **11.4 ms/region** | 11,016 regions ≈ **126 s** |
| Per-region counting, random order | 500 regions in 16.5 s = **32.9 ms/region** | 11,016 regions ≈ **363 s** |
| Contig read (starts/L/gc/mapq) | chr22, 4,351,071 frags, **0.35 s**, ~39 MB | 356,737,933 autosomal frags ≈ **29 s**, ~3.2 GB read |
| Dedup, int64 key | chr22, 4,130,133 → 1,025,848 in **0.19 s** | ≈ **16 s** |
| Dedup, `np.unique(axis=1)` | same input, **6.18 s** | ≈ **8.5 min** — do not use |

**Totals.** FL pass ≈ 45-60 s. Counting pass ≈ 2 min in BED order. Combined under
~3 minutes against a run whose per-region sampling loop alone is the dominant cost
(the existing driver reports ms/region for Steps 3-5 and extrapolates to tens of
minutes at 66,649 regions).

**Memory.** The counting pass is negligible — the largest region returned 1,317
fragments. The FL pass is bounded by the largest contig: chr1 at 31,038,889
fragments *(measured)* × (4 B starts + 2 B lengths + 1 B gc + 2 B mapq) ≈ **280 MB**
of raw arrays, plus an 8-byte int64 key ≈ 248 MB, plus `np.unique`'s sort copy.
Peak ≈ **1.0-1.3 GB** per contig, released between contigs. No chunking needed;
if it ever is, the contig loop is already the chunk boundary.

### Must the counting pass be a cached artifact?

**No. Run it inline.** Three reasons, in order of weight:

1. **It is cheap relative to the run it feeds.** ~2 minutes against a multi-ten-minute
   sampling loop. Caching buys nothing measurable.
2. **A cache is a second provenance surface.** The whole point of decision 126 is
   that the pipeline runs from basic inputs. A `*.region_counts.tsv.gz` sitting on
   EFS can go stale against the h5, against the region set, or against the mapq
   rule, and nothing in the simulator would notice — it would read a well-formed
   file of plausible integers. That is the same failure class as §1b.
3. **The counts are already persisted where they matter** — `per_region_counts` in
   the manifest. So a *rerun* is reproducible without re-reading the h5, while a
   *fresh run* always re-derives from source.

Order the region set in BED order for the pass (it already is), since random
access costs 2.9× *(measured: 32.9 vs 11.4 ms/region)*.

The **FL pass** is likewise inline, and is additionally unfit for caching: it
consumes `predict_lut`, so a cached FL would silently decouple from a refitted
capture surface — reintroducing §1b in a new disguise.

---

## 8. What breaks

### Tests

| Location | What breaks |
|---|---|
| `tests/test_simulator_phase2.py::TestMarginalFL` — `test_sum_exactly_one`, `test_all_nonneg`, `test_exactly_156_entries` | All three call `load_duphist("RD-56670")` then `build_marginal_fl(df)`. Both signatures change: `load_duphist` takes a path, `build_marginal_fl` moves to the h5 and takes `predict_lut`. The *assertions* (sums to 1, non-negative, shape 156) survive and should be re-pointed, not deleted — but note they are exactly the checks a no-op deconvolution also passes, so Phase 2's tests must be added, not substituted. |
| `tests/test_simulator_phase2.py::TestMarginalFL::test_empty_duphist_raises` | Asserts `ValueError` matching `"No molecule_keys"` on a DataFrame with out-of-range lengths. That contract disappears with the DataFrame input. Needs an h5-shaped equivalent. |
| `tests/test_simulator_phase2.py::test_fit_and_build_on_real_sample` | `fit_and_build("RD-56670")` — the function is deleted (§5.0). |
| `tests/test_simulator_phase3.py` (~L688-695) | Three asserts: `target_count_for_region(2560) == 54`, `(1536) == 37`, `KeyError` on 999. Deleted with the function. |
| `tests/test_simulator_phase3.py` (~L299, L312) | Manifest round trip with `per_region_counts={"chr1:1000-3560": 54}`. Survives structurally, but needs updating for the v2 version bump and the three new `*_source` blocks. |

Every one of these fails on an **import or signature error**, not a wrong value —
which is the good case. The dangerous breakage is the opposite kind and is in §8.3.

### Scripts

| Script | What breaks |
|---|---|
| `scripts/run_simulator.py` | `DEFAULT_SAMPLE`, `DEFAULT_REGION_SET`, `fit_and_build`, `target_count_for_region`, the two self-check asserts (fixed separately — see Phase 5 note). |
| `scripts/validate_parameter_recovery.py` | `DEFAULT_SAMPLE` (L99), `fit_and_build` (L161). |
| `scripts/profile_simulator.py` | `DEFAULT_SAMPLE` (L48), `target_count_for_region` (L86), `fit_and_build` (L89). |
| `scripts/verify_vectorize.py` | `target_count_for_region` (L152), `fit_and_build("RD-56670")` (L154). |
| `scripts/_prof2.py` | `build_marginal_fl(load_duphist("RD-56670"))` (L49), `target_count_for_region` (L223). |
| `scripts/cut_site_oracle.py` | Survives. `sum(manifest["per_region_counts"].values())` (L126) still works and is now correct; `W_D` takes a different value. |

### 8.3 Stored provenance that assumes constant counts or a non-deconvolved FL

These do not fail loudly. They are the ones to actively retire.

- **Every existing `*.manifest.json`** under
  `/efs/analytics/nathanboley/background_model/sim_smoke/` and
  `sim_run_tile1536_20260930/`. Their `marginal_fl` is double-counted and their
  `per_region_counts` is a constant. The v1-rejection in §5e is what makes this
  visible instead of silent.
- **`sim_tile1536.zarr` and `sim_tile1536_v2.zarr`**, and any anchors derived from
  them.
- **The anchor tables in `simulator_and_fragment_nll.md` §"Measured anchors"**
  (oracle 12.240380 / 12.244812, gap 0.765109 / 0.766759). Already marked
  SUPERSEDED for the midpoint-rule change; now superseded for a second,
  independent reason. The note there says they "will be recomputed once the
  per-region-count alignment change also lands" — this is that change.
- **`sim_sample_sheet.tsv` / `sim_12track_tile1536_config.json`** if they name
  RD-56670 or the chrX-inclusive region set. Not inspected; flagged.
- **The `_wg` duphist**, `RD-56804__duphist_wg.tsv.gz`. It is not deleted (other
  work may use it) but nothing in this pipeline may read it after Phase 1.

---

## 9. Design reconciliation with `simulator_and_fragment_nll.md`

That document is the simulator's specification and it currently asserts the
opposite of this design in five places. Reconciling it is part of the job.

| Where | Existing text | Status |
|---|---|---|
| Step 5, "In:" | "the per-region target count (**54** at region_len 2560, **37** at 1536)" | **Superseded.** Counts come from the h5, per region, with the measured distribution in §1a. |
| Step 2 | "for each `L`, sum `molecule_keys` across all GC values; restrict to `L = 25..180`; normalise to sum 1. **Nothing is deconvolved.**" and "**Out:** the empirical unweighted length marginal" | **Reversed** by decision A. This is the §1b bug. Decision 67's "nothing is deconvolved out of it… do not reopen it" no longer stands. |
| Step 1, "In:" | "duphist `duphist_merged/<sid>__duphist_wg.tsv.gz`" | **Superseded.** Autosome duphist, explicit path, refitted. |
| Data table, "region sets" | "`…tile2560` (11,505 tiles), `…tile1536` (66,649) — **both include chrX** (decision 94)" | **Superseded** by decision D. The tile2560 set becomes the 11,016-row autosome set. Decision 94 is reversed. The tile1536 set has **no** autosome variant yet — see §10. |
| Data table, "length marginal" | "`duphist_merged/<sid>__duphist_wg.tsv.gz`, deduped `molecule_keys`" | **Superseded.** Fragment h5, deduplicated, LUT-weighted. |
| Manifest §, "per-region counts" | listed as a stored field | **Stays**, now carrying real values; joined by three new `*_source` blocks (§5e). |
| Step 4 / Appendix A / Appendix E | the `E_s`/`Z_s`/`S_s` factorisation and the `Σ_Ω w = 1` proof | **Unchanged.** The normalisation algebra does not depend on where `marginal_fl` came from. This is worth stating: it is why the bug was invisible. |
| Appendix F | bin boundary semantics | **Unchanged**, and now load-bearing in a second place (§5b item 2). |
| "Measured anchors" § | the v1/v2 anchor tables | **Superseded** (§8.3). |
| Validation §, "retired `observed_len_p` / `capture_marginal` / fixed point" | removed forms | **Stay removed.** Decision A revives only the single LUT weighting. |

`h5_derived_counts_and_fl.md` is superseded wholesale by this document. Its §4d
recommendations (autosome scope, `min_mapq=0` for FL) were correct and are now
decisions D and E; its §4f "the duphist's mapq filter is unknown" is resolved by
decision E making it a stipulation rather than an inference; its §3b GC trap
survives as §5b.

---

## 10. Self-assessment

**Grade: B.**

What earns it: the two defects are verified against source rather than restated;
every cost, memory and distribution number in §1a, §5b, §5c and §7 was measured
in this session against the real artifacts rather than estimated; §3.1 is a
genuine finding that neither the brief nor the superseded design contains; and
§5d's guard is built on *measured* h5 attributes (`_source_format`, `_bam_header`,
`_build_argv`) rather than the directory heuristic the prior design proposed,
which makes it a positive test for real data instead of a negative test for our
own output.

What holds it down:

- **§2.1 is an unresolved algorithmic question discovered late**, and it gates
  Phase 3. A design that cannot say what number it is targeting to better than
  3.3% is not finished.
- **No code was run against the design.** Every measurement is of the *existing*
  system; nothing validates the proposed estimator.
- **§3.1 may invalidate the premise.** If the h5 and the duphist are different
  libraries, decision B's cancellation — the justification for the capped
  `predict` — does not hold, and Phases 1-5 are built on sand. I could not settle
  it from the artifacts available.

### What I could not verify

- **Whether `/efs/analytics/nathanboley/ibd/frag_h5s/RD-56804.fragments.h5` is the
  h5 the duphist was built from.** The sidecar names a different path
  (`ibd/frag_h5s/NC-14185/RD-56804-Lib1.hg38.fragments.h5`) and does not record
  which file its `independent_flgc_recount_mapq0` read. §3.1, Phase 0.
- **The 75,811,761 / 1,077,130 duphist figures.** I read them from the sidecar
  JSON; I did not recount them. The brief states they were verified two ways;
  I am relaying that, not confirming it.
- **The claim that the autosome BED's retained rows are byte-identical to the
  source minus chrX.** I measured the line counts (11,016 and 11,505, differing
  by 489) and the first two rows' geometry (region_len 2560). I did not diff the
  retained rows.
- **The `max_weight` cap-binding fraction.** Unmeasured. Decision B settles which
  value to use and the cancellation holds regardless, so it does not block — but
  if the cap binds on substantial mass the realised FL will visibly differ from
  target. §5a asks for it as a reported number.
- **Whether `min_mapq=0` through `RegionFragmentArray.from_fragments_h5` reproduces
  `flgc._mapq_pass`'s retaining semantics for the -1 sentinel.** The two are
  specified to agree; I did not test them against each other. *(measured)* the
  chr22 `mapq` array contains no `255` values at all, so the sentinel path is
  untested by that data — which means a local spot-check cannot settle it either.
- **The per-contig duplicate ratio.** *(measured)* raw autosomal records total
  356,737,933 against the sidecar's 75,811,761 distinct molecules, an overall
  ratio of ~4.7. But the per-contig ratios implied by the two vectors range from
  roughly 3.9 (chr7) to 7.5 (chr8). That spread may be ordinary coverage
  variation or may be a symptom of §3.1. I did not investigate it, and it should
  be looked at during Phase 0 rather than filed as noise.
- **`sim_sample_sheet.tsv` and `sim_12track_tile1536_config.json`** — listed in
  §8.3 as possibly carrying stale sample/region-set references. Not opened.
- **Whether a `tile1536` autosome region set is needed.** The existing 66,649-tile
  set includes chrX. Decision D is about the *quantities*, and the brief only
  names the tile2560 autosome BED as built. If Layer 2 runs at 1536, a matching
  autosome set does not exist yet. Flagged, not assumed.

---

## Review Notes (2026-10-05)

**Verdict**: APPROVED WITH CONDITIONS
**Grade**: A-

### Risks
1. **Manifest version collision (must-fix).** §5e says "bumps 1→2" but `emit.py`
   already has `MANIFEST_VERSION = 2` (for the `pad` field). The deconvolved
   `marginal_fl` needs 2→3, or existing v2 manifests with double-counted FL load
   without error under the new code — the exact silent failure this design exists
   to prevent.
2. **§5c pseudocode omits the L-restriction (must-fix).** Decision 134 mandates
   `L ∈ [25,180]` but the counting snippet has no length filter; `from_fragments_h5`
   defaults to `max_frag_len=511`. As written it produces ~251.4, not ~243.3.
3. **mapq sentinel gap.** `min_mapq=0` excludes `-1` sentinels from the FL estimator
   while the duphist includes them in `predict_lut`. Benign for RD-56804 (no
   sentinel values observed) but a latent population mismatch for other samples.

### Conditions
1. Bump manifest version to 3 (not 2). Reject v1 AND v2 in `load_manifest`.
2. Add the L-restriction to §5c's pseudocode and to Phase 3's "must FAIL" tests.
3. Rewrite §3.1 to reflect that the duphist provenance question is settled (flat-path
   recount agrees, sidecar updated). Phase 0 shrinks to a runtime assertion.
4. State an explicit mapq sentinel policy for `marginal_fl`, even if the policy is
   "accepted gap for this sample."

### Key Tradeoffs
- **Inline counting over caching**: correct — 2 min inline vs a second provenance
  surface that can go stale. The argument at §7 is well-made.
- **Contig-wide dedup for FL vs the library's per-region `drop_duplicate_fragments`**:
  the sanctioned-entry-point divergence is justified (per-region dedup double-counts
  boundary fragments), the key equivalence test is well-designed, and the int64 key
  is 32× faster than `np.unique(axis=1)`. Written down as CLAUDE.md requires.
- **Circularity guard built on positive h5 attributes vs directory heuristic**: better
  than the prior design's approach. Checks 2-3 (`_source_format="BAM"`, `_bam_header`
  chain) are the real guards; check 4 (`_build_argv` absent) is fragile for future h5s
  but conservative.
- **Recovery gate reinterpretation (Phase 5)**: the doc correctly identifies that the
  gate cannot confirm the deconvolution — it is a self-consistency check that passes
  both before and after. Phase 2's tests are the actual guard. This is an important
  insight that prevents a false sense of validation.

### Verified against source
- `gc_bin_index` FLOOR rule: `weights.py:116` — `floor(gc / 5)` with clip
- `_bin_index` ROUND rule: `flgc/model.py:57` — `int(np.round(value))`
- `drop_duplicate_fragments` key: `fragment_array.py:1036-1040` — `(starts_0, stops_0)`, strand ignored
- `MANIFEST_VERSION = 2`: `emit.py:399` — already bumped for `pad` field
- `DEFAULT_MAX_FRAG_LEN = 511`: `constants.py:17`
- `DEFAULT_SAMPLE = "RD-56670"` in all three scripts: confirmed
- Self-check breaks: `run_simulator.py:664` (constant-target), `:486` (containment)
- `_source_format`, `_bam_header`, `_build_argv` h5 schema: confirmed via `fragments_h5/AGENT_CONTEXT.md`

### Not verified (would require code execution)
- The *(measured)* numbers: 243.3/251.4 means, 4.17% inflation, 11.4 ms/region,
  75,811,761 molecules, chr1 31,038,889 fragments
- Byte identity of autosome BED rows vs source minus chrX
- The `max_weight` cap-binding fraction
