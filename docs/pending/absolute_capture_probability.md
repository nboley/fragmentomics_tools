# Estimating absolute capture probability by (length, GC)

**Status:** problem statement. No implementation, no chosen approach.
**Scope:** the model lives in `~/src/biomarker/flgc/`; the consumer lives in
`fragmentomics_tools` (`fragment_array/weights.py`, `GCFlWeights`).

Everything below labelled VERIFIED was checked by execution against the real
data on 2026-09-24. Everything labelled INFERENCE was not.

---

## 1. What we want

A per-fragment weight that supports an **unbiased estimate of the original
molecule count** — i.e. inverse-probability (Horvitz–Thompson) weighting:

```
weight(L, G) = 1 / P(seen | L, G)
```

where `P(seen | L, G)` is the probability that a cfDNA molecule of length `L`
and GC fraction `G`, present in the input plasma, produces at least one
observation in the final data.

A weighted pileup would then estimate original copy number at a locus,
corrected for length/GC-dependent losses, rather than merely rebalancing the
observed reads.

## 2. What we currently have — two estimators, different semantics

`GCFlDistModel` exposes both through one `predict(length, gc)` interface, which
makes them easy to confuse. They are not interchangeable.

| | `fit()` — ZTNB | `fit_from_spikes()` — spike grid |
|---|---|---|
| fitted on | native cfDNA duplicate counts | spike-in read counts |
| estimates | **absolute** `P(seen)` | recovery **relative to panel mean** |
| formula | `1/P(seen)`, `P(seen) = 1-(1-w)·P_NB(0)` | `exp(-log(count / grand_mean))` |
| range | **≥ 1** by construction | can be **< 1** |
| effect on totals | inflates | approximately preserves |

VERIFIED, measured:

```
ZTNB       lengths 35, 45 @ gc 35/45%  ->  [1.717, 2.367]
spike-grid lengths 30..70 @ gc 50%     ->  [1.028, 0.761, 0.689, 0.761, 1.028]
```

The spike-grid U-shape has its minimum at the best-recovered cell: cells
recovered better than the panel average are *deflated*. That is a coherent
correction, but it is not `1/P(seen)` and cannot be used as one.

Source: `flgc/_ztnb.py:42-48`, `flgc/model.py:301, 376, 411`.

### Consequences of the ZTNB path worth knowing

- **The no-data default is `MAX_WEIGHT`, not 1.0** (`model.py:403`). A fragment
  outside every fitted bin, or in a bin below `MIN_CELL_SIZE = 200`, is weighted
  3.0 — maximally up-weighted, not left neutral. On a low-depth library where
  most cells fail that threshold, the result is a near-uniform 3× inflation
  wearing the costume of a correction.
- **The `MAX_WEIGHT = 3.0` clamp reintroduces the bias it bounds.** 3.0
  corresponds to `P(seen) = 0.33`; any cell genuinely recovered worse than that
  stays permanently under-corrected. The correction is weakest exactly where
  bias is worst.

## 3. The data we actually have

### 3.1 Spike metadata

Loaded via `flgc.spike_data.load_spikes(profile)` (`spike_data.py:52`), which
drops ID spikes at `spike_data.py:35` and derives `length`, `gc_perc`,
`use_with_model`, `molarity_scale_factor`, `is_spank`.

The source TSVs have **13 columns**; the loader reads **4**. Dropped on load:

| col | content |
|---|---|
| 1 | `M in plasma` |
| 2 | **`MPM in plasma = 1x`** — molecules per mL |
| 3 | `M in 100x` — **this is what the code calls `molarity`** |

So the `molarity` used in code is the **100× stock concentration**, not the
in-plasma quantity, and the direct molecules-per-mL figure is discarded.

VERIFIED, SNMv3 model panel (28 rows): `M in plasma` = 5.00e-14,
`MPM in plasma = 1x` = **3.00e+04**, `M in 100x` = 5.00e-12 — all uniform.

### 3.2 Molarity structure differs by profile

VERIFIED across all four profiles, `use_with_model` subset only:

| profile | n | molarities | `5e-12/molarity` | scaling |
|---|---|---|---|---|
| SNMv3 | 20 | `[5.0e-12]` | `[1.0]` | **inert** |
| SNMv4 | 29 | `[5.0e-11, 5.0e-10]` | `[0.01, 0.1]` | active |
| SNMv4B | 29 | `[5.0e-13, 2.5e-12]` | `[2.0, 10.0]` | active |
| SNMv4C | 29 | `[5.0e-13, 2.5e-12]` | `[2.0, 10.0]` | active |

`molarity_scaled_count = (5e-12 / molarity) · (count + 1)`
(`spike_data.py:43, 214`).

For **SNMv3 only**, that scale factor is exactly 1.0, so the column reduces to
`count + 1` and the molarity correction does nothing. For the v4 family it
applies real 5–10× corrections between two concentration tiers.

**A plan must state which profile it targets.** The panels differ in
concentration, in whether molarity scaling is active, and in length coverage
(SNMv3 32–75 bp; v4 family 24–175 bp).

### 3.3 The blocker: no UMIs on the model panel

VERIFIED:

```
GC-dSpark model panel:  0 of 28 sequences contain any N   -> no UMI
SPANK family:           2 rows, 16 N bases each           -> UMI
```

Every molecule of a given GC-dSpark has an **identical sequence**, so all
copies align to the same coordinates. These are indistinguishable in the data:

- 30,000 molecules recovered, each sequenced once
- 100 molecules recovered, each sequenced 300×

Coordinate-based deduplication — which works on native cfDNA because real
fragments have varied endpoints — collapses the entire spike to one position.

**Therefore `observed_molecules` is not measurable for the model panel.** Only
read counts are available, and those conflate capture probability with PCR
amplification and sequencing depth.

This, not the input quantity, is what blocks absolute `P(seen)` from spikes.
The input side is solved: 3e4 molecules/mL is known, and plasma volume is a
standard assay parameter.

### 3.4 What already produces an absolute number

`preprocessing/spank/spikein_ztnb_fit.py:334-341` fits ZTNB to **SPANK UMI
duplicate distributions** and computes exactly the target quantity:

```python
pi_mix = 1.0 - p0_mix              # absolute P(seen)
c_mix  = 1.0 / max(pi_mix, 1e-15)  # Horvitz-Thompson factor
```

Limits: SPANK is tiny and profile-dependent. VERIFIED via `load_spikes`:

| profile | SPANK spikes |
|---|---|
| SNMv3 | **1** (`spank-75B`) |
| SNMv4 | **1** (`v4-Spank-ss-A`) |
| SNMv4B / SNMv4C | 2 (`SPANK-52C`, `spank-75B`) |

So on SNMv3 or SNMv4 there is a **single** anchor point, not two. It cannot be
stratified across the grid, and using it as an anchor assumes recovery is
linear in concentration across a 10× gap.

(An earlier revision of this document said 2 rows for SNMv3. That came from a
hand-rolled parse of the raw TSV which matched `Spank75A` — a row the file
itself marks "NOT INCLUDED IN SNMv3". `load_spikes` excludes it correctly. The
loader is the right entry point; the hand-rolled read was the error.)

The native-cfDNA ZTNB path (`fit()`) also yields absolute `P(seen)` and is
stratified by (length, GC). It is arguably the better instrument, since it
measures the molecules of interest rather than synthetic proxies. Its weakness
is that it infers input from the duplicate-count distribution rather than
knowing it.

## 4. The problem, stated precisely

We have two independent routes to `P(seen)`, each with a complementary gap:

- **Native ZTNB**: stratified by (length, GC), but input is *inferred* from
  duplicate structure, under a NB assumption, with `K_FIT = 25` truncation and
  `MIN_CELL_SIZE = 200`.
- **Spikes**: input is *known* exactly, but recovered-molecule count is
  unmeasurable without UMIs, and the panel is synthetic.

The question is whether these can be combined into a better-founded estimate
than either alone, and what it would cost.

## 5. Open questions a plan must resolve

1. **Which spike profile is production actually using?** This determines
   whether molarity scaling is inert or active, and the available length range.
2. ~~**Can the two-tier v4 design be exploited?**~~ **RESOLVED: no.** This was
   proposed on the assumption that the same (length, GC) cell appears at both
   concentrations, making a linearity check possible on existing data. It does
   not. VERIFIED across SNMv4, SNMv4B, SNMv4C:

   ```
   low tier :  25 spikes, lengths  24-75
   high tier:   4 spikes, lengths 100-175
   OVERLAPPING (length, gc) cells: 0
   ```

   The tiers are a **length split, not a dose series** — long fragments are
   spiked at 5-10× to compensate for expected poorer recovery. No cell exists at
   both concentrations, so concentration-dependence cannot be tested from this
   panel, and `molarity_scale_factor` is doing necessary work putting a
   deliberately non-uniform design onto a common basis.
3. ~~**Is the two-tier slope informative without UMIs?**~~ **Moot**, per the
   above — there is no slope to take.
4. **Do synthetic spikes transfer to native cfDNA at all?** Spikes are blunt-
   ended oligos with no nucleosome context, no epigenetic modification, and
   different secondary structure. Even the *relative* surface may not transfer.
   This is worth quantifying regardless of which route is chosen.
5. **What validates any of this?** A weighted pileup at a known-truth locus, or
   agreement between independently-derived estimates, or a titration. Without a
   check, an unbiased-looking estimator cannot be distinguished from a confident
   wrong one.
6. **What breaks downstream?** Changing spike-grid semantics from relative to
   absolute flips weights from possibly-<1 to ≥1, changes totals from roughly
   preserved to inflated, and interacts with the 0.10 floor and `MAX_WEIGHT`
   clamp. Consumers assuming the current semantics would silently misread it.

## 6. Constraint worth stating plainly

If the answer turns out to be "add UMIs to the GC-dSpark panel", that is a
**wet-lab change, not a code change** — and it is the only change that removes
the blocker rather than working around it. The input quantity is already known;
the molecule identity is what was never encoded. A plan should say clearly
whether it is proposing a statistical workaround for a measurement that was not
made, and what that workaround costs in assumptions.

---

## 7. Outcome of the planning pass (2026-09-24)

A planning agent worked this problem and recommended, with high confidence,
**not pursuing it as a code project.** The reasoning, which holds up:

- The existing native-cfDNA ZTNB path **already produces absolute P(seen)**
  stratified by (length, GC). Its weakness is that input is *inferred* from
  duplicate structure rather than measured.
- The SPANK-anchored alternative does not remove that weakness, it relocates
  it: it trades ZTNB extrapolation for an assumption of linearity across a 10×
  concentration gap plus interpolation from **one or two** points. Neither of
  those is checkable from existing data, whereas the ZTNB assumptions partly
  are (fit residuals, K_FIT sensitivity, MIN_CELL_SIZE sensitivity).
- Estimated cost of the anchored path is ~2 weeks, producing an estimator that
  is more complex, rests on uncheckable assumptions, and **still cannot be
  validated against ground truth.**

### The cheaper thing that captures most of the value

Characterise the uncertainty of the estimator that already exists. The ZTNB fit
yields `r_hat`, `p_hat`, `w_hat` and residuals alongside `pi_mix`, so per-cell
confidence intervals on P(seen) are derivable and could be surfaced through
`GCFlWeights`. A well-characterised uncertainty bound on the current estimator
beats a point estimate from a more elaborate one with uncharacterised
systematic error. Estimated 2-3 days.

### On validation, stated plainly

No available data can prove any of these estimates correct. Comparing two
estimators establishes agreement, not accuracy. Genuine validation needs
ground-truth molecule counts across the grid — which is the panel redesign
again.

**The recommendation therefore stands as §6 framed it:** if this matters
enough to fund, fund the UMIs. The input quantity is already known; the
molecule identity was never encoded, and no amount of statistics recovers a
measurement that was not made.

---

## 8. Reopened: use the SPANK UMIs to measure duplication

§7's "do not pursue" verdict is **superseded**. It rested on framing SPANK as an
*anchor for P(seen)* — a quantity that varies strongly with (length, GC),
because that variation is precisely the bias being modelled, so transferring it
from two points was indefensible.

The better framing: SPANK's UMIs measure the **duplication rate**, and that is a
different quantity with a much better chance of being flat across the grid.

```
molecules_seen(L,G) ~= reads(L,G) / E[reads | seen]      # E[...] measured from SPANK UMIs
P(seen | L,G)        = molecules_seen(L,G) / (3e4 x volume)
```

The load-bearing assumption becomes "duplication rate is roughly
(length, GC)-independent" rather than "recovery is linear across a 10x
concentration gap". Weaker, more plausible — and, unlike the old one,
**measurable from data already on disk.**

### Experiment A — how much does duplication actually vary?

`fit_ztnb_per_cell` (`flgc/model.py:69-95`) already stores per-cell duplication
parameters for every bin meeting `MIN_CELL_SIZE`:

```python
records.append({
    "len_bin_idx", "gc_bin_idx", "len_range", "gc_range",
    "pi_mix",                      # P(seen)
    "r_hat", "p_hat", "w_hat",     # the duplication structure, per cell
})
```

**Input:** any already-fitted ZTNB `GCFlDistModel` (`._records`), ideally
several samples across depths.
**Compute:** `E[reads | seen]` per cell from `(r_hat, p_hat, w_hat)`; then its
spread across cells — absolute range, and whether it trends with length or GC
rather than scattering.
**Decides:** flat -> the SPANK transfer is justified and the whole approach
works. Structured -> it does not, but you can bound the induced error, and a
trend in length or GC is itself a publishable characterisation of the assay.
**Cost:** a script. No new data, no library change.

This is the experiment that should run first, because a negative result kills
the approach cheaply and a positive one licenses everything after it.

### Experiment B — is the ZTNB extrapolation trustworthy?

At the SPANK points there are **two independent estimates of P(seen)**:

| source | how obtained |
|---|---|
| SPANK | **measured** — UMI families give `seen_unique` directly (`spikein_ztnb_fit.py:334-341`) |
| native ZTNB | **inferred** — zero-class extrapolated from duplicate structure |

**Input:** a cohort with both SPANK UMI counts and a fitted native ZTNB model.
**Compute:** `pi_mix` from each, at the bins containing the SPANK lengths
(52 bp and 75 bp; note SNMv3 and SNMv4 carry only ONE SPANK spike — see §3.4).
Compare per sample, then look for systematic offset versus scatter.
**Decides:** agreement is **empirical support for the ZTNB extrapolation across
the whole grid**, which is the exact weakness that motivated this document — it
would largely dissolve the "input is inferred, not measured" objection.
Systematic disagreement is a finding about a model that is already shipping.
**Cost:** a script plus cohort selection.

This is higher value than anything in §7, and it validates the estimator rather
than merely bounding its uncertainty.

### What these do NOT resolve

- **Duplication and capture are not separable from reads alone.**
  `E[reads | seen]` folds PCR amplification and sequencing depth together, so
  the conversion assumes both are shared between SPANK and the GC-dSparks. Same
  library, so plausible — but assumed, not shown.
- **Spike -> native transfer is untouched.** Even a perfect spike P(seen)
  surface assumes blunt-ended synthetic oligos recover like nucleosome-bound,
  methylated native fragments with varied termini. That gap survives every
  route considered here and is the one worth quantifying independently.

### Revised recommendation

Run Experiment A, then B. Both are scripts over existing data, and between them
they either license the spike-based absolute estimate or explain precisely why
it fails. Only if both land favourably does implementation work become
justified — and at that point the UMI panel redesign (§6) may still be the
better investment, because it removes the assumptions rather than validating
them.

---

## 9. The calibration design, and the ss/ds rule (SETTLED — do not relitigate)

The comparison is **spike-to-spike at matched length**, not spike-to-native. At a
length carried by both panels you have SPANK's UMI-measured duplication rate and
the GC-dSpark's raw read count; that ratio calibrates reads -> molecules, and it
propagates across the grid. This never touches native cfDNA, so the
spike -> native question does not enter the calibration step.

### Matched lengths (VERIFIED via `load_spikes`)

| profile | SPANK | ss/ds | GC-dSpark ss/ds | matched lengths |
|---|---|---|---|---|
| SNMv3 | `spank-75B` | ds | ds (20) | 75 |
| SNMv4 | `v4-Spank-ss-A` | ss | ss (29) | 52 |
| SNMv4B | `SPANK-52C`, `spank-75B` | ds | ss (29) | 52, 75 |
| SNMv4C | `SPANK-52C`, `SPANK-75B` | ds | ss (29) | 52, 75 |

At each matched length there are **5 GC-dSpark peers** spanning ~30-70% GC, and
SPANK lands almost exactly on one (52 bp: SPANK 50.0% vs peer 50.0%; 75 bp:
SPANK 47.5% vs peer 49.3%). So the comparison can be made at matched length AND
matched GC.

~~Bonus: those 5 peers permit a test of whether duplication varies with GC at
fixed length, using the spike panel alone.~~ **WRONG — retracted.** Duplication
is only measurable where there is a UMI, and the GC-dSpark peers have none
(§3.3). The peers give *read counts* across GC, which informs the shape of the
recovery surface, not the duplication rate.

**The only duplication-variance check available from the spikes is `d(52)` vs
`d(75)`** — two points, a LENGTH check, with no GC check at all.
GC-independence of duplication is assumed and is NOT testable within the spike
panel. It can only be approached through the per-cell `r_hat`/`p_hat` of native
ZTNB fits (Experiment A, §8) — a different population, but one that does vary
in GC.

### The ss/ds rule — OWNER-CONFIRMED, TESTED

**A ds spike contributes 2x the molecules its listed molarity implies, because
the two strands are not grouped into one UMI family. The correction applies to
the INPUT quantity.** This has been tested. It is settled; do not re-derive it
from SAM semantics or reopen it.

Effective input in ss terms, and the resulting gap at the matched lengths:

| profile | SPANK listed | effective (ss) | vs GC-dSpark peer | gap |
|---|---|---|---|---|
| SNMv3 | 5.0e-11 ds | 1.0e-10 | 5.0e-12 | 20x |
| SNMv4 | 5.0e-10 ss | 5.0e-10 | 5.0e-11 | 10x |
| SNMv4B/C | 1.0e-12 ds | 2.0e-12 | 5.0e-13 | **4x** |

**SNMv4B/C is the best vehicle**: the smallest effective gap, two matched
lengths rather than one, near-exact GC matches at both, and the model panel
spans 24-175 bp so the length extrapolation from the calibration points is
measurable across the whole grid. The ds/ss difference is a scalar on input and
does not disturb reads-per-observed-molecule.

### What remains assumed

- Duplication rate is roughly (length, GC)-independent. The LENGTH half is
  testable from the spikes (`d(52)` vs `d(75)`, §10 Check 1). The **GC half is
  not testable within the spike panel at all** — only the peers vary in GC and
  they carry no UMIs. Approaching it requires the per-cell `r_hat`/`p_hat` of
  native ZTNB fits (Experiment A, §8), which is a different population.
- Duplication is insensitive to concentration across the 4x gap. The failure
  mode is saturation, whose signature is SPANK's duplicate distribution being
  right-SHIFTED rather than merely scaled.
- Spike -> native transfer, for applying a spike-derived surface to real
  fragments. Unchanged by any of this, and the one worth quantifying separately.

---

## 10. The calibration, stated exactly

### Inputs per sample

| source | quantity | where |
|---|---|---|
| SPANK (52 bp ds, 75 bp ds) | `accepted_reads`, `unique_umis` | `spank_pipeline_b.py` summary rows |
| GC-dSpark (9 lengths x 5 GC) | raw read counts | `metrics.json` via `load_spike_counts` |
| both | molarity, length, gc, ss/ds | `load_spikes(profile)` |
| assay | plasma volume | **NOT in the data — outstanding** |

### Step 1 — measure duplication at the calibration points

```
d(L) = accepted_reads(L) / unique_umis(L)          for L in {52, 75}
```

Reads per observed molecule. Measured, not modelled. This is the only thing
SPANK's UMIs are needed for.

### Step 2 — known input, with the settled ss/ds correction (§9)

```
N_input(spike) ∝ MPM_per_uL x (2 if ds else 1)
```

**Units: per µL, not per mL** (§11). An earlier revision of this step said
`volume_mL`, a 1000x error of exactly the kind that yields a plausible wrong
answer rather than an obvious one.

**Plasma volume is not required** — see §11. Working in concentration space,
as production already does, cancels it.

### Step 3 — P(seen) for SPANK, directly

```
P_seen_SPANK(L) = unique_umis(L) / N_input(SPANK, L)
```

Note what this is: with input known AND molecules countable, this is P(seen)
**by definition** — no extrapolation, no NB assumption, no zero-class inference.
It is the ground truth this document was originally written for want of.

### Step 4 — transfer to the GC-dSparks

```
molecules_seen(L,G) = reads(L,G) / d(L*)           L* = nearer of {52, 75}
P_seen(L,G)         = molecules_seen(L,G) / N_input(L,G)
weight(L,G)         = 1 / P_seen(L,G)              clamped as now
```

### Two checks that fall out at no extra cost

**Check 1 — `d(52)` vs `d(75)`.** Two independent measurements of the
duplication rate at different lengths. Agreement supports length-flatness and
licenses the extrapolation to 24 bp and 175 bp; divergence kills the transfer
and quantifies by how much. **Run this first** — it is a handful of numbers and
can end the project before any surface is built.

**Check 2 — measured vs inferred P(seen) at 52 and 75.** Step 3 gives a
*measured* P(seen); the native ZTNB fit gives an *extrapolated* `pi_mix` at the
same cells. Comparing them validates the ZTNB extrapolation that is already
shipping. This is Experiment B (§8), and it is cleaner than described there,
because knowing the input removes the need to infer anything on the spike side.

### Outstanding before this can run

**SUPERSEDED — see §11 for the current list.** Items 1 and 2 below have since
been withdrawn: plasma volume is not needed (the ratio cancels it) and the
input quantities are tabulated in the SNMv4C component table. Retained only to
show what was blocking and why it stopped.

1. ~~**Plasma volume**~~ — withdrawn (§11).
2. ~~**The input-quantity column for SNMv4B/C**~~ — resolved (§11). Note the
   v4 panels record **molarity**, not MPM, unlike SNMv3; the component table
   supplies MPM directly for both.
3. **Result IDs on a v4B/C profile** — a cohort was found (§11).

---

## 11. Production already does the unstratified version of this

Source: `biomarker-projects/mtdna_chimerism/projects/transplant_poc/docs/methods/spikein_normalization.md`,
and `bin/crr-caller`. VERIFIED by reading both.

```python
def compute_mpm(edr, ddspank, spanks_per_ul):
    return (edr / ddspank) * spanks_per_ul        # SPANKS_PER_UL = 1.2e6
```

> "Reproduces 10 reported pathogen MPMs at ratio 0.9915-1.0027"
> "All 184 nuclear-cohort samples have ddspank_source = 52C (SNMv4B/C) ...
>  SPANKS_PER_UL = 1.2e6 is correct"

This is *pathogen-signal / spike-signal x known-spike-concentration* — the same
ratio calibration proposed here, running in production and validated to ~1%.

### The SNMv4C input table (resolves the input-quantity blocker)

| Component | Type | Length | Molarity (plasma) | MPM (1x) |
|---|---|---|---|---|
| SPANK-52C | ds | 52 bp | 1e-12 M | **6e5** |
| SPANK-75B | ds | 75 bp | 1e-12 M | **6e5** |
| SPARK-032..075 | ds | 32-75 bp | 5e-14 M | 3e4 |
| SPARK-125, 150 | ds | 125-150 bp | 5e-13 M | 3e5 |
| **SPARK-v4-\*** (56) | **ss** | 24-175 bp | 5e-13 to 2.5e-12 M | **3e5 to 1.5e6** |
| ID spikes (200) | ss | 50 bp | 4e-11 M | 2.4e7 |

`SPANKS_PER_UL = 1.2e6` is simply the **sum of the two SPANKs** at 6e5 each —
not a fitted constant. That is why it reproduces production MPMs so closely.

Input quantities are tabulated for every component including the v4 grid, so no
Avogadro conversion is needed.

### Plasma volume is NOT required

The formula works in concentration space: numerator and denominator come from
the same aliquot, so volume cancels. The answer is expressed per µL rather than
as an absolute count, which is the natural unit anyway. **Blocker 1 (§10) is
withdrawn.**

### MUST be computed per sample — the variance is pooling

`ddspank` varies 191x across the cohort (448 to 85,509), with 74.5% of that
variation WITHIN run. **OWNER-CONFIRMED: this reflects pooling differences
between samples, not measurement noise.**

The consequence is a hard design constraint: **the calibration must be run
per sample.** A cohort-level or run-level `d` would average over genuine
per-sample differences in how much of each library was loaded, and would be
wrong for every individual sample. Production already respects this — `ddspank`
enters `compute_mpm` per sample.

So `d(L) = accepted_reads(L) / unique_umis(L)` (§10 Step 1) is a **per-sample**
quantity throughout. It is not a constant to be fitted once and reused, and
`d(52)` vs `d(75)` (§10 Check 1) is a within-sample comparison, aggregated
across samples only afterwards to assess consistency.

### What this reframes

Production does the **(length, GC)-agnostic** version: one global `ddspank`, one
global constant, per sample. The GC-dSpark grid is what makes it **stratified**
— the same ratio calibration computed per (length, GC) cell rather than once.

That changes the work from "build a new estimator" to "extend a validated one
into a second dimension", which is a far better starting position and inherits
the existing validation.

### Remaining outstanding

1. ~~Plasma volume~~ — **withdrawn**, not needed (above).
2. ~~Input-quantity column for v4B/C~~ — **resolved** by the table above.
3. **Result IDs** — a cohort exists: 876 samples, result IDs 166k-228k, carrying
   both `ddspank` and GC-dSpark counts, with scripts already written
   (`stage0_spikein_variance.py`, `compute_mpm_donor_full_cohort.py`).
4. **NEW — per-cell depth.** Stratifying divides an already-variable denominator
   across 45 cells. Check the per-cell GC-dSpark read counts before committing
   to the design; thin cells would make per-sample per-cell estimates unusable
   even though the unstratified version works fine.

---

## 12. Implemented, and what it showed (`scripts/spike_recovery.py`)

Per-sample, per-(length, GC) recovery from spikes, built to compare against
ZTNB. Runs on real data: `load_spikes_for_result` for SPANK reads/deduped and
GC-dSpark counts, `load_spikes_df` for the metadata join (it owns the
`v4-` -> `SPARK-v4-` name translation; do not re-derive it).

### Check 1 PASSED — duplication is length-flat

```
     rid   d(52)   d(75)   ratio
  200000   1.321   1.300   1.016
  200001   1.315   1.296   1.015
  200002   1.335   1.326   1.007
  200005   1.346   1.313   1.025
  200010   1.301   1.293   1.006
  190000   1.021   1.017   1.004
  180000   1.300   1.245   1.044

  ratio d(52)/d(75): mean=1.0167 sd=0.0129 range=[1.004, 1.044]
```

Duplication agrees between 52 bp and 75 bp to ~1.7%. The transfer assumption
holds over that span.

It also confirms the per-sample rule empirically: `d` ranges 1.02 to 1.35
BETWEEN samples while staying flat WITHIN each one. A cohort-level constant
would have been wrong for every individual sample.

### Depth is not a problem

All 29 model cells populate, minimum 1580 (molarity-scaled), median 6190. The
concern raised in §11 is withdrawn.

### The absolute scale is NOT obtainable this way

```
P(seen | L,G) = molecules_observed / (MPM_input x V_uL)
```

SPANK supplies `molecules_observed` and `MPM_input`, leaving **one equation in
two unknowns**: SPANK's own P(seen) and the effective volume V. Production's
formula avoids this because it estimates an unknown input concentration for a
pathogen; here the input is known and recovery is unknown — the opposite
direction. Assuming 3 mL plasma yields P(seen) ~ 3e-6, which is not credible
enough to build on.

So the script reports recovery **normalised to the cell nearest the SPANK
calibration point**. That is exactly what a shape-vs-shape comparison with ZTNB
needs; the absolute scale stays open.

### The finding that matters: the surface is reproducible but CONFOUNDED

Cross-sample Spearman of the recovery surface is **0.82-0.95**, so it is a
stable property, not noise. But it is not a smooth function of GC:

```
                200000  200001  200002  200005  190000
v4-32-40-ss-02    4.02    3.17    3.68    4.15    1.97   <- high everywhere
v4-32-50-ss-03    1.87    1.95    2.44    2.27    1.65   <- low everywhere
v4-32-60-ss-02    4.31    3.20    3.36    3.87    1.59   <- high everywhere
```

A GC bias should be smooth in GC. This alternates high/low/high between
adjacent GC cells, which is **oligo identity, not GC**. The panel carries
exactly ONE oligo per (length, GC) cell, so sequence-specific recovery is
**perfectly confounded** with the cell. There is no way, from this panel, to
tell whether `v4-32-40-ss-02` recovers well because 32 bp / 40% GC is
favourable or because that specific sequence is.

(Some rows do look like clean bias — 24 bp declines monotonically 1.29 -> 0.33
across GC — but those are equally reproducible, so "looks like bias" and "is
bias" cannot be separated here.)

### Consequence for the ZTNB comparison

The two methods do not estimate the same quantity:

| | measures |
|---|---|
| ZTNB | population-averaged (length, GC) recovery over thousands of NATIVE fragments per cell; sequence idiosyncrasy averages out |
| Spikes | recovery of 29 specific SYNTHETIC sequences, reproducibly, with sequence effects inseparable from cell effects |

Where they disagree, that is **not automatically ZTNB's error.** The spike
surface carries per-sequence structure that native data does not.

Escaping this needs multiple distinct oligos per cell, so sequence effects can
be averaged within a cell — another panel-design change, and the same
conclusion §6 reached by a different route.

### Note

Sample 190000 has `d ~ 1.02` against ~1.31 for the others and correlates least
with them (0.44-0.72). A different duplication regime; treat separately or
exclude.
