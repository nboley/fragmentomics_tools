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

Limits: SPANK exists at only **2 (length, GC) points** (52 bp, 75 bp, ~50% GC),
at 10× the GC-dSpark concentration. It cannot be stratified across the grid,
and using it as an anchor assumes recovery is linear in concentration across
that gap.

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
2. **Can the two-tier v4 design be exploited?** The same (length, GC) cell
   appears at two known concentrations. If recovery is concentration-independent,
   molarity-scaled counts should agree between tiers; disagreement would expose
   saturation or competition — and would invalidate any SPANK-anchoring scheme
   that assumes linearity. INFERENCE: this looks like a test runnable on
   existing data, but it has not been attempted.
3. **Is the two-tier slope informative without UMIs?** Reads-per-input-molecule
   still folds in amplification. Whether anything separable falls out is
   unresolved.
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
