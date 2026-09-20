# Simulation Study — Plan

**Status:** draft for owner sign-off. High-level and procedural; implementation
detail deliberately omitted until the plan is agreed.

## 1. Why

On real data we cannot separate *"the model recovered the true bias"* from
*"the model produced something plausible"*, because the truth is unknown. Every
open question from the CTCF work is blocked on that:

- Does the loss choice matter at all? (three models gave near-identical corrections)
- Is the predicted dip at −50..−10 real bias removal, or extrapolation?
- Are we data-limited or iteration-limited? (train−val gap flipped by epoch 2)
- What clamp bounds are right? (deferred; currently unquantified)

A generative simulation with a known cut-site bias makes all four measurable.

It also tests **the whole stack**, not just the model: because the simulation
emits *fragments*, the same pipeline runs end to end —
`fragment_array` → store → train → `apply_fragment_weights` → pileup.

## 2. Ground truth

- Cut-site bias is a **hexamer weight table** `w6` (4,096 entries), drawn
  log-normal and scaled to a realistic dynamic range (real cfDNA cut bias is
  roughly 2–5× favoured vs disfavoured; realised range to be reported).
- `w6` is the ground truth: the true per-position endpoint propensity is known
  exactly at every base of every region.
- Chosen deliberately to be **structurally unlike the dilated CNN**, so recovery
  is a real test rather than an architecture recognising itself.

## 3. Generative procedure

Per region, propose-and-reject until the target fragment count is reached:

1. Draw a start position **uniformly** within the region.
2. Draw a fragment length from the **empirical cfDNA length distribution**
   (taken from a real h5, not invented).
3. Compute the hexamer at each end — the far end uses the **reverse-complement**
   hexamer, since it is a cut on the opposite strand.
4. Compute the fragment's **GC content and length**, and look up the empirical
   **GC×FL bias weight** (§3.1).
5. **Accept** with probability `w6(left) × w6(right) × wGC(gc, len)`
   (all weights normalised to max 1).
6. Region fragment counts follow the **empirical per-tile count distribution**
   from real data.

Regions: 1,000 real genomic regions (real sequence composition and k-mer structure).

### 3.1 GC × fragment-length bias

Two bias mechanisms act on a real fragment and the simulation includes both:

| Mechanism | Acts on | Source |
|---|---|---|
| **Cut-site bias** | the 6 bp at each end | synthetic `w6` table (§2) |
| **GC×FL bias** | the whole fragment (amplification/sequencing efficiency) | **empirical**, from the production GC/FL bias model ("spark grid") in the biomarker repo |

Taking GC bias from the real production model rather than inventing it means the
simulation carries a bias whose shape we did **not** choose, which is a
meaningfully harder and more honest test of recovery.

Note these two act at different scales — cut-site bias is local and
base-resolution, GC×FL is fragment-scale — so the model must learn both a sharp
local effect and a broad compositional one.

**Sign convention (critical).** The production model stores
`cf = log(count / grand_mean)` and applies `weight = exp(-cf)` — it *divides the
bias out*. A simulator needs the bias itself:

```
bias(GC) = exp(+cf) = 1 / production_weight
```

Multiplying acceptance by the production `predict()` output would anti-bias the
simulation — exactly backwards. Fitted by `scripts/sim_fit_gc_bias.py`.

**GC bias is a 2-D surface — GC interacts with fragment length.** Fitted from 30
real production samples (per-oligo SPARK counts from
`s3://results.prod.kariusdx.com/{result_id}/analyze-alignments/metrics.json`).
Row-centred bias (rows = length, cols = GC%):

| | 30 | 40 | 50 | 60 | 70 |
|---|---|---|---|---|---|
| **24 bp** | 1.258 | 0.848 | 1.283 | 0.789 | 0.822 |
| **32 bp** | 0.631 | 1.158 | 0.949 | 1.202 | 1.060 |
| **42 bp** | 0.761 | 0.994 | 1.126 | 1.188 | 0.930 |
| **52 bp** | 0.751 | 0.758 | 0.785 | 1.320 | 1.386 |
| **75 bp** | 0.459 | 1.111 | 0.722 | 0.838 | 1.871 |

The interaction is the dominant structure — the GC slope (70%/30%) runs
**0.654 → 1.679 → 1.222 → 1.847 → 4.080** across lengths 24→75 bp. It *flips
sign*: short fragments are mildly GC-disfavouring, long ones strongly
GC-favouring. Every contrast is significant at >2 SEM (per-cell SEM 0.03–0.10),
so this is real, not noise. **A 1-D GC marginal would average the sign flip
away and is not used.**

**Row-centring removes the length main effect** (each row divided by its own
mean). That effect is contaminated by single-stranded oligo degradation —
longer ss SPARKs are recovered less because they *degrade*, not because of
library bias — and the simulator already draws lengths from the empirical
real-data distribution, so re-imposing it would double-count. What survives is
purely the GC×length interaction.

**Extrapolation policy: HOLD at the grid edges.** `L<24` uses the 24 bp row,
`L>75` uses the 75 bp row, GC clamps to [30,70]. The grid has no coverage above
75 bp while real cfDNA reaches 167 bp+, and the GC slope is *still rising* at
the edge — extrapolating that trend would invent a very strong bias the data
cannot support. Holding is conservative and explicit.

Note also that this is *not* the unimodal mid-GC peak described in secondary
documentation; the fitted data disagrees, which is why the numbers were refit
rather than quoted.

**Empirical overdispersion.** Median across-sample SD of log-bias = **0.286**
(≈ ±33%), giving regime B (§4) a measured jitter magnitude rather than an
invented one.

## 4. Regimes

| Regime | Between-sample variation | Question it answers |
|---|---|---|
| **A. Base** | none — every sample drawn from the same `w6` | Can the bias be recovered at all? Multinomial and DM should tie here by construction. |
| **B. Overdispersed** | per-sample jitter `w6 × exp(ε_s)`, known magnitude | Does DM's dispersion head earn its complexity, and does it recover the true dispersion? |

Regime B is what makes the loss comparison decisive: if DM cannot beat
multinomial even when the data is genuinely overdispersed, that settles it.

## 5. Paired comparison (owner requirement)

**All models are trained and tested on the identical fragment set.**

- Simulate **once** per regime → one fragment set → **one store**.
- Identical splits, identical seeds, identical held-out simulated samples.
- The **only** difference between arms is `--loss`.

This removes simulation sampling noise from the comparison entirely: any
difference observed between models is attributable to the loss, not to luck of
the draw. (Same discipline as the real bake-off, which already shared a store.)

## 6. Procedure

1. **Build `w6`** and record it as the ground-truth artefact.
2. **Fit the realistic inputs** — fragment-length distribution and per-region
   count distribution — from a real h5.
3. **Simulate fragments** for all samples in the regime; write them in the
   normal fragments-h5 form so the existing pipeline consumes them unchanged.
4. **Validate the simulator before anything else** — recompute observed endpoint
   frequency per hexamer from the simulated fragments and confirm it reproduces
   `w6`. If this fails, stop: nothing downstream is meaningful.
5. **Build the store** using the existing preprocess path.
6. **Train all three losses to convergence** on that one store — early stopping
   must actually fire (unlike the real bake-off, which was timeout-truncated).
7. **Evaluate** against ground truth (§7).
8. **Report** with plots, using the library track plotting.

## 7. What gets measured

| Metric | Meaning |
|---|---|
| **Shape recovery** | divergence between predicted `probs` and true per-position propensity |
| **Hexamer recovery** | inferred vs true `w6` — does the model learn the actual table? |
| **Dispersion recovery** | fitted γ vs true dispersion (regime B only) |
| **Corrected flatness** | **the product metric** — after applying weights, the corrected profile should be flat; residual structure *is* the error |
| **Data-scaling curve** | error vs number of samples / regions — tests the "data-limited, not iteration-limited" hypothesis directly |
| **Clamp sweep** | which clamp bounds minimise error, now measurable rather than guessed |

## 8. Known limitations

- Validates the **estimator**, not the real-data conclusion: it shows what the
  models do when the bias has this form, not that real bias has this form.
- Risk of flattering results if the truth is too easy. Mitigated by the
  non-CNN generator (§2) and by reporting the realised dynamic range; a
  misspecified variant (bias structure the model cannot express) can be added
  if the base result looks suspiciously clean.

## 9. Decisions

Resolved with defaults (auto mode); flag if any should change:

| Decision | Default taken | Why |
|---|---|---|
| Hexamer placement | **3 in / 3 out**, spanning the cut | standard convention in cut-bias work; symmetric about the cut |
| Regime B (overdispersed) | **included from the start** | regime A alone can only produce a tie, so it cannot answer the loss question |
| Scale | **~20 simulated samples, ~2,048 bp tiles, 1,000 regions** | small enough to train to genuine convergence in minutes — avoids repeating the 5-epoch truncation |
| GC bias | **empirical, from the production GC/FL model** | a bias shape we did not choose is a harder, more honest test |

Still genuinely open, pending the research summary:

1. Exact GC×FL parameterisation — whether the production weights are biases
   (multiply) or corrections (divide), and what window GC is computed over.
   Getting this backwards would invert the bias, so it is a hard gate before
   implementation.
