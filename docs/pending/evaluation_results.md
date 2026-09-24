# Background Model v2 — Evaluation Results

2026-09-20

## Executive Summary

The background model v2 recovers a textbook CTCF footprint from held-out cfDNA samples using a model trained only on inactive genomic regions, confirming it learns real sequence-driven fragmentation bias. Three loss functions (multinomial, Dirichlet-multinomial, NB-offset) produce near-identical corrections at CTCF sites. A no-blood control experiment confirms the correction does not manufacture signal where biology is absent.

| Result | Finding |
| --- | --- |
| Footprint recovery | Short-band endpoint spike at motif centre: ~200 vs baseline ~15 (>10x enrichment) |
| Nucleosome phasing | Mono-band flanking peaks at +/-180 bp, consistent with known nucleosome array |
| Model agreement | Three losses nearly indistinguishable in correction effect |
| Uniform control | Max abs diff = 1.19e-07 (PASS) — correction machinery provably non-distorting |
| Mirror check | Shape correlation r = 0.85 between + and - oriented pileups |
| No-blood control | Flat stays flat: raw amplitude 1.05x (no footprint), corrected 1.08x (not manufactured) |
| Helical periodicity | 10.19 bp/turn recovered from sequence alone in flanking regions |
| Predicted bias shape | Flanks flat (~0.30-0.32), dip at motif centre — correction amplifies footprint (correct behaviour) |

**Training summary.** 5-epoch bake-off on 40 train / 10 held-out samples, 5,000 inactive-region tiles, B=8, lr=1e-4. DM val loss 9.3191 < multinomial 9.3243 (gap shrinking). NB-offset non-monotonic (instability signal). None converged — early-stop never fired. Raw val-loss cannot select across likelihood families; formal QQ evaluation on held-out data still needed.

**Status.** Phases 0-4 complete and reviewed. Remaining: simulation study (known ground truth), QQ-based loss selection, converged training runs, weight clamping decision.

---

## Raw vs Predicted (Counts) — All Models

Raw observed counts (black) vs model-predicted expected counts (coloured, all 3 trained models overlaid). The gap between observed and predicted is the biological signal that correction preserves.

### Zoomed ±128 bp — shared y-axis

![Raw vs predicted zoom128](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/zoom128__raw_vs_predicted.png)

### Zoomed ±128 bp — twin y-axes (separate scales for raw and predicted)

![Raw vs predicted twin zoom128](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/zoom128__raw_vs_predicted_twin.png)

---

## Observed vs Predicted Shape — All Models

Normalised shape profiles (probability distributions). Observed shape shows the biological footprint + nucleosome phasing; predicted shape captures only the sequence-bias component.

### Zoomed ±128 bp

![Obs vs pred shape zoom128](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/zoom128__observed_vs_predicted_shape.png)

### Track-style ±128 bp — All 3 models overlaid (shape)

![Tracks shape allmodels zoom128](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/tracks_zoom128__shape__allmodels.png)

### Track-style ±128 bp — Multinomial only (shape)

![Tracks shape multinomial zoom128](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/tracks_zoom128__shape__multinomial.png)

### Track-style ±16 bp — All 3 models (motif close-up)

![Tracks shape allmodels zoom16](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/tracks_zoom16__shape__allmodels.png)

### Track-style ±128 bp — Multinomial (counts)

![Tracks counts multinomial zoom128](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/tracks_zoom128__counts__multinomial.png)

---

## Expected Bias Shape (Predicted Only) — All Models

The predicted bias shape — the component that correction divides out. Flat flanks (~0.30-0.32), sharp dip at motif centre. All three models agree closely.

### Wide ±1000 bp

![Expected shape wide](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/expected_shape.png)

### Zoomed ±128 bp

![Expected shape zoom128](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/zoom128__expected_shape.png)

---

## Uncorrected vs Corrected — Per Model (Bounded [0.2, 5.0])

### Multinomial — wide ±1000 bp

![Uncorr vs corr multinomial wide](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/uncorr_vs_corr__multinomial__bounded_0.2_5.0.png)

### Multinomial — zoomed ±128 bp

![Uncorr vs corr multinomial zoom128](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/zoom128__uncorr_vs_corr__multinomial__bounded_0.2_5.0.png)

### Dirichlet-Multinomial — wide ±1000 bp

![Uncorr vs corr DM wide](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/uncorr_vs_corr__dirichlet_multinomial__bounded_0.2_5.0.png)

### NB-Offset — wide ±1000 bp

![Uncorr vs corr NB wide](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/uncorr_vs_corr__nb_offset__bounded_0.2_5.0.png)

---

## Uncorrected vs Corrected — Per Model (Identity / No Clamp)

### Multinomial — identity clamp

![Uncorr vs corr multinomial identity](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/uncorr_vs_corr__multinomial__identity.png)

### Dirichlet-Multinomial — identity clamp

![Uncorr vs corr DM identity](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/uncorr_vs_corr__dirichlet_multinomial__identity.png)

### NB-Offset — identity clamp

![Uncorr vs corr NB identity](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/uncorr_vs_corr__nb_offset__identity.png)

---

## 3-Model Comparison — Correction Ratio (Corrected / Uncorrected)

### Bounded [0.2, 5.0] — wide ±1000 bp

![3-model ratio bounded wide](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/three_model_ratio__bounded_0.2_5.0.png)

### Bounded [0.2, 5.0] — zoomed ±128 bp

![3-model ratio bounded zoom128](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/zoom128__three_model_ratio__bounded_0.2_5.0.png)

### Identity (no clamp) — wide ±1000 bp

![3-model ratio identity wide](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/three_model_ratio__identity.png)

---

## Validation Controls

### Uniform Control

A uniform model (all positions predicted equally likely) must reproduce the uncorrected pileup exactly. **PASS:** max abs diff = 1.192e-07.

![Uniform control](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/uniform_control.png)

### Mirror Check (Orientation Validation)

+ strand oriented vs - strand oriented (RC) pileups. **PASS:** shape correlation r = 0.8538.

![Mirror check](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/mirror_check.png)

---

## No-Blood Control Experiment

CTCF sites from non-blood cell types — biology absent, correction should not manufacture signal. 5,000 control sites (328,690 raw → filtered cascade → 5,000 final).

### Blood vs No-Blood Side-by-Side

Blood (blue) vs no-blood (red): raw, predicted, and corrected. Blood footprint amplified; no-blood stays flat.

![Blood vs no-blood](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/noblood/compare_blood_vs_noblood.png)

| Metric | Blood (accessible) | No-blood (control) |
| --- | --- | --- |
| Raw footprint amplitude | 2.43x (strong footprint) | 1.05x (flat — no footprint) |
| Predicted bias at motif | 0.90 (dip) | 0.95 (still a dip) |
| Corrected amplitude | 3.59x | 1.13x |
| Correction amplification (corr/raw at footprint) | 1.46x | 1.08x |

### No-Blood: Raw vs Predicted — zoomed ±128 bp (all models, shared y-axis)

![No-blood raw vs pred zoom128](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/noblood/zoom128__raw_vs_predicted.png)

### No-Blood: Raw vs Predicted — zoomed ±128 bp (twin y-axes)

![No-blood raw vs pred twin zoom128](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/noblood/zoom128__raw_vs_predicted_twin.png)

### No-Blood: Observed vs Predicted Shape — zoomed ±128 bp

![No-blood obs vs pred shape zoom128](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/noblood/zoom128__observed_vs_predicted_shape.png)

### No-Blood: Expected Bias Shape — zoomed ±128 bp

![No-blood expected shape zoom128](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/noblood/zoom128__expected_shape.png)

### No-Blood: Uncorrected vs Corrected — zoomed ±128 bp (multinomial, bounded)

![No-blood uncorr vs corr zoom128](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/noblood/zoom128__uncorr_vs_corr__multinomial__bounded_0.2_5.0.png)

### No-Blood: 3-Model Ratio — zoomed ±128 bp (bounded)

![No-blood 3-model ratio zoom128](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/noblood/zoom128__three_model_ratio__bounded_0.2_5.0.png)

### No-Blood: Track-style Shape — ±128 bp (all models)

![No-blood tracks shape zoom128](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/noblood/tracks_zoom128__shape__allmodels.png)

### No-Blood: Track-style Counts — ±128 bp (all models)

![No-blood tracks counts zoom128](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/noblood/tracks_zoom128__counts__allmodels.png)

### No-Blood: Track-style Shape — ±128 bp (short tracks only)

![No-blood tracks short zoom128](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/noblood/tracks_zoom128__shape__allmodels__short_first-short_last.png)

### No-Blood: Track-style Shape — ±16 bp (short tracks, motif close-up)

![No-blood tracks short zoom16](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/noblood/tracks_zoom16__shape__allmodels__short_first-short_last.png)

### No-Blood: Wide ±1000 bp — short tracks, no smoothing

![No-blood tracks short zoom1000](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/noblood/tracks_zoom1000__shape__allmodels__short_first-short_last.png)

### No-Blood: Wide ±1000 bp — smoothed 25bp

![No-blood tracks short zoom1000 smooth25](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/noblood/tracks_zoom1000__shape__allmodels__short_first-short_last__smooth25.png)

### No-Blood: Wide ±1000 bp — smoothed 100bp

![No-blood tracks short zoom1000 smooth100](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/noblood/tracks_zoom1000__shape__allmodels__short_first-short_last__smooth100.png)

### No-Blood: Wide ±1000 bp — masked ±32bp

![No-blood tracks short zoom1000 mask32](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/noblood/tracks_zoom1000__shape__allmodels__short_first-short_last__mask32.png)

### No-Blood: Wide ±1000 bp — smoothed 25bp + masked ±32bp

![No-blood tracks short zoom1000 smooth25 mask32](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/noblood/tracks_zoom1000__shape__allmodels__short_first-short_last__smooth25__mask32.png)

### No-Blood: Wide ±1000 bp — smoothed 100bp + masked ±32bp

![No-blood tracks short zoom1000 smooth100 mask32](/home/nathanboley/src/fragmentomics_tools/data/ctcf_plots/noblood/tracks_zoom1000__shape__allmodels__short_first-short_last__smooth100__mask32.png)

---

## Open Questions

| Question | What we know | What resolves it |
| --- | --- | --- |
| Does loss choice matter? | Three models gave near-identical CTCF corrections | Simulation with known ground truth |
| Is the predicted dip at -50..-10 real? | Model predicts a sequence-bias dip upstream of motif | Simulation — is the dip in the ground truth? |
| Data-limited or iteration-limited? | Train-val gap flipped by epoch 2; no run plateaued | Simulation — train to convergence, measure data-scaling curve |
| What clamp bounds are right? | Bounded [0.2, 5.0] visually better than identity | Simulation — sweep bounds, measure error |
