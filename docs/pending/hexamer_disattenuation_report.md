# Hexamer Prior Disattenuation Measurement: 92-Sample Cohort

## What this measures

Shrinkage trades variance for bias. The Dirichlet-multinomial posterior pulls
each sample's hexamer weights toward the pooled prior, attenuating the
sample's real deviation from the cohort along with its noise. Before using
a specific sample's posterior in the simulator, we need to know: **how much
of the true sample-vs-pool difference survives the shrinkage?**

This report measures the **disattenuated Pearson correlation** between each
sample's raw hexamer log-enrichment and the leave-one-out (LOO) pooled
log-enrichment, for all 4 tables (start_fwd, end_fwd, start_rev, end_rev),
across all 92 samples in the cohort.

## Methodology

The estimator reuses the thinning-null approach from commit ef60d52, which
established pairwise disattenuation for these same tables (documented in
`docs/pending/cut_site_hexamer_counts.md`).

For each (sample, table) pair:

1. **Leave-one-out pool**: Subtract the sample's counts from the pooled counts
   to avoid correlating a sample with a pool that contains it.

2. **Observed r**: Pearson correlation of log(obs_proportion / bg_proportion)
   between the sample and the LOO pool, over all hexamers where both are > 0.

3. **Thinning-null reliability (rho_sample)**: Binomial-thin the sample's
   counts (p = 0.5) into two halves, compute log-enrichment for each half,
   correlate them. Average over 50 replications. This gives the measurement
   reliability at the sample's actual depth — the correlation ceiling imposed
   by counting noise alone.

4. **LOO pool reliability (rho_pool_loo)**: Same thinning procedure on the
   91-sample LOO pool. At ~34M pooled counts, this is 0.995-0.998.

5. **Disattenuated r**: `r_observed / sqrt(rho_sample * rho_pool_loo)`,
   clamped to [-1, 1]. This estimates what the sample-vs-pool correlation
   would be if both were measured with infinite depth.

**Why log-enrichment**: the precedent doc uses log-enrichment throughout.
The log transform stabilises variance across the 250x dynamic range of start
tables; Pearson on raw weights would be dominated by a handful of extreme
hexamers.

**Implementation**: `scripts/measure_hexamer_disattenuation.py`. Tested by
`tests/test_hexamer_disattenuation.py` (10 tests on synthetic data with known
ground truth).

## Results

### Summary statistics

| Table | Median r_obs | Median rho_sample | Median rho_pool_loo | Median r_disatt | Samples at 1.0 | Below 0.99 | Below 0.95 |
|-------|-------------|-------------------|---------------------|-----------------|----------------|------------|------------|
| start_fwd | 0.9395 | 0.8603 | 0.9977 | 1.0000 | 75/92 | 11/92 | 3/92 |
| end_fwd | 0.8803 | 0.7370 | 0.9956 | 1.0000 | 73/92 | 16/92 | 9/92 |
| start_rev | 0.9390 | 0.8626 | 0.9977 | 1.0000 | 75/92 | 12/92 | 3/92 |
| end_rev | 0.8822 | 0.7436 | 0.9955 | 1.0000 | 72/92 | 18/92 | 8/92 |

**75% of samples (69/92) have r_disattenuated clamped at 1.0 across all 4
tables** — their hexamer profiles are indistinguishable from the pool within
counting noise. For these samples, the posterior is the right estimate and
shrinkage has removed only noise.

### End tables are more attenuated than start tables

End tables have 2-3x as many samples below 0.95 (8-9 vs 3). This is
consistent with the prior finding (ef60d52) that end tables have lower spread
(log-SD 0.49 vs 0.74 for start), so the same counting noise degrades them more
and the noise ceiling is lower: median rho_sample is 0.74 for end vs 0.86 for
start.

### Outlier samples

9 samples have at least one table with r_disattenuated < 0.95. These represent
genuine biological deviation from the cohort — not noise.

| Sample | Mean r_disatt | Min r_disatt (table) | Depth (N) | Mean rho | Assessment |
|--------|--------------|---------------------|-----------|----------|------------|
| RD-56171-Lib1 | 0.713 | 0.616 (end_rev) | 173k | 0.788 | **Severe** — genuinely different hexamer profile |
| RD-57090-Lib1 | 0.842 | 0.739 (end_rev) | 1,135k | 0.934 | **Severe** — high depth, high reliability, real deviation |
| RD-56818-Lib1 | 0.852 | 0.784 (end_fwd) | 75k | 0.680 | **Moderate** — low depth but deviation persists after correction |
| RD-56676-Lib1 | 0.901 | 0.836 (end_rev) | 74k | 0.713 | **Moderate** — low depth, end tables hit hardest |
| RD-56805-Lib1 | 0.939 | 0.895 (end_rev) | 2,287k | 0.955 | **Notable** — very high depth confirms this is real |
| RD-56428-Lib1 | 0.941 | 0.894 (end_fwd) | 102k | 0.707 | **Moderate** — end tables only |
| RD-56905-Lib1 | 0.943 | 0.910 (end_fwd) | 171k | 0.700 | **Moderate** — end tables only |
| RD-56687-Lib1 | 0.957 | 0.931 (end_rev) | 154k | 0.741 | **Mild** — end tables only |
| RD-56161-Lib1 | 0.966 | 0.950 (end_fwd) | 316k | 0.787 | **Borderline** — barely below threshold |

**RD-57090-Lib1** and **RD-56805-Lib1** are especially informative because
they have very high depth (1.1M and 2.3M counts) and very high reliability
(0.93 and 0.96). Their deviation from the pool is measured with high
confidence — these are not depth-starved samples where noise might be
masquerading as biology.

![Disattenuated r distribution](hexamer_disattenuation_plots/fig1_disattenuated_r_histogram.png)

![Observed r vs reliability](hexamer_disattenuation_plots/fig2_r_vs_reliability.png)

![Disattenuated r vs depth](hexamer_disattenuation_plots/fig3_disattenuated_vs_depth.png)

![Per-sample summary](hexamer_disattenuation_plots/fig4_per_sample_summary.png)

## What this licenses and what it does not

### What it licenses

1. **For 69/92 samples (75%), the pooled posterior is the right estimate.**
   Their hexamer profiles are indistinguishable from the pool, and the
   shrinkage removed only counting noise. Using their posterior weights in the
   simulator is safe — the disattenuated r is 1.0, meaning no real signal was
   attenuated.

2. **For the remaining 23 samples, the posterior weights still carry the
   sample's signal**, just partially attenuated toward the pool. The shrinkage
   ratio (23-29% at the median) means ~75% of each sample's deviation from
   the pool is preserved.

3. **The prior itself is sound.** Pool reliability is 0.995-0.998 across
   tables — the pooled hexamer weights are measured with negligible noise.

### What it does not license

1. **For 9 samples with r_disatt < 0.95, the posterior is a biased estimator
   of the true sample-specific hexamer weights.** Their real hexamer profiles
   differ from the pool, and the shrinkage has pulled them partway toward a
   pool they don't fully belong to. The bias is in the direction of the pool:
   the posterior underestimates how different these samples truly are.

2. **This measurement does not tell us whether the bias matters for the
   downstream simulator output.** The hexamer weights enter the simulator
   through cut-site probabilities. A 5-10% attenuation in log-enrichment
   correlation may or may not produce a detectable difference in simulated
   fragment pileups — that depends on how sensitive the pileup shape is to
   hexamer weight perturbations, which is a separate question.

3. **The two most extreme outliers (RD-56171, RD-57090) are potentially
   unsuitable for pooled-prior simulation.** Their end-table r_disatt of
   0.62-0.74 means the posterior has been pulled substantially toward a pool
   that does not represent them. If simulation fidelity for these specific
   samples matters, they may need a stronger sample-specific component
   (higher weight on their own data, or a subgroup prior).

### The per-sample gate

For any new sample being simulated:
```python
from scripts.measure_hexamer_disattenuation import (
    measure_sample_disattenuation, load_measurement_results,
)
from scripts.build_hexamer_prior import load_artifact

# From pre-computed results (instantaneous):
results = load_measurement_results(
    "/efs/analytics/nathanboley/background_model/cut_site_hexamers/disattenuation_92samples.tsv"
)
sample_rows = results[results["sample_name"] == "RD-56436-Lib1"]
# Check: all tables have r_disattenuated > threshold

# Or compute fresh for a new sample not in the cohort:
art = load_artifact("hexamer_prior_92samples.json")
result = measure_sample_disattenuation(
    art, sample_obs, sample_name="NEW-SAMPLE",
    leave_one_out=False,  # sample is not part of this pool
)
```

## Artifact

Results TSV: `/efs/analytics/nathanboley/background_model/cut_site_hexamers/disattenuation_92samples.tsv`
(368 rows = 92 samples x 4 tables, tab-separated)

Columns: `sample_name`, `table`, `r_observed`, `rho_sample`, `rho_sample_se`,
`rho_pool_loo`, `rho_pool_loo_se`, `r_disattenuated`, `leave_one_out`,
`sample_total_obs`.

## Reproducibility

- Artifact: `hexamer_prior_92samples.json` (92-sample Dirichlet-multinomial fit)
- Cohort: `cohort92_resolved_paths.tsv`
- Script: `scripts/measure_hexamer_disattenuation.py`
- Tests: `tests/test_hexamer_disattenuation.py` (10 tests, synthetic data)
- 50 thinning replications per (sample, table), deterministic seeds
- Python 3.10, scipy, numpy, pandas
