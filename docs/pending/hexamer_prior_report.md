# Pooled Hexamer Prior: 92-Sample Fit Report

## Shrinkage form

**Dirichlet-multinomial conjugate model**, fitted independently per table
(start_fwd, end_fwd, start_rev, end_rev), aggregated across all 16
fragment-length bands.

For each table:

- **Prior direction**: the pooled proportion `p_pool[h] = sum_i(obs_i[h]) / sum_i(N_i)`.
- **Concentration**: `alpha[h] = alpha_0 * p_pool[h]`, where `alpha_0` is
  estimated from the 92-sample spread via method of moments (Ronning 1989).
- **Per-sample posterior mean**: `theta_i[h] = (obs_i[h] + alpha[h]) / (N_i + alpha_0)`.
- **Hexamer weight**: `w_i[h] = theta_i[h] / bg_proportion[h]`, normalised to sum 1.

**Why this form**: the Dirichlet-multinomial is the conjugate prior for
multinomial count data, which is what observed cut-site hexamer counts are
(each cut event picks one of 4096 hexamers). The concentration `alpha_0`
controls the bias-variance tradeoff: larger `alpha_0` means more trust in the
pool, less in the individual sample. The method-of-moments estimator matches
the observed cross-sample variance to the Dirichlet-multinomial prediction,
automatically calibrating the shrinkage to the data.

**Note on statistical semantics**: this is the PROPOSED form. The
Dirichlet-multinomial is standard and well-understood, but the owner has final
say on whether this is the right shrinkage for the simulator's hexamer tables.

## What 92 samples bought

### Depth

| Metric | 5-sample pool (est.) | 92-sample pool | Ratio |
|--------|---------------------|----------------|-------|
| Per-hexamer median count (start tables) | ~309 | 5,694 | 18.4x |
| Per-hexamer median count (end tables) | ~403 | 7,424 | 18.4x |
| Zero-count hexamers (pooled) | 0 | 0 | -- |
| Zero-count hexamers (per-sample median) | -- | 27.5 (start), 18 (end) | -- |
| Zero-count hexamers (per-sample max) | -- | 215 (start), 211 (end) | -- |
| Total observed cuts (pooled, per table) | ~1.9M | ~34.9M | 18.4x |

The 5-sample pool was reported to have median ~270 counts per hexamer and 5-6%
CV at the median hexamer. With 92 samples the depth is 18.4x larger, which:

1. **Eliminates pooled zeros entirely.** Even the rarest hexamers have hundreds
   of pooled counts. The zero-count problem is a per-sample issue (median 27
   zeros per sample in start tables), not a pool-level issue.

2. **Makes the pooled prior effectively exact.** At ~5700 counts per hexamer,
   the sampling uncertainty in the pooled proportion is ~1.3% (1/sqrt(5700)),
   far below the biological inter-sample CV of ~15%.

### Per-sample zeros

Individual samples have ~338k total observed cuts per table, or ~82 counts per
hexamer at the median. At this depth, 27-28 of 4096 hexamers (0.7%) have zero
counts in a typical sample, and the worst sample has 215 zeros (5.2%). The
Dirichlet posterior assigns non-zero weight to all of these — the attenuation
test suite verifies this property holds on synthetic data, and the real
artifact confirms: **zero posterior-weight hexamers = 0** across all 92 samples
and all 4 tables.

## Estimated concentration parameter (alpha_0)

| Table | alpha_0 | alpha_0 / (alpha_0 + median_N) | Interpretation |
|-------|---------|-------------------------------|----------------|
| start_fwd | 102,481 | 0.233 | 23% prior, 77% data |
| end_fwd | 134,102 | 0.284 | 28% prior, 72% data |
| start_rev | 111,569 | 0.248 | 25% prior, 75% data |
| end_rev | 133,607 | 0.283 | 28% prior, 72% data |

The end tables have higher `alpha_0`, meaning less biological variation
(consistent with their lower raw CV of ~0.54 vs ~0.84 for start tables).
A typical sample's posterior is ~75% driven by its own data and ~25% by the
pool.

## Do per-sample posteriors still carry signal?

**Yes.** The posteriors do not collapse onto the prior.

| Table | Posterior inter-sample CV (median) | Pairwise r | Shrinkage ratio (median) |
|-------|-----------------------------------|------------|--------------------------|
| start_fwd | 14.6% | 0.893 | 0.233 |
| end_fwd | 12.5% | 0.728 | 0.286 |
| start_rev | 14.2% | 0.897 | 0.249 |
| end_rev | 12.5% | 0.725 | 0.284 |

- **Inter-sample CV**: the posterior weights still vary 12-15% across samples
  at the median hexamer. This is biological signal that survived shrinkage.
- **Pairwise correlation**: 0.73-0.90. If posteriors collapsed onto the prior,
  this would be ~1.0.
- **Shrinkage ratio**: each sample moves ~23-29% of the way from its raw
  estimate toward the prior. The range across samples (min 4-6%, max 57-64%)
  confirms the monotone-attenuation property: low-depth samples shrink more.

## Artifact format

The artifact is a JSON file at:
`/efs/analytics/nathanboley/background_model/cut_site_hexamers/hexamer_prior_92samples.json`

Structure:
```
{
  "version": 1,
  "hex_order": ["AAAAAA", ...],    // 4096 hexamer strings, canonical order
  "table_names": ["start_fwd", "end_fwd", "start_rev", "end_rev"],
  "prior": {
    "<table>": {
      "alpha_0": float,             // concentration parameter
      "alpha": [float x 4096],      // Dirichlet concentration vector
      "pooled_obs": [int x 4096],   // sum of observed counts across 92 samples
      "pooled_total": int,
      "pooled_weight": [float x 4096]  // obs/bg, normalised — the prior weight
    }
  },
  "posteriors": {
    "<sample_name>": {
      "<table>": [float x 4096]    // posterior weight, ready for HexamerTables
    }
  },
  "background": {
    "<table>": [int x 4096]        // background counts (same for all samples)
  },
  "diagnostics": { ... },
  "provenance": {
    "regions_bed_md5": "761b711c1c98034d164f5f77b22481f0",
    "shrinkage_form": "Dirichlet-multinomial conjugate ...",
    ...
  }
}
```

To load into the simulator:
```python
from scripts.build_hexamer_prior import load_artifact, artifact_to_hex_tables
art = load_artifact("hexamer_prior_92samples.json")
hex_tables, hex_order = artifact_to_hex_tables(art, sample_name="RD-56436-Lib1")
# or for the pooled prior:
hex_tables, hex_order = artifact_to_hex_tables(art, sample_name=None)
```

## Attenuation test

`tests/test_hexamer_prior_attenuation.py` — 7 tests, all passing:

1. **No zeros in posterior** — every posterior weight is strictly positive
2. **Shrinkage toward prior** — every sample's posterior is closer to the prior than its raw estimate (L2)
3. **Monotone attenuation** — low-count samples shrink more than high-count samples
4. **Sparse sample coverage** — a sample with >90% zero-count hexamers still gets all-positive posteriors
5. **Large-N limit** — with 10M observations, posterior ≈ raw estimate (max diff < 1e-4)
6. **alpha_0 positive** — the concentration parameter is always positive
7. **alpha_0 monotonicity** — less biological variance → larger alpha_0
