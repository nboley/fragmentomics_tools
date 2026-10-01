#!/usr/bin/env python
"""Measure disattenuated correlation between each sample's hexamer weights and the pool.

Disattenuation estimator
------------------------
This uses the same thinning-null approach established in commit ef60d52
("docs: add the thinning null") for the pairwise sample-sample analysis.

For each sample i and table t:

1. **Observed r**: Pearson correlation of log(w_raw_i) vs log(w_pool) over all
   hexamers where both are > 0.

2. **Thinning-null reliability**: Split sample i's raw observed counts via
   binomial thinning (p = 0.5) into two halves. Compute log-enrichment for
   each half (half_obs / bg_proportion). Correlate the two halves. This gives
   the reliability rho_xx of sample i's measurement at its actual depth.

   The pool's reliability is ~1.0 because it sums 92 samples (~5700 median
   counts per hexamer). We verify this rather than assuming it.

3. **Disattenuated r**: r_obs / sqrt(rho_sample * rho_pool).
   This estimates what the sample-vs-pool correlation would be if both were
   measured with infinite depth.

Why log-enrichment (not raw weight):
The correlations in the precedent doc (cut_site_hexamer_counts.md) are all on
log-enrichment. The log transform stabilises variance across the 250x dynamic
range of start tables; Pearson on raw weights would be dominated by a handful
of extreme hexamers.

Usage
-----
    python scripts/measure_hexamer_disattenuation.py \\
        --artifact /efs/.../hexamer_prior_92samples.json \\
        --parquet-dir /efs/.../cut_site_hexamers \\
        --output-tsv results.tsv \\
        [--n-thinning-reps 50]

Per-sample API (for the simulator gate):
    from scripts.measure_hexamer_disattenuation import (
        measure_sample_disattenuation, load_measurement_results,
    )
    # From pre-computed results:
    results = load_measurement_results("results.tsv")
    row = results[results["sample_name"] == "RD-56436-Lib1"]

    # Or compute fresh for one sample:
    art = load_artifact("hexamer_prior_92samples.json")
    result = measure_sample_disattenuation(art, sample_obs, sample_name="RD-56436-Lib1")
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_hexamer_prior import (
    load_all_samples_ordered,
    load_artifact,
)
from scripts.count_cut_site_hexamers import TABLE_NAMES

logger = logging.getLogger(__name__)

# Default number of thinning replications for the reliability estimate.
# 50 gives stable estimates (SE of the mean reliability < 0.002 at typical depths).
DEFAULT_N_THINNING_REPS = 50


def _log_enrichment(obs: np.ndarray, bg_prop: np.ndarray) -> np.ndarray:
    """Compute log-enrichment for hexamers where both obs and bg are > 0.

    Returns (log_enrich, mask) where mask indicates valid entries.
    """
    obs_f = obs.astype(np.float64)
    n = obs_f.sum()
    if n == 0:
        return np.full(len(obs), np.nan), np.zeros(len(obs), dtype=bool)
    p = obs_f / n
    mask = (p > 0) & (bg_prop > 0)
    log_enrich = np.full(len(obs), np.nan)
    log_enrich[mask] = np.log(p[mask] / bg_prop[mask])
    return log_enrich, mask


def _thinning_reliability(
    obs: np.ndarray,
    bg_prop: np.ndarray,
    n_reps: int = DEFAULT_N_THINNING_REPS,
    rng: np.random.RandomState | None = None,
) -> tuple[float, float]:
    """Estimate the split-half reliability of log-enrichment via binomial thinning.

    Splits the observed counts into two halves using binomial(n=obs_h, p=0.5)
    for each hexamer h, computes log-enrichment for each half, and returns
    the mean Pearson correlation across replications.

    Returns (mean_reliability, se_reliability).
    """
    if rng is None:
        rng = np.random.RandomState(42)

    obs_int = obs.astype(np.int64)
    total = obs_int.sum()
    if total < 20:
        return np.nan, np.nan

    corrs = []
    for _ in range(n_reps):
        half_a = rng.binomial(obs_int, 0.5)
        half_b = obs_int - half_a

        le_a, mask_a = _log_enrichment(half_a, bg_prop)
        le_b, mask_b = _log_enrichment(half_b, bg_prop)
        joint_mask = mask_a & mask_b

        if joint_mask.sum() < 10:
            continue

        r, _ = sp_stats.pearsonr(le_a[joint_mask], le_b[joint_mask])
        corrs.append(r)

    if len(corrs) < 5:
        return np.nan, np.nan

    return float(np.mean(corrs)), float(np.std(corrs) / np.sqrt(len(corrs)))


def _observed_correlation(
    sample_log_enrich: np.ndarray,
    pool_log_enrich: np.ndarray,
    sample_mask: np.ndarray,
    pool_mask: np.ndarray,
) -> float:
    """Pearson r between sample and pool log-enrichments on their joint support."""
    joint = sample_mask & pool_mask
    if joint.sum() < 10:
        return np.nan
    r, _ = sp_stats.pearsonr(sample_log_enrich[joint], pool_log_enrich[joint])
    return float(r)


def _pool_reliability(
    pooled_obs: np.ndarray,
    bg_prop: np.ndarray,
    n_reps: int = DEFAULT_N_THINNING_REPS,
    rng: np.random.RandomState | None = None,
) -> tuple[float, float]:
    """Thinning reliability for the pooled counts (expected to be ~1.0)."""
    return _thinning_reliability(pooled_obs, bg_prop, n_reps=n_reps, rng=rng)


def measure_sample_disattenuation(
    artifact: dict,
    sample_obs: dict[str, np.ndarray],
    sample_name: str = "unknown",
    n_reps: int = DEFAULT_N_THINNING_REPS,
    rng_seed: int = 42,
    leave_one_out: bool = True,
) -> pd.DataFrame:
    """Measure disattenuated r for one sample against the pooled prior.

    Parameters
    ----------
    artifact : dict
        Loaded hexamer prior artifact (from load_artifact).
    sample_obs : dict[str, ndarray]
        Raw observed counts per table: sample_obs[table_name] = (4096,) int array.
    sample_name : str
        Label for the sample.
    n_reps : int
        Number of thinning replications.
    rng_seed : int
        Random seed for reproducibility.
    leave_one_out : bool
        If True (default), subtract the sample's counts from the pooled counts
        before computing the pool's log-enrichment and reliability. This avoids
        inflating the observed r by correlating a sample with a pool that
        contains it. Set to False only when the sample is NOT part of the pool
        (e.g. a new sample being evaluated against an existing prior).

    Returns
    -------
    DataFrame with columns: sample_name, table, r_observed, rho_sample,
    rho_sample_se, rho_pool_loo, rho_pool_loo_se, r_disattenuated,
    sample_total_obs, leave_one_out.
    """
    bg = artifact["background"]
    rows = []

    for tn in TABLE_NAMES:
        bg_arr = np.array(bg[tn], dtype=np.float64)
        bg_prop = bg_arr / bg_arr.sum()

        obs_i = np.array(sample_obs[tn], dtype=np.int64)
        pooled_obs = np.array(artifact["prior"][tn]["pooled_obs"], dtype=np.int64)

        # Leave-one-out pool: subtract this sample's counts
        if leave_one_out:
            loo_pool = pooled_obs - obs_i
            # Guard: if subtraction goes negative (shouldn't happen if sample
            # is genuinely in the pool), clamp to 0
            loo_pool = np.maximum(loo_pool, 0)
        else:
            loo_pool = pooled_obs

        # Log-enrichment for sample and (LOO) pool
        le_sample, mask_sample = _log_enrichment(obs_i, bg_prop)
        le_pool, mask_pool = _log_enrichment(loo_pool, bg_prop)

        # Observed correlation
        r_obs = _observed_correlation(le_sample, le_pool, mask_sample, mask_pool)

        # Sample reliability
        rng = np.random.RandomState(rng_seed)
        rho_s, rho_s_se = _thinning_reliability(obs_i, bg_prop, n_reps, rng)

        # Pool reliability (on the LOO pool, not the full pool)
        rng_pool = np.random.RandomState(rng_seed + 1)
        rho_p, rho_p_se = _pool_reliability(loo_pool, bg_prop, n_reps, rng_pool)

        # Disattenuated r
        if np.isnan(rho_s) or np.isnan(rho_p) or rho_s <= 0 or rho_p <= 0:
            r_disatt = np.nan
        else:
            r_disatt = r_obs / np.sqrt(rho_s * rho_p)
            # Clamp to [-1, 1] — disattenuation can slightly exceed 1.0
            # due to sampling; this is expected, not an error.
            r_disatt = float(np.clip(r_disatt, -1.0, 1.0))

        rows.append({
            "sample_name": sample_name,
            "table": tn,
            "r_observed": r_obs,
            "rho_sample": rho_s,
            "rho_sample_se": rho_s_se,
            "rho_pool_loo": rho_p,
            "rho_pool_loo_se": rho_p_se,
            "r_disattenuated": r_disatt,
            "leave_one_out": leave_one_out,
            "sample_total_obs": int(obs_i.sum()),
        })

    return pd.DataFrame(rows)


def measure_cohort_disattenuation(
    artifact: dict,
    obs_by_sample: dict[str, dict[str, np.ndarray]],
    n_reps: int = DEFAULT_N_THINNING_REPS,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Measure disattenuated r for all samples in the cohort.

    Returns a DataFrame with one row per (sample, table).
    """
    frames = []
    for i, (sn, obs) in enumerate(sorted(obs_by_sample.items())):
        # Use different seed per sample so thinning draws are independent
        df = measure_sample_disattenuation(
            artifact, obs, sample_name=sn,
            n_reps=n_reps, rng_seed=rng_seed + i * 100,
        )
        frames.append(df)
        if (i + 1) % 10 == 0:
            logger.info("  measured %d / %d samples", i + 1, len(obs_by_sample))

    return pd.concat(frames, ignore_index=True)


def load_measurement_results(path: str) -> pd.DataFrame:
    """Load previously saved measurement results."""
    return pd.read_csv(path, sep="\t")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--artifact", required=True,
        help="Path to hexamer_prior_92samples.json",
    )
    parser.add_argument(
        "--parquet-dir", required=True,
        help="Directory containing per-sample .cut_site_hexamers.parquet files.",
    )
    parser.add_argument(
        "--sample-sheet", required=True,
        help="TSV with sample_name column.",
    )
    parser.add_argument(
        "--output-tsv", required=True,
        help="Output TSV path for per-sample results.",
    )
    parser.add_argument(
        "--n-thinning-reps", type=int, default=DEFAULT_N_THINNING_REPS,
        help=f"Number of thinning replications (default: {DEFAULT_N_THINNING_REPS}).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    logger.info("Loading artifact: %s", args.artifact)
    artifact = load_artifact(args.artifact)

    cohort = pd.read_csv(args.sample_sheet, sep="\t")
    sample_names = cohort["sample_name"].tolist()
    logger.info("Loading %d samples from %s", len(sample_names), args.parquet_dir)

    obs_by_sample, bg, metas, hex_order = load_all_samples_ordered(
        args.parquet_dir, sample_names,
    )

    logger.info("Measuring disattenuation (%d reps per sample)...", args.n_thinning_reps)
    results = measure_cohort_disattenuation(
        artifact, obs_by_sample,
        n_reps=args.n_thinning_reps,
    )

    results.to_csv(args.output_tsv, sep="\t", index=False, float_format="%.6f")
    logger.info("Wrote %d rows to %s", len(results), args.output_tsv)

    # Summary
    for tn in TABLE_NAMES:
        sub = results[results["table"] == tn]
        logger.info(
            "%s: r_obs median=%.4f, rho_sample median=%.4f, "
            "rho_pool_loo=%.4f, r_disatt median=%.4f [%.4f, %.4f]",
            tn,
            sub["r_observed"].median(),
            sub["rho_sample"].median(),
            sub["rho_pool_loo"].median(),
            sub["r_disattenuated"].median(),
            sub["r_disattenuated"].min(),
            sub["r_disattenuated"].max(),
        )


if __name__ == "__main__":
    main()
