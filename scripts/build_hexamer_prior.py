#!/usr/bin/env python
"""Build a pooled hexamer prior with per-sample posteriors from 92 counted samples.

Shrinkage form
--------------
Dirichlet-multinomial conjugate model, fitted independently per table
(start_fwd, end_fwd, start_rev, end_rev).

For each table:

1. **Observed proportions**: For sample i, p_i[h] = obs_i[h] / N_i where
   N_i = sum_h obs_i[h].

2. **Pooled prior**: The Dirichlet concentration vector alpha[h] = alpha_0 *
   p_pool[h], where p_pool is the pooled proportion (sum of all samples' obs
   counts, normalised) and alpha_0 is estimated via method of moments from the
   92-sample spread.

3. **Per-sample posterior mean**:
   theta_i[h] = (obs_i[h] + alpha[h]) / (N_i + alpha_0)
   This is the Dirichlet(alpha + obs_i) posterior mean.

4. **Hexamer weight** (what the simulator consumes):
   w_i[h] = theta_i[h] / bg_proportion[h]
   normalised so sum_h w_i[h] = 1.

The method-of-moments estimate for alpha_0 uses:
   alpha_0 = (p_pool * (1 - p_pool) - mean_var) / (mean_var - p_pool * (1 - p_pool) / N_mean)
where mean_var = mean_i(var_h(p_i)) and N_mean = harmonic_mean(N_i).
This is the standard Ronning (1989) estimator.

Output
------
A JSON file containing:
- ``prior``: the four alpha vectors and alpha_0
- ``posteriors``: per-sample posterior weights (4 tables x 4096 each)
- ``diagnostics``: CV, zero counts, attenuation metrics
- ``provenance``: region set, sample list, parameters

The artifact is designed to be consumed by the simulator as HexamerTables:
each sample's posterior weights can be loaded via ``emit.dataframe_to_hex_table``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.count_cut_site_hexamers import TABLE_NAMES, read_output  # noqa: E402

logger = logging.getLogger(__name__)


def _hexamer_order(df: pd.DataFrame) -> list[str]:
    """Get the hexamer order from the first (table, band) group."""
    first_table = df["table"].iloc[0]
    first_band = df["band_lo"].iloc[0]
    mask = (df["table"] == first_table) & (df["band_lo"] == first_band)
    return df.loc[mask, "hexamer"].tolist()


def load_all_samples_ordered(parquet_dir: str, sample_names: list[str]):
    """Load all samples, returning arrays in a consistent hexamer order.

    Returns the hexamer string order alongside the arrays so the consumer
    can rebuild index-keyed arrays through its own vocabulary.
    """
    obs_by_sample = {}
    bg_accum = None
    metas = {}
    hex_order = None

    for sample_name in sample_names:
        path = os.path.join(parquet_dir, f"{sample_name}.cut_site_hexamers.parquet")
        df, meta = read_output(path)
        metas[sample_name] = meta

        if hex_order is None:
            hex_order = _hexamer_order(df)

        obs = {}
        bg_local = {}
        for table_name in TABLE_NAMES:
            tdf = df[df["table"] == table_name]
            agg = tdf.groupby("hexamer", sort=False)[["observed", "background"]].sum()
            # Reindex to the canonical order
            agg = agg.reindex(hex_order)
            obs[table_name] = agg["observed"].values.astype(np.int64)
            bg_local[table_name] = agg["background"].values.astype(np.int64)

        obs_by_sample[sample_name] = obs

        if bg_accum is None:
            bg_accum = bg_local
        else:
            for tn in TABLE_NAMES:
                if not np.array_equal(bg_accum[tn], bg_local[tn]):
                    raise ValueError(
                        f"Background differs between samples for {tn}."
                    )

    return obs_by_sample, bg_accum, metas, hex_order


def estimate_dirichlet_alpha0(
    obs_by_sample: dict[str, dict[str, np.ndarray]],
    table_name: str,
) -> float:
    """Estimate Dirichlet concentration alpha_0 via method of moments.

    Uses the Ronning (1989) estimator: match the mean within-sample variance
    of proportions to the Dirichlet-multinomial prediction.
    """
    sample_names = list(obs_by_sample.keys())
    n_samples = len(sample_names)

    # Collect per-sample proportions and total counts
    proportions = []
    total_counts = []
    for sn in sample_names:
        counts = obs_by_sample[sn][table_name].astype(np.float64)
        n = counts.sum()
        if n == 0:
            continue
        proportions.append(counts / n)
        total_counts.append(n)

    proportions = np.array(proportions)  # (n_samples, 4096)
    total_counts = np.array(total_counts)

    # Pooled proportion (prior mean direction) — count-weighted, consistent
    # with the pooled p_pool in build_prior_and_posteriors and with the
    # Ronning (1989) estimator this code cites.
    all_counts = np.array([obs_by_sample[sn][table_name].astype(np.float64)
                           for sn in sample_names if obs_by_sample[sn][table_name].sum() > 0])
    p_pool = all_counts.sum(axis=0) / all_counts.sum()

    # Mean within-sample variance of proportions
    # Var_multinomial(p_h) = p_h(1-p_h)/N for a single sample
    # With Dirichlet: Var(p_h) = p_pool_h(1-p_pool_h) * (1 + N)/(alpha_0 + N) / N
    # Observed variance across samples:
    var_across = np.var(proportions, axis=0, ddof=1)  # (4096,)

    # Expected variance under multinomial alone (no biological variation):
    # E[var] = p_pool * (1 - p_pool) / N_typical
    N_harm = len(total_counts) / np.sum(1.0 / total_counts)  # harmonic mean

    # The observed cross-sample variance has two components:
    # 1. Sampling variance: p(1-p)/N
    # 2. Biological variance: p(1-p) / (alpha_0 + 1)
    # Total: p(1-p) * (1/N + 1/(alpha_0+1))
    # = p(1-p) * (alpha_0 + 1 + N) / (N * (alpha_0 + 1))

    # Method of moments: match sum of variances
    S_obs = var_across.sum()  # observed total variance
    pq = (p_pool * (1 - p_pool))
    S_pq = pq.sum()

    # S_obs ≈ S_pq * (alpha_0 + 1 + N_harm) / (N_harm * (alpha_0 + 1))
    # => N_harm * (alpha_0 + 1) * S_obs = S_pq * (alpha_0 + 1 + N_harm)
    # => (alpha_0 + 1) * (N_harm * S_obs - S_pq) = S_pq * N_harm
    # => alpha_0 + 1 = S_pq * N_harm / (N_harm * S_obs - S_pq)
    # => alpha_0 = S_pq * N_harm / (N_harm * S_obs - S_pq) - 1

    denom = N_harm * S_obs - S_pq
    if denom <= 0:
        # Observed variance is less than multinomial alone — no overdispersion
        # detected. Return a large alpha_0 (weak shrinkage).
        logger.warning(
            "%s: no overdispersion detected (sampling variance exceeds "
            "observed cross-sample variance). Using alpha_0 = 1e6.",
            table_name,
        )
        return 1e6

    alpha_0 = S_pq * N_harm / denom - 1.0
    if alpha_0 < 1.0:
        # Floor at 1.0 — below this the prior is so diffuse it provides
        # almost no regularisation and individual hexamer posteriors can
        # be dominated by a single observation.
        alpha_0 = max(alpha_0, 1.0)

    return float(alpha_0)


def build_prior_and_posteriors(
    obs_by_sample: dict[str, dict[str, np.ndarray]],
    bg: dict[str, np.ndarray],
    hex_order: list[str],
):
    """Build pooled prior and per-sample posteriors.

    Returns
    -------
    prior : dict
        Keys: alpha_0 (per table), alpha (per table, 4096-vector),
        pooled_weight (per table, 4096-vector).
    posteriors : dict[str, dict[str, ndarray]]
        posteriors[sample_name][table_name] = (4096,) weight array
    diagnostics : dict
    """
    sample_names = list(obs_by_sample.keys())
    n_samples = len(sample_names)

    prior = {}
    posteriors = {sn: {} for sn in sample_names}
    diagnostics = {}

    for tn in TABLE_NAMES:
        # Pool observed counts across all samples
        pooled_obs = np.zeros(len(hex_order), dtype=np.int64)
        for sn in sample_names:
            pooled_obs += obs_by_sample[sn][tn]

        pooled_total = pooled_obs.sum()
        p_pool = pooled_obs.astype(np.float64) / pooled_total

        # Estimate concentration parameter
        alpha_0 = estimate_dirichlet_alpha0(obs_by_sample, tn)
        alpha = alpha_0 * p_pool  # Dirichlet concentration vector

        # Pooled weight: obs/bg, normalised
        bg_arr = bg[tn].astype(np.float64)
        bg_prop = bg_arr / bg_arr.sum()

        pooled_weight = np.where(bg_prop > 0, p_pool / bg_prop, 0.0)
        pooled_weight /= pooled_weight.sum()

        prior[tn] = {
            "alpha_0": alpha_0,
            "alpha": alpha.tolist(),
            "pooled_obs": pooled_obs.tolist(),
            "pooled_total": int(pooled_total),
            "pooled_weight": pooled_weight.tolist(),
        }

        # Per-sample diagnostics
        n_zeros_raw = []
        cv_raw = []
        cv_posterior = []
        shrinkage_ratios = []

        for sn in sample_names:
            obs_i = obs_by_sample[sn][tn].astype(np.float64)
            n_i = obs_i.sum()

            # Raw weight
            raw_p = obs_i / n_i if n_i > 0 else np.zeros_like(obs_i)
            raw_weight = np.where(bg_prop > 0, raw_p / bg_prop, 0.0)
            if raw_weight.sum() > 0:
                raw_weight /= raw_weight.sum()

            # Posterior mean
            theta_i = (obs_i + alpha) / (n_i + alpha_0)
            post_weight = np.where(bg_prop > 0, theta_i / bg_prop, 0.0)
            post_weight /= post_weight.sum()

            posteriors[sn][tn] = post_weight

            # Diagnostics
            n_zeros_raw.append(int((obs_i == 0).sum()))

            nonzero_mask = raw_weight > 0
            if nonzero_mask.sum() > 1:
                cv_raw.append(float(np.std(raw_weight[nonzero_mask]) / np.mean(raw_weight[nonzero_mask])))
            else:
                cv_raw.append(float("nan"))

            nonzero_post = post_weight > 0
            if nonzero_post.sum() > 1:
                cv_posterior.append(float(np.std(post_weight[nonzero_post]) / np.mean(post_weight[nonzero_post])))
            else:
                cv_posterior.append(float("nan"))

            # Shrinkage ratio: how far each sample moved from raw toward prior
            if raw_weight.sum() > 0:
                dist_raw_to_prior = np.sqrt(((raw_weight - pooled_weight) ** 2).sum())
                dist_post_to_prior = np.sqrt(((post_weight - pooled_weight) ** 2).sum())
                if dist_raw_to_prior > 0:
                    shrinkage_ratios.append(float(1.0 - dist_post_to_prior / dist_raw_to_prior))
                else:
                    shrinkage_ratios.append(float("nan"))
            else:
                shrinkage_ratios.append(float("nan"))

        # Pairwise posterior correlation (M2: provenance for the report)
        post_weights_all = np.array([posteriors[sn][tn] for sn in sample_names])
        pairwise_rs = []
        for ii in range(n_samples):
            for jj in range(ii + 1, n_samples):
                r, _ = sp_stats.pearsonr(post_weights_all[ii], post_weights_all[jj])
                pairwise_rs.append(r)

        diagnostics[tn] = {
            "alpha_0": alpha_0,
            "n_samples": n_samples,
            "pooled_total_obs": int(pooled_total),
            "median_per_sample_obs": float(np.median([obs_by_sample[sn][tn].sum() for sn in sample_names])),
            "n_zeros_raw_median": float(np.median(n_zeros_raw)),
            "n_zeros_raw_max": int(max(n_zeros_raw)),
            "n_zeros_posterior": 0,  # posterior always > 0 if alpha > 0
            "cv_raw_median": float(np.nanmedian(cv_raw)),
            "cv_posterior_median": float(np.nanmedian(cv_posterior)),
            "shrinkage_ratio_median": float(np.nanmedian(shrinkage_ratios)),
            "shrinkage_ratio_min": float(np.nanmin(shrinkage_ratios)) if shrinkage_ratios else float("nan"),
            "shrinkage_ratio_max": float(np.nanmax(shrinkage_ratios)) if shrinkage_ratios else float("nan"),
            "pairwise_posterior_r_mean": float(np.mean(pairwise_rs)),
            "pairwise_posterior_r_median": float(np.median(pairwise_rs)),
        }

    return prior, posteriors, diagnostics


def write_artifact(
    out_path: str,
    prior: dict,
    posteriors: dict[str, dict[str, np.ndarray]],
    diagnostics: dict,
    hex_order: list[str],
    bg: dict[str, np.ndarray],
    provenance: dict,
):
    """Write the prior artifact as a JSON file."""
    # Convert posteriors to serialisable form
    post_json = {}
    for sn, tables in posteriors.items():
        post_json[sn] = {
            tn: arr.tolist() for tn, arr in tables.items()
        }

    bg_json = {tn: arr.tolist() for tn, arr in bg.items()}

    artifact = {
        "version": 1,
        "description": (
            "Pooled Dirichlet-multinomial hexamer prior with per-sample "
            "posteriors, fitted over the region-set domain. Each posterior "
            "weight w[h] = theta_posterior[h] / bg_proportion[h], normalised. "
            "The simulator consumes these as HexamerTables."
        ),
        "hex_order": hex_order,
        "table_names": list(TABLE_NAMES),
        "prior": prior,
        "posteriors": post_json,
        "background": bg_json,
        "diagnostics": diagnostics,
        "provenance": provenance,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=False)
    logger.info("Wrote artifact to %s (%.1f MB)", out_path,
                os.path.getsize(out_path) / 1e6)


def verify_provenance(*script_paths: str) -> dict:
    """Verify that every path in *script_paths* matches its committed blob at HEAD.

    Returns a dict::

        {
            "commit_sha": "<HEAD>",
            "script_shas": {
                "scripts/build_hexamer_prior.py": "<sha256>",
                "scripts/count_cut_site_hexamers.py": "<sha256>",
                ...
            },
        }

    Raises RuntimeError if any file is untracked or differs from HEAD.

    Design note: the guard scopes to the listed files, not the whole
    worktree.  A worktree that is permanently dirty due to another active
    stream (e.g. an in-progress doc rewrite) must not block a run.  The
    property worth guaranteeing is: every sha an artifact records can be
    resolved by checking out the recorded commit.
    """
    import hashlib
    import subprocess

    if not script_paths:
        raise ValueError("verify_provenance requires at least one path")

    first = os.path.abspath(script_paths[0])
    repo_dir = os.path.dirname(first)

    # 1. Resolve HEAD commit.
    try:
        commit_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"Cannot resolve HEAD (git unavailable or not a repo): {exc}"
        ) from exc

    # 2. Repo root for relative paths.
    try:
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=repo_dir, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Cannot find repo root: {exc}") from exc

    # 3. Verify each file.
    script_shas = {}
    for sp in script_paths:
        sp_abs = os.path.abspath(sp)
        rel_path = os.path.relpath(sp_abs, repo_root)

        try:
            committed_content = subprocess.check_output(
                ["git", "show", f"HEAD:{rel_path}"],
                cwd=repo_root, stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            raise RuntimeError(
                f"Provenance guard: {rel_path} is not tracked at HEAD "
                f"({commit_sha[:12]}). Add and commit it before running."
            )

        with open(sp_abs, "rb") as f:
            disk_content = f.read()

        committed_sha = hashlib.sha256(committed_content).hexdigest()
        disk_sha = hashlib.sha256(disk_content).hexdigest()

        if committed_sha != disk_sha:
            raise RuntimeError(
                f"Provenance guard: {rel_path} on disk "
                f"(sha256 {disk_sha[:16]}...) differs from HEAD "
                f"({commit_sha[:12]}, sha256 {committed_sha[:16]}...). "
                f"Commit the script before running so the recorded provenance "
                f"resolves to a reproducible state."
            )

        script_shas[rel_path] = disk_sha

    return {"commit_sha": commit_sha, "script_shas": script_shas}


def verify_script_provenance(script_path: str) -> tuple[str, str]:
    """Verify *script_path* matches its committed blob at HEAD.

    Returns (commit_sha, script_sha256) on success.
    Thin wrapper around :func:`verify_provenance` for backward compatibility.
    """
    prov = verify_provenance(script_path)
    sha = next(iter(prov["script_shas"].values()))
    return prov["commit_sha"], sha


def load_artifact(path: str):
    """Load a prior artifact. Returns (artifact_dict, HexamerTables_for_pooled)."""
    with open(path) as fh:
        artifact = json.load(fh)
    return artifact


class HexamerTables(NamedTuple):
    """The four cut-site tables, one ``(4096,)`` array each.

    A plain container, local to this script.  It used to be imported from the
    previous-generation simulator's ``weights.py`` (now in
    ``attic/pre_rewrite_simulator/``), where it also carried that sampler's
    strand semantics.  The live simulator passes tables as a dict keyed by
    ``TABLE_NAMES`` and has no container type, so there was nothing live to
    import it from.
    """
    start_fwd: np.ndarray
    end_fwd: np.ndarray
    start_rev: np.ndarray
    end_rev: np.ndarray


def artifact_to_hex_tables(artifact: dict, sample_name: str | None = None):
    """Extract HexamerTables from the artifact.

    If sample_name is None, returns the pooled prior weights.
    Otherwise, returns the per-sample posterior weights.
    """
    hex_order = artifact["hex_order"]

    if sample_name is None:
        # Pooled prior
        tables = {}
        for tn in TABLE_NAMES:
            tables[tn] = np.array(artifact["prior"][tn]["pooled_weight"])
        return HexamerTables(**tables), hex_order
    else:
        # Per-sample posterior
        tables = {}
        for tn in TABLE_NAMES:
            tables[tn] = np.array(artifact["posteriors"][sample_name][tn])
        return HexamerTables(**tables), hex_order


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--parquet-dir", required=True,
        help="Directory containing per-sample .cut_site_hexamers.parquet files.",
    )
    parser.add_argument(
        "--sample-sheet", required=True,
        help="TSV with columns: sample_id, sample_name, endo_category, h5_resolved.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSON path for the prior artifact.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    # Verify provenance BEFORE any computation: every script in the
    # dependency chain must match HEAD so recorded shas resolve later.
    counting_script = os.path.join(os.path.dirname(__file__), "count_cut_site_hexamers.py")
    prov = verify_provenance(__file__, counting_script)

    cohort = pd.read_csv(args.sample_sheet, sep="\t")
    sample_names = cohort["sample_name"].tolist()
    logger.info("Loading %d samples from %s", len(sample_names), args.parquet_dir)

    obs_by_sample, bg, metas, hex_order = load_all_samples_ordered(
        args.parquet_dir, sample_names,
    )

    logger.info("Building prior and posteriors...")
    prior, posteriors, diagnostics = build_prior_and_posteriors(
        obs_by_sample, bg, hex_order,
    )

    # Log summary diagnostics
    for tn in TABLE_NAMES:
        d = diagnostics[tn]
        logger.info(
            "%s: alpha_0=%.1f, pooled_total=%d, median_per_sample=%d, "
            "zeros_raw_median=%.0f, cv_raw=%.4f, cv_post=%.4f, "
            "shrinkage_median=%.4f",
            tn, d["alpha_0"], d["pooled_total_obs"],
            d["median_per_sample_obs"], d["n_zeros_raw_median"],
            d["cv_raw_median"], d["cv_posterior_median"],
            d["shrinkage_ratio_median"],
        )

    # Provenance — verified by verify_provenance() at the start.
    first_meta = next(iter(metas.values()))
    provenance = {
        "n_samples": len(sample_names),
        "sample_names": sample_names,
        "parquet_dir": os.path.abspath(args.parquet_dir),
        "sample_sheet": os.path.abspath(args.sample_sheet),
        "regions_bed": first_meta.get("regions_bed"),
        "regions_bed_md5": first_meta.get("regions_bed_md5"),
        "n_regions": first_meta.get("n_regions_in_bed"),
        "l_min": first_meta.get("l_min"),
        "l_max_inclusive": first_meta.get("l_max_inclusive"),
        "min_mapq": first_meta.get("min_mapq"),
        "commit_sha": prov["commit_sha"],
        "script_shas": prov["script_shas"],
        "shrinkage_form": (
            "Dirichlet-multinomial conjugate: alpha = alpha_0 * p_pool, "
            "posterior mean theta_i = (obs_i + alpha) / (N_i + alpha_0), "
            "weight w_i = theta_i / bg_proportion, normalised. "
            "alpha_0 estimated via method of moments (Ronning 1989)."
        ),
    }

    write_artifact(args.output, prior, posteriors, diagnostics, hex_order, bg, provenance)


if __name__ == "__main__":
    main()
