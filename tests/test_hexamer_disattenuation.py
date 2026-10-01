"""Tests for the disattenuation measurement (scripts/measure_hexamer_disattenuation.py).

These tests verify the thinning-null disattenuation estimator on synthetic data
where ground truth is known. They do NOT require the 92-sample artifact or
parquet files.
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.measure_hexamer_disattenuation import (
    _log_enrichment,
    _thinning_reliability,
    _observed_correlation,
    measure_sample_disattenuation,
)
from scripts.build_hexamer_prior import build_prior_and_posteriors
from scripts.count_cut_site_hexamers import TABLE_NAMES

NHEX = 4096


def _make_artifact_and_obs(
    n_samples: int = 20,
    total_per_sample: int = 5000,
    rng_seed: int = 42,
):
    """Build a synthetic artifact and per-sample obs for testing."""
    rng = np.random.RandomState(rng_seed)

    true_bias = rng.dirichlet(np.ones(NHEX) * 5.0)
    bg_flat = np.ones(NHEX, dtype=np.int64) * 1000
    bg = {tn: bg_flat.copy() for tn in TABLE_NAMES}
    hex_order = [f"HEX{i:04d}" for i in range(NHEX)]

    obs_by_sample = {}
    for i in range(n_samples):
        n_i = total_per_sample + rng.randint(-1000, 1000)
        obs_counts = rng.multinomial(n_i, true_bias)
        obs_by_sample[f"sample_{i:03d}"] = {
            tn: obs_counts.copy() for tn in TABLE_NAMES
        }

    prior, posteriors, diagnostics = build_prior_and_posteriors(
        obs_by_sample, bg, hex_order,
    )

    # Build a minimal artifact dict matching the real artifact structure
    artifact = {
        "hex_order": hex_order,
        "table_names": list(TABLE_NAMES),
        "prior": prior,
        "posteriors": {
            sn: {tn: post.tolist() for tn, post in tables.items()}
            for sn, tables in posteriors.items()
        },
        "background": {tn: arr.tolist() for tn, arr in bg.items()},
        "diagnostics": diagnostics,
    }
    return artifact, obs_by_sample


class TestLogEnrichment:
    def test_basic(self):
        obs = np.array([100, 200, 0, 50])
        bg_prop = np.array([0.25, 0.25, 0.25, 0.25])
        le, mask = _log_enrichment(obs, bg_prop)
        assert mask[0] and mask[1] and not mask[2] and mask[3]
        # obs proportions: 100/350, 200/350, 0, 50/350
        # enrichment[0] = (100/350) / 0.25
        np.testing.assert_allclose(le[0], np.log((100 / 350) / 0.25), rtol=1e-10)

    def test_all_zero_obs(self):
        obs = np.zeros(10, dtype=np.int64)
        bg_prop = np.ones(10) / 10
        le, mask = _log_enrichment(obs, bg_prop)
        assert not mask.any()
        assert np.all(np.isnan(le))


class TestThinningReliability:
    """Tests for the thinning-null reliability estimator.

    These use NHEX_SMALL=100 hexamers to keep counts per hexamer adequate
    even at low total depth. The estimator's correctness does not depend
    on the vocabulary size.
    """
    NHEX_SMALL = 100

    def test_high_depth_high_reliability(self):
        """At very high depth, reliability should be close to 1.0."""
        rng = np.random.RandomState(123)
        true_p = rng.dirichlet(np.ones(self.NHEX_SMALL) * 5.0)
        obs = rng.multinomial(10_000_000, true_p)
        bg_prop = np.ones(self.NHEX_SMALL) / self.NHEX_SMALL

        rho, se = _thinning_reliability(obs, bg_prop, n_reps=20, rng=np.random.RandomState(0))
        assert rho > 0.999, f"Expected reliability > 0.999 at 10M depth, got {rho:.4f}"

    def test_low_depth_lower_reliability(self):
        """At low depth (~10 counts/hexamer), reliability should be noticeably below 1."""
        rng = np.random.RandomState(123)
        true_p = rng.dirichlet(np.ones(self.NHEX_SMALL) * 5.0)
        # 1000 counts over 100 hexamers = ~10/hexamer
        obs = rng.multinomial(1000, true_p)
        bg_prop = np.ones(self.NHEX_SMALL) / self.NHEX_SMALL

        rho, se = _thinning_reliability(obs, bg_prop, n_reps=30, rng=np.random.RandomState(0))
        assert rho < 0.98, f"Expected reliability < 0.98 at ~10 counts/hex, got {rho:.4f}"

    def test_reliability_monotone_in_depth(self):
        """Higher depth -> higher reliability."""
        rng = np.random.RandomState(123)
        true_p = rng.dirichlet(np.ones(self.NHEX_SMALL) * 5.0)
        bg_prop = np.ones(self.NHEX_SMALL) / self.NHEX_SMALL

        rho_low, _ = _thinning_reliability(
            rng.multinomial(2000, true_p), bg_prop, n_reps=30,
            rng=np.random.RandomState(0),
        )
        rho_high, _ = _thinning_reliability(
            rng.multinomial(500_000, true_p), bg_prop, n_reps=30,
            rng=np.random.RandomState(1),
        )
        assert rho_high > rho_low, (
            f"rho_high={rho_high:.4f} should exceed rho_low={rho_low:.4f}"
        )


class TestMeasureSampleDisattenuation:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.artifact, self.obs_by_sample = _make_artifact_and_obs()

    def test_returns_dataframe_with_expected_columns(self):
        sn = list(self.obs_by_sample.keys())[0]
        df = measure_sample_disattenuation(
            self.artifact, self.obs_by_sample[sn],
            sample_name=sn, n_reps=10,
        )
        expected_cols = {
            "sample_name", "table", "r_observed", "rho_sample",
            "rho_sample_se", "rho_pool_loo", "rho_pool_loo_se",
            "r_disattenuated", "sample_total_obs", "leave_one_out",
        }
        assert set(df.columns) == expected_cols
        assert len(df) == len(TABLE_NAMES)

    def test_all_correlations_positive(self):
        """All samples drawn from the same multinomial should correlate positively
        with the pool."""
        sn = list(self.obs_by_sample.keys())[0]
        df = measure_sample_disattenuation(
            self.artifact, self.obs_by_sample[sn],
            sample_name=sn, n_reps=10,
        )
        assert (df["r_observed"] > 0).all()
        assert (df["r_disattenuated"] > 0).all()

    def test_disattenuated_geq_observed(self):
        """Disattenuated r should be >= observed r (noise only depresses r)."""
        sn = list(self.obs_by_sample.keys())[0]
        df = measure_sample_disattenuation(
            self.artifact, self.obs_by_sample[sn],
            sample_name=sn, n_reps=10,
        )
        for _, row in df.iterrows():
            assert row["r_disattenuated"] >= row["r_observed"] - 0.01, (
                f"Disattenuated r ({row['r_disattenuated']:.4f}) should be >= "
                f"observed r ({row['r_observed']:.4f}) for {row['table']}"
            )

    def test_disattenuated_higher_than_observed_at_high_depth(self):
        """A high-depth sample from the same distribution as the pool should
        have disattenuated r close to (or clamped at) 1.0, and higher than
        its observed r."""
        # Rebuild with high-depth fixture so pool reliability is adequate
        art, obs = _make_artifact_and_obs(
            n_samples=20, total_per_sample=200_000, rng_seed=42,
        )
        sn = list(obs.keys())[0]
        df = measure_sample_disattenuation(
            art, obs[sn], sample_name=sn, n_reps=20,
        )
        for _, row in df.iterrows():
            assert row["r_disattenuated"] > 0.90, (
                f"Same-distribution high-depth sample should have disattenuated r > 0.90, "
                f"got {row['r_disattenuated']:.4f} for {row['table']}"
            )
            assert row["r_disattenuated"] >= row["r_observed"] - 0.005

    def test_pool_reliability_scales_with_depth(self):
        """A high-count pool should have higher reliability than a low-count one."""
        art_lo, obs_lo = _make_artifact_and_obs(
            n_samples=5, total_per_sample=10_000, rng_seed=77,
        )
        art_hi, obs_hi = _make_artifact_and_obs(
            n_samples=30, total_per_sample=200_000, rng_seed=77,
        )
        sn_lo = list(obs_lo.keys())[0]
        sn_hi = list(obs_hi.keys())[0]
        df_lo = measure_sample_disattenuation(art_lo, obs_lo[sn_lo], n_reps=20)
        df_hi = measure_sample_disattenuation(art_hi, obs_hi[sn_hi], n_reps=20)
        for tn in TABLE_NAMES:
            rho_lo = df_lo.loc[df_lo["table"] == tn, "rho_pool_loo"].values[0]
            rho_hi = df_hi.loc[df_hi["table"] == tn, "rho_pool_loo"].values[0]
            assert rho_hi > rho_lo, (
                f"{tn}: high-count pool rho ({rho_hi:.4f}) should exceed "
                f"low-count pool rho ({rho_lo:.4f})"
            )


class TestDisattenuationGoldenValues:
    """Pin the full measurement chain to exact values so formula errors are caught.

    The property tests above all pass for a wide range of return values and
    CANNOT detect, e.g., removing the Spearman-Brown step-up, double-applying
    it, or applying it to the pool but not the sample.  This test pins
    r_observed, rho_sample, rho_pool_loo, and r_disattenuated for a single
    deterministic (sample, table) to ~3 significant figures.

    Fixture: 30 samples drawn from a shared multinomial (NHEX=256,
    depth ~100k each), plus one "deviant" sample whose true p is a 70/30
    mixture of the pool p and an independent Dirichlet draw (depth 80k).
    The deviant's r_disattenuated is ~0.911 (unclamped), which means the
    step-up matters: removing it shifts rho_sample from 0.941 to 0.889
    and r_disattenuated from 0.911 to 0.938, both far outside tolerance.

    Which formula error each pinned value catches:
      r_observed    — wrong log-enrichment, wrong mask, wrong Pearson call
      rho_sample    — step-up removed or double-applied on the SAMPLE side
      rho_pool_loo  — step-up removed or double-applied on the POOL side,
                      or LOO subtraction broken
      r_disattenuated — any of the above, plus wrong denominator formula
                        (e.g. product vs geometric mean)
    """
    NHEX_GOLDEN = 256

    @pytest.fixture(autouse=True)
    def setup(self):
        rng = np.random.RandomState(99)
        pool_p = rng.dirichlet(np.ones(self.NHEX_GOLDEN) * 10.0)

        bg_flat = np.ones(self.NHEX_GOLDEN, dtype=np.int64) * 5000
        bg = {tn: bg_flat.copy() for tn in TABLE_NAMES}
        hex_order = [f"HEX{i:04d}" for i in range(self.NHEX_GOLDEN)]

        obs_by_sample = {}
        for i in range(30):
            n_i = 100_000 + rng.randint(-20_000, 20_000)
            obs_counts = rng.multinomial(n_i, pool_p)
            obs_by_sample[f"sample_{i:03d}"] = {
                tn: obs_counts.copy() for tn in TABLE_NAMES
            }

        # Deviant sample: 70% pool + 30% independent — genuine deviation
        deviant_p = 0.7 * pool_p + 0.3 * rng.dirichlet(
            np.ones(self.NHEX_GOLDEN) * 10.0
        )
        deviant_p /= deviant_p.sum()
        deviant_obs = rng.multinomial(80_000, deviant_p)
        obs_by_sample["sample_deviant"] = {
            tn: deviant_obs.copy() for tn in TABLE_NAMES
        }

        prior, posteriors, diagnostics = build_prior_and_posteriors(
            obs_by_sample, bg, hex_order,
        )

        self.artifact = {
            "hex_order": hex_order,
            "table_names": list(TABLE_NAMES),
            "prior": prior,
            "posteriors": {
                sn: {tn: post.tolist() for tn, post in tables.items()}
                for sn, tables in posteriors.items()
            },
            "background": {tn: arr.tolist() for tn, arr in bg.items()},
            "diagnostics": diagnostics,
        }
        self.obs_by_sample = obs_by_sample

    def test_golden_measurement_chain(self):
        """Pin r_observed, rho_sample, rho_pool_loo, r_disattenuated."""
        df = measure_sample_disattenuation(
            self.artifact,
            self.obs_by_sample["sample_deviant"],
            sample_name="sample_deviant",
            n_reps=50,
            rng_seed=42,
        )
        row = df[df["table"] == "start_fwd"].iloc[0]

        # r_observed: catches log-enrichment or mask bugs
        np.testing.assert_allclose(
            row["r_observed"], 0.8835, rtol=5e-3,
            err_msg="r_observed shifted — check _log_enrichment or _observed_correlation",
        )
        # rho_sample: catches Spearman-Brown step-up removal/double-application
        # on the sample side.  Without step-up this is ~0.889 (shift 0.052).
        np.testing.assert_allclose(
            row["rho_sample"], 0.9414, rtol=5e-3,
            err_msg="rho_sample shifted — check Spearman-Brown step-up in _thinning_reliability",
        )
        # rho_pool_loo: catches step-up on pool side, or broken LOO subtraction
        np.testing.assert_allclose(
            row["rho_pool_loo"], 0.9990, rtol=2e-3,
            err_msg="rho_pool_loo shifted — check pool reliability or LOO subtraction",
        )
        # r_disattenuated: catches wrong denominator formula (e.g. product instead
        # of geometric mean, or applying step-up only to one side).
        np.testing.assert_allclose(
            row["r_disattenuated"], 0.9111, rtol=5e-3,
            err_msg="r_disattenuated shifted — check disattenuation formula",
        )
