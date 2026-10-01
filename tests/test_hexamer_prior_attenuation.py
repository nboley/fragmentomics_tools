"""Test that the Dirichlet-multinomial shrinkage actually attenuates.

These are structural tests on the shrinkage machinery — they use synthetic
data and do not require the 92-sample parquets.  The attenuation properties
they assert are:

1. **No zeros**: every posterior weight is strictly positive, even when the
   raw per-sample count is zero for some hexamers.
2. **Shrinkage toward prior**: every per-sample posterior is closer to the
   pooled prior than the raw per-sample estimate is (in L2 distance).
3. **Monotone attenuation**: samples with fewer observations shrink MORE
   toward the prior than samples with more observations.
4. **Limit behaviour**: as N -> inf, the posterior converges to the raw
   estimate; as N -> 0, it converges to the prior.
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_hexamer_prior import (
    build_prior_and_posteriors,
    estimate_dirichlet_alpha0,
)
from scripts.count_cut_site_hexamers import TABLE_NAMES

NHEX = 4096


def _make_synthetic_data(
    n_samples: int = 20,
    n_hexamers: int = NHEX,
    total_per_sample: int = 5000,
    rng_seed: int = 42,
):
    """Generate synthetic observed and background counts.

    The 'true' hexamer bias is drawn from a Dirichlet, then each sample's
    observed counts are drawn from a multinomial.  Background is uniform.
    """
    rng = np.random.RandomState(rng_seed)

    # True bias: draw from a Dirichlet with moderate concentration
    true_bias = rng.dirichlet(np.ones(n_hexamers) * 5.0)

    # Background: uniform (each hexamer equally represented in the region set)
    bg_flat = np.ones(n_hexamers, dtype=np.int64) * 1000

    obs_by_sample = {}
    for i in range(n_samples):
        # Draw observed counts from multinomial with the true bias
        n_i = total_per_sample + rng.randint(-1000, 1000)
        obs_counts = rng.multinomial(n_i, true_bias)
        obs_by_sample[f"sample_{i:03d}"] = {
            tn: obs_counts.copy() for tn in TABLE_NAMES
        }

    bg = {tn: bg_flat.copy() for tn in TABLE_NAMES}
    hex_order = [f"HEX{i:04d}" for i in range(n_hexamers)]

    return obs_by_sample, bg, hex_order, true_bias


def _make_sparse_sample(n_hexamers: int = NHEX, n_obs: int = 100, rng_seed: int = 99):
    """A single sample with many zero-count hexamers."""
    rng = np.random.RandomState(rng_seed)
    obs = np.zeros(n_hexamers, dtype=np.int64)
    # Put all counts into a small subset of hexamers
    chosen = rng.choice(n_hexamers, size=min(n_obs, n_hexamers // 10), replace=False)
    counts = rng.multinomial(n_obs, np.ones(len(chosen)) / len(chosen))
    obs[chosen] = counts
    return obs


class TestAttenuationProperties:
    """Structural tests on the Dirichlet-multinomial shrinkage."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.obs_by_sample, self.bg, self.hex_order, self.true_bias = (
            _make_synthetic_data()
        )
        self.prior, self.posteriors, self.diagnostics = build_prior_and_posteriors(
            self.obs_by_sample, self.bg, self.hex_order,
        )

    def test_no_zeros_in_posterior(self):
        """Every posterior weight must be strictly positive."""
        for sn, tables in self.posteriors.items():
            for tn, weights in tables.items():
                assert np.all(weights > 0), (
                    f"{sn}/{tn}: posterior has {(weights == 0).sum()} zeros"
                )

    def test_shrinkage_toward_prior(self):
        """Every sample's posterior is closer to the prior than its raw estimate."""
        for tn in TABLE_NAMES:
            pooled_weight = np.array(self.prior[tn]["pooled_weight"])
            bg_arr = self.bg[tn].astype(np.float64)
            bg_prop = bg_arr / bg_arr.sum()

            for sn in self.obs_by_sample:
                obs_i = self.obs_by_sample[sn][tn].astype(np.float64)
                n_i = obs_i.sum()
                if n_i == 0:
                    continue

                # Raw weight
                raw_p = obs_i / n_i
                raw_weight = np.where(bg_prop > 0, raw_p / bg_prop, 0.0)
                if raw_weight.sum() > 0:
                    raw_weight /= raw_weight.sum()

                post_weight = self.posteriors[sn][tn]

                dist_raw = np.sqrt(((raw_weight - pooled_weight) ** 2).sum())
                dist_post = np.sqrt(((post_weight - pooled_weight) ** 2).sum())

                assert dist_post < dist_raw, (
                    f"{sn}/{tn}: posterior (dist={dist_post:.6f}) is NOT closer "
                    f"to prior than raw (dist={dist_raw:.6f})"
                )

    def test_monotone_attenuation(self):
        """Samples with fewer observations shrink more toward the prior."""
        # Create two samples: one with 500 obs, one with 50000 obs
        rng = np.random.RandomState(123)
        true_p = rng.dirichlet(np.ones(NHEX) * 5.0)

        bg_flat = np.ones(NHEX, dtype=np.int64) * 1000
        bg = {tn: bg_flat.copy() for tn in TABLE_NAMES}
        hex_order = [f"HEX{i:04d}" for i in range(NHEX)]

        obs_by_sample = {}
        obs_low = rng.multinomial(500, true_p)
        obs_high = rng.multinomial(50000, true_p)
        for tn in TABLE_NAMES:
            obs_by_sample.setdefault("low_count", {})[tn] = obs_low.copy()
            obs_by_sample.setdefault("high_count", {})[tn] = obs_high.copy()

        _, posteriors, _ = build_prior_and_posteriors(obs_by_sample, bg, hex_order)

        for tn in TABLE_NAMES:
            bg_arr = bg[tn].astype(np.float64)
            bg_prop = bg_arr / bg_arr.sum()

            # Compute raw weights
            raw_low = obs_low.astype(np.float64) / obs_low.sum()
            raw_low_w = np.where(bg_prop > 0, raw_low / bg_prop, 0.0)
            raw_low_w /= raw_low_w.sum()

            raw_high = obs_high.astype(np.float64) / obs_high.sum()
            raw_high_w = np.where(bg_prop > 0, raw_high / bg_prop, 0.0)
            raw_high_w /= raw_high_w.sum()

            post_low = posteriors["low_count"][tn]
            post_high = posteriors["high_count"][tn]

            # Pooled weight (midpoint between the two)
            pooled_obs = (obs_low + obs_high).astype(np.float64)
            pooled_p = pooled_obs / pooled_obs.sum()
            pooled_w = np.where(bg_prop > 0, pooled_p / bg_prop, 0.0)
            pooled_w /= pooled_w.sum()

            # Shrinkage ratio: 1 - dist_post/dist_raw
            dist_raw_low = np.sqrt(((raw_low_w - pooled_w) ** 2).sum())
            dist_post_low = np.sqrt(((post_low - pooled_w) ** 2).sum())
            shrink_low = 1.0 - dist_post_low / dist_raw_low

            dist_raw_high = np.sqrt(((raw_high_w - pooled_w) ** 2).sum())
            dist_post_high = np.sqrt(((post_high - pooled_w) ** 2).sum())
            shrink_high = 1.0 - dist_post_high / dist_raw_high

            assert shrink_low > shrink_high, (
                f"{tn}: low-count sample should shrink MORE (ratio={shrink_low:.4f}) "
                f"than high-count (ratio={shrink_high:.4f})"
            )

    def test_sparse_sample_no_zeros(self):
        """A sample with many zero-count hexamers still has all-positive posteriors."""
        sparse_obs = _make_sparse_sample()
        n_zeros = (sparse_obs == 0).sum()
        assert n_zeros > NHEX * 0.9, f"Test setup: expected >90% zeros, got {n_zeros}"

        # Add sparse sample to the existing data
        obs = dict(self.obs_by_sample)
        obs["sparse"] = {tn: sparse_obs.copy() for tn in TABLE_NAMES}

        _, posteriors, _ = build_prior_and_posteriors(obs, self.bg, self.hex_order)

        for tn in TABLE_NAMES:
            post = posteriors["sparse"][tn]
            assert np.all(post > 0), (
                f"sparse/{tn}: {(post == 0).sum()} zero-weight hexamers in posterior"
            )

    def test_limit_large_n(self):
        """With very large N, posterior ≈ raw estimate (weak shrinkage).

        Needs multiple samples for alpha_0 estimation (var with ddof=1).
        We use several moderate samples plus one huge one and check the
        huge sample's posterior is nearly identical to its raw estimate.
        """
        rng = np.random.RandomState(456)
        true_p = rng.dirichlet(np.ones(NHEX) * 5.0)

        bg_flat = np.ones(NHEX, dtype=np.int64) * 1000
        bg = {tn: bg_flat.copy() for tn in TABLE_NAMES}
        hex_order = [f"HEX{i:04d}" for i in range(NHEX)]

        obs_huge = rng.multinomial(10_000_000, true_p)
        obs_by_sample = {"huge": {tn: obs_huge.copy() for tn in TABLE_NAMES}}
        # Add moderate-sized samples so alpha_0 estimation works
        for i in range(5):
            obs_mod = rng.multinomial(5000, true_p)
            obs_by_sample[f"mod_{i}"] = {tn: obs_mod.copy() for tn in TABLE_NAMES}

        _, posteriors, _ = build_prior_and_posteriors(obs_by_sample, bg, hex_order)

        for tn in TABLE_NAMES:
            bg_arr = bg[tn].astype(np.float64)
            bg_prop = bg_arr / bg_arr.sum()

            raw_p = obs_huge.astype(np.float64) / obs_huge.sum()
            raw_w = np.where(bg_prop > 0, raw_p / bg_prop, 0.0)
            raw_w /= raw_w.sum()

            post = posteriors["huge"][tn]

            # Should be very close
            max_diff = np.abs(post - raw_w).max()
            assert max_diff < 1e-4, (
                f"{tn}: large-N posterior differs from raw by {max_diff:.6f}"
            )


class TestAlpha0Estimation:
    """Tests for the Dirichlet concentration parameter estimator."""

    def test_alpha0_positive(self):
        obs_by_sample, _, _, _ = _make_synthetic_data()
        for tn in TABLE_NAMES:
            a0 = estimate_dirichlet_alpha0(obs_by_sample, tn)
            assert a0 > 0, f"{tn}: alpha_0 must be positive, got {a0}"

    def test_alpha0_increases_with_less_variance(self):
        """Less biological variance → larger alpha_0 (more concentrated prior)."""
        rng = np.random.RandomState(789)
        true_p = rng.dirichlet(np.ones(NHEX) * 5.0)

        # High variance: each sample has a DIFFERENT true_p
        obs_highvar = {}
        for i in range(20):
            p_i = rng.dirichlet(np.ones(NHEX) * 2.0)
            obs_highvar[f"s{i}"] = {
                "start_fwd": rng.multinomial(5000, p_i),
            }

        # Low variance: all samples share the same true_p
        obs_lowvar = {}
        for i in range(20):
            obs_lowvar[f"s{i}"] = {
                "start_fwd": rng.multinomial(5000, true_p),
            }

        a0_high = estimate_dirichlet_alpha0(obs_highvar, "start_fwd")
        a0_low = estimate_dirichlet_alpha0(obs_lowvar, "start_fwd")

        assert a0_low > a0_high, (
            f"Low-variance alpha_0 ({a0_low:.1f}) should exceed "
            f"high-variance alpha_0 ({a0_high:.1f})"
        )
