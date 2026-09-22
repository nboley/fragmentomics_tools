"""Numerical test suite for ``background_model_core.py``.

Run from the repo root (the root ``conftest.py`` puts the repo root on
``sys.path``, which makes both ``background_model_core`` and the
``fragmentomics_tools`` package importable)::

    conda run -n biomarker_env python -m pytest tests/test_background_model_core.py

All tests are CPU-only, all randomness is seeded, and model configs are tiny.
Loss-function unit tests run in float64 so comparisons against scipy
references can use tight tolerances.
"""

import numpy as np
import pytest
import torch
from scipy import stats
from scipy.sparse import coo_matrix, csr_matrix
from scipy.special import gammaln, softmax

import background_model_core as bmc

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

L_OUT = 512
DISPERSION_WINDOW = 64
TINY = dict(
    n_kernels=16,
    kernel_size=8,  # even kernel is legal: block dilation rates (2, 4) are even
    num_residual_layers=2,
    dropout=0.1,
    dispersion_window_size=DISPERSION_WINDOW,
)
N_TRACKS = len(bmc.DEFAULT_OUTPUT_TRACKS)


def make_model(loss):
    torch.manual_seed(0)
    return bmc.BackgroundModel(loss=loss, **TINY)


def random_one_hot(rng, batch, length):
    idx = rng.integers(0, 4, size=(batch, length))
    x = np.zeros((batch, 4, length), dtype=np.float32)
    for b in range(batch):
        x[b, idx[b], np.arange(length)] = 1.0
    return torch.from_numpy(x)


def t64(arr):
    """np -> float64 torch tensor."""
    return torch.as_tensor(np.asarray(arr), dtype=torch.float64)


def multinomial_log_constant(x):
    """log(N! / prod x_i!) — the combinatorial term the losses drop."""
    x = np.asarray(x, dtype=float)
    return gammaln(x.sum() + 1.0) - gammaln(x + 1.0).sum()


# ---------------------------------------------------------------------------
# 1. geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("loss", bmc.LOSSES)
def test_geometry_forward_matches_calc_input_region_size(loss):
    rng = np.random.default_rng(0)
    model = make_model(loss)
    model.eval()
    l_in = model.calc_input_region_size(L_OUT)
    x = random_one_hot(rng, 1, l_in)
    with torch.no_grad():
        shape_logits, dispersion = model(x)
    assert shape_logits.shape == (1, N_TRACKS, L_OUT)
    if loss == "multinomial":
        assert dispersion is None
    else:
        assert dispersion.shape == (1, N_TRACKS, L_OUT)


# ---------------------------------------------------------------------------
# 2. multinomial loss vs scipy
# ---------------------------------------------------------------------------


class TestMultinomialLoss:
    def _loss(self, logits, x, mask=None):
        return bmc.MaskedMultinomialNLLLoss()(logits, x, mask)

    def test_matches_scipy_with_constant_added(self):
        rng = np.random.default_rng(1)
        L = 48
        logits = rng.normal(size=L)
        p = softmax(logits)
        x = rng.multinomial(300, p).astype(float)
        n = x.sum()

        loss = self._loss(t64(logits)[None, None], t64(x)[None, None]).item()
        # implemented loss == -(logpmf - combinatorial constant) / N
        full_ll = -loss * n + multinomial_log_constant(x)
        ref = stats.multinomial.logpmf(x, int(n), p)
        assert np.isclose(full_ll, ref, rtol=0, atol=1e-8)

    def test_logit_differences_match_scipy(self):
        rng = np.random.default_rng(2)
        L = 32
        logits_a = rng.normal(size=L)
        logits_b = rng.normal(size=L)
        x = rng.multinomial(150, softmax(logits_a)).astype(float)
        n = x.sum()

        loss_a = self._loss(t64(logits_a)[None, None], t64(x)[None, None]).item()
        loss_b = self._loss(t64(logits_b)[None, None], t64(x)[None, None]).item()
        ref_a = stats.multinomial.logpmf(x, int(n), softmax(logits_a))
        ref_b = stats.multinomial.logpmf(x, int(n), softmax(logits_b))
        # constants cancel in the difference
        assert np.isclose((loss_a - loss_b) * n, -(ref_a - ref_b), atol=1e-8)

    def test_masked_equals_subsetted(self):
        rng = np.random.default_rng(3)
        B, C, L = 1, 2, 40
        logits = rng.normal(size=(B, C, L))
        mask = rng.random(L) > 0.25
        x = np.zeros((B, C, L))
        for c in range(C):
            x[0, c, mask] = rng.multinomial(200, softmax(logits[0, c, mask]))

        masked = self._loss(t64(logits), t64(x), torch.as_tensor(mask)[None, :]).item()
        subset = self._loss(t64(logits[..., mask]), t64(x[..., mask])).item()
        assert np.isclose(masked, subset, atol=1e-10)

    def test_mean_over_batch_and_channels(self):
        rng = np.random.default_rng(4)
        B, C, L = 2, 3, 24
        logits = rng.normal(size=(B, C, L))
        x = rng.integers(0, 8, size=(B, C, L)).astype(float)
        combined = self._loss(t64(logits), t64(x)).item()
        singles = [
            self._loss(t64(logits[b, c])[None, None], t64(x[b, c])[None, None]).item()
            for b in range(B)
            for c in range(C)
        ]
        assert np.isclose(combined, np.mean(singles), atol=1e-10)


# ---------------------------------------------------------------------------
# 3. dirichlet-multinomial loss vs scipy
# ---------------------------------------------------------------------------


class TestDirichletMultinomialLoss:
    def _loss(self, logits, log_gamma, x, mask=None):
        return bmc.MaskedDirichletMultinomialNLLLoss()(logits, log_gamma, x, mask)

    @pytest.mark.parametrize("gamma,seed,sparsity", [(5.0, 10, 0.0), (50.0, 11, 0.6), (500.0, 12, 0.8)])
    def test_matches_scipy_dirichlet_multinomial(self, gamma, seed, sparsity):
        rng = np.random.default_rng(seed)
        L = 40
        logits = rng.normal(size=L)
        p = softmax(logits)
        x = rng.multinomial(250, p).astype(float)
        # force many zeros for the sparse cases
        x[rng.random(L) < sparsity] = 0.0
        n = x.sum()
        assert n > 0

        loss = self._loss(
            t64(logits)[None, None], t64([[np.log(gamma)]]), t64(x)[None, None]
        ).item()
        # implemented loss drops the x-only combinatorial constant and
        # normalizes by N: loss == -(logpmf - const)/N
        full_ll = -loss * n + multinomial_log_constant(x)
        ref = stats.dirichlet_multinomial.logpmf(x.astype(int), gamma * p, int(n))
        assert np.isclose(full_ll, ref, rtol=0, atol=1e-7)

    def test_zero_count_positions_contribute_zero(self):
        rng = np.random.default_rng(13)
        L, gamma = 30, 40.0
        logits = rng.normal(size=L)
        p = softmax(logits)
        x = rng.multinomial(60, p).astype(float)
        x[::2] = 0.0  # force zeros
        n = x.sum()

        loss = self._loss(
            t64(logits)[None, None], t64([[np.log(gamma)]]), t64(x)[None, None]
        ).item()
        alpha = gamma * p
        nz = x > 0
        ll_nonzero_only = (
            gammaln(gamma)
            - gammaln(n + gamma)
            + (gammaln(x[nz] + alpha[nz]) - gammaln(alpha[nz])).sum()
        )
        assert np.isclose(-loss * n, ll_nonzero_only, atol=1e-8)

    def test_masked_equals_subsetted(self):
        rng = np.random.default_rng(14)
        B, C, L = 1, 2, 36
        logits = rng.normal(size=(B, C, L))
        log_gamma = np.log([[30.0, 300.0]])
        mask = rng.random(L) > 0.3
        x = np.zeros((B, C, L))
        for c in range(C):
            x[0, c, mask] = rng.multinomial(120, softmax(logits[0, c, mask]))

        masked = self._loss(
            t64(logits), t64(log_gamma), t64(x), torch.as_tensor(mask)[None, :]
        ).item()
        subset = self._loss(
            t64(logits[..., mask]), t64(log_gamma), t64(x[..., mask])
        ).item()
        assert np.isclose(masked, subset, atol=1e-10)

    def test_mean_over_batch_and_channels(self):
        rng = np.random.default_rng(15)
        B, C, L = 2, 3, 20
        logits = rng.normal(size=(B, C, L))
        log_gamma = rng.normal(size=(B, C)) + 3.0
        x = rng.integers(0, 6, size=(B, C, L)).astype(float)
        combined = self._loss(t64(logits), t64(log_gamma), t64(x)).item()
        singles = [
            self._loss(
                t64(logits[b, c])[None, None],
                t64([[log_gamma[b, c]]]),
                t64(x[b, c])[None, None],
            ).item()
            for b in range(B)
            for c in range(C)
        ]
        assert np.isclose(combined, np.mean(singles), atol=1e-10)


# ---------------------------------------------------------------------------
# 4. nb-offset loss vs scipy
# ---------------------------------------------------------------------------


class TestNegativeBinomialOffsetLoss:
    def _loss(self, logits, log_r, x, mask=None, max_dispersion_ratio=None):
        return bmc.MaskedNegativeBinomialOffsetNLLLoss(
            max_dispersion_ratio=max_dispersion_ratio,
        )(logits, log_r, x, mask)

    @staticmethod
    def _reference(logits, r_bp, x):
        """-sum_i nbinom.logpmf(x_i; n=r_i, p=r_i/(r_i+mu_i)) / N."""
        p = softmax(logits)
        n = x.sum()
        mu = n * p
        ll = stats.nbinom.logpmf(x.astype(int), r_bp, r_bp / (r_bp + mu))
        return -ll.sum() / n

    def test_matches_scipy_single_window(self):
        rng = np.random.default_rng(20)
        L = 32
        logits = rng.normal(size=L)
        x = rng.multinomial(400, softmax(logits)).astype(float)
        r = 25.0
        loss = self._loss(
            t64(logits)[None, None], t64([[[np.log(r)]]]), t64(x)[None, None]
        ).item()
        ref = self._reference(logits, np.full(L, r), x)
        assert np.isclose(loss, ref, rtol=0, atol=1e-8)

    def test_per_window_dispersion_broadcast(self):
        rng = np.random.default_rng(21)
        L, W = 32, 4  # 8 bp per dispersion window
        logits = rng.normal(size=L)
        x = rng.multinomial(500, softmax(logits)).astype(float)
        r_windows = np.array([5.0, 50.0, 500.0, 5000.0])
        loss = self._loss(
            t64(logits)[None, None],
            t64(np.log(r_windows))[None, None],
            t64(x)[None, None],
        ).item()
        r_bp = np.repeat(r_windows, L // W)
        ref = self._reference(logits, r_bp, x)
        assert np.isclose(loss, ref, rtol=0, atol=1e-8)
        # sanity: a wrong (reversed) window order must NOT match
        wrong = self._reference(logits, np.repeat(r_windows[::-1], L // W), x)
        assert not np.isclose(loss, wrong, atol=1e-3)

    def test_masked_equals_subsetted(self):
        # use W == L (per-position dispersion) so the subsetted call stays aligned
        rng = np.random.default_rng(22)
        B, C, L = 1, 2, 30
        logits = rng.normal(size=(B, C, L))
        log_r = rng.normal(size=(B, C, L)) + 3.0
        mask = rng.random(L) > 0.3
        x = np.zeros((B, C, L))
        for c in range(C):
            x[0, c, mask] = rng.multinomial(200, softmax(logits[0, c, mask]))

        masked = self._loss(
            t64(logits), t64(log_r), t64(x), torch.as_tensor(mask)[None, :]
        ).item()
        subset = self._loss(
            t64(logits[..., mask]), t64(log_r[..., mask]), t64(x[..., mask])
        ).item()
        assert np.isclose(masked, subset, atol=1e-10)

    def test_dispersion_clamp_raises_loss(self):
        """With a very low r (high overdispersion), clamping should raise the
        loss toward the unclamped-with-higher-r value because the clamp
        forces r up to the floor."""
        rng = np.random.default_rng(25)
        L = 32
        logits = rng.normal(size=L)
        x = rng.multinomial(400, softmax(logits)).astype(float)
        very_low_r = 0.5  # way below multinomial-equivalent
        log_r = np.log(very_low_r)

        unclamped = self._loss(
            t64(logits)[None, None], t64([[[log_r]]]), t64(x)[None, None],
            max_dispersion_ratio=None,
        ).item()
        clamped = self._loss(
            t64(logits)[None, None], t64([[[log_r]]]), t64(x)[None, None],
            max_dispersion_ratio=2.0,
        ).item()
        # The clamp pushes r up, changing the loss value
        assert clamped != unclamped, "clamp should be active at r=0.5"

    def test_dispersion_clamp_inactive_at_high_r(self):
        """At high r (near-multinomial), the clamp should not engage."""
        rng = np.random.default_rng(26)
        L = 32
        logits = rng.normal(size=L)
        x = rng.multinomial(400, softmax(logits)).astype(float)
        high_r = 10000.0
        log_r = np.log(high_r)

        unclamped = self._loss(
            t64(logits)[None, None], t64([[[log_r]]]), t64(x)[None, None],
            max_dispersion_ratio=None,
        ).item()
        clamped = self._loss(
            t64(logits)[None, None], t64([[[log_r]]]), t64(x)[None, None],
            max_dispersion_ratio=2.0,
        ).item()
        assert np.isclose(unclamped, clamped, atol=1e-8), (
            f"clamp should be inactive at r={high_r}: {unclamped} vs {clamped}"
        )

    def test_dispersion_clamp_gradient_flows(self):
        """Gradients should be finite with clamping active."""
        rng = np.random.default_rng(27)
        L = 32
        logits = torch.tensor(rng.normal(size=(1, 1, L)), dtype=torch.float64,
                              requires_grad=True)
        log_r = torch.tensor([[[np.log(0.5)]]], dtype=torch.float64,
                             requires_grad=True)
        x = torch.tensor(
            rng.multinomial(400, softmax(rng.normal(size=L)))[None, None].astype(float),
            dtype=torch.float64,
        )
        loss = bmc.MaskedNegativeBinomialOffsetNLLLoss(
            max_dispersion_ratio=2.0,
        )(logits, log_r, x)
        loss.backward()
        assert torch.isfinite(logits.grad).all()
        assert torch.isfinite(log_r.grad).all()


# ---------------------------------------------------------------------------
# 5. gradient finiteness through the full model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("loss", bmc.LOSSES)
def test_gradients_finite_with_mask(loss):
    rng = np.random.default_rng(30)
    torch.manual_seed(30)
    model = make_model(loss)
    l_in = model.calc_input_region_size(L_OUT)
    B = 2

    x = random_one_hot(rng, B, l_in)
    mask = np.ones((B, L_OUT), dtype=bool)
    mask[:, 128:192] = False  # fully-masked stretch spanning a dispersion window
    mask &= rng.random((B, L_OUT)) > 0.1  # ~10% additional random masking
    y = rng.integers(0, 6, size=(B, N_TRACKS, L_OUT)).astype(np.float32)
    y *= mask[:, None, :]

    batch = (x, torch.as_tensor(y), torch.as_tensor(mask))
    loss_val = model.training_step(batch, 0)
    assert torch.isfinite(loss_val)
    loss_val.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"


# ---------------------------------------------------------------------------
# 6. reverse-complement permutation
# ---------------------------------------------------------------------------


class TestReverseComplementPermutation:
    def test_involution(self):
        perm = bmc.reverse_complement_track_permutation(bmc.DEFAULT_OUTPUT_TRACKS)
        perm = np.asarray(perm)
        assert sorted(perm) == list(range(N_TRACKS))
        assert np.array_equal(perm[perm], np.arange(N_TRACKS))

    def test_documented_partners(self):
        tracks = bmc.DEFAULT_OUTPUT_TRACKS
        perm = bmc.reverse_complement_track_permutation(tracks)

        def partner_of(name):
            return tracks[perm[tracks.index(name)]]

        assert (
            partner_of("strand_+__fl_40_65__coverage_first")
            == "strand_-__fl_40_65__coverage_last"
        )
        assert (
            partner_of("strand_-__fl_120_175__coverage_first")
            == "strand_+__fl_120_175__coverage_last"
        )
        # midpoint maps to itself modulo strand swap; fl band unchanged
        assert (
            partner_of("strand_+__fl_120_175__coverage_midpoint")
            == "strand_-__fl_120_175__coverage_midpoint"
        )

    def test_asymmetric_track_list_raises(self):
        with pytest.raises(ValueError, match="reverse-complement partner"):
            bmc.reverse_complement_track_permutation(bmc.DEFAULT_OUTPUT_TRACKS[:-1])


# ---------------------------------------------------------------------------
# 7. jitter
# ---------------------------------------------------------------------------


class TestJitterMatrix:
    def test_center_crop_matches_manual_slice(self):
        rng = np.random.default_rng(40)
        arr = rng.normal(size=(3, 20))
        out = bmc.jitter_matrix(arr, jitter_value=0, output_length=10)
        np.testing.assert_array_equal(out, arr[:, 5:15])

    def test_nonzero_jitter_shifts(self):
        rng = np.random.default_rng(41)
        arr = rng.normal(size=(2, 20))
        np.testing.assert_array_equal(
            bmc.jitter_matrix(arr, jitter_value=3, output_length=10), arr[:, 8:18]
        )
        np.testing.assert_array_equal(
            bmc.jitter_matrix(arr, jitter_value=-3, output_length=10), arr[:, 2:12]
        )

    def test_dense_torch_sparse_agree(self):
        rng = np.random.default_rng(42)
        dense = rng.integers(0, 4, size=(3, 21)).astype(np.float64)
        dense[dense < 2] = 0.0  # make it actually sparse
        out_np = bmc.jitter_matrix(dense, jitter_value=2, output_length=11)
        out_torch = bmc.jitter_matrix(
            torch.as_tensor(dense), jitter_value=2, output_length=11
        )
        out_coo = bmc.jitter_matrix(
            coo_matrix(dense), jitter_value=2, output_length=11
        )
        np.testing.assert_array_equal(out_np, out_torch.numpy())
        np.testing.assert_array_equal(out_np, out_coo.toarray())
        assert out_coo.shape == (3, 11)

    def test_too_narrow_raises(self):
        arr = np.zeros((2, 10))
        with pytest.raises(ValueError, match="not wide enough"):
            bmc.jitter_matrix(arr, jitter_value=3, output_length=8)

    def test_non_integer_jitter_raises(self):
        with pytest.raises(ValueError, match="whole number"):
            bmc.jitter_matrix(np.zeros((2, 10)), jitter_value=1.5, output_length=4)

    def test_unsupported_type_raises(self):
        import types

        fake = types.SimpleNamespace(shape=(2, 10))  # has .shape, unsupported type
        with pytest.raises(TypeError, match="invalid"):
            bmc.jitter_matrix(fake, jitter_value=0, output_length=4)


# ---------------------------------------------------------------------------
# 8. masked_mean_pool
# ---------------------------------------------------------------------------


class TestMaskedMeanPool:
    def test_hand_computed_example(self):
        x = t64([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]])
        mask = torch.tensor([[[True, True, False, True, False, False, False, False]]])
        out = bmc.masked_mean_pool(x, mask, out_size=2)
        assert out.shape == (1, 1, 2)
        # window 0: mean of valid {1, 2, 4} = 7/3; window 1: fully masked -> 0, not nan
        np.testing.assert_allclose(out[0, 0].numpy(), [7.0 / 3.0, 0.0])
        assert torch.isfinite(out).all()

    def test_no_mask_is_plain_mean(self):
        rng = np.random.default_rng(50)
        x = t64(rng.normal(size=(2, 3, 12)))
        out = bmc.masked_mean_pool(x, None, out_size=4)
        np.testing.assert_allclose(
            out.numpy(), x.numpy().reshape(2, 3, 4, 3).mean(axis=-1)
        )


# ---------------------------------------------------------------------------
# 9. calibration end-to-end
# ---------------------------------------------------------------------------


class TestCalibration:
    L = 512
    W = 64
    N = 5000
    GAMMA = 500.0
    TILES = 200

    def _fixed_p(self, rng):
        return softmax(rng.normal(scale=0.3, size=self.L))

    def test_beta_binomial_pvalues_uniform_under_dm_null(self):
        rng = np.random.default_rng(60)
        p = self._fixed_p(rng)
        pvals = []
        for _ in range(self.TILES):
            p_tilde = rng.dirichlet(self.GAMMA * p)
            x = rng.multinomial(self.N, p_tilde)
            pvals.append(
                bmc.beta_binomial_window_pvalues(x, p, self.GAMMA, self.W)
            )
        pv = np.concatenate(pvals)
        assert pv.shape == (self.TILES * self.L // self.W,)
        ks = stats.kstest(pv, "uniform")
        assert ks.pvalue > 0.01, f"KS rejects uniformity: {ks}"
        assert 0.45 < pv.mean() < 0.55

    def test_beta_binomial_pvalues_power_on_perturbed_window(self):
        rng = np.random.default_rng(61)
        p = self._fixed_p(rng)
        p_alt = p.copy()
        p_alt[: self.W] *= 3.0  # window 0 mass up 3x
        p_alt /= p_alt.sum()
        pv0 = []
        for _ in range(self.TILES):
            p_tilde = rng.dirichlet(self.GAMMA * p_alt)
            x = rng.multinomial(self.N, p_tilde)
            pv = bmc.beta_binomial_window_pvalues(x, p, self.GAMMA, self.W)
            pv0.append(pv[0])
        pv0 = np.asarray(pv0)
        assert np.median(pv0) < 1e-4
        assert (pv0 < 0.05).mean() > 0.95

    def test_nb_pvalues_uniform_under_nb_null(self):
        rng = np.random.default_rng(62)
        p = self._fixed_p(rng)
        n_windows = self.L // self.W
        r_windows = rng.uniform(15.0, 40.0, size=n_windows)
        r_bp = np.repeat(r_windows, self.W)
        mu = self.N * p
        pvals = []
        for _ in range(self.TILES):
            x = rng.negative_binomial(r_bp, r_bp / (r_bp + mu)).astype(float)
            pvals.append(bmc.nb_window_pvalues(x, p, r_windows, self.W))
        pv = np.concatenate(pvals)
        ks = stats.kstest(pv, "uniform")
        assert ks.pvalue > 0.01, f"KS rejects uniformity: {ks}"
        assert 0.45 < pv.mean() < 0.55


# ---------------------------------------------------------------------------
# 10. predict_profile
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("loss", bmc.LOSSES)
def test_predict_profile(loss):
    rng = np.random.default_rng(70)
    model = make_model(loss)
    l_in = model.calc_input_region_size(L_OUT)
    seq = random_one_hot(rng, 1, l_in)[0].numpy()

    out = model.predict_profile(seq)
    assert out["probs"].shape == (N_TRACKS, L_OUT)
    np.testing.assert_allclose(out["probs"].sum(axis=-1), 1.0, atol=1e-4)

    mask = np.ones(L_OUT, dtype=bool)
    mask[100:180] = False
    out_m = model.predict_profile(seq, mask=mask)
    assert out_m["probs"].shape == (N_TRACKS, L_OUT)
    assert (out_m["probs"][:, ~mask] == 0).all()
    np.testing.assert_allclose(out_m["probs"][:, mask].sum(axis=-1), 1.0, atol=1e-4)

    if loss == "multinomial":
        assert out["log_dispersion"] is None
        assert out_m["log_dispersion"] is None
    elif loss == "dirichlet_multinomial":
        assert out["log_dispersion"].shape == (N_TRACKS,)
        assert out_m["log_dispersion"].shape == (N_TRACKS,)
    else:  # nb_offset
        assert out["log_dispersion"].shape == (N_TRACKS, L_OUT // DISPERSION_WINDOW)
        assert out_m["log_dispersion"].shape == (N_TRACKS, L_OUT // DISPERSION_WINDOW)
    for o in (out, out_m):
        if o["log_dispersion"] is not None:
            assert np.isfinite(o["log_dispersion"]).all()


# ---------------------------------------------------------------------------
# SpatialDropout semantics (docstring: drop whole channels)
# ---------------------------------------------------------------------------


def test_spatial_dropout_drops_whole_channels():
    torch.manual_seed(80)
    p = 0.5
    sd = bmc.SpatialDropout(p)
    sd.train()
    x = torch.ones(4, 16, 64)
    out = sd(x)
    # every (batch, channel) row is either entirely zero or entirely 1/(1-p)
    row_zero = (out == 0).all(dim=-1)
    row_scaled = torch.isclose(out, torch.full_like(out, 1.0 / (1.0 - p))).all(dim=-1)
    assert (row_zero | row_scaled).all(), "dropout must act on whole channels"
    assert row_zero.any() and row_scaled.any()
    sd.eval()
    assert torch.equal(sd(x), x)
