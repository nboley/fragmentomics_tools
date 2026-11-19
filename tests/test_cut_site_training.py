"""Phase 3 tests: end-to-end training integration.

Exercises: synthetic store → dataset → collate → model forward → loss →
backward → optimizer step.  Also tests the multinomial NLL, annealed NB
loss, and which input bases one (start, length, strand) logit reads.
"""

import numpy as np
import pytest
import torch

from background_model.cut_site_model.model import (
    CutSiteFragmentModel,
    DEFAULT_L_MIN,
    DEFAULT_L_MAX,
    WINDOW_HALF,
    _compute_geometry,
)

L_MIN_INBAND = DEFAULT_L_MIN
L_MAX_INBAND = DEFAULT_L_MAX
N_L, _, _, _ = _compute_geometry(L_MIN_INBAND, L_MAX_INBAND)
from background_model.cut_site_model.dataset import (
    CutSiteFragmentDataset,
    cut_site_collate_fn,
)
from tests.test_cut_site_store_dataset import _create_synthetic_store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_training_setup(tmp_path):
    """Create a small model + dataset for training tests."""
    store_path, cfg = _create_synthetic_store(
        tmp_path, tile_size=256, jitter=32, n_tiles=4,
        n_samples=1, n_frags_per_tile=100, seed=42,
    )
    model = CutSiteFragmentModel(
        tile_size=cfg.tile_size, N1=8, K=8, N2=4, H=4, L_chunk=32,
        dispersion_window_size=cfg.tile_size,  # single window for small tile
    )
    ds = CutSiteFragmentDataset(
        store_path=store_path,
        model_input_size=model.model_input_size,
        model_L_min=cfg.L_min, model_L_max=cfg.L_max,
        split="train",
        min_N=0,
        train_mode=False,
    )
    return model, ds, cfg


# ---------------------------------------------------------------------------
# Multinomial NLL
# ---------------------------------------------------------------------------


class TestMultinomialNLL:
    def test_uniform_logits_give_log_D(self, small_training_setup):
        """With uniform logits, NLL per fragment should be log(|D|)."""
        model, ds, cfg = small_training_setup
        P = cfg.tile_size

        item = ds[0]
        batch = cut_site_collate_fn([item])

        # Run forward to get logit shape, then override with zeros (uniform)
        with torch.no_grad():
            logits, ep_shape, disp = model(batch["x"])
            # Uniform logits → log_softmax → -log(1/|D|) = log(|D|)
            D_size = P * N_L * 2
            log_D = np.log(D_size)

            flat = torch.zeros_like(logits.reshape(1, -1))
            log_probs = torch.log_softmax(flat, dim=-1).reshape(1, P, N_L, 2)

            # Gather for actual fragments
            frag_m = batch["frag_m"][0]
            frag_L_idx = batch["frag_L_idx"][0]
            frag_s = batch["frag_s"][0]
            frag_N = batch["frag_N"][0].item()

            if frag_N > 0:
                lp = log_probs[0, frag_m, frag_L_idx, frag_s]
                nll = -lp.mean().item()
                assert abs(nll - log_D) < 0.01, (
                    f"Uniform NLL {nll:.4f} != log|D| {log_D:.4f}"
                )

    def test_nll_decreases_with_peaked_logits(self, small_training_setup):
        """Logits peaked at observed fragments should have lower NLL than uniform."""
        model, ds, cfg = small_training_setup
        P = cfg.tile_size

        item = ds[0]
        batch = cut_site_collate_fn([item])

        with torch.no_grad():
            logits, _, _ = model(batch["x"])
            D_size = P * N_L * 2

            # Uniform NLL
            flat_uniform = torch.zeros(1, D_size)
            lp_uniform = torch.log_softmax(flat_uniform, dim=-1).reshape(1, P, N_L, 2)

            # Peaked: add +10 at observed fragment positions
            flat_peaked = torch.zeros(1, D_size)
            peaked_logits = flat_peaked.reshape(1, P, N_L, 2)

            frag_m = batch["frag_m"][0]
            frag_L_idx = batch["frag_L_idx"][0]
            frag_s = batch["frag_s"][0]
            frag_N = batch["frag_N"][0].item()

            if frag_N > 0:
                peaked_logits[0, frag_m, frag_L_idx, frag_s] += 10.0
                lp_peaked = torch.log_softmax(
                    peaked_logits.reshape(1, -1), dim=-1
                ).reshape(1, P, N_L, 2)

                nll_uniform = -lp_uniform[0, frag_m, frag_L_idx, frag_s].mean().item()
                nll_peaked = -lp_peaked[0, frag_m, frag_L_idx, frag_s].mean().item()
                assert nll_peaked < nll_uniform, (
                    f"Peaked NLL {nll_peaked:.4f} >= uniform {nll_uniform:.4f}"
                )


# ---------------------------------------------------------------------------
# Annealed NB loss
# ---------------------------------------------------------------------------


class _EpochModel(CutSiteFragmentModel):
    """Sets ``current_epoch`` directly and records ``self.log`` calls, so
    ``_step`` runs its real annealing schedule without a Trainer."""

    epoch = 0

    @property
    def current_epoch(self):
        return self.epoch

    def log(self, name, value, *args, **kwargs):
        self.logged[name] = value


class TestAnnealedNBLoss:
    """Runs the model's own ``_step`` and reads back the NB weight it logs
    and the loss it returns."""

    def _model_and_batch(self, small_training_setup, nb_w0, nb_T_anneal):
        _, ds, cfg = small_training_setup
        torch.manual_seed(0)
        model = _EpochModel(
            tile_size=cfg.tile_size, N1=8, K=8, N2=4, H=4, L_chunk=32,
            dispersion_window_size=cfg.tile_size,
            nb_w0=nb_w0, nb_T_anneal=nb_T_anneal, mult_weight=0.5)
        model.logged = {}
        return model, cut_site_collate_fn([ds[0], ds[1]])

    def test_weight_schedule_and_loss(self, small_training_setup):
        """w0 = 2, T_anneal = 4: the weight falls linearly to 0 at epoch 4
        and stays 0 after.  The loss is mult_weight * L_mult + w_t * L_NB."""
        model, batch = self._model_and_batch(small_training_setup, 2.0, 4)
        weights = []
        with torch.no_grad():
            for epoch in range(6):
                model.epoch = epoch
                loss = model._step(batch, "train")
                lg = model.logged
                w_t = lg["train_nb_weight"]
                weights.append(w_t)
                # Nonzero, so the loss check means something at w_t == 0 too.
                assert lg["train_nb_loss"] != 0
                torch.testing.assert_close(
                    loss, 0.5 * lg["train_mult_nll"] + w_t * lg["train_nb_loss"])
        assert weights == [2.0, 1.5, 1.0, 0.5, 0.0, 0.0]

    def test_T_anneal_zero_disables_nb(self, small_training_setup):
        model, batch = self._model_and_batch(small_training_setup, 2.0, 0)
        with torch.no_grad():
            model._step(batch, "train")
        assert model.logged["train_nb_weight"] == 0.0


# ---------------------------------------------------------------------------
# End-to-end training step
# ---------------------------------------------------------------------------


class TestEndToEndTraining:
    def test_training_step_runs(self, small_training_setup):
        """A full training step (forward + loss + backward) should run."""
        model, ds, cfg = small_training_setup
        batch = cut_site_collate_fn([ds[0]])

        # Manually run the training step logic
        model.train()
        logits, ep_shape, disp = model(batch["x"])

        # Multinomial NLL
        B, P, n_L, _ = logits.shape
        flat_logits = logits.reshape(B, -1)
        log_probs = torch.log_softmax(flat_logits, dim=-1).reshape(B, P, n_L, 2)

        frag_m = batch["frag_m"][0]
        frag_L_idx = batch["frag_L_idx"][0]
        frag_s = batch["frag_s"][0]
        frag_N = batch["frag_N"][0]

        if frag_N > 0:
            lp = log_probs[0, frag_m, frag_L_idx, frag_s]
            L_mult = (-lp.sum() / frag_N.float())
        else:
            L_mult = torch.tensor(0.0)

        loss = L_mult
        loss.backward()

        # Check gradients exist on trainable params
        grad_params = [
            (n, p) for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert len(grad_params) > 0, "No gradients after training step"

    def test_optimizer_step_reduces_loss(self, small_training_setup):
        """A few optimizer steps should reduce the multinomial NLL."""
        model, ds, cfg = small_training_setup
        batch = cut_site_collate_fn([ds[0], ds[1]])

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2
        )

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            logits, _, _ = model(batch["x"])
            B, P, n_L, _ = logits.shape
            flat = logits.reshape(B, -1)
            lp = torch.log_softmax(flat, dim=-1).reshape(B, P, n_L, 2)

            nll = torch.zeros(B)
            for i in range(B):
                m_i = batch["frag_m"][i]
                l_i = batch["frag_L_idx"][i]
                s_i = batch["frag_s"][i]
                N_i = batch["frag_N"][i]
                if N_i > 0:
                    nll[i] = -lp[i, m_i, l_i, s_i].sum() / N_i.float()
            loss = nll.mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease (not necessarily monotonically, but final < initial)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# Tap geometry, read off the model's own gradients
# ---------------------------------------------------------------------------


class TestLogitInputSupport:
    """Which input bases one logit depends on, measured through the model.

    The tap offsets live inside ``_compute_chunk``; a test that restates
    them as local constants cannot fail.  Instead, back-propagate a single
    logit ``(m, L, s)`` and read its support on the input: the left tap
    reads ``x[W+m : W+m+6]`` (width-6 conv), the interior reads
    ``x[W+m : W+m+L]``, and the right tap reads ``x[W+m+L : W+m+L+6]``, so
    the support is exactly ``[W+m, W+m+L+5]`` with ``W = WINDOW_HALF``.
    The upper end pins the right tap at ``m+L``, i.e. the cut AFTER the
    last base ``m+L-1``; an off-by-one there moves it.  This pins CURRENT
    geometry: the design doc's open issue 1 records that the hex prior keys
    the 3' cut at ``m+L-1``.  If the owner resolves that, change this test.
    Running m = 0 and m = P-1 also shows every tap is inside ``h`` (an
    out-of-range gather raises).

    The lower edge does NOT pin the left tap: the interior also starts at
    ``W+m``, so a shifted left tap hides inside the interior's support.
    ``test_each_tap_bank_on_its_own`` below isolates the banks for that.
    """

    @staticmethod
    def _model(P):
        torch.manual_seed(0)
        model = CutSiteFragmentModel(
            tile_size=P, N1=8, K=8, N2=4, H=4, L_chunk=32).eval()
        # w_out is zero-initialised (f_theta == 0 at init), which zeroes
        # every input gradient.  Any nonzero value exposes the support.
        with torch.no_grad():
            model.w_out.normal_()
        return model

    @staticmethod
    def _support(model, m, L_val, strand):
        x = torch.rand(1, 4, model.model_input_size, requires_grad=True)
        logits, _, _ = model(x)
        logits[0, m, L_val - L_MIN_INBAND, strand].backward()
        return torch.nonzero(x.grad[0].abs().sum(0)).flatten().tolist()

    @pytest.mark.parametrize("tap", ["left", "right"])
    @pytest.mark.parametrize("strand", [0, 1])
    @pytest.mark.parametrize("m", [0, 63])
    @pytest.mark.parametrize("L_val", [L_MIN_INBAND, L_MIN_INBAND + 1, 100, L_MAX_INBAND])
    def test_each_tap_bank_on_its_own(self, L_val, m, strand, tap):
        """Pins the CURRENT tap positions; it does not say where the taps
        should be.  With the interior branch zeroed (``interior_F = 0``) and
        the other tap's banks zeroed (``bank_W``), one logit's input support
        is that tap's width-6 conv window alone:

            left  tap: x[W+m   : W+m+6]
            right tap: x[W+m+L : W+m+L+6]

        A +-1 shift of either tap's gather index moves its window."""
        model = self._model(64)
        other = 1 if tap == "left" else 0
        with torch.no_grad():
            model.interior_F.zero_()
            for s in (0, 1):
                model.bank_W[other * 2 + s].zero_()
        lo = WINDOW_HALF + m + (0 if tap == "left" else L_val)
        assert self._support(model, m, L_val, strand) == list(range(lo, lo + 6))

    @pytest.mark.parametrize("strand", [0, 1])
    @pytest.mark.parametrize("m", [0, 63])
    @pytest.mark.parametrize("L_val", [L_MIN_INBAND, L_MIN_INBAND + 1, 100, L_MAX_INBAND])
    def test_support_is_start_to_right_tap(self, L_val, m, strand):
        support = self._support(self._model(64), m, L_val, strand)
        lo = WINDOW_HALF + m
        assert support == list(range(lo, lo + L_val + 6))
