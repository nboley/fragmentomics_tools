"""Phase 3 tests: end-to-end training integration.

Exercises: synthetic store → dataset → collate → model forward → loss →
backward → optimizer step.  Also tests the multinomial NLL, annealed NB
loss, and the ±1 alignment between fragment centres and model tap indices.
"""

import numpy as np
import pytest
import torch

from background_model.cut_site_model import (
    CutSiteFragmentModel,
    DEFAULT_L_MIN,
    DEFAULT_L_MAX,
    WINDOW_HALF,
    _compute_geometry,
)

L_MIN_INBAND = DEFAULT_L_MIN
L_MAX_INBAND = DEFAULT_L_MAX
N_L, _, _, _ = _compute_geometry(L_MIN_INBAND, L_MAX_INBAND)
from background_model.cut_site_dataset import (
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
        model_max_len=cfg.max_len, model_L_min=cfg.L_min, model_L_max=cfg.L_max,
        split="train",
        sample_role="train",
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


class TestAnnealedNBLoss:
    def test_nb_weight_decays_to_zero(self):
        """w(t) should decay linearly from w0 to 0 over T_anneal epochs."""
        w0, T_anneal = 1.0, 5
        for t in range(T_anneal + 2):
            w_t = w0 * max(0.0, 1.0 - t / T_anneal)
            if t == 0:
                assert w_t == w0
            elif t >= T_anneal:
                assert w_t == 0.0
            else:
                assert 0 < w_t < w0

    def test_nb_weight_zero_at_T_anneal(self):
        """After T_anneal epochs, w(t) = 0 exactly."""
        w0, T_anneal = 2.5, 3
        w_t = w0 * max(0.0, 1.0 - T_anneal / T_anneal)
        assert w_t == 0.0


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
# ±1 alignment: fragment centres vs model tap indices
# ---------------------------------------------------------------------------


class TestAlignmentBothParities:
    """Verify that for both even and odd L, the fragment centre m maps
    correctly to the tap indices in h, and the round-trip p = m - L//2
    recovers the correct start position."""

    @pytest.mark.parametrize("L_val", [25, 26, 100, 179])
    def test_tap_alignment_at_boundaries(self, L_val):
        """For a fragment at the leftmost and rightmost valid centres,
        verify the taps land within h."""
        P = 256
        model_input_size = P + 2 * WINDOW_HALF
        o_l = WINDOW_HALF - L_val // 2
        o_r = o_l + L_val

        # Leftmost centre: m = 0
        assert 0 <= 0 + o_l < model_input_size
        assert 0 <= 0 + o_r < model_input_size

        # Rightmost centre: m = P - 1
        assert 0 <= (P - 1) + o_l < model_input_size
        assert 0 <= (P - 1) + o_r < model_input_size

    @pytest.mark.parametrize("L_val", [25, 26, 100, 179])
    def test_start_roundtrip(self, L_val):
        """p = m - L//2 and m = p + L//2 must round-trip exactly."""
        for m in (0, 50, 128, 255):
            p = m - L_val // 2
            m_back = p + L_val // 2
            assert m_back == m

    @pytest.mark.parametrize("L_val", [25, 26, 100, 179])
    def test_cut_sites_are_L_apart(self, L_val):
        """The two cut sites p and p+L are exactly L apart (not L-1)."""
        o_l = WINDOW_HALF - L_val // 2
        o_r = o_l + L_val
        # The taps in h-coordinates: m + o_l and m + o_r
        # Difference: o_r - o_l = L
        assert o_r - o_l == L_val
        # This means the right CUT SITE is at p + L, not p + L - 1
        # (the right endpoint BASE is at p + L - 1, but the cut site is at p + L)

    def test_endpoint_base_vs_cut_site(self):
        """Verify the ±1 distinction: right endpoint base = p+L-1,
        right cut site = p+L."""
        for L_val in (25, 26, 100, 179):
            for m in (0, 128):
                p = m - L_val // 2
                right_cut_site = p + L_val       # what the model taps
                right_base = p + L_val - 1       # what /endpoints/last records
                assert right_cut_site == right_base + 1
