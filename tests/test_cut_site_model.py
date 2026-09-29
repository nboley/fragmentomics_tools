"""Tests for the cut-site fragment model (Phase 1: architecture).

Covers: geometry, forward pass shapes, gradient flow, parameter counts,
centre-based domain sizing, and the fold-down linearity condition.
"""

import pytest
import torch

from background_model.cut_site_model import (
    CutSiteFragmentModel,
    L_MAX_INBAND,
    L_MIN_INBAND,
    N_L,
    N_LENGTH_GROUPS,
    N_OFFSETS,
    O_L_MIN,
    O_R_MAX,
    WINDOW_HALF,
    length_to_group,
)


# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------


class TestGeometryConstants:
    """Verify the FL_BANDS-derived constants match the design doc."""

    def test_inband_range(self):
        assert L_MIN_INBAND == 25
        assert L_MAX_INBAND == 179
        assert N_L == 155  # 179 - 25 + 1

    def test_offset_range(self):
        assert O_L_MIN == 128 - 179 // 2  # 39
        assert O_R_MAX == O_L_MIN + L_MAX_INBAND  # 218
        assert N_OFFSETS == L_MAX_INBAND  # 179

    def test_n_offsets_is_live_width(self):
        """All N_OFFSETS entries receive gradient (no dead offsets)."""
        # The minimum o_l for any in-band L is O_L_MIN (at L_MAX).
        # The maximum o_r for any in-band L is O_R_MAX (at L_MAX).
        # Live range is [O_L_MIN, O_R_MAX) = L_MAX_INBAND entries.
        assert N_OFFSETS == O_R_MAX - O_L_MIN

    def test_length_groups(self):
        """10 groups, 15 wide, last group is 20 wide ([160,180))."""
        assert N_LENGTH_GROUPS == 10
        # First group
        assert length_to_group(torch.tensor(25)).item() == 0
        assert length_to_group(torch.tensor(39)).item() == 0
        # Second group
        assert length_to_group(torch.tensor(40)).item() == 1
        # Last group (clamped)
        assert length_to_group(torch.tensor(160)).item() == 9
        assert length_to_group(torch.tensor(175)).item() == 9
        assert length_to_group(torch.tensor(179)).item() == 9


# ---------------------------------------------------------------------------
# Model geometry
# ---------------------------------------------------------------------------


class TestModelGeometry:
    """Verify calc_input_region_size and model_input_size."""

    def test_model_input_size(self):
        m = CutSiteFragmentModel(tile_size=2048)
        assert m.model_input_size == 2048 + 256  # P + W

    def test_calc_input_region_size(self):
        m = CutSiteFragmentModel(tile_size=2048)
        l_in = m.calc_input_region_size(2048)
        # model_input_size + 5 (width-6 conv trim)
        assert l_in == 2048 + 256 + 5  # 2309

    def test_forward_output_matches_tile_size(self):
        """The model's output P dimension equals tile_size."""
        for tile_size in (256, 512, 1024, 2048):
            m = CutSiteFragmentModel(tile_size=tile_size, L_chunk=32)
            l_in = m.calc_input_region_size(tile_size)
            x = torch.randn(1, 4, l_in)
            logits, ep_shape, disp = m(x)
            assert logits.shape == (1, tile_size, N_L, 2)
            assert disp.shape == (1, 2, tile_size)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


class TestForwardPass:
    @pytest.fixture
    def small_model(self):
        return CutSiteFragmentModel(
            tile_size=256, N1=16, K=16, N2=8, H=8, L_chunk=32,
        )

    def test_output_shapes(self, small_model):
        l_in = small_model.calc_input_region_size(256)
        x = torch.randn(4, 4, l_in)
        logits, ep_shape, disp = small_model(x)
        assert logits.shape == (4, 256, N_L, 2)
        assert ep_shape.shape == (4, 2, 256)
        assert disp.shape == (4, 2, 256)

    def test_logits_finite(self, small_model):
        l_in = small_model.calc_input_region_size(256)
        x = torch.randn(2, 4, l_in)
        logits, _, _ = small_model(x)
        assert torch.isfinite(logits).all()

    def test_strand_axis_differs(self, small_model):
        """The two strands should generally produce different logits."""
        l_in = small_model.calc_input_region_size(256)
        x = torch.randn(1, 4, l_in)
        logits, _, _ = small_model(x)
        # With random init, strand 0 and 1 should differ
        assert not torch.allclose(logits[..., 0], logits[..., 1], atol=1e-6)

    def test_batch_independence(self, small_model):
        """Each batch element should be independent."""
        l_in = small_model.calc_input_region_size(256)
        x1 = torch.randn(1, 4, l_in)
        x2 = torch.randn(1, 4, l_in)
        x_cat = torch.cat([x1, x2], dim=0)

        logits_cat, _, _ = small_model(x_cat)
        logits_1, _, _ = small_model(x1)
        logits_2, _, _ = small_model(x2)

        assert torch.allclose(logits_cat[0], logits_1[0], atol=1e-5)
        assert torch.allclose(logits_cat[1], logits_2[0], atol=1e-5)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_all_trainable_params_receive_gradients(self):
        m = CutSiteFragmentModel(tile_size=64, N1=16, K=16, N2=8, H=8, L_chunk=32)
        l_in = m.calc_input_region_size(64)
        x = torch.randn(2, 4, l_in)
        logits, ep_shape, _ = m(x)
        # Sum all trainable outputs so every head receives gradient
        loss = logits.sum() + ep_shape.sum()
        loss.backward()

        for name, p in m.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"{name} has no gradient"
                assert not (p.grad == 0).all(), f"{name} has all-zero gradient"

    def test_frozen_dispersion_no_grad(self):
        m = CutSiteFragmentModel(tile_size=64, N1=16, K=16, N2=8, H=8, L_chunk=32)
        l_in = m.calc_input_region_size(64)
        x = torch.randn(2, 4, l_in)
        logits, ep_shape, disp = m(x)
        (logits.sum() + ep_shape.sum() + disp.sum()).backward()

        for name, p in m.named_parameters():
            if "dispersion_head" in name:
                assert not p.requires_grad, f"{name} should be frozen"


# ---------------------------------------------------------------------------
# Centre-based domain
# ---------------------------------------------------------------------------


class TestCentreBasedDomain:
    """Verify the scoring domain D = {(m, L, s) : m in [0,P), L in-band, s in {0,1}}."""

    def test_domain_size_per_L_is_P(self):
        """For every in-band L, the number of valid centres is exactly P."""
        P = 256
        m = CutSiteFragmentModel(tile_size=P, N1=16, K=16, N2=8, H=8, L_chunk=32)
        l_in = m.calc_input_region_size(P)
        x = torch.randn(1, 4, l_in)
        logits, _, _ = m(x)
        # logits has shape (1, P, n_L, 2) — every (m, L) slot exists
        assert logits.shape[1] == P  # all P centres
        assert logits.shape[2] == N_L  # all in-band lengths

    def test_taps_within_h_for_all_m_L(self):
        """For m in [0, P) and all in-band L, both taps fall within h."""
        P = 256
        model_input_size = P + 2 * WINDOW_HALF  # h length
        for L in range(L_MIN_INBAND, L_MAX_INBAND + 1):
            o_l = WINDOW_HALF - L // 2
            o_r = o_l + L
            for m in (0, P - 1):
                # h indices
                h_left = m + o_l
                h_right = m + o_r
                assert 0 <= h_left < model_input_size, (
                    f"Left tap out of bounds: m={m}, L={L}, h_left={h_left}"
                )
                assert 0 <= h_right < model_input_size, (
                    f"Right tap out of bounds: m={m}, L={L}, h_right={h_right}"
                )

    def test_both_L_parities(self):
        """Exercise both even and odd L values."""
        m = CutSiteFragmentModel(tile_size=64, N1=16, K=16, N2=8, H=8, L_chunk=32)
        l_in = m.calc_input_region_size(64)
        x = torch.randn(1, 4, l_in)
        logits, _, _ = m(x)

        # Check even L (e.g., L=26, idx=1) and odd L (e.g., L=25, idx=0)
        even_idx = 1   # L=26
        odd_idx = 0    # L=25
        assert torch.isfinite(logits[:, :, even_idx, :]).all()
        assert torch.isfinite(logits[:, :, odd_idx, :]).all()

    def test_centre_to_start_roundtrip_both_parities(self):
        """p = m - L//2; m = p + L//2 round-trips for both even and odd L."""
        for L in (25, 26, 100, 179):  # odd, even, even, odd
            for m in (0, 50, 127):
                p = m - L // 2
                m_back = p + L // 2
                assert m_back == m, f"Round-trip failed: m={m}, L={L}, p={p}"


# ---------------------------------------------------------------------------
# Parameter counts
# ---------------------------------------------------------------------------


class TestParameterCounts:
    def test_endpoint_bank_params(self):
        """4 banks × (K*N1 + K*10 + H*K) = 87,040 at default sizes."""
        m = CutSiteFragmentModel()
        per_bank = 128 * 128 + 128 * 10 + 32 * 128  # 16384 + 1280 + 4096 = 21760
        total = per_bank * 4  # 87040
        actual = sum(
            p.numel()
            for name, p in m.named_parameters()
            if name.startswith("bank_")
        )
        assert actual == total, f"Expected {total}, got {actual}"

    def test_interior_branch_params(self):
        """F (N2, 4, n_offsets) = 22,912 at default sizes."""
        m = CutSiteFragmentModel()
        expected_F = 32 * 4 * N_OFFSETS  # 22912
        expected_W = 32 * 32 * 2  # W_span + W_mean, no bias
        actual_F = m.interior_F.numel()
        actual_W = sum(
            p.numel()
            for name, p in m.named_parameters()
            if name.startswith("W_span") or name.startswith("W_mean")
        )
        assert actual_F == expected_F, f"F: expected {expected_F}, got {actual_F}"
        assert actual_W == expected_W, f"W: expected {expected_W}, got {actual_W}"

    def test_len_bias_size(self):
        m = CutSiteFragmentModel()
        assert m.len_bias.shape == (N_L,)  # 155
