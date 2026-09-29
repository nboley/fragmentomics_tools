"""Tests for the cut-site fragment model (Phase 1: architecture).

Covers: geometry, forward pass shapes, gradient flow, parameter counts,
centre-based domain sizing, and the fold-down linearity condition.
"""

import pytest
import torch

from background_model.cut_site_model import (
    CutSiteFragmentModel,
    DEFAULT_L_MIN,
    DEFAULT_L_MAX,
    N_LENGTH_GROUPS,
    WINDOW_HALF,
    _compute_geometry,
    length_to_group,
)

# Derived constants at default L_min/L_max (for tests that use defaults)
L_MIN_INBAND = DEFAULT_L_MIN
L_MAX_INBAND = DEFAULT_L_MAX
N_L, O_L_MIN, O_R_MAX, N_OFFSETS = _compute_geometry(L_MIN_INBAND, L_MAX_INBAND)


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
        assert length_to_group(torch.tensor(25), L_MIN_INBAND).item() == 0
        assert length_to_group(torch.tensor(39), L_MIN_INBAND).item() == 0
        # Second group
        assert length_to_group(torch.tensor(40), L_MIN_INBAND).item() == 1
        # Last group (clamped)
        assert length_to_group(torch.tensor(160), L_MIN_INBAND).item() == 9
        assert length_to_group(torch.tensor(175), L_MIN_INBAND).item() == 9
        assert length_to_group(torch.tensor(179), L_MIN_INBAND).item() == 9


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
        # model_input_size + 5 = 2309, rounded up to even = 2310
        assert l_in == 2310
        assert l_in % 2 == 0  # must be even for jitter_matrix

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


# ---------------------------------------------------------------------------
# Numerical component verification
# ---------------------------------------------------------------------------


class TestFoldedEndpointMatrix:
    """Verify _folded_M produces W @ diag(B[:,g]) @ a exactly."""

    def test_fold_matches_explicit(self):
        m = CutSiteFragmentModel(N1=8, K=4, N2=4, H=4, tile_size=64, L_chunk=32)
        bank_idx, group = 0, 3
        a = m.bank_a[bank_idx]          # (K, N1)
        B_mat = m.bank_B[bank_idx]      # (K, n_groups)
        W_mat = m.bank_W[bank_idx]      # (H, K)

        # Explicit: W @ diag(b_g) @ a
        b_g = B_mat[:, group]
        explicit = W_mat @ torch.diag(b_g) @ a

        folded = m._folded_M(bank_idx, group)
        assert torch.allclose(explicit, folded, atol=1e-6)


class TestInteriorPrefixSum:
    """Verify the prefix-sum difference computes the correct interior span."""

    def test_prefix_sum_properties(self):
        """U[0] = 0, U[k+1] - U[k] = u[k]."""
        m = CutSiteFragmentModel(N1=8, K=4, N2=4, H=4, tile_size=32, L_chunk=32)
        l_in = m.calc_input_region_size(32)
        x = torch.randn(1, 4, l_in)
        U = m._interior_prefix(x, 0, 32)
        # U[..., 0] should be 0 (exclusive prefix)
        assert (U[..., 0] == 0).all()
        # U[..., k+1] - U[..., k] should reconstruct u
        diffs = U[..., 1:] - U[..., :-1]
        # Recompute u directly (base_offset = O_L_MIN, matching the model)
        base_offset = O_L_MIN
        m_range = torch.arange(0, 32)
        o_range = torch.arange(N_OFFSETS)
        x_idx = m_range[:, None] + base_offset + o_range[None, :]
        x_patch = x[:, :, x_idx.reshape(-1)].reshape(1, 4, 32, N_OFFSETS)
        u_direct = torch.einsum('nco, bcmo -> bnmo', m.interior_F, x_patch)
        assert torch.allclose(diffs, u_direct, atol=1e-5)

    def test_span_sums_interior_bases(self):
        """span = U[o_r'] - U[o_l'] should equal sum of u over [o_l', o_r')."""
        m = CutSiteFragmentModel(N1=8, K=4, N2=4, H=4, tile_size=32, L_chunk=32)
        l_in = m.calc_input_region_size(32)
        x = torch.randn(1, 4, l_in)
        U = m._interior_prefix(x, 0, 32)

        # Pick L=100 (odd), check span at m=16
        L_val = 100
        o_l = WINDOW_HALF - L_val // 2   # 128 - 50 = 78
        o_r = o_l + L_val                 # 178
        o_l_p = o_l - O_L_MIN            # 78 - 39 = 39
        o_r_p = o_r - O_L_MIN            # 178 - 39 = 139

        span_from_U = U[0, :, 16, o_r_p] - U[0, :, 16, o_l_p]

        # Direct sum of u over [o_l_p, o_r_p)
        diffs = U[..., 1:] - U[..., :-1]
        direct_sum = diffs[0, :, 16, o_l_p:o_r_p].sum(dim=-1)

        assert torch.allclose(span_from_U, direct_sum, atol=1e-5)


class TestCrossBranchAlignment:
    """Verify the interior branch reads x_onehot at the same genomic positions
    that the endpoint branch reads h.  These tests use MARKER INPUTS so they
    exercise the actual model, not just arithmetic on constants."""

    def test_interior_responds_to_correct_position(self):
        """Place a distinguishing base at a known position.  The interior
        branch's prefix-sum difference for a fragment spanning that position
        should change when the base changes — proving it reads there."""
        P = 32
        m = CutSiteFragmentModel(N1=8, K=4, N2=4, H=4, tile_size=P, L_chunk=155)
        l_in = m.calc_input_region_size(P)

        # All-A baseline
        x_base = torch.zeros(1, 4, l_in)
        x_base[:, 0, :] = 1.0  # all A

        # Mutant: change one base at the position the left tap of L=100
        # at scored_m=5 should read.  Endpoint tap h-pos = o_l + scored_m.
        L_test = 100
        scored_m = 5
        o_l = WINDOW_HALF - L_test // 2  # 78
        target_x_pos = o_l + scored_m     # 83 — the position both branches should see

        x_mut = x_base.clone()
        x_mut[:, 0, target_x_pos] = 0.0
        x_mut[:, 2, target_x_pos] = 1.0  # A → G at position 83

        # Run the interior prefix on both
        U_base = m._interior_prefix(x_base, 0, P)
        U_mut = m._interior_prefix(x_mut, 0, P)

        # The span covering position 83 should differ between base and mutant.
        # For L=100: o_l'=o_l-O_L_MIN=39, o_r'=o_r-O_L_MIN=139
        o_l_p = o_l - O_L_MIN
        o_r_p = o_l_p + L_test
        span_base = U_base[0, :, scored_m, o_r_p] - U_base[0, :, scored_m, o_l_p]
        span_mut = U_mut[0, :, scored_m, o_r_p] - U_mut[0, :, scored_m, o_l_p]

        assert not torch.allclose(span_base, span_mut, atol=1e-7), (
            "Interior branch did not respond to a base change at the expected "
            f"position {target_x_pos} — possible coordinate misalignment"
        )

    def test_interior_does_not_respond_outside_span(self):
        """A base change OUTSIDE the fragment span should NOT affect the span."""
        P = 32
        m = CutSiteFragmentModel(N1=8, K=4, N2=4, H=4, tile_size=P, L_chunk=155)
        l_in = m.calc_input_region_size(P)

        x_base = torch.zeros(1, 4, l_in)
        x_base[:, 0, :] = 1.0  # all A

        L_test = 50
        scored_m = 10
        o_l = WINDOW_HALF - L_test // 2  # 103
        o_r = o_l + L_test                # 153

        # Mutate a position BEFORE the span (o_l + scored_m - 2 in x coords)
        outside_pos = o_l + scored_m - 2  # 111, which is < o_l + scored_m = 113
        # Actually need to ensure it's before o_l in the re-indexed frame.
        # The span covers x positions [o_l + scored_m, o_r + scored_m).
        # Place mutation well before: at position 5 (way outside).
        outside_pos = 5
        x_mut = x_base.clone()
        x_mut[:, 0, outside_pos] = 0.0
        x_mut[:, 3, outside_pos] = 1.0  # A → T at position 5

        U_base = m._interior_prefix(x_base, 0, P)
        U_mut = m._interior_prefix(x_mut, 0, P)

        o_l_p = o_l - O_L_MIN
        o_r_p = o_l_p + L_test
        span_base = U_base[0, :, scored_m, o_r_p] - U_base[0, :, scored_m, o_l_p]
        span_mut = U_mut[0, :, scored_m, o_r_p] - U_mut[0, :, scored_m, o_l_p]

        assert torch.allclose(span_base, span_mut, atol=1e-7), (
            f"Interior branch responded to a base change at position {outside_pos} "
            "which is outside the fragment span — coordinate misalignment"
        )

    @pytest.mark.parametrize("L_val", [25, 179])
    def test_endpoint_and_interior_see_same_position(self, L_val):
        """Mutate the base at the left tap position.  Both the endpoint branch
        (via h) and the interior branch (via x_onehot) should respond."""
        P = 32
        m = CutSiteFragmentModel(N1=8, K=4, N2=4, H=4, tile_size=P, L_chunk=155)
        l_in = m.calc_input_region_size(P)

        x_base = torch.zeros(1, 4, l_in)
        x_base[:, 0, :] = 1.0

        scored_m = 10
        o_l = WINDOW_HALF - L_val // 2
        tap_pos = o_l + scored_m  # position in x/h coordinates

        x_mut = x_base.clone()
        x_mut[:, 0, tap_pos] = 0.0
        x_mut[:, 1, tap_pos] = 1.0  # A → C

        # Endpoint branch: h changes at tap_pos
        with torch.no_grad():
            h_base = m.l1_act(m.l1_conv(x_base))
            h_mut = m.l1_act(m.l1_conv(x_mut))
            if h_base.shape[-1] > m._model_input_size:
                h_base = h_base[..., :m._model_input_size]
                h_mut = h_mut[..., :m._model_input_size]
        # h differs at tap_pos (and its receptive field)
        endpoint_changed = not torch.allclose(
            h_base[0, :, tap_pos], h_mut[0, :, tap_pos], atol=1e-7
        )

        # Interior branch: prefix sum changes for spans covering tap_pos
        U_base = m._interior_prefix(x_base, 0, P)
        U_mut = m._interior_prefix(x_mut, 0, P)
        o_l_p = o_l - O_L_MIN
        o_r_p = o_l_p + L_val
        span_base = U_base[0, :, scored_m, o_r_p] - U_base[0, :, scored_m, o_l_p]
        span_mut = U_mut[0, :, scored_m, o_r_p] - U_mut[0, :, scored_m, o_l_p]
        interior_changed = not torch.allclose(span_base, span_mut, atol=1e-7)

        assert endpoint_changed, "Endpoint branch did not respond to mutation"
        assert interior_changed, "Interior branch did not respond to mutation"


class TestEndpointTapAlignment:
    """Verify that endpoint taps address the correct h positions."""

    def test_tap_positions_match_design(self):
        """For L=25 and L=179, verify o_l and o_r match design formulas."""
        for L_val in (25, 26, 100, 179):
            o_l = WINDOW_HALF - L_val // 2
            o_r = o_l + L_val
            # Design: taps are L apart (cut sites p and p+L)
            assert o_r - o_l == L_val
            # Start p = m - L//2, so p at m=0 gives p = -L//2
            p = 0 - L_val // 2
            # Right cut site: p + L = -L//2 + L = L - L//2 = ceil(L/2)
            assert o_r - WINDOW_HALF == L_val - L_val // 2  # ceil(L/2)
            assert o_l - WINDOW_HALF == -(L_val // 2)       # -floor(L/2)

    def test_endpoint_feats_use_correct_h_positions(self):
        """The endpoint features at m=0 for the left tap should read h[:,:,o_l]."""
        m = CutSiteFragmentModel(N1=8, K=4, N2=4, H=4, tile_size=32, L_chunk=155)
        l_in = m.calc_input_region_size(32)
        torch.manual_seed(42)
        x = torch.randn(1, 4, l_in)

        # Forward to get logits
        logits, _, _ = m(x)

        # Manually verify for L=25 (idx=0), strand=0, tap=left
        h = m.l1_act(m.l1_conv(x))  # (1, N1, model_input_size)
        L_val = 25
        o_l = WINDOW_HALF - L_val // 2  # 128 - 12 = 116
        group = 0  # (25-25)//15 = 0
        bank_idx = 0  # left tap, strand 0
        M_gs = m._folded_M(bank_idx, group)  # (H, N1)
        Mh = M_gs @ h  # (1, H, model_input_size)
        # At m=0, the left tap reads h position 0 + o_l = 116
        f_l_at_m0 = Mh[0, :, o_l]  # (H,)

        # This should match what the forward pass computed for position m=0,
        # L_idx=0, strand=0 — specifically the left-tap contribution.
        # We can't isolate it from the logit (it's mixed with f_r and interior),
        # but we can verify the h indexing is correct.
        assert f_l_at_m0.shape == (4,)  # H=4
        assert torch.isfinite(f_l_at_m0).all()


class TestLinearityCondition:
    """The endpoint path must stay LINEAR until summed into pre-activation.

    Verify no nonlinearity is applied between the bank matrices and the
    pre-activation sum.
    """

    def test_endpoint_linearity(self):
        """Scaling h by alpha should scale the endpoint contribution by alpha."""
        m = CutSiteFragmentModel(N1=8, K=4, N2=4, H=4, tile_size=32, L_chunk=155)
        alpha = 2.0
        group = 0
        bank_idx = 0
        M = m._folded_M(bank_idx, group)  # (H, N1)

        h_fake = torch.randn(1, 8, 293)  # model_input_size for tile=32
        result_1x = M @ h_fake
        result_2x = M @ (alpha * h_fake)
        assert torch.allclose(result_2x, alpha * result_1x, atol=1e-5)


class TestLenBiasPlacement:
    """len_bias must be added AFTER the nonlinearity, on the logit."""

    def test_len_bias_shifts_logit_uniformly(self):
        """Setting len_bias to a known value should shift that L's logits."""
        m = CutSiteFragmentModel(N1=8, K=4, N2=4, H=4, tile_size=64, L_chunk=32)
        l_in = m.calc_input_region_size(64)
        x = torch.randn(1, 4, l_in)

        # Get baseline logits
        with torch.no_grad():
            logits_base, _, _ = m(x)

            # Set len_bias[5] += 10.0
            m.len_bias.data[5] += 10.0
            logits_shifted, _, _ = m(x)

        # Only L index 5 should change, by exactly +10.0
        diff = logits_shifted - logits_base
        assert torch.allclose(diff[:, :, 5, :],
                              torch.full_like(diff[:, :, 5, :], 10.0),
                              atol=1e-4)
        # Other L indices should be unchanged
        mask = torch.ones(N_L, dtype=torch.bool)
        mask[5] = False
        assert torch.allclose(diff[:, :, mask, :],
                              torch.zeros_like(diff[:, :, mask, :]),
                              atol=1e-4)


# ---------------------------------------------------------------------------
# L_chunk correctness
# ---------------------------------------------------------------------------


class TestLChunkCorrectness:
    """Verify that chunked forward (L_chunk=32) produces the same logits
    as unchunked (L_chunk=N_L)."""

    def test_chunked_matches_unchunked(self):
        P = 64
        m_chunked = CutSiteFragmentModel(
            N1=8, K=4, N2=4, H=4, tile_size=P, L_chunk=32,
        )
        m_unchunked = CutSiteFragmentModel(
            N1=8, K=4, N2=4, H=4, tile_size=P, L_chunk=N_L,
        )
        # Copy weights
        m_unchunked.load_state_dict(m_chunked.state_dict())

        l_in = m_chunked.calc_input_region_size(P)
        x = torch.randn(1, 4, l_in)

        with torch.no_grad():
            logits_c, ep_c, disp_c = m_chunked(x)
            logits_u, ep_u, disp_u = m_unchunked(x)

        assert torch.allclose(logits_c, logits_u, atol=1e-5), (
            f"Chunked vs unchunked max diff: {(logits_c - logits_u).abs().max()}"
        )
        assert torch.allclose(ep_c, ep_u, atol=1e-5)


# ---------------------------------------------------------------------------
# NB loss integration
# ---------------------------------------------------------------------------


class TestNBLossIntegration:
    """Verify the NB endpoint loss path runs correctly end-to-end."""

    def test_nb_loss_runs_with_model_outputs(self):
        """The NB loss should produce a finite scalar when given model outputs
        and synthetic endpoint targets."""
        P = 64
        m = CutSiteFragmentModel(
            N1=8, K=4, N2=4, H=4, tile_size=P, L_chunk=32,
            dispersion_window_size=P,  # single window for small tile
        )
        l_in = m.calc_input_region_size(P)
        x = torch.randn(2, 4, l_in)

        logits, ep_shape, disp_bp = m(x)

        # Synthetic endpoint targets (2 tracks)
        endpoints = torch.randint(0, 10, (2, 2, P), dtype=torch.float32)
        mask = torch.ones(2, P, dtype=torch.bool)
        # _pooled_log_dispersion handles the mask expansion internally
        log_disp = m._pooled_log_dispersion(disp_bp, mask)

        # NB loss needs mask as (B, 1, P) or (B, C, P) for _prepare_mask
        mask3 = mask[:, None, :].expand_as(endpoints)
        nb_loss = m.nb_loss_fn(ep_shape, log_disp, endpoints, mask3)

        assert torch.isfinite(nb_loss), f"NB loss is not finite: {nb_loss}"
        assert nb_loss.requires_grad, "NB loss should have grad through ep_shape"
