"""Cut-site fragment model — per-fragment (m, L, strand) logits.

Implements the architecture described in docs/pending/cut_site_fragment_model.md.
The model emits one logit per (start m, length L, strand s) triple and trains
on the fragment multinomial.  Start admission: a fragment is admitted iff
min(start, end) lies in [0, tile_size).  m is the start position, not the
centre.

Two-stage training (docs/pending/two_stage_training.md):
  Stage 1: Analytical hex-table baseline — log p₀(m, L, s) from empirical
           hexamer cut-site frequencies + marginal p(L).  Zero training.
  Stage 2: Neural correction — model outputs additive log-corrections f_θ,
           so log p(m,L,s) = log p₀(m,L,s) + f_θ(seq, m, L, s), re-normalized.
           Hex tables are trainable parameters initialized from Stage 1.

Consumes primitives from the frozen background_model_core.py (losses, masking,
dispersion pooling) without modifying them.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import lightning as L
import torch
import torch.nn as nn

from background_model.tracks import FL_BANDS
from background_model_core import (
    MaskedNegativeBinomialOffsetNLLLoss,
    masked_mean_pool,
)


# ---------------------------------------------------------------------------
# Custom autograd for contiguous-range gather
# ---------------------------------------------------------------------------


class EndpointGather(torch.autograd.Function):
    """Gather P-length contiguous ranges from the last dimension of a tensor.

    Forward: ``out[..., i, j] = Mh[..., pos_idx[i, j]]``
    where ``pos_idx[i, j] = offset[i] + j`` (contiguous per row).

    Backward: exploits the contiguous structure to replace the generic
    scatter_add with slice accumulation, avoiding the full-size zero-tensor
    allocation that IndexBackward0 performs.
    """

    @staticmethod
    def forward(ctx, Mh, pos_idx):
        ctx.save_for_backward(pos_idx)
        ctx.Mh_shape = Mh.shape
        return Mh[..., pos_idx]

    @staticmethod
    def backward(ctx, grad_output):
        (pos_idx,) = ctx.saved_tensors
        grad_Mh = grad_output.new_zeros(ctx.Mh_shape)

        n_g, P = pos_idx.shape
        offsets = pos_idx[:, 0]

        for i in range(n_g):
            start = offsets[i].item()
            grad_Mh[..., start:start + P] += grad_output[..., i, :]

        return grad_Mh, None

# ---------------------------------------------------------------------------
# Window geometry (fixed)
# ---------------------------------------------------------------------------

# WINDOW_HALF >= L_max - 1 = 179.  Under start admission the max 3' cut is
# at position (P-1) + L_max - 1 = P + 178, so the model needs hex indices
# that far past the tile edge.  181 gives rf_budget = WINDOW_HALF + 5 = 186.
WINDOW_HALF = 181                                            # W/2

# Length groups: 10, half-open, 15 wide from L_min with the last group wider
N_LENGTH_GROUPS = 10
LENGTH_GROUP_WIDTH = 15

# ---------------------------------------------------------------------------
# Default length bounds (from FL_BANDS, but overridable)
# ---------------------------------------------------------------------------

DEFAULT_L_MIN = min(lo for lo, hi in FL_BANDS)               # 25
DEFAULT_L_MAX = max(hi for lo, hi in FL_BANDS)               # 180 (inclusive)


def _compute_geometry(L_min: int, L_max: int):
    """Derive all geometry from explicit length bounds.

    Under start admission the left tap is at WINDOW_HALF + m (independent of L),
    so the offset range for the interior branch spans [WINDOW_HALF, WINDOW_HALF + L_max).
    """
    n_L = L_max - L_min + 1
    o_l_min = WINDOW_HALF
    o_r_max = WINDOW_HALF + L_max
    n_offsets = L_max  # live width = L_max
    return n_L, o_l_min, o_r_max, n_offsets


def length_to_group(L: torch.Tensor, L_min: int) -> torch.Tensor:
    """Map fragment length to group index: g = min((L - L_min) // 15, 9)."""
    return torch.clamp((L - L_min) // LENGTH_GROUP_WIDTH,
                        max=N_LENGTH_GROUPS - 1)


# ---------------------------------------------------------------------------
# Hex-table baseline (Stage 1)
# ---------------------------------------------------------------------------

HEX_K = 6
N_HEX = 4 ** HEX_K   # 4096

# Position of the cut BASE within the k-mer, junction-centred: for a cut at
# base s the k-mer spans s-3 .. s+2, so the cut junction (between s-1 and s)
# sits exactly mid-k-mer.  Owner-specified 2026-10-05.
#
# This MUST be the single definition.  It previously lived implicitly in crop
# widths in two files and they silently disagreed -- the builder registered at
# cut-2 while the model read at cut+0, so the tables were keyed on one k-mer and
# looked up with another.  Guarded by tests/test_hex_registration.py.
CUT_OFFSET_IN_KMER = 3


def _onehot_to_hex_indices(x_onehot: torch.Tensor, k: int = HEX_K) -> torch.Tensor:
    """Convert (B, 4, L) one-hot sequence to (B, L-k+1) hexamer indices.

    Each index is in [0, 4^k).  Positions with ambiguous bases (no channel
    active, i.e. N) get index 0 (safe — just maps to a single hex entry).
    """
    # argmax over channel dim: (B, L) base indices 0-3
    bases = x_onehot.argmax(dim=1)  # (B, L)
    powers = (4 ** torch.arange(k, device=x_onehot.device)).long()  # (k,)
    # Sliding window: sum base[p+i] * 4^i for i in 0..k-1
    # Use unfold for efficiency
    bases_unf = bases.unfold(1, k, 1)  # (B, L-k+1, k)
    indices = (bases_unf * powers[None, None, :]).sum(dim=-1)  # (B, L-k+1)
    return indices


def compute_log_p0(
    x_onehot: torch.Tensor,
    hex_log_start_fwd: torch.Tensor,
    hex_log_stop_fwd: torch.Tensor,
    hex_log_start_rev: torch.Tensor,
    hex_log_stop_rev: torch.Tensor,
    log_pL: torch.Tensor,
    tile_size: int,
    L_min: int,
    L_max: int,
) -> torch.Tensor:
    """Compute factored baseline log p₀(m, L, s) from hex tables.

    Under start admission, m is the fragment START position (not centre).

    Args:
        x_onehot: (B, 4, l_in) one-hot sequence (includes conv padding)
        hex_log_*: (4096,) log-probability tables (trainable parameters)
        log_pL: (n_L,) log marginal length distribution
        tile_size: P (number of scored positions)
        L_min, L_max: fragment length range

    Returns:
        log_p0: (B, P, n_L, 2) unnormalized log-scores for each (m, L, s).
                NOT log-softmax-ed — the caller normalises after adding f_θ.
    """
    B = x_onehot.shape[0]
    n_L = L_max - L_min + 1

    # Hex indices over the full sequence (including margins for endpoint lookup)
    hex_idx = _onehot_to_hex_indices(x_onehot)  # (B, l_in - k + 1)

    # Under start admission, m IS the fragment start.  The hex lookup for a
    # fragment (m, L) on the + strand reads k-mers at positions m (5' cut)
    # and m + L - 1 (3' cut) in tile space, which map to hex_idx positions
    # WINDOW_HALF + m and WINDOW_HALF + m + L - 1.

    # Look up per-position log-probs for all 4 tables
    log_sf = hex_log_start_fwd[hex_idx]   # (B, n_hex_pos)
    log_ef = hex_log_stop_fwd[hex_idx]    # (B, n_hex_pos)
    log_sr = hex_log_start_rev[hex_idx]   # (B, n_hex_pos)
    log_er = hex_log_stop_rev[hex_idx]    # (B, n_hex_pos)

    # Build log_p0 (B, P, n_L, 2) by gathering endpoint log-probs
    L_vals = torch.arange(L_min, L_max + 1, device=x_onehot.device)  # (n_L,)
    m_vals = torch.arange(tile_size, device=x_onehot.device)          # (P,)

    # start_pos[m, L] = WINDOW_HALF + m           (5' cut in hex_idx coords)
    # stop_pos[m, L]  = WINDOW_HALF + m + L - 1   (3' cut in hex_idx coords)
    # start_pos is the same for all L under start admission.
    stop_pos = (WINDOW_HALF + m_vals[:, None] + L_vals[None, :] - 1).long()    # (P, n_L)
    start_pos = (WINDOW_HALF + m_vals[:, None]).long().expand_as(stop_pos)      # (P, n_L)

    # Shape check: the input tensor must be wide enough for every (m, L)
    # cell.  Uses Python ints (hex_idx.shape[1]), not tensor reductions,
    # so there is no device-to-host sync on the hot path.
    required_width = WINDOW_HALF + tile_size + L_max - 1
    if hex_idx.shape[1] < required_width:
        raise ValueError(
            f"hex_idx width {hex_idx.shape[1]} < required {required_width}: "
            f"input tensor too narrow for the geometry"
        )

    # Flatten for gather: (P * n_L,)
    flat_start = start_pos.contiguous().reshape(-1)   # (P * n_L,)
    flat_stop = stop_pos.reshape(-1)

    # Gather: (B, P * n_L) -> reshape to (B, P, n_L)
    log_sf_g = log_sf[:, flat_start].reshape(B, tile_size, n_L)
    log_ef_g = log_ef[:, flat_stop].reshape(B, tile_size, n_L)
    log_sr_g = log_sr[:, flat_stop].reshape(B, tile_size, n_L)
    log_er_g = log_er[:, flat_start].reshape(B, tile_size, n_L)

    # log p₀(m, L, s=0) = log_start_fwd[start] + log_stop_fwd[stop] + log_pL[L]
    # log p₀(m, L, s=1) = log_start_rev[stop] + log_stop_rev[start] + log_pL[L]
    log_pL_broad = log_pL[None, None, :]   # (1, 1, n_L)

    log_p0_fwd = log_sf_g + log_ef_g + log_pL_broad  # (B, P, n_L)
    log_p0_rev = log_sr_g + log_er_g + log_pL_broad   # (B, P, n_L)

    log_p0 = torch.stack([log_p0_fwd, log_p0_rev], dim=-1)  # (B, P, n_L, 2)
    return log_p0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class CutSiteFragmentModel(L.LightningModule):
    """Per-fragment (m, L, strand) logit model for cfDNA cut-site bias.

    Architecture:
        L1:  Conv1d(4, N1, k=6) + act → h (B, N1, model_input_size)
        L2a: 4 endpoint banks (tap × strand), folded M[g,s] = W diag(B[:,g]) a
        L2b: Interior branch F (N2, 4, n_offsets), prefix-sum + difference
        L3:  pre → act → w_out · hid + len_bias[L] → logit (B, P, n_L, 2)

    The forward pass is chunked over L (L_chunk=32) to avoid OOM.
    """

    def __init__(
        self,
        # Length bounds (the model's scoring domain)
        L_min: int = DEFAULT_L_MIN,
        L_max: int = DEFAULT_L_MAX,
        # Architecture
        N1: int = 128,
        K: int = 128,
        N2: int = 32,
        H: int = 32,
        L_chunk: int = 32,
        tile_size: int = 2048,
        # Training
        learning_rate: float = 1e-4,
        lr_patience: int = 4,
        max_lr_reductions: int = 3,
        lr_factor: float = 0.5,
        # NB endpoint loss (ancillary, frozen dispersion)
        dispersion_window_size: int = 256,
        log_dispersion_init: float = 7.0,
        max_dispersion_ratio: Optional[float] = 2.0,
        clamp_margin: float = 1.0,
        nb_w0: float = 1.0,
        nb_T_anneal: int = 5,
        mult_weight: float = 1.0,
        # Stage 2: hex-table baseline (path to .npz from stage1_hex_baseline.py)
        hex_tables_path: Optional[str] = None,
        freeze_hex: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["hex_tables_path", "freeze_hex"])

        # -- Geometry derived from L_min / L_max ----------------------------
        n_L, o_l_min, o_r_max, n_offsets = _compute_geometry(L_min, L_max)
        self._n_L = n_L
        self._o_l_min = o_l_min
        self._n_offsets = n_offsets

        # Under start admission the max 3' hex lookup is at
        # WINDOW_HALF + (P-1) + (L_max-1).  This must be a valid hex_idx
        # index, which requires WINDOW_HALF >= L_max - 1.
        if WINDOW_HALF < L_max - 1:
            raise ValueError(
                f"WINDOW_HALF={WINDOW_HALF} < L_max-1={L_max - 1}: "
                f"the geometry cannot reach every (m, L) cell.  "
                f"Increase WINDOW_HALF to at least {L_max - 1}."
            )

        # Verify the k-mer registration offset.  CUT_OFFSET_IN_KMER=3
        # holds only when raw = P + 2*WINDOW_HALF + 5 is ODD, which
        # requires an even tile_size.  An odd tile_size silently slides
        # the offset by 1, breaking hexamer registration with no error.
        l_in = self.calc_input_region_size(tile_size)
        pad = (l_in - tile_size) // 2
        if pad - WINDOW_HALF != CUT_OFFSET_IN_KMER:
            raise ValueError(
                f"Geometry error: padding {pad} - WINDOW_HALF {WINDOW_HALF} = "
                f"{pad - WINDOW_HALF}, expected CUT_OFFSET_IN_KMER = "
                f"{CUT_OFFSET_IN_KMER}.  tile_size must be even."
            )

        # Strictly-lower-triangular ones, (n_offsets, n_offsets + 1), so that
        # `u @ _excl_prefix` is the exclusive prefix sum of u along its last
        # axis: column k sums the offsets o' < k, giving column 0 == 0.
        # Replaces a cumsum -- see _interior_prefix for the measurements.
        # A buffer, not rebuilt per forward, so it follows .to(device) and is
        # built once; it is tiny (n_offsets x n_offsets+1) next to the (B, N2,
        # m_count, n_offsets) operand it multiplies.
        self.register_buffer(
            "_excl_prefix",
            torch.triu(
                torch.ones(n_offsets, n_offsets + 1, dtype=torch.float32),
                diagonal=1,
            ),
        )

        W = 2 * WINDOW_HALF
        self._h_width = tile_size + W

        # -- L1: cut-site propensity ----------------------------------------
        self.l1_conv = nn.Conv1d(4, N1, kernel_size=6, padding=0)
        self.l1_act = nn.LeakyReLU()

        # -- L2a: endpoint banks (4 = 2 taps × 2 strands) ------------------
        # Each bank: a (K, N1), B (K, n_groups), W_proj (H, K)
        # Index: [tap][strand] where tap ∈ {0=left, 1=right}, strand ∈ {0=+, 1=-}
        self.bank_a = nn.ParameterList([
            nn.Parameter(torch.randn(K, N1) * (N1 ** -0.5))
            for _ in range(4)
        ])
        self.bank_B = nn.ParameterList([
            nn.Parameter(torch.randn(K, N_LENGTH_GROUPS) * 0.01)
            for _ in range(4)
        ])
        self.bank_W = nn.ParameterList([
            nn.Parameter(torch.randn(H, K) * (K ** -0.5))
            for _ in range(4)
        ])

        # -- L2b: interior branch ------------------------------------------
        # F (N2, 4, n_offsets) with learned per-offset weights
        self.interior_F = nn.Parameter(
            torch.randn(N2, 4, n_offsets) * 0.01
        )
        self.W_span = nn.Linear(N2, H, bias=False)
        self.W_mean = nn.Linear(N2, H, bias=False)

        # -- L3: output head -----------------------------------------------
        self.l3_act = nn.LeakyReLU()
        # w_out is ZERO-initialised, so the correction f_theta is exactly 0 at
        # init and the logits are exactly log_p0. Stage 2 therefore starts AT
        # its hex-table prior and can only improve on it, rather than starting
        # at prior+noise and having to undo the noise first.
        #
        # Known and accepted consequence: logit = (hid * w_out).sum(-1), so
        # d(logit)/d(hid) = w_out = 0 and NO upstream parameter receives
        # gradient on step 0. This is transient, not fatal -- d(L)/d(w_out) =
        # sum((dL/dlogit) * hid) is nonzero, so w_out leaves zero on the first
        # step and the stem trains from step 1. Confirmed empirically: a
        # zero-init run trained normally (12.214 -> 12.099 over 16 epochs).
        # test_zero_init_contract pins both halves of this.
        self.w_out = nn.Parameter(torch.zeros(H))
        self.len_bias = nn.Parameter(torch.zeros(n_L))

        # -- Endpoint heads for NB loss (shape + frozen dispersion) ---------
        # endpoint_shape_head: trainable, predicts per-position shape for the
        # 2 band-free endpoint tracks (first/last).  Same role as shape_head
        # in the 12-track models, but C=2 instead of C=12.
        self.endpoint_shape_head = nn.Conv1d(N1, 2, kernel_size=1, padding=0)
        # dispersion_head: FROZEN at log_dispersion_init (near-multinomial)
        self.dispersion_head = nn.Conv1d(N1, 2, kernel_size=1, padding=0)
        for p in self.dispersion_head.parameters():
            p.requires_grad = False

        # NB loss (reused from frozen core)
        self.nb_loss_fn = MaskedNegativeBinomialOffsetNLLLoss(
            max_dispersion_ratio=max_dispersion_ratio,
            clamp_margin=clamp_margin,
        )

        # -- Stage 2: hex-table baseline (trainable or frozen) -----------------
        n_L = L_max - L_min + 1
        self._use_hex_baseline = hex_tables_path is not None
        self._freeze_hex = freeze_hex
        if hex_tables_path is not None:
            data = np.load(hex_tables_path)
            hex_tensors = {
                "hex_log_start_fwd": torch.from_numpy(np.log(np.maximum(data["start_fwd"], 1e-30))).float(),
                "hex_log_stop_fwd": torch.from_numpy(np.log(np.maximum(data["stop_fwd"], 1e-30))).float(),
                "hex_log_start_rev": torch.from_numpy(np.log(np.maximum(data["start_rev"], 1e-30))).float(),
                "hex_log_stop_rev": torch.from_numpy(np.log(np.maximum(data["stop_rev"], 1e-30))).float(),
            }
            if freeze_hex:
                for name, tensor in hex_tensors.items():
                    self.register_buffer(name, tensor)
            else:
                for name, tensor in hex_tensors.items():
                    setattr(self, name, nn.Parameter(tensor))
            self.register_buffer(
                "_log_pL",
                torch.from_numpy(np.log(np.maximum(data["p_L"], 1e-30))).float(),
            )
        else:
            # Dummy parameters (not used when hex baseline is off)
            self.hex_log_start_fwd = None
            self.hex_log_stop_fwd = None
            self.hex_log_start_rev = None
            self.hex_log_stop_rev = None
            self.register_buffer("_log_pL", torch.zeros(n_L))

        # Pre-compute length tensors as buffers
        self.register_buffer(
            "_L_vals",
            torch.arange(L_min, L_max + 1, dtype=torch.long),
        )
        self.register_buffer(
            "_L_groups",
            length_to_group(
                torch.arange(L_min, L_max + 1, dtype=torch.long), L_min
            ),
        )

    # -- Geometry -----------------------------------------------------------

    def calc_input_region_size(self, output_region_size: int) -> int:
        """One-hot sequence length needed to emit `output_region_size` scored
        positions.  The width-6 conv trims 5.  Rounded up to even for
        jitter_matrix's matched-parity requirement (same pattern as
        BackgroundModelKEN).
        """
        raw = output_region_size + 2 * WINDOW_HALF + 5
        return raw + (raw % 2)  # round up to even

    @property
    def model_input_size(self) -> int:
        """Full one-hot input width, including conv margin and even-parity rounding.

        This is the SINGLE definition of the model's input size.  Callers
        (dataset, training script, inference) use this value directly — do NOT
        add +5 or round separately.
        """
        return self.calc_input_region_size(self.hparams.tile_size)

    # -- Folded endpoint matrices -------------------------------------------

    def _folded_M(self, bank_idx: int, group: int) -> torch.Tensor:
        """Compute M[g,s] = W diag(B[:,g]) a → (H, N1).

        This is the folded endpoint matrix for one (tap, strand, group).
        """
        a = self.bank_a[bank_idx]          # (K, N1)
        b_g = self.bank_B[bank_idx][:, group]  # (K,)
        w = self.bank_W[bank_idx]          # (H, K)
        return w @ (b_g[:, None] * a)      # (H, N1)

    # -- Interior branch ----------------------------------------------------

    def _interior_prefix(
        self, x_onehot: torch.Tensor, m_start: int, m_count: int,
    ) -> torch.Tensor:
        """Compute the exclusive prefix sum U for a chunk of m positions.

        Args:
            x_onehot: (B, 4, l_in) one-hot sequence
            m_start: first scored position in this chunk
            m_count: number of scored positions in this chunk

        Returns:
            U: (B, N2, m_count, n_offsets+1) exclusive prefix sum
        """
        # x_onehot index for scored position m and re-indexed offset o':
        #   x_idx = m + O_L_MIN + o'
        # Under start admission O_L_MIN = WINDOW_HALF, so
        #   x_idx = m + WINDOW_HALF + o'

        B = x_onehot.shape[0]
        N2 = self.interior_F.shape[0]
        n_offsets = self._n_offsets
        base_offset = self._o_l_min

        # Gather the x_onehot slice for all (m, o') pairs
        # Positions: for m in [m_start, m_start+m_count), o' in [0, n_offsets)
        # x_idx = m + base_offset + o'
        m_range = torch.arange(m_start, m_start + m_count,
                               device=x_onehot.device)  # (m_count,)
        o_range = torch.arange(n_offsets, device=x_onehot.device)  # (n_offsets,)
        # (m_count, n_offsets) indices into x_onehot's last dim
        x_idx = m_range[:, None] + base_offset + o_range[None, :]  # (m_count, n_offsets)

        # x_patch: (B, 4, m_count, n_offsets)
        x_patch = x_onehot[:, :, x_idx.reshape(-1)].reshape(
            B, 4, m_count, n_offsets
        )

        # u[b,n,m,o'] = sum_c F[n,c,o'] * x_patch[b,c,m,o']
        # F is (N2, 4, n_offsets), x_patch is (B, 4, m_count, n_offsets)
        # einsum: 'nco, bcmo -> bnmo'
        u = torch.einsum('nco, bcmo -> bnmo', self.interior_F, x_patch)

        # Exclusive prefix sum along the offset axis (last dim):
        # U[k] = sum_{o'<k} u[o'], so U[0] = 0 and U[n_offsets] = sum(u).
        #
        # Done as a matmul with a strictly-lower-triangular ones matrix rather
        # than `zeros(); U[...,1:] = cumsum(u, -1)`.  Measured on an A10G at
        # the real shape (32, 32, 1024, 180), bf16, warmed fwd AND bwd:
        #
        #     zeros+slice+cumsum   fwd 17.25 ms   fwd+bwd 39.06 ms   3.15 GiB
        #     F.pad(cumsum)        fwd 17.24 ms   fwd+bwd 37.49 ms   3.15 GiB
        #     matmul (this)        fwd  2.75 ms   fwd+bwd  6.61 ms   2.12 GiB
        #
        # 5.9x faster end-to-end and 33% less memory.  The scan is bandwidth-
        # bound over a short innermost axis, which is the worst case for it,
        # while the GEMM runs on tensor cores that `aten::mm` profiling showed
        # sitting at 2.6% of the step -- i.e. essentially idle.
        #
        # It is also MORE accurate, which is why this is a safe substitution
        # and not merely a fast one: a sequential bf16 scan accumulates over
        # n_offsets terms in 8 mantissa bits, whereas the GEMM accumulates in
        # fp32 on the tensor cores.  Max deviation from an fp32 reference was
        # 0.244 for the matmul vs 0.536 for the scan -- 2.2x better.
        #
        # Changes computed results (different summation order); approved by the
        # owner 2026-10-05.
        U = torch.matmul(u, self._excl_prefix.to(u.dtype))

        return U  # (B, N2, m_count, n_offsets + 1)

    # -- Forward pass -------------------------------------------------------

    def forward(
        self, x_onehot: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_onehot: (B, 4, l_in) one-hot sequence

        Returns:
            logits: (B, P, n_L, 2) per-(start, length, strand) logits
            endpoint_shape: (B, 2, P) per-position endpoint shape logits
            disp_bp: (B, 2, P) per-position dispersion (frozen)
        """
        B = x_onehot.shape[0]
        P = self.hparams.tile_size
        H = self.hparams.H
        L_chunk = self.hparams.L_chunk

        # -- L1: cut-site propensity ----------------------------------------
        h = self.l1_act(self.l1_conv(x_onehot))  # (B, N1, l_in - 5)
        # Trim extra position from even-parity rounding (if l_in was rounded up)
        if h.shape[-1] > self._h_width:
            h = h[..., :self._h_width]

        # -- Endpoint heads (NB loss) on centre P positions of h ------------
        h_centre = h[:, :, WINDOW_HALF:WINDOW_HALF + P]   # (B, N1, P)
        # endpoint_shape_head must remain OUTSIDE the no_grad block below —
        # moving it inside would silently kill gradients to the shape head.
        endpoint_shape = self.endpoint_shape_head(h_centre)  # (B, 2, P)
        with torch.no_grad():
            disp_bp = self.dispersion_head(h_centre)         # (B, 2, P), frozen

        # -- L2b: interior prefix sum (L-independent, compute once) ----------
        U = self._interior_prefix(x_onehot, 0, P)  # (B, N2, P, n_offsets+1)

        # -- L2a: pre-compute M[g,s] @ h for ALL groups (batched) -----------
        # Stack all 40 M matrices into (40, H, N1), do one batched matmul
        # with h expanded to (40, N1, S), then unpack into dict.
        all_groups = self._L_groups.unique().tolist()
        n_groups = len(all_groups)
        M_list = []
        _cache_keys = []
        for bank_idx in range(4):
            for g in all_groups:
                M_list.append(self._folded_M(bank_idx, g))  # (H, N1)
                _cache_keys.append((bank_idx, g))
        M_stacked = torch.stack(M_list)  # (4*n_groups, H, N1)

        # Batched matmul: (4*n_groups, H, N1) @ (B, N1, S) → per-batch
        # Use einsum to handle the batch dim: 'khn, bns -> kbhs'
        Mh_all = torch.einsum('khn, bns -> kbhs', M_stacked, h)  # (40, B, H, S)

        _Mh_cache = {k: Mh_all[i] for i, k in enumerate(_cache_keys)}

        # -- Chunked forward over L ----------------------------------------
        all_logits = []

        L_vals = self._L_vals       # (n_L,)
        L_groups = self._L_groups   # (n_L,)
        use_checkpoint = self.training and torch.is_grad_enabled()

        n_L = self._n_L
        for L_start_idx in range(0, n_L, L_chunk):
            L_end_idx = min(L_start_idx + L_chunk, n_L)
            chunk_L = L_vals[L_start_idx:L_end_idx]        # (chunk_size,)
            chunk_g = L_groups[L_start_idx:L_end_idx]       # (chunk_size,)

            if use_checkpoint:
                logit_chunk = torch.utils.checkpoint.checkpoint(
                    self._compute_chunk,
                    h, U, _Mh_cache, chunk_L, chunk_g,
                    L_start_idx, L_end_idx, B, P, H,
                    use_reentrant=False,
                )
            else:
                logit_chunk = self._compute_chunk(
                    h, U, _Mh_cache, chunk_L, chunk_g,
                    L_start_idx, L_end_idx, B, P, H,
                )
            all_logits.append(logit_chunk)  # (B, P, chunk_size, 2)

        logits = torch.cat(all_logits, dim=2)  # (B, P, n_L, 2)
        return logits, endpoint_shape, disp_bp

    def _compute_chunk(
        self, h, U, _Mh_cache, chunk_L, chunk_g,
        L_start_idx, L_end_idx, B, P, H,
    ):
        """Compute logits for one L-chunk. Extracted for gradient checkpointing."""
        chunk_size = chunk_L.shape[0]

        # Unique groups in this chunk
        unique_groups = chunk_g.unique()

        # -- L2a: endpoint branch (fused tap×strand gather) ----------------
        # Groups are contiguous within each chunk (L values are sorted, groups
        # are consecutive L ranges). We gather per group and concatenate —
        # no scatter into a pre-allocated buffer, eliminating CopySlices backward.
        # Under start admission the left tap is at WINDOW_HALF + m (independent
        # of L), and the right tap is at WINDOW_HALF + m + L.
        o_l = torch.full_like(chunk_L, WINDOW_HALF)  # (chunk_size,)
        o_r = WINDOW_HALF + chunk_L                   # (chunk_size,)
        p_range = torch.arange(P, device=h.device)

        # Collect per-group gathered results in order, then cat
        gathered_parts = [[] for _ in range(4)]  # one list per bank

        for g in unique_groups.tolist():
            g_mask = (chunk_g == g)
            if not g_mask.any():
                continue

            # Build position indices for left and right taps
            offsets_l = o_l[g_mask]  # (n_g,)
            offsets_r = o_r[g_mask]  # (n_g,)
            pos_l = offsets_l[:, None] + p_range[None, :]  # (n_g, P)
            pos_r = offsets_r[:, None] + p_range[None, :]  # (n_g, P)

            # Left-tap banks (0=left_fwd, 1=left_rev)
            Mh_left = torch.stack([_Mh_cache[(0, g)], _Mh_cache[(1, g)]])
            gathered_l = EndpointGather.apply(Mh_left, pos_l.long())  # (2, B, H, n_g, P)
            # permute n_g to first dim: (n_g, B, H, P)
            gathered_parts[0].append(gathered_l[0].permute(2, 0, 1, 3))
            gathered_parts[1].append(gathered_l[1].permute(2, 0, 1, 3))

            # Right-tap banks (2=right_fwd, 3=right_rev)
            Mh_right = torch.stack([_Mh_cache[(2, g)], _Mh_cache[(3, g)]])
            gathered_r = EndpointGather.apply(Mh_right, pos_r.long())
            gathered_parts[2].append(gathered_r[0].permute(2, 0, 1, 3))
            gathered_parts[3].append(gathered_r[1].permute(2, 0, 1, 3))

        # Concatenate along the L dim (groups are contiguous, so cat = correct order)
        # Each: (chunk_size, B, H, P) → permute to (B, H, P, chunk_size)
        endpoint_feats = [[None, None], [None, None]]
        for tap in (0, 1):
            for strand in (0, 1):
                bank_idx = tap * 2 + strand
                stacked = torch.cat(gathered_parts[bank_idx], dim=0)  # (chunk_size, B, H, P)
                endpoint_feats[tap][strand] = stacked.permute(1, 2, 3, 0)

        # -- L2b span: vectorized gather from prefix sum ----------------
        o_l_prime = o_l - self._o_l_min
        o_r_prime = o_r - self._o_l_min

        U_r = U[:, :, :, o_r_prime.long()]
        U_l = U[:, :, :, o_l_prime.long()]
        span = U_r - U_l

        span_t = span.permute(0, 2, 3, 1)  # (B, P, chunk_size, N2)
        interior_span = self.W_span(span_t)
        L_float = chunk_L.float().to(span.device)
        mean_t = span_t / L_float[None, None, :, None]
        interior_mean = self.W_mean(mean_t)

        # -- Fused pre-activation + nonlinearity + output per strand -----
        # Compute per-strand logits directly without materializing the full
        # (B, P, chunk, H, 2) pre-activation tensor. Each strand is half
        # the size, and we contract over H immediately after the nonlinearity.
        logit_parts = []
        for strand in (0, 1):
            f_l = endpoint_feats[0][strand].permute(0, 2, 3, 1)  # (B, P, chunk, H)
            f_r = endpoint_feats[1][strand].permute(0, 2, 3, 1)
            pre_s = f_l + f_r + interior_span + interior_mean     # (B, P, chunk, H)
            hid_s = self.l3_act(pre_s)                            # (B, P, chunk, H)
            logit_s = (hid_s * self.w_out).sum(dim=-1)            # (B, P, chunk)
            logit_parts.append(logit_s)

        logit_chunk = torch.stack(logit_parts, dim=-1)  # (B, P, chunk, 2)
        lb = self.len_bias[L_start_idx:L_end_idx]
        logit_chunk = logit_chunk + lb[None, None, :, None]

        return logit_chunk

    # -- Dispersion pooling (copied from frozen core) -----------------------

    def _pooled_log_dispersion(
        self, dispersion_bp: torch.Tensor, mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Pool bp-resolution dispersion to NB window level."""
        bp_len = dispersion_bp.shape[-1]
        w = self.hparams.dispersion_window_size
        assert bp_len % w == 0, f"tile size {bp_len} not divisible by dispersion_window_size {w}"
        out_size = bp_len // w
        if mask is not None and mask.dim() == 2:
            mask = mask[:, None, :]
        pooled = masked_mean_pool(dispersion_bp, mask, out_size)
        pooled = pooled + self.hparams.log_dispersion_init
        return pooled

    # -- Lightning interface ------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def _step(self, batch, prefix: str) -> torch.Tensor:
        """Shared train/val step.  Batch from CutSiteFragmentDataset + collate."""
        x = batch["x"]                    # (B, 4, l_in)
        frag_m = batch["frag_m"]          # list of B tensors, each (n_frag,)
        frag_L_idx = batch["frag_L_idx"]  # list of B tensors, each (n_frag,)
        frag_s = batch["frag_s"]          # list of B tensors, each (n_frag,)
        frag_N = batch["frag_N"]          # (B,) total fragment count
        endpoints = batch["endpoints"]    # (B, 2, tile_size)
        mask = batch["mask"]              # (B, tile_size)

        logits, endpoint_shape, disp_bp = self(x)

        # -- Stage 2: add hex-table baseline log p₀ to logits --------------
        if self._use_hex_baseline:
            log_p0 = compute_log_p0(
                x, self.hex_log_start_fwd, self.hex_log_stop_fwd,
                self.hex_log_start_rev, self.hex_log_stop_rev,
                self._log_pL, self.hparams.tile_size,
                self.hparams.L_min, self.hparams.L_max,
            )
            logits = logits + log_p0

        # -- Fragment multinomial NLL ---------------------------------------
        # log_softmax over the full domain D
        B, P, n_L, _ = logits.shape
        flat_logits = logits.reshape(B, -1)  # (B, P * n_L * 2)
        log_probs = torch.log_softmax(flat_logits, dim=-1)
        log_probs_flat = log_probs.reshape(B, P * n_L * 2)

        # Pad variable-length fragment lists to a single tensor for vectorized gather.
        # Flat index: m * (n_L * 2) + L_idx * 2 + s
        max_n = max(f.shape[0] for f in frag_m) if B > 0 else 0
        if max_n > 0:
            flat_idx = torch.zeros(B, max_n, dtype=torch.long, device=x.device)
            frag_mask = torch.zeros(B, max_n, dtype=torch.bool, device=x.device)
            for i in range(B):
                n_i = frag_m[i].shape[0]
                if n_i > 0:
                    flat_idx[i, :n_i] = (
                        frag_m[i] * (n_L * 2) + frag_L_idx[i] * 2 + frag_s[i]
                    )
                    frag_mask[i, :n_i] = True
            # Gather: (B, max_n)
            gathered_lp = log_probs_flat.gather(1, flat_idx)
            gathered_lp = gathered_lp * frag_mask  # zero out padding
            # Per-sample NLL: -sum(lp) / N
            nll_sum = -gathered_lp.sum(dim=1)  # (B,)
            N_float = frag_N.float().clamp(min=1.0)
            L_mult = (nll_sum / N_float).mean()
        else:
            L_mult = torch.tensor(0.0, device=x.device)

        # -- NB endpoint loss (annealed) ------------------------------------
        # _pooled_log_dispersion takes (B, P) or (B, 1, P) mask — pass 2D.
        # The NB loss takes (B, C, P) mask via _prepare_mask.
        log_disp = self._pooled_log_dispersion(disp_bp, mask)
        mask3 = mask[:, None, :].expand_as(endpoints) if mask is not None else None
        L_NB = self.nb_loss_fn(endpoint_shape, log_disp, endpoints, mask3)

        # Annealing weight
        t = self.current_epoch
        T_anneal = self.hparams.nb_T_anneal
        w0 = self.hparams.nb_w0
        w_t = w0 * max(0.0, 1.0 - t / T_anneal) if T_anneal > 0 else 0.0

        mw = self.hparams.mult_weight
        loss = mw * L_mult + w_t * L_NB

        self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}_mult_nll", L_mult, sync_dist=True)
        self.log(f"{prefix}_nb_loss", L_NB, sync_dist=True)
        self.log(f"{prefix}_nb_weight", w_t, sync_dist=True)
        return loss

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=lr,
        )
        factor = self.hparams.lr_factor
        max_red = self.hparams.max_lr_reductions
        patience = self.hparams.lr_patience
        min_lrs = [
            g["lr"] * factor ** max_red for g in optimizer.param_groups
        ]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor,
            patience=patience, min_lr=min_lrs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
