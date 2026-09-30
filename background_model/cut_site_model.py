"""Cut-site fragment model — per-fragment (m, L, strand) logits.

Implements the architecture described in docs/pending/cut_site_fragment_model.md.
The model emits one logit per (centre m, length L, strand s) triple and trains
on the fragment multinomial.  It joins the bake-off alongside the 12-track
model class but does NOT subclass any existing model or mixin.

Consumes primitives from the frozen background_model_core.py (losses, masking,
dispersion pooling) without modifying them.
"""

from __future__ import annotations

from typing import Optional, Tuple

import lightning as L
import torch
import torch.nn as nn

from background_model.tracks import FL_BANDS
from background_model_core import (
    MaskedNegativeBinomialOffsetNLLLoss,
    masked_mean_pool,
)

# ---------------------------------------------------------------------------
# Window geometry (fixed)
# ---------------------------------------------------------------------------

WINDOW_HALF = 128                                            # W/2

# Length groups: 10, half-open, 15 wide from L_min with the last group wider
N_LENGTH_GROUPS = 10
LENGTH_GROUP_WIDTH = 15

# ---------------------------------------------------------------------------
# Default length bounds (from FL_BANDS, but overridable)
# ---------------------------------------------------------------------------

DEFAULT_L_MIN = min(lo for lo, hi in FL_BANDS)               # 25
DEFAULT_L_MAX = max(hi for lo, hi in FL_BANDS) - 1           # 179 (half-open)


def _compute_geometry(L_min: int, L_max: int):
    """Derive all geometry from explicit length bounds."""
    n_L = L_max - L_min + 1
    o_l_min = WINDOW_HALF - L_max // 2
    o_r_max = o_l_min + L_max
    n_offsets = L_max  # live width = L_max
    return n_L, o_l_min, o_r_max, n_offsets


def length_to_group(L: torch.Tensor, L_min: int) -> torch.Tensor:
    """Map fragment length to group index: g = min((L - L_min) // 15, 9)."""
    return torch.clamp((L - L_min) // LENGTH_GROUP_WIDTH,
                        max=N_LENGTH_GROUPS - 1)


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
    ):
        super().__init__()
        self.save_hyperparameters()

        # -- Geometry derived from L_min / L_max ----------------------------
        n_L, o_l_min, o_r_max, n_offsets = _compute_geometry(L_min, L_max)
        self._n_L = n_L
        self._o_l_min = o_l_min
        self._n_offsets = n_offsets

        W = 2 * WINDOW_HALF  # 256
        self._model_input_size = tile_size + W  # 2304

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
        self.w_out = nn.Parameter(torch.randn(H) * (H ** -0.5))
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
        return self._model_input_size

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
        # The endpoint branch reads h at index (o_l + scored_m) where
        # o_l = WINDOW_HALF - L//2.  h[j] = conv(x_onehot[j:j+6]), so
        # h-position j starts at x_onehot[j].  The interior branch must
        # read x_onehot at the same h-aligned positions: for re-indexed
        # offset o' (absolute offset o = o' + O_L_MIN), x_onehot position
        # is scored_m + O_L_MIN + o'.

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

        # Exclusive prefix sum along the offset axis (last dim)
        # U[k] = sum_{o'<k} u[o'], so U[0] = 0 and U[n_offsets] = sum(u)
        U = torch.zeros(B, N2, m_count, n_offsets + 1,
                         device=u.device, dtype=u.dtype)
        U[..., 1:] = torch.cumsum(u, dim=-1)

        return U  # (B, N2, m_count, n_offsets + 1)

    # -- Forward pass -------------------------------------------------------

    def forward(
        self, x_onehot: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_onehot: (B, 4, l_in) one-hot sequence

        Returns:
            logits: (B, P, n_L, 2) per-(centre, length, strand) logits
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
        if h.shape[-1] > self._model_input_size:
            h = h[..., :self._model_input_size]

        # -- Endpoint heads (NB loss) on centre P positions of h ------------
        h_centre = h[:, :, WINDOW_HALF:WINDOW_HALF + P]   # (B, N1, P)
        # endpoint_shape_head must remain OUTSIDE the no_grad block below —
        # moving it inside would silently kill gradients to the shape head.
        endpoint_shape = self.endpoint_shape_head(h_centre)  # (B, 2, P)
        with torch.no_grad():
            disp_bp = self.dispersion_head(h_centre)         # (B, 2, P), frozen

        # -- L2b: interior prefix sum (L-independent, compute once) ----------
        U = self._interior_prefix(x_onehot, 0, P)  # (B, N2, P, n_offsets+1)

        # -- L2a: pre-compute folded M matrices per (bank, group) ------------
        # Store M matrices (not M@h) — we gather h slices first, then matmul.
        # This avoids the expensive IndexBackward0 on the large Mh tensor.
        _M_cache = {}
        all_groups = self._L_groups.unique().tolist()
        for bank_idx in range(4):
            for g in all_groups:
                _M_cache[(bank_idx, g)] = self._folded_M(bank_idx, g)  # (H, N1)

        # -- Chunked forward over L ----------------------------------------
        all_logits = []

        L_vals = self._L_vals       # (n_L,)
        L_groups = self._L_groups   # (n_L,)
        p_range = torch.arange(P, device=h.device)  # (P,) reused across chunks

        n_L = self._n_L
        for L_start_idx in range(0, n_L, L_chunk):
            L_end_idx = min(L_start_idx + L_chunk, n_L)
            chunk_L = L_vals[L_start_idx:L_end_idx]        # (chunk_size,)
            chunk_g = L_groups[L_start_idx:L_end_idx]       # (chunk_size,)
            chunk_size = chunk_L.shape[0]

            # Unique groups in this chunk
            unique_groups = chunk_g.unique()

            # -- L2a: endpoint branch (einsum fusion + fused tap×strand) ----
            # Instead of M@h → gather positions (expensive IndexBackward),
            # we gather h slices first → then apply M (cheap mm backward).
            o_l = WINDOW_HALF - chunk_L // 2       # (chunk_size,)
            o_r = o_l + chunk_L                     # (chunk_size,)

            # Build position indices for both taps: (2, chunk_size, P)
            # tap 0 = left (o_l), tap 1 = right (o_r)
            pos_left = o_l[:, None] + p_range[None, :]   # (chunk_size, P)
            pos_right = o_r[:, None] + p_range[None, :]  # (chunk_size, P)

            # Fuse tap×strand: compute all 4 endpoint features per group
            # in one pass, avoiding 4 separate IndexBackward ops.
            # endpoint_feats[tap][strand] = (B, P, chunk_size, H)
            endpoint_feats = [[None, None], [None, None]]
            for g in unique_groups.tolist():
                g_mask = (chunk_g == g)
                if not g_mask.any():
                    continue
                g_indices = g_mask.nonzero(as_tuple=True)[0]
                n_g = g_indices.shape[0]

                # Gather h slices at left and right tap positions for this group
                # h is (B, N1, model_input_size)
                pos_l_g = pos_left[g_mask]   # (n_g, P)
                pos_r_g = pos_right[g_mask]  # (n_g, P)

                # Single gather for both taps: (2, n_g, P) positions
                # → h_slices: (2, B, N1, n_g, P) via one fancy-index op
                pos_both = torch.stack([pos_l_g, pos_r_g])  # (2, n_g, P)
                h_slices = h[:, :, pos_both.reshape(-1).long()].reshape(
                    B, -1, 2, n_g, P
                )  # (B, N1, 2, n_g, P)

                # Apply M for all 4 banks: M[bank] @ h_slices → (B, H, 2, n_g, P)
                for tap in (0, 1):
                    for strand in (0, 1):
                        bank_idx = tap * 2 + strand
                        M_gs = _M_cache[(bank_idx, g)]  # (H, N1)
                        # h_tap: (B, N1, n_g, P) → einsum with M → (B, H, n_g, P)
                        h_tap = h_slices[:, :, tap, :, :]  # (B, N1, n_g, P)
                        feat = torch.einsum('hn, bnlp -> bhlp', M_gs, h_tap)
                        # Scatter into result
                        if endpoint_feats[tap][strand] is None:
                            endpoint_feats[tap][strand] = torch.empty(
                                B, H, chunk_size, P, device=h.device, dtype=h.dtype
                            )
                        endpoint_feats[tap][strand][:, :, g_indices, :] = feat

            # Permute to (B, H, P, chunk_size) to match pre-activation assembly
            for tap in (0, 1):
                for strand in (0, 1):
                    endpoint_feats[tap][strand] = endpoint_feats[tap][strand].permute(
                        0, 1, 3, 2
                    )  # (B, H, chunk_size, P) → (B, H, P, chunk_size)

            # -- L2b span: vectorized gather from prefix sum ----------------
            o_l_prime = o_l - self._o_l_min   # (chunk_size,)
            o_r_prime = o_r - self._o_l_min   # (chunk_size,)

            # U is (B, N2, P, n_offsets+1). Gather at o_r' and o_l' for all
            # chunk positions at once.
            U_r = U[:, :, :, o_r_prime.long()]  # (B, N2, P, chunk_size)
            U_l = U[:, :, :, o_l_prime.long()]  # (B, N2, P, chunk_size)
            span = U_r - U_l                    # (B, N2, P, chunk_size)

            # Project span and mean = span / L
            # W_span: (N2) → (H), W_mean: (N2) → (H)
            # span is (B, N2, P, chunk_size), need to apply linear on dim=1
            span_t = span.permute(0, 2, 3, 1)  # (B, P, chunk_size, N2)
            interior_span = self.W_span(span_t)  # (B, P, chunk_size, H)
            L_float = chunk_L.float().to(span.device)  # (chunk_size,)
            mean_t = span_t / L_float[None, None, :, None]
            interior_mean = self.W_mean(mean_t)   # (B, P, chunk_size, H)

            # -- Assemble pre-activation per (m, L, strand) -----------------
            # pre[h,m,L,s] = f_l[s][:,h,m,L] + f_r[s][:,h,m,L]
            #              + interior_span[:,m,L,h] + interior_mean[:,m,L,h]
            # Shape target: (B, H, P, chunk_size, 2) for the 2 strands

            pre = torch.zeros(B, P, chunk_size, H, 2, device=h.device, dtype=h.dtype)
            for strand in (0, 1):
                # endpoint_feats[tap][strand] is (B, H, P, chunk_size)
                f_l = endpoint_feats[0][strand].permute(0, 2, 3, 1)  # (B, P, chunk_size, H)
                f_r = endpoint_feats[1][strand].permute(0, 2, 3, 1)
                pre[:, :, :, :, strand] = (
                    f_l + f_r + interior_span + interior_mean
                )

            # -- L3: nonlinearity + output ----------------------------------
            hid = self.l3_act(pre)  # (B, P, chunk_size, H, 2)
            # logit = w_out · hid + len_bias[L]
            # w_out: (H,), dot over H dim
            logit_chunk = torch.einsum('bplhs, h -> bpls', hid, self.w_out)
            # Add len_bias: (chunk_size,) broadcast over (B, P, ·, 2)
            lb = self.len_bias[L_start_idx:L_end_idx]  # (chunk_size,)
            logit_chunk = logit_chunk + lb[None, None, :, None]

            all_logits.append(logit_chunk)  # (B, P, chunk_size, 2)

        logits = torch.cat(all_logits, dim=2)  # (B, P, n_L, 2)
        return logits, endpoint_shape, disp_bp

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

        loss = L_mult + w_t * L_NB

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
