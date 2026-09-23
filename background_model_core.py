"""Core model for the cfDNA fragment-endpoint background ("bias") model, v2.

Distilled from ``fragmentomics_tools/bias_correction/{model,layers,loss}.py``
per BIAS_CORRECTION_REVIEW.md, then revised through design discussion
(2026-08-25).  This file is the record of that discussion and its result.

Design summary (what was decided, and why)
==========================================

**Purpose.** This is a *technical-bias* null, not a healthy-population null:
it models how DNA sequence alone shapes fragment-endpoint profiles — and how
much those profiles vary between samples — in regions with no active
regulation.  Genetic variants and regulatory activity are deliberately
EXCLUDED from the null: they are the residuals we want to detect.  Two
consequences:

- The model is trained on inactive regions only (definition of "inactive" is
  a pending decision), and
- all parameters are indexed by *sequence*, never by locus, so the null
  generalizes to regions never seen in training.  Locus-indexed empirical
  dispersion was considered and rejected: at active loci it would absorb the
  biology into the null.

**Scale / the role of N.**  The observed total
``N[s,t,c] = sum of counts for sample s, tile t, track c`` (over unmasked
positions) is the plug-in scale.  Sequence cannot predict depth, copy
number, or tile-level accessibility, so the network never models absolute
counts: N enters as a fixed offset (``log mu_i = log N + log p_i``, NB view)
or equivalently as the multinomial total (conditioning view).  These are the
same thing: for independent NBs the likelihood factorizes as
``P(N) * P(x|N)`` where ``P(x|N)`` is Dirichlet-multinomial; conditioning
just drops the nuisance factor exactly.  v1's NB failure was not the NB
family — it was asking the network to supply this scale
(``total_count=1000``, sigmoid-bounded means).

**Variance.**  Between-sample profile variance is real (accessibility varies
across people) and is predicted from sequence by a second head, amortized
across the genome — identifiable because the same sequence function is fit
across millions of positions and many samples (NOT one free parameter per
position).  This requires **per-sample training targets**: merged counts
contain a single realization and carry no between-sample variance
information (v1's fatal flaw).

**Result: one trunk, two heads, three switchable likelihoods** (the two
overdispersed ones are near-equivalent by the factorization above — both are
implemented so they can be tested empirically):

1. ``multinomial`` — baseline, gamma -> infinity (no between-sample variance).
2. ``dirichlet_multinomial`` — exact conditional likelihood; one
   concentration gamma per (tile, track), pooled from the dispersion head.
   gamma is a pseudo-count: "the sequence prediction is worth gamma
   fragments of evidence"; Var[share of window w] = p_w(1-p_w)/(gamma+1).
3. ``nb_offset`` — independent NB2 per position with observed-N offset,
   per-window dispersion (default 256 bp) from the same head.  A
   pseudo-likelihood (ignores the fixed-N constraint, which is physically
   negligible: fragments do not meaningfully compete within a tile), but
   allows dispersion to vary at sub-tile scale, unlike the exact DM whose
   single gamma is tile-wide.  This scale difference is part of what the
   empirical comparison should settle.

**Evaluation plan.**  On held-out inactive regions x held-out samples:
window-level tail p-values (``beta_binomial_window_pvalues`` /
``nb_window_pvalues``) must be uniform (QQ) per track and per accessibility
stratum.  Positive controls (CTCF sites, immune/epithelial marker genes —
v1's test sets) should show strong deviations.  Conditioning on N makes the
test blind to coverage-level and fragment-length-band-ratio signals by
construction; those are separate statistics, out of scope here.

Other retained decisions
------------------------
- One track-naming scheme: ``strand_{s}__fl_{lo}_{hi}__coverage_{type}``.
- Standard Lightning checkpointing via ``save_hyperparameters()``.
- Unpadded convolutions: the model consumes exactly
  ``calc_input_region_size(L_out)`` bases to emit ``L_out`` positions.
- Augmentation is first-class: tiles stored with margin; ``jitter_matrix``
  crops shifted windows at load time; ``reverse_complement_track_permutation``
  gives the channel permutation for RC augmentation.
- Blacklist positions are masked out of the likelihood (excluded from both
  N and the softmax support), not merely zeroed in the targets.

Not in this file (next): dataset materialization / plumbing.  The store must
carry per-sample counts, and low-N (sample, tile, track) terms should be
excluded from training by a configurable minimum-count threshold.
"""

import re
from typing import List, Optional, Tuple, Type, Union

import lightning as L
import numpy as np
import scipy.sparse
import torch
from scipy.sparse import coo_matrix
from scipy.stats import betabinom, nbinom

from fragmentomics_tools.region import Region


# --------------------------------------------------------------------------
# Track naming
# --------------------------------------------------------------------------

STRANDS = ("+", "-")
FL_BANDS = ((40, 65), (120, 175))  # short (TF footprint) / mononucleosomal
COVERAGE_TYPES = ("first", "last", "midpoint")

_TRACK_PAT = re.compile(
    r"strand_([+-.])__fl_(\d+)_(\d+)__coverage_(first|last|midpoint)"
)


def index_key_to_track_name(strand: str, fl_band: Tuple[int, int], coverage_type: str) -> str:
    return f"strand_{strand}__fl_{fl_band[0]}_{fl_band[1]}__coverage_{coverage_type}"


def track_name_to_index_key(track_name: str):
    m = _TRACK_PAT.fullmatch(track_name)
    if m is None:
        raise ValueError(
            f"track name '{track_name}' does not match '{_TRACK_PAT.pattern}'"
        )
    strand, lo, hi, cov = m.groups()
    return strand, (int(lo), int(hi)), cov


DEFAULT_OUTPUT_TRACKS: List[str] = [
    index_key_to_track_name(s, fl, c)
    for s in STRANDS
    for fl in FL_BANDS
    for c in COVERAGE_TYPES
]


def reverse_complement_track_permutation(output_tracks: List[str]) -> List[int]:
    """Index permutation mapping each track to its reverse-complement partner.

    Under reverse complement, genomic coordinates reverse: + and - strands
    swap, and the first/last covered base swap (midpoint maps to itself; the
    fragment-length band is unchanged).  Usage on a target/logit array of
    shape (..., n_tracks, L):  ``y_rc = y[..., perm, ::-1]``.
    """
    swap_strand = {"+": "-", "-": "+", ".": "."}
    swap_cov = {"first": "last", "last": "first", "midpoint": "midpoint"}
    perm = []
    for t in output_tracks:
        s, fl, c = track_name_to_index_key(t)
        partner = index_key_to_track_name(swap_strand[s], fl, swap_cov[c])
        try:
            perm.append(output_tracks.index(partner))
        except ValueError:
            raise ValueError(
                f"track '{t}' has no reverse-complement partner '{partner}' in "
                "output_tracks; RC augmentation requires a strand/coverage-"
                "symmetric track set"
            ) from None
    return perm


# --------------------------------------------------------------------------
# Jitter (window-shift augmentation and residual cropping)
# --------------------------------------------------------------------------


def calculate_start_and_stop_from_jitter(
    input_length: int,
    output_length: int,
    jitter_value: int,
    strand: Optional[str] = None,
):
    """Slice coordinates for an ``output_length`` window shifted ``jitter_value``
    bp from the (strand-aware) center of an ``input_length`` array.

    Only defined for input/output lengths of the same parity.  With
    ``jitter_value=0`` this is a plain center crop.
    """
    if input_length < output_length + abs(jitter_value):
        raise ValueError(
            f"input array (len {input_length}) is not wide enough for an output "
            f"of length {output_length} with jitter_value {jitter_value}"
        )
    resize_start = Region.get_resize_start(
        start=0, current_size=input_length, new_size=output_length, strand=strand
    )
    jitter_start = resize_start + jitter_value
    return jitter_start, jitter_start + output_length


def jitter_matrix(
    input_arr,
    jitter_value: int,
    output_length: int,
    strand: Optional[str] = None,
):
    """Crop a jittered window from the last axis of a dense or sparse array.

    This is the load-time augmentation primitive: tiles are stored with a
    margin, and the loader draws ``jitter_value`` (0 for val/test) and crops.
    Accepts numpy arrays, torch tensors, and scipy COO sparse matrices.
    """
    input_length = input_arr.shape[-1]
    if round(jitter_value) != jitter_value:
        raise ValueError(f"jitter_value must be a whole number, got {jitter_value}")

    new_start, new_stop = calculate_start_and_stop_from_jitter(
        input_length=input_length,
        output_length=output_length,
        jitter_value=jitter_value,
        strand=strand,
    )
    assert new_stop - new_start == output_length
    assert 0 <= new_start and new_stop <= input_length

    if isinstance(input_arr, (np.ndarray, torch.Tensor)):
        return input_arr[..., new_start:new_stop]
    elif scipy.sparse.issparse(input_arr):
        indices = np.where(
            (input_arr.col >= new_start) & (input_arr.col < new_stop)
        )[0]
        return coo_matrix(
            (
                input_arr.data[indices],
                (input_arr.row[indices], input_arr.col[indices] - new_start),
            ),
            shape=(input_arr.shape[0], output_length),
            dtype=input_arr.dtype,
        )
    else:
        raise TypeError(f"input_arr type {type(input_arr)} is invalid")


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------


class SpatialDropout(torch.nn.Module):
    """Drop whole channels (keras SpatialDropout1D, pytorch channel order)."""

    def __init__(self, p: float = 0.2):
        super().__init__()
        # Dropout1d on (B, C, L) zeroes whole channels.  (Dropout2d on the
        # (B, L, C)-permuted input treated L as the channel dim under
        # torch >= 1.12's 3D semantics, i.e. dropped positions, not channels.)
        self.dropout = torch.nn.Dropout1d(p)

    def forward(self, x):
        if not self.training:
            return x
        return self.dropout(x)


class ResNetDilatedBlock(torch.nn.Module):
    """Dilated residual conv block (channels in == channels out).

    Config used by all v1 trained checkpoints: ``activation=LeakyReLU,
    activation_post_sum=True, skip_batchnorm=True,
    preact_residual_normalization=False, padding=0`` — these are the defaults
    ``BackgroundModel`` passes.  The other configurations are retained for
    experimentation (e.g., normalization for deeper stacks).

    :param input_channels: conv in == out channel count.
    :param profile_kernel_size: kernel size; must be odd unless the dilation
        rate is even, so the receptive-field trim is symmetric.
    :param dilation_rate: conv dilation.
    :param activation: activation module class.
    :param activation_post_sum: apply activation after the residual add
        (standard pytorch ResNet) rather than before it.
    :param skip_batchnorm: if True, no BatchNorm after the conv.  Note that
        batch statistics across genomic tiles leak inter-region information
        into a model that should be a pure function of local sequence; prefer
        skipping BN (or using GroupNorm) unless depth demands it.
    :param preact_residual_normalization: pre-activation BN+activation before
        the conv, to limit variance accumulation in deep networks.  See
        https://iclr-blog-track.github.io/2022/03/25/unnormalized-resnets/#moment-control
    :param padding: "same" preserves length; 0 (unpadded) trims the receptive
        field and center-crops the residual to match (via ``jitter_matrix``
        with ``jitter_value=0``).
    """

    def __init__(
        self,
        input_channels: int = 64,
        profile_kernel_size: int = 20,
        dilation_rate: int = 1,
        activation: Type[torch.nn.Module] = torch.nn.ReLU,
        activation_post_sum: bool = False,
        skip_batchnorm: bool = False,
        preact_residual_normalization: bool = False,
        padding: Union[str, int] = "same",
    ):
        super().__init__()
        assert profile_kernel_size % 2 == 1 or dilation_rate % 2 == 0, (
            f"kernel size must be odd unless dilation is even "
            f"(got kernel {profile_kernel_size}, dilation {dilation_rate})"
        )
        self.conv1 = torch.nn.Conv1d(
            in_channels=input_channels,
            out_channels=input_channels,
            stride=1,
            kernel_size=profile_kernel_size,
            padding=(dilation_rate * (profile_kernel_size - 1)) // 2
            if padding == "same"
            else padding,
            dilation=dilation_rate,
        )
        self.activation_post_sum = activation_post_sum
        self.skip_batchnorm = skip_batchnorm
        self.bn = None if skip_batchnorm else torch.nn.BatchNorm1d(input_channels)
        self.preact_residual_normalization = preact_residual_normalization
        self.bn_preact = (
            torch.nn.BatchNorm1d(input_channels)
            if preact_residual_normalization
            else None
        )
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.preact_residual_normalization:
            x = self.activation(self.bn_preact(x))
        out = self.conv1(x)
        if not self.skip_batchnorm:
            out = self.bn(out)
        if out.shape[-1] < x.shape[-1]:
            # unpadded conv shrank the output: center-crop the residual
            x = jitter_matrix(x, jitter_value=0, output_length=out.shape[-1])
        if self.activation_post_sum:
            return self.activation(out + x)
        return self.activation(out) + x


# --------------------------------------------------------------------------
# K-mer index computation (used by BackgroundModelKEN)
# --------------------------------------------------------------------------


def rc_kmer_permutation(k: int) -> np.ndarray:
    """Fixed permutation mapping each k-mer index to its RC partner.

    Complement in 2-bit code is (3 - code) (A0<->T3, C1<->G2); RC also
    reverses base order.  Involution: perm[perm[i]] == i.
    """
    n = 4 ** k
    idx = np.arange(n, dtype=np.int64)
    codes = np.zeros((n, k), dtype=np.int64)
    rem = idx.copy()
    for j in range(k):
        codes[:, k - 1 - j] = rem % 4
        rem //= 4
    rc_codes = (3 - codes)[:, ::-1]
    powers = 4 ** np.arange(k - 1, -1, -1)
    return (rc_codes * powers).sum(axis=1)


def one_hot_to_kmer_indices(one_hot: torch.Tensor, k: int,
                            powers: torch.Tensor) -> torch.Tensor:
    """Convert one-hot DNA (B, 4, L) to integer k-mer indices (B, L-k+1).

    Each position i in the output gets the k-mer centered on it (after
    the dataset's center-crop ensures the correct input offset). The
    k-mer is encoded as a base-4 big-endian integer, matching the
    simulation's hexamer_indices encoding.
    """
    base_idx = one_hot.argmax(dim=1)
    patches = base_idx.unfold(dimension=1, size=k, step=1)
    return (patches * powers).sum(dim=-1)


# --------------------------------------------------------------------------
# Losses
#
# All three share the same masked-softmax shape convention:
#   shape_logits, target: (B, C, L);  mask: (B, L) or (B, 1, L) bool
# Masked positions are excluded from the softmax support and from N; targets
# must be zero there.  All losses normalize the NLL by N so tiles of
# different depth contribute comparably.
# --------------------------------------------------------------------------


def _prepare_mask(mask: Optional[torch.Tensor], target: torch.Tensor):
    if mask is None:
        return None
    if mask.dim() == 2:
        mask = mask[:, None, :]
    assert not target.masked_select(~mask).any(), (
        "targets must be zero at masked positions"
    )
    return mask


def masked_mean_pool(
    x: torch.Tensor, mask: Optional[torch.Tensor], out_size: int
) -> torch.Tensor:
    """Mean-pool (B, C, L) -> (B, C, out_size), counting only valid positions.

    mask: (B, 1, L) bool or None.  Windows that are fully masked get the
    pool of zero contributions (their likelihood terms are masked anyway).
    """
    B, C, L = x.shape
    assert L % out_size == 0, f"L={L} not divisible by out_size={out_size}"
    k = L // out_size
    if mask is None:
        return x.reshape(B, C, out_size, k).mean(dim=-1)
    m = mask.to(x.dtype)
    num = (x * m).reshape(B, C, out_size, k).sum(dim=-1)
    den = m.reshape(B, 1, out_size, k).sum(dim=-1)
    return num / den.clamp(min=1.0)


class MaskedMultinomialNLLLoss(torch.nn.Module):
    """Baseline (gamma -> infinity): per-track multinomial NLL over positions.

    The combinatorial term log(N! / prod x_i!) is constant w.r.t. parameters
    and omitted; gradients are identical to the full NLL.
    """

    def forward(
        self,
        shape_logits: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = _prepare_mask(mask, target)
        if mask is not None:
            shape_logits = shape_logits.masked_fill(~mask, float("-inf"))
        logp = torch.log_softmax(shape_logits, dim=-1)
        if mask is not None:
            logp = logp.masked_fill(~mask, 0.0)  # avoid 0 * -inf -> nan
        totals = target.sum(dim=-1).clamp(min=1.0)
        nll = -(target * logp).sum(dim=-1) / totals
        return nll.mean()


class MaskedDirichletMultinomialNLLLoss(torch.nn.Module):
    """Exact conditional likelihood: x | N ~ DirMult(N, gamma * p) per track.

    log_concentration: (B, C) — one log(gamma) per (tile, track).  With
    alpha_i = gamma * p_i and sum_valid(p_i) = 1:

        log P(x|N) = lgamma(gamma) - lgamma(N + gamma)
                     + sum_valid [ lgamma(x_i + alpha_i) - lgamma(alpha_i) ]

    (dropping the x-only combinatorial constant).  Zero-count positions
    contribute exactly 0 to the sum, so the loss is naturally sparse.
    Masked positions use a safe alpha=1 inside lgamma (their terms are
    exactly 0 and carry no gradient) to avoid nan/inf gradient leaks from
    lgamma near 0.
    """

    def forward(
        self,
        shape_logits: torch.Tensor,
        log_concentration: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = _prepare_mask(mask, target)
        if mask is not None:
            shape_logits = shape_logits.masked_fill(~mask, float("-inf"))
        logp = torch.log_softmax(shape_logits, dim=-1)  # (B, C, L)

        gamma = log_concentration.exp()  # (B, C)
        log_alpha = logp + log_concentration[..., None]
        if mask is not None:
            # safe value at masked positions: x=0, alpha=1 -> term == 0, and
            # masked_fill cuts the gradient path (avoids digamma(0) * 0 = nan)
            log_alpha = log_alpha.masked_fill(~mask, 0.0)
        alpha = log_alpha.exp()

        N = target.sum(dim=-1)  # (B, C)
        pos_terms = torch.lgamma(target + alpha) - torch.lgamma(alpha)
        if mask is not None:
            pos_terms = pos_terms.masked_fill(~mask, 0.0)
        ll = torch.lgamma(gamma) - torch.lgamma(N + gamma) + pos_terms.sum(dim=-1)
        nll = -ll / N.clamp(min=1.0)
        return nll.mean()


class _DispersionClamp(torch.autograd.Function):
    """Hard value clamp with soft gradient scaling for dispersion stability.

    Forward: ``max(log_r, log_r_floor)`` (hard floor on dispersion).
    Backward: gradient to ``log_r`` is scaled by ``sigmoid((log_r -
    log_r_floor) / margin)``, giving a smooth transition from full gradient
    (well above floor) to zero gradient (at/below floor).  The complement
    flows to ``log_r_floor`` (and thus to the shape head via the chain rule).
    """

    @staticmethod
    def forward(ctx, log_r, log_r_floor, margin):
        ctx.save_for_backward(log_r, log_r_floor)
        ctx.margin = margin
        return torch.maximum(log_r, log_r_floor)

    @staticmethod
    def backward(ctx, grad_output):
        log_r, log_r_floor = ctx.saved_tensors
        scale = torch.sigmoid((log_r - log_r_floor) / ctx.margin)
        return grad_output * scale, grad_output * (1 - scale), None


class MaskedNegativeBinomialOffsetNLLLoss(torch.nn.Module):
    """Pseudo-likelihood: independent NB2 per position with observed-N offset.

    mu_i = N * p_i (N observed, p from the masked softmax — the offset view:
    log mu_i = log N + log p_i with the log N coefficient fixed at 1).
    Dispersion r is per window: log_dispersion (B, C, W) broadcast to L
    (requires L % W == 0).  Var_i = mu_i + mu_i^2 / r_i.

    torch's NegativeBinomial convention: mean = total_count * exp(logits),
    so logits = log mu - log r with total_count = r.

    Dispersion clamping (``max_dispersion_ratio``):  when set, ``log_r`` is
    clamped so that Var_NB <= ratio * Var_multinomial at every position.  This
    prevents the dispersion head from swinging wildly at high-count positions,
    stabilising training without removing the overdispersion signal.  The
    clamp uses soft gradient scaling (sigmoid transition over ``clamp_margin``
    nats) so gradients taper smoothly near the floor rather than switching
    abruptly between full and zero.
    """

    def __init__(
        self,
        max_dispersion_ratio: Optional[float] = 2.0,
        clamp_margin: float = 1.0,
    ):
        super().__init__()
        self.max_dispersion_ratio = max_dispersion_ratio
        self.clamp_margin = clamp_margin

    def forward(
        self,
        shape_logits: torch.Tensor,
        log_dispersion: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, L = shape_logits.shape
        W = log_dispersion.shape[-1]
        assert L % W == 0, f"L={L} not divisible by n dispersion windows W={W}"

        mask = _prepare_mask(mask, target)
        if mask is not None:
            shape_logits = shape_logits.masked_fill(~mask, float("-inf"))
        logp = torch.log_softmax(shape_logits, dim=-1)

        N = target.sum(dim=-1)  # (B, C)
        log_mu = torch.log(N.clamp(min=1.0))[..., None] + logp  # -inf at masked

        # -- dispersion clamping -------------------------------------------
        # Clamp r so Var_NB = mu + mu²/r  <=  ratio * Var_multi = ratio*mu*(1-p)
        #   => r >= mu / (ratio*(1-p) - 1)
        # Computed per position, pooled to window level via max (tightest
        # constraint), then applied as a floor on log_dispersion.
        if self.max_dispersion_ratio is not None:
            with torch.no_grad():
                p = logp.exp()                    # (B, C, L)
                mu = log_mu.exp()                 # 0 at masked positions
                denom = (self.max_dispersion_ratio * (1.0 - p) - 1.0).clamp(min=0.01)
                r_floor_pos = mu / denom          # (B, C, L)
                if mask is not None:
                    r_floor_pos = r_floor_pos.masked_fill(~mask, 0.0)
                # Max over positions in each window — tightest constraint
                r_floor_win = r_floor_pos.reshape(B, C, W, L // W).max(dim=-1).values
                log_r_floor = torch.log(r_floor_win.clamp(min=1e-6))
            log_dispersion = _DispersionClamp.apply(
                log_dispersion, log_r_floor, self.clamp_margin
            )

        log_r = log_dispersion.repeat_interleave(L // W, dim=-1)
        nb_logits = log_mu - log_r
        if mask is not None:
            # safe finite value at masked positions; terms are zeroed below
            nb_logits = nb_logits.masked_fill(~mask, 0.0)

        dist = torch.distributions.NegativeBinomial(
            total_count=log_r.exp(), logits=nb_logits, validate_args=False
        )
        nll = -dist.log_prob(target)
        if mask is not None:
            nll = nll.masked_fill(~mask, 0.0)
        return (nll.sum(dim=-1) / N.clamp(min=1.0)).mean()


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

LOSSES = ("multinomial", "dirichlet_multinomial", "nb_offset")


def _with_lr_schedule(optimizer, hparams):
    """Wrap an optimizer with ReduceLROnPlateau, returning the Lightning dict.

    Computes per-group ``min_lr`` from each group's initial LR so that the
    inter-group ratios (e.g. ``dispersion_lr_scale``) are preserved at the
    floor.  A scalar ``min_lr`` would flatten all groups to a single value
    and silently destroy the ratio — see design §3.1 / §5.0.

    **Known behaviour at the floor:** once all param groups sit at their
    ``min_lr``, ``ReduceLROnPlateau`` still fires its (no-op) reduction and
    **resets ``num_bad_epochs`` to 0**, cycling indefinitely.  The LR never
    changes and nothing is reported — the floor silently absorbs the event.
    This matters for Phase 3's divergence recovery: a floor cannot report a
    refusal, so recovery must use an explicit reduction counter rather than
    relying on ``min_lr`` to cap reductions.
    """
    factor = hparams.lr_factor
    max_reductions = hparams.max_lr_reductions
    patience = hparams.lr_patience
    min_lrs = [
        g["lr"] * factor ** max_reductions for g in optimizer.param_groups
    ]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        min_lr=min_lrs,
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_loss",
        },
    }


class BackgroundModel(L.LightningModule):
    """Sequence -> per-position profile logits (+ per-window dispersion).

    Input:  one-hot sequence (B, 4, L_in)
    Output: shape logits (B, n_tracks, L_out), and (unless
            loss == "multinomial") a dispersion output pooled per loss type;
            L_in == calc_input_region_size(L_out).

    Batch convention (per-sample training): each batch element is one
    (sample, tile) pair — ``(x, y, mask)`` with x the one-hot sequence
    (shared across samples of the same tile), y that SAMPLE'S counts
    (B, C, L_out), and mask the valid-position mask (B, L_out).  Merged
    counts must not be used as targets: they carry no between-sample
    variance information.
    """

    def __init__(
        self,
        output_tracks: Optional[List[str]] = None,
        n_kernels: int = 512,
        kernel_size: int = 32,
        num_residual_layers: int = 2,
        dropout: float = 0.15,
        learning_rate: float = 1e-4,
        loss: str = "dirichlet_multinomial",
        dispersion_window_size: int = 256,
        log_dispersion_init: float = 7.0,
        max_dispersion_ratio: Optional[float] = 2.0,
        clamp_margin: float = 1.0,
        freeze_dispersion: bool = False,
        dispersion_lr_scale: float = 1.0,
        lr_patience: int = 4,
        max_lr_reductions: int = 3,
        lr_factor: float = 0.5,
        block_kwargs: Optional[dict] = None,
    ):
        """
        :param loss: one of ``multinomial`` (baseline, no dispersion head),
            ``dirichlet_multinomial`` (exact; tile-level gamma), ``nb_offset``
            (pseudo-likelihood; window-level dispersion).
        :param dispersion_window_size: window (bp) for nb_offset dispersion;
            must divide the training tile size.  Ignored by other losses.
        :param log_dispersion_init: constant added to the dispersion head
            output, setting the initial scale (gamma ~ e^7 ~ 1100, i.e.
            near-multinomial at init, so training starts from the baseline
            and learns overdispersion where the data demand it).
        :param max_dispersion_ratio: for nb_offset only — clamp r so that
            NB variance does not exceed this multiple of the multinomial
            variance.  None disables clamping.  Default 2.0.
        :param clamp_margin: nats of headroom over which the gradient
            tapers from full to zero near the dispersion floor.
        :param freeze_dispersion: if True, dispersion head parameters are
            frozen (requires_grad=False).  The NB/DM loss still runs with
            dispersion at its init value (~multinomial).  Use for
            pre-training shape before fine-tuning dispersion.
        :param dispersion_lr_scale: relative learning rate for the
            dispersion head (0.1 = 10x slower than trunk/shape).  Only
            used when freeze_dispersion is False.
        :param block_kwargs: overrides for ``ResNetDilatedBlock`` config
            (activation, activation_post_sum, skip_batchnorm,
            preact_residual_normalization).  ``padding`` may not be
            overridden: ``calc_input_region_size`` assumes unpadded blocks.
        """
        super().__init__()
        if loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES} (got '{loss}')")
        if output_tracks is None:
            output_tracks = list(DEFAULT_OUTPUT_TRACKS)
        for t in output_tracks:
            track_name_to_index_key(t)  # validate names early
        self.save_hyperparameters()
        self.output_tracks = output_tracks

        resolved_block_kwargs = dict(
            activation=torch.nn.LeakyReLU,
            activation_post_sum=True,
            skip_batchnorm=True,
            preact_residual_normalization=False,
        )
        resolved_block_kwargs.update(block_kwargs or {})
        assert resolved_block_kwargs.get("padding", 0) == 0, (
            "block padding is fixed at 0: calc_input_region_size assumes "
            "unpadded blocks"
        )
        resolved_block_kwargs["padding"] = 0

        self.trunk = torch.nn.Sequential(
            torch.nn.Conv1d(4, n_kernels, kernel_size, padding=0),
            torch.nn.LeakyReLU(),
            SpatialDropout(dropout),
            *[
                ResNetDilatedBlock(
                    input_channels=n_kernels,
                    profile_kernel_size=kernel_size,
                    dilation_rate=2**i,
                    **resolved_block_kwargs,
                )
                for i in range(1, num_residual_layers + 1)
            ],
        )
        n_tracks = len(output_tracks)
        self.shape_head = torch.nn.Conv1d(n_kernels, n_tracks, kernel_size, padding=0)
        # dispersion head only exists for overdispersed losses (avoids unused
        # parameters under DDP for the multinomial baseline)
        self.dispersion_head = (
            None
            if loss == "multinomial"
            else torch.nn.Conv1d(n_kernels, n_tracks, kernel_size, padding=0)
        )

        if loss == "multinomial":
            self.loss_fn = MaskedMultinomialNLLLoss()
        elif loss == "dirichlet_multinomial":
            self.loss_fn = MaskedDirichletMultinomialNLLLoss()
        else:
            self.loss_fn = MaskedNegativeBinomialOffsetNLLLoss(
                max_dispersion_ratio=max_dispersion_ratio,
                clamp_margin=clamp_margin,
            )

        if freeze_dispersion and self.dispersion_head is not None:
            for p in self.dispersion_head.parameters():
                p.requires_grad = False

    # -- geometry ----------------------------------------------------------

    def calc_input_region_size(self, output_region_size: int) -> int:
        """Sequence length required to emit `output_region_size` positions.

        Initial conv and each head conv trim (k-1); residual block i trims
        (k-1) * 2**i.  Verified against a forward pass in the test suite.
        """
        k = self.hparams.kernel_size
        return (
            output_region_size
            + 2 * (k - 1)
            + sum((k - 1) * 2**i for i in range(1, self.hparams.num_residual_layers + 1))
        )

    # -- lightning ---------------------------------------------------------

    def forward(self, x):
        h = self.trunk(x)
        shape_logits = self.shape_head(h)
        if self.dispersion_head is None:
            return shape_logits, None
        return shape_logits, self.dispersion_head(h)

    def _pooled_log_dispersion(self, dispersion_bp, mask):
        """Pool the bp-resolution dispersion output per the configured loss."""
        L = dispersion_bp.shape[-1]
        if self.hparams.loss == "dirichlet_multinomial":
            out_size = 1
        else:  # nb_offset
            w = self.hparams.dispersion_window_size
            assert L % w == 0, (
                f"tile size {L} not divisible by dispersion_window_size {w}"
            )
            out_size = L // w
        pooled = masked_mean_pool(dispersion_bp, mask, out_size)
        pooled = pooled + self.hparams.log_dispersion_init
        if self.hparams.loss == "dirichlet_multinomial":
            pooled = pooled.squeeze(-1)  # (B, C)
        return pooled

    def _step(self, batch, log_name):
        x, y, mask = batch
        mask3 = _prepare_mask(mask, y)
        shape_logits, dispersion_bp = self(x)
        if self.hparams.loss == "multinomial":
            loss = self.loss_fn(shape_logits, y, mask3)
        else:
            log_disp = self._pooled_log_dispersion(dispersion_bp, mask3)
            loss = self.loss_fn(shape_logits, log_disp, y, mask3)
        self.log(log_name, loss, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train_loss")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val_loss")

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        scale = self.hparams.dispersion_lr_scale
        if (
            self.dispersion_head is not None
            and not self.hparams.freeze_dispersion
            and scale != 1.0
        ):
            disp_ids = {id(p) for p in self.dispersion_head.parameters()}
            main_params = [p for p in self.parameters() if id(p) not in disp_ids]
            disp_params = list(self.dispersion_head.parameters())
            optimizer = torch.optim.Adam([
                {"params": main_params, "lr": lr},
                {"params": disp_params, "lr": lr * scale},
            ])
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), lr=lr
            )
        return _with_lr_schedule(optimizer, self.hparams)

    # -- inference ---------------------------------------------------------

    @torch.no_grad()
    def predict_profile(self, one_hot_seq: np.ndarray, mask: Optional[np.ndarray] = None):
        """(4, L_in) one-hot -> dict with the null parameters for one tile.

        Returns:
            probs: (n_tracks, L_out) per-position probabilities (each track
                sums to 1 over valid positions) — the profile *shape*.
                Multiply by an observed N for expected counts.
            log_dispersion: (n_tracks,) log gamma  [dirichlet_multinomial],
                (n_tracks, W) log r per window     [nb_offset],
                or None                            [multinomial].

        mask: optional (L_out,) bool of valid positions.
        """
        self.eval()
        x = torch.as_tensor(one_hot_seq, dtype=torch.float32, device=self.device)
        shape_logits, dispersion_bp = self(x[None])
        mask3 = None
        if mask is not None:
            mask3 = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            mask3 = mask3[None, None, :]
            shape_logits = shape_logits.masked_fill(~mask3, float("-inf"))
        probs = torch.softmax(shape_logits, dim=-1)[0].cpu().numpy()
        log_dispersion = None
        if dispersion_bp is not None:
            log_dispersion = (
                self._pooled_log_dispersion(dispersion_bp, mask3)[0].cpu().numpy()
            )
        return {"probs": probs, "log_dispersion": log_dispersion}


class BackgroundModelKEN(L.LightningModule):
    """K-mer Embedding Network for cfDNA fragment-endpoint background modeling.

    Replaces the CNN trunk with an explicit k-mer lookup table
    (nn.Embedding) followed by optional context Conv1d layers.
    Same loss functions, masking, and predict_profile interface as
    BackgroundModel.

    The model predicts joint endpoint profiles: the observed count at each
    position reflects hexamer bias at BOTH fragment ends (left cut forward,
    right cut RC), convolved with the fragment-length distribution. The
    embedding table learns the effective per-hexamer contribution to this
    joint profile, not raw individual cut-site weights.

    RC weight tying: each k-mer and its reverse-complement share the same
    embedding row (2,080 canonical entries for k=6 instead of 4,096).
    """

    def __init__(
        self,
        output_tracks: Optional[List[str]] = None,
        k: int = 6,
        d_embed: int = 64,
        d_context: int = 128,
        n_context_layers: int = 2,
        context_kernel_size: int = 15,
        dropout: float = 0.15,
        learning_rate: float = 1e-3,
        loss: str = "multinomial",
        dispersion_window_size: int = 256,
        log_dispersion_init: float = 7.0,
        max_dispersion_ratio: Optional[float] = 2.0,
        clamp_margin: float = 1.0,
        freeze_dispersion: bool = False,
        dispersion_lr_scale: float = 1.0,
        weight_decay: float = 0.0,
        lr_patience: int = 4,
        max_lr_reductions: int = 3,
        lr_factor: float = 0.5,
    ):
        super().__init__()
        if loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES} (got '{loss}')")
        if output_tracks is None:
            output_tracks = list(DEFAULT_OUTPUT_TRACKS)
        for t in output_tracks:
            track_name_to_index_key(t)
        self.save_hyperparameters()
        self.output_tracks = output_tracks

        n_tracks = len(output_tracks)
        vocab_size = 4 ** k

        # Stage 1: k-mer index computation (fixed)
        self.register_buffer(
            "_powers",
            4 ** torch.arange(k - 1, -1, -1, dtype=torch.long),
        )

        # RC weight tying: map each k-mer to its canonical representative
        rc_perm = rc_kmer_permutation(k)
        canonical = np.minimum(np.arange(vocab_size, dtype=np.int64), rc_perm)
        _, to_canonical = np.unique(canonical, return_inverse=True)
        self.register_buffer(
            "_to_canonical",
            torch.from_numpy(to_canonical.astype(np.int64)),
        )
        n_canonical = int(to_canonical.max()) + 1  # 2080 for k=6

        # Stage 2: embedding table (canonical entries only)
        self.embed = torch.nn.Embedding(n_canonical, d_embed)
        self.embed_dropout = SpatialDropout(dropout)

        # Stage 3: context CNN (unpadded)
        layers = []
        in_ch = d_embed
        for _ in range(n_context_layers):
            layers.append(
                torch.nn.Conv1d(in_ch, d_context, context_kernel_size,
                                padding=0)
            )
            layers.append(torch.nn.GELU())
            in_ch = d_context
        self.context = torch.nn.Sequential(*layers) if layers else torch.nn.Identity()
        trunk_out_ch = d_context if n_context_layers > 0 else d_embed

        # Stage 4: shape head
        self.shape_head = torch.nn.Conv1d(trunk_out_ch, n_tracks, 1)

        # Stage 5: dispersion head (only for overdispersed losses)
        self.dispersion_head = (
            None if loss == "multinomial"
            else torch.nn.Conv1d(trunk_out_ch, n_tracks, 1)
        )

        # Loss
        if loss == "multinomial":
            self.loss_fn = MaskedMultinomialNLLLoss()
        elif loss == "dirichlet_multinomial":
            self.loss_fn = MaskedDirichletMultinomialNLLLoss()
        else:
            self.loss_fn = MaskedNegativeBinomialOffsetNLLLoss(
                max_dispersion_ratio=max_dispersion_ratio,
                clamp_margin=clamp_margin,
            )

        if freeze_dispersion and self.dispersion_head is not None:
            for p in self.dispersion_head.parameters():
                p.requires_grad = False

        # Even-k trimming flag: when k is even, calc_input_region_size
        # rounds up to even, producing one extra unfold position.
        self._trim = (k - 1) % 2  # 1 when k is even, 0 when k is odd

    def calc_input_region_size(self, output_region_size: int) -> int:
        """Input length for the given output length.

        The k-mer unfold needs (k-1) extra bases. Each unpadded context
        layer trims (context_kernel_size - 1) positions. For even k the
        raw total may be odd, violating the dataset's even-parity
        requirement -- round up.
        """
        k = self.hparams.k
        n_ctx = self.hparams.n_context_layers
        k_ctx = self.hparams.context_kernel_size
        raw = output_region_size + (k - 1) + n_ctx * (k_ctx - 1)
        return raw + (raw % 2)

    def forward(self, x):
        # x: (B, 4, L_in) one-hot
        kmer_idx = one_hot_to_kmer_indices(x, self.hparams.k, self._powers)
        canonical_idx = self._to_canonical[kmer_idx]
        h = self.embed(canonical_idx)             # (B, L_unfold, d_embed)
        h = h.transpose(1, 2)                     # (B, d_embed, L_unfold)
        h = self.embed_dropout(h)
        h = self.context(h)                       # (B, d_context, L_out')

        # Trim the extra position from even-parity rounding (even k only)
        if self._trim:
            h = h[..., :-1]

        shape_logits = self.shape_head(h)
        if self.dispersion_head is None:
            return shape_logits, None
        return shape_logits, self.dispersion_head(h)

    def _pooled_log_dispersion(self, dispersion_bp, mask):
        L = dispersion_bp.shape[-1]
        if self.hparams.loss == "dirichlet_multinomial":
            out_size = 1
        else:
            w = self.hparams.dispersion_window_size
            assert L % w == 0
            out_size = L // w
        pooled = masked_mean_pool(dispersion_bp, mask, out_size)
        pooled = pooled + self.hparams.log_dispersion_init
        if self.hparams.loss == "dirichlet_multinomial":
            pooled = pooled.squeeze(-1)
        return pooled

    def _step(self, batch, log_name):
        x, y, mask = batch
        mask3 = _prepare_mask(mask, y)
        shape_logits, dispersion_bp = self(x)
        if self.hparams.loss == "multinomial":
            loss = self.loss_fn(shape_logits, y, mask3)
        else:
            log_disp = self._pooled_log_dispersion(dispersion_bp, mask3)
            loss = self.loss_fn(shape_logits, log_disp, y, mask3)
        self.log(log_name, loss, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train_loss")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val_loss")

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        wd = self.hparams.weight_decay
        scale = self.hparams.dispersion_lr_scale

        # Apply weight decay only to the embedding table
        embed_params = list(self.embed.parameters())
        embed_ids = {id(p) for p in embed_params}

        if self.dispersion_head is not None and not self.hparams.freeze_dispersion and scale != 1.0:
            disp_ids = {id(p) for p in self.dispersion_head.parameters()}
            main_params = [p for p in self.parameters()
                          if id(p) not in embed_ids and id(p) not in disp_ids]
            optimizer = torch.optim.Adam([
                {"params": embed_params, "lr": lr, "weight_decay": wd},
                {"params": main_params, "lr": lr, "weight_decay": 0.0},
                {"params": list(self.dispersion_head.parameters()),
                 "lr": lr * scale, "weight_decay": 0.0},
            ])
        else:
            main_params = [p for p in self.parameters() if id(p) not in embed_ids]
            optimizer = torch.optim.Adam([
                {"params": embed_params, "lr": lr, "weight_decay": wd},
                {"params": main_params, "lr": lr, "weight_decay": 0.0},
            ])
        return _with_lr_schedule(optimizer, self.hparams)

    @torch.no_grad()
    def predict_profile(self, one_hot_seq: np.ndarray,
                        mask: Optional[np.ndarray] = None):
        self.eval()
        x = torch.as_tensor(one_hot_seq, dtype=torch.float32, device=self.device)
        shape_logits, dispersion_bp = self(x[None])
        mask3 = None
        if mask is not None:
            mask3 = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            mask3 = mask3[None, None, :]
            shape_logits = shape_logits.masked_fill(~mask3, float("-inf"))
        probs = torch.softmax(shape_logits, dim=-1)[0].cpu().numpy()
        log_dispersion = None
        if dispersion_bp is not None:
            log_dispersion = (
                self._pooled_log_dispersion(dispersion_bp, mask3)[0].cpu().numpy()
            )
        return {"probs": probs, "log_dispersion": log_dispersion}


class BackgroundModelHybrid(L.LightningModule):
    """K-mer embedding + one-hot conv stem, fused into a dilated ResNet trunk.

    Two parallel input representations are concatenated on the channel axis
    and fed to the dilated residual stack:

      * **k-mer embedding** — an exact ``Embedding`` lookup of the k-mer
        centered on each position.  Supplies the combinatorial cut-site
        weight that a convolution over one-hot bases can only approximate.
      * **conv stem** — ``Conv1d`` over the one-hot sequence.  Supplies what
        a fixed-width lookup cannot: variable-width motifs, partial matches,
        positional base preferences.

    The dilated residual blocks then aggregate both over a long receptive
    field, which is what GC x fragment-length bias requires (fragments span
    40-175 bp, far beyond the k-mer itself).  Residual skip connections carry
    the local signal through largely intact while the dilated branch adds
    contextual corrections — matching the structure of the bias: cut-site
    weight is the dominant term, GC x FL a modulation on top.

    RC weight tying is applied to the embedding (each k-mer shares a row with
    its reverse complement); the conv branch relies on RC data augmentation,
    as in ``BackgroundModel``.
    """

    def __init__(
        self,
        output_tracks: Optional[List[str]] = None,
        k: int = 6,
        d_embed: int = 64,
        n_kernels: int = 128,
        kernel_size: int = 32,
        num_residual_layers: int = 3,
        dropout: float = 0.15,
        learning_rate: float = 1e-3,
        loss: str = "multinomial",
        dispersion_window_size: int = 256,
        log_dispersion_init: float = 7.0,
        max_dispersion_ratio: Optional[float] = 2.0,
        clamp_margin: float = 1.0,
        freeze_dispersion: bool = False,
        dispersion_lr_scale: float = 1.0,
        weight_decay: float = 0.0,
        lr_patience: int = 4,
        max_lr_reductions: int = 3,
        lr_factor: float = 0.5,
        block_kwargs: Optional[dict] = None,
    ):
        """
        :param k: k-mer width for the embedding branch.  Must not exceed
            ``kernel_size`` — the conv stem trims more than the k-mer unfold,
            so the embedding output is center-cropped down to match it.
        :param d_embed: embedding dimension per k-mer.
        :param n_kernels: channel count for the conv stem and the residual
            trunk.  The concatenated (n_kernels + d_embed) channels are
            projected back to n_kernels by a 1x1 conv before the trunk.
        :param weight_decay: applied to the embedding table only; the rest of
            the network trains unregularized.
        """
        super().__init__()
        if loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES} (got '{loss}')")
        if k > kernel_size:
            raise ValueError(
                f"k ({k}) must not exceed kernel_size ({kernel_size}): the "
                f"embedding branch is center-cropped to the conv stem length"
            )
        if output_tracks is None:
            output_tracks = list(DEFAULT_OUTPUT_TRACKS)
        for t in output_tracks:
            track_name_to_index_key(t)
        self.save_hyperparameters()
        self.output_tracks = output_tracks

        n_tracks = len(output_tracks)
        vocab_size = 4 ** k

        # -- embedding branch ---------------------------------------------
        self.register_buffer(
            "_powers",
            4 ** torch.arange(k - 1, -1, -1, dtype=torch.long),
        )
        rc_perm = rc_kmer_permutation(k)
        canonical = np.minimum(np.arange(vocab_size, dtype=np.int64), rc_perm)
        _, to_canonical = np.unique(canonical, return_inverse=True)
        self.register_buffer(
            "_to_canonical",
            torch.from_numpy(to_canonical.astype(np.int64)),
        )
        n_canonical = int(to_canonical.max()) + 1  # 2080 for k=6
        self.embed = torch.nn.Embedding(n_canonical, d_embed)
        self.embed_dropout = SpatialDropout(dropout)

        # -- conv stem branch ---------------------------------------------
        self.conv_stem = torch.nn.Sequential(
            torch.nn.Conv1d(4, n_kernels, kernel_size, padding=0),
            torch.nn.LeakyReLU(),
            SpatialDropout(dropout),
        )

        # -- fusion + dilated trunk ----------------------------------------
        resolved_block_kwargs = dict(
            activation=torch.nn.LeakyReLU,
            activation_post_sum=True,
            skip_batchnorm=True,
            preact_residual_normalization=False,
        )
        resolved_block_kwargs.update(block_kwargs or {})
        assert resolved_block_kwargs.get("padding", 0) == 0, (
            "block padding is fixed at 0: calc_input_region_size assumes "
            "unpadded blocks"
        )
        resolved_block_kwargs["padding"] = 0

        self.fuse = torch.nn.Conv1d(n_kernels + d_embed, n_kernels, 1)
        self.trunk = torch.nn.Sequential(
            *[
                ResNetDilatedBlock(
                    input_channels=n_kernels,
                    profile_kernel_size=kernel_size,
                    dilation_rate=2**i,
                    **resolved_block_kwargs,
                )
                for i in range(1, num_residual_layers + 1)
            ],
        )

        self.shape_head = torch.nn.Conv1d(n_kernels, n_tracks, kernel_size, padding=0)
        self.dispersion_head = (
            None
            if loss == "multinomial"
            else torch.nn.Conv1d(n_kernels, n_tracks, kernel_size, padding=0)
        )

        if loss == "multinomial":
            self.loss_fn = MaskedMultinomialNLLLoss()
        elif loss == "dirichlet_multinomial":
            self.loss_fn = MaskedDirichletMultinomialNLLLoss()
        else:
            self.loss_fn = MaskedNegativeBinomialOffsetNLLLoss(
                max_dispersion_ratio=max_dispersion_ratio,
                clamp_margin=clamp_margin,
            )

        if freeze_dispersion and self.dispersion_head is not None:
            for p in self.dispersion_head.parameters():
                p.requires_grad = False

    # -- geometry ----------------------------------------------------------

    def calc_input_region_size(self, output_region_size: int) -> int:
        """Sequence length required to emit ``output_region_size`` positions.

        The conv stem and shape head each trim (kernel_size - 1); residual
        block i trims (kernel_size - 1) * 2**i.  The embedding branch trims
        only (k - 1) and is center-cropped down to the stem's length, so it
        never binds — which is why ``k <= kernel_size`` is required.

        Every term is even, so the result has the same parity as
        ``output_region_size``; even tile sizes yield the even input length
        the dataset requires, with no rounding.
        """
        ks = self.hparams.kernel_size
        return (
            output_region_size
            + 2 * (ks - 1)
            + sum(
                (ks - 1) * 2**i
                for i in range(1, self.hparams.num_residual_layers + 1)
            )
        )

    # -- lightning ---------------------------------------------------------

    def forward(self, x):
        # x: (B, 4, L_in) one-hot
        h_conv = self.conv_stem(x)                       # (B, n_kernels, L_c)

        kmer_idx = one_hot_to_kmer_indices(x, self.hparams.k, self._powers)
        canonical_idx = self._to_canonical[kmer_idx]
        h_embed = self.embed(canonical_idx)              # (B, L_k, d_embed)
        h_embed = h_embed.transpose(1, 2)                # (B, d_embed, L_k)
        h_embed = self.embed_dropout(h_embed)

        # The embedding branch trims (k-1); the stem trims (kernel_size-1).
        # Center-crop the longer embedding output down to the stem's length.
        target = h_conv.shape[-1]
        diff = h_embed.shape[-1] - target
        if diff > 0:
            lo = diff // 2
            h_embed = h_embed[..., lo:lo + target]

        h = torch.cat([h_conv, h_embed], dim=1)          # (B, n_k + d_e, L_c)
        h = self.fuse(h)                                 # (B, n_kernels, L_c)
        h = self.trunk(h)

        shape_logits = self.shape_head(h)
        if self.dispersion_head is None:
            return shape_logits, None
        return shape_logits, self.dispersion_head(h)

    def _pooled_log_dispersion(self, dispersion_bp, mask):
        L = dispersion_bp.shape[-1]
        if self.hparams.loss == "dirichlet_multinomial":
            out_size = 1
        else:
            w = self.hparams.dispersion_window_size
            assert L % w == 0, (
                f"tile size {L} not divisible by dispersion_window_size {w}"
            )
            out_size = L // w
        pooled = masked_mean_pool(dispersion_bp, mask, out_size)
        pooled = pooled + self.hparams.log_dispersion_init
        if self.hparams.loss == "dirichlet_multinomial":
            pooled = pooled.squeeze(-1)
        return pooled

    def _step(self, batch, log_name):
        x, y, mask = batch
        mask3 = _prepare_mask(mask, y)
        shape_logits, dispersion_bp = self(x)
        if self.hparams.loss == "multinomial":
            loss = self.loss_fn(shape_logits, y, mask3)
        else:
            log_disp = self._pooled_log_dispersion(dispersion_bp, mask3)
            loss = self.loss_fn(shape_logits, log_disp, y, mask3)
        self.log(log_name, loss, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train_loss")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val_loss")

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        wd = self.hparams.weight_decay
        scale = self.hparams.dispersion_lr_scale

        embed_params = list(self.embed.parameters())
        embed_ids = {id(p) for p in embed_params}

        if (
            self.dispersion_head is not None
            and not self.hparams.freeze_dispersion
            and scale != 1.0
        ):
            disp_params = list(self.dispersion_head.parameters())
            disp_ids = {id(p) for p in disp_params}
            main_params = [
                p for p in self.parameters()
                if id(p) not in embed_ids and id(p) not in disp_ids
            ]
            optimizer = torch.optim.Adam([
                {"params": embed_params, "lr": lr, "weight_decay": wd},
                {"params": main_params, "lr": lr, "weight_decay": 0.0},
                {"params": disp_params, "lr": lr * scale, "weight_decay": 0.0},
            ])
        else:
            main_params = [p for p in self.parameters() if id(p) not in embed_ids]
            optimizer = torch.optim.Adam([
                {"params": embed_params, "lr": lr, "weight_decay": wd},
                {"params": main_params, "lr": lr, "weight_decay": 0.0},
            ])
        return _with_lr_schedule(optimizer, self.hparams)

    @torch.no_grad()
    def predict_profile(self, one_hot_seq: np.ndarray,
                        mask: Optional[np.ndarray] = None):
        self.eval()
        x = torch.as_tensor(one_hot_seq, dtype=torch.float32, device=self.device)
        shape_logits, dispersion_bp = self(x[None])
        mask3 = None
        if mask is not None:
            mask3 = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            mask3 = mask3[None, None, :]
            shape_logits = shape_logits.masked_fill(~mask3, float("-inf"))
        probs = torch.softmax(shape_logits, dim=-1)[0].cpu().numpy()
        log_dispersion = None
        if dispersion_bp is not None:
            log_dispersion = (
                self._pooled_log_dispersion(dispersion_bp, mask3)[0].cpu().numpy()
            )
        return {"probs": probs, "log_dispersion": log_dispersion}


# --------------------------------------------------------------------------
# Deviation testing / calibration diagnostics (post-hoc, numpy/scipy)
#
# The acceptance gate for both overdispersed variants: on held-out INACTIVE
# regions x held-out samples, these window p-values must be QQ-uniform per
# track (and per accessibility stratum).  On positive controls (CTCF sites,
# marker genes) they should deviate strongly.
# --------------------------------------------------------------------------


def _window_sums(arr: np.ndarray, window_size: int) -> np.ndarray:
    L = arr.shape[-1]
    assert L % window_size == 0
    return arr.reshape(*arr.shape[:-1], L // window_size, window_size).sum(axis=-1)


def beta_binomial_window_pvalues(
    counts: np.ndarray,
    probs: np.ndarray,
    gamma: float,
    window_size: int,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Upper-tail P(Y >= y_w) per window under the DM null (exact aggregation).

    counts, probs: (L,) for one track; gamma: scalar concentration for the
    tile; mask: optional (L,) bool.  Window sums of a DM are beta-binomial:
    y_w ~ BB(N, gamma * q_w, gamma * (1 - q_w)) with q_w the window's
    predicted mass (renormalized over valid positions).
    """
    counts = np.asarray(counts, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if mask is not None:
        counts = np.where(mask, counts, 0.0)
        probs = np.where(mask, probs, 0.0)
        probs = probs / probs.sum()
    n = int(counts.sum())
    y = _window_sums(counts, window_size)
    q = np.clip(_window_sums(probs, window_size), 1e-12, 1 - 1e-12)
    a = gamma * q
    b = gamma * (1.0 - q)
    return betabinom.sf(y - 1, n, a, b)


def nb_window_pvalues(
    counts: np.ndarray,
    probs: np.ndarray,
    r_windows: np.ndarray,
    dispersion_window_size: int,
    test_window_size: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Upper-tail P(Y >= y_w) per window under the NB-offset null.

    r_windows: (W,) per-window dispersion (natural scale) at
    ``dispersion_window_size``.  If the test window differs, window variances
    are combined by moment matching (sum of independent NBs is not NB;
    Var_w = sum(mu_i + mu_i^2 / r_i) -> effective r = mu_w^2/(Var_w - mu_w)).
    """
    if test_window_size is None:
        test_window_size = dispersion_window_size
    counts = np.asarray(counts, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if mask is not None:
        counts = np.where(mask, counts, 0.0)
        probs = np.where(mask, probs, 0.0)
        probs = probs / probs.sum()
    n = counts.sum()
    mu = n * probs  # (L,)
    r_bp = np.repeat(r_windows, dispersion_window_size)  # (L,)
    var_bp = mu + mu**2 / r_bp

    y = _window_sums(counts, test_window_size)
    mu_w = np.clip(_window_sums(mu, test_window_size), 1e-12, None)
    var_w = _window_sums(var_bp, test_window_size)
    excess = np.clip(var_w - mu_w, 1e-12, None)
    r_eff = mu_w**2 / excess
    p_nb = r_eff / (r_eff + mu_w)
    return nbinom.sf(y - 1, r_eff, p_nb)


def qq_uniformity(pvalues: np.ndarray):
    """Sorted observed p-values vs uniform quantiles, for QQ plotting.

    Returns (expected, observed); calibrated nulls lie on the diagonal.
    """
    p = np.sort(np.asarray(pvalues).ravel())
    expected = (np.arange(1, len(p) + 1) - 0.5) / len(p)
    return expected, p
