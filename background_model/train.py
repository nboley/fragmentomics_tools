"""Training driver for the background model (Phase 3.4).

Device-agnostic (``accelerator="auto"``) — runs on CPU or GPU with no code
change; nothing here calls ``.cuda()`` and the Dataset does its own worker-safe,
CUDA-free reads.  The frozen statistical core (``background_model_core.py``) is
NOT modified: all diagnostics live in the :class:`InstrumentedBackgroundModel`
subclass and in callbacks.  Per-track losses reuse the frozen ``loss_fn`` on
channel slices (they do not reimplement any likelihood).

Usage::

    python -m background_model.train \
        --loss multinomial --run-name my_run \
        --max-epochs 10 --batch-size 8 --lr 1e-4

CLI (see ``build_arg_parser``): ``--loss --run-name --max-epochs --batch-size
--lr --limit-batches`` plus ``--num-workers --seed --lr-patience
--max-lr-reductions --max-recoveries --store --runs-root --resume-from``
conveniences.

Reproducibility contract: every run dir records ``config_hash``,
``split_version``, git sha and the full resolved config in ``run_meta.json``.
NOTE: the Dataset's jitter/RC augmentation RNG is keyed on ``[seed, worker_pid]``
(dataset.py docstring), so bit-identical loss curves across two *separate
processes* are only guaranteed with augmentation off; within one process two
freshly-built datasets under the same seed reproduce (used by the L6 rung).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass

import numpy as np
import lightning as L
import torch
from lightning.pytorch.callbacks import (
    Callback,
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch.utils.data import DataLoader

from background_model.config import TILE, PlumbingConfig
from background_model.dataset import BackgroundTileDataset
from background_model_core import (
    LOSSES,
    BackgroundModel,
    BackgroundModelHybrid,
    BackgroundModelKEN,
    MaskedMultinomialNLLLoss,
    _prepare_mask,
)

DEFAULT_STORE = (
    "/efs/analytics/nathanboley/background_model/stores/bg_store_b67d7c95.zarr"
)
DEFAULT_RUNS_ROOT = "/efs/analytics/nathanboley/background_model/runs"

# Track-to-FL-band mapping: which FL band each of the 12 tracks belongs to.
# Layout: 2 strands × 2 FL bands × 3 coverage types
# Tracks 0-2 = (+, short), 3-5 = (+, mono), 6-8 = (-, short), 9-11 = (-, mono)
_TRACK_TO_BAND = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])


def _apply_fl_reweighting(loss_unweighted, fl_fracs, shape_logits, y, mask3,
                           loss_fn, log_disp, loss_name):
    """Recompute loss with per-sample, per-track FL band weighting.

    Each sample's loss contribution is weighted by its own FL band fractions,
    so tracks with more fragments in THAT sample's length distribution
    contribute proportionally more to the gradient.

    fl_fracs: (B, n_bands) — fraction of fragments in each FL band per sample
    Returns: scalar weighted loss (with gradients)
    """
    B, C, L = shape_logits.shape
    band_idx = _TRACK_TO_BAND.to(shape_logits.device)  # (C,)
    # Per-sample, per-track weights
    w = fl_fracs[:, band_idx]  # (B, C)
    w = w / w.sum(dim=1, keepdim=True) * C  # normalize per sample

    # Compute per-sample, per-track multinomial NLL inline
    sl = shape_logits
    if mask3 is not None:
        sl = sl.masked_fill(~mask3, float("-inf"))
    logp = torch.log_softmax(sl, dim=-1)
    if mask3 is not None:
        logp = logp.masked_fill(~mask3, 0.0)
    N = y.sum(dim=-1).clamp(min=1.0)  # (B, C)
    nll = -(y * logp).sum(dim=-1) / N  # (B, C) per-sample per-track

    if loss_name != "multinomial" and log_disp is not None:
        # For NB-offset: use the full NB loss per sample per track.
        # Fall back to the unweighted loss — FL reweighting is most
        # important for shape learning (multinomial component).
        # TODO: implement per-sample NB loss if needed.
        return loss_unweighted

    # Weighted mean: each (sample, track) pair weighted by that sample's
    # FL band fraction for that track's band
    return (nll * w).sum() / (B * C)


# --------------------------------------------------------------------------
# Instrumented model (frozen core + detached diagnostics only)
# --------------------------------------------------------------------------


class InstrumentedBackgroundModel(BackgroundModel):
    """BackgroundModel + per-track loss, dispersion-trajectory and grad-norm
    logging.  The training loss and gradients are byte-for-byte those of the
    frozen ``BackgroundModel._step`` (same single forward, same ``loss_fn``);
    all extra logging is ``detach``-ed and adds no gradient path.
    """

    def _step(self, batch, log_name):
        if len(batch) == 4:
            x, y, mask, fl_fracs = batch
        else:
            x, y, mask = batch
            fl_fracs = None
        mask3 = _prepare_mask(mask, y)
        shape_logits, dispersion_bp = self(x)

        if self.hparams.loss == "multinomial":
            log_disp = None
            loss = self.loss_fn(shape_logits, y, mask3)
        else:
            log_disp = self._pooled_log_dispersion(dispersion_bp, mask3)
            loss = self.loss_fn(shape_logits, log_disp, y, mask3)

        # FL-conditioned per-track reweighting: scale the loss by each
        # sample's FL band fractions so tracks with more fragments in that
        # sample's length distribution contribute proportionally more.
        if fl_fracs is not None:
            loss = _apply_fl_reweighting(loss, fl_fracs, shape_logits, y, mask3,
                                         self.loss_fn, log_disp, self.hparams.loss)

        self.log(log_name, loss, prog_bar=True, sync_dist=True)
        self._log_per_track(log_name, shape_logits, log_disp, y, mask3)
        self._log_dispersion_trajectory(log_name, log_disp)
        self._log_multinomial_nll(log_name, shape_logits, y, mask3)
        return loss

    @torch.no_grad()
    def _log_multinomial_nll(self, log_name, shape_logits, y, mask3):
        """Log multinomial NLL for all models (comparable to oracle=7.5745).

        For multinomial loss this equals val_loss. For other losses it provides
        a cross-family comparison metric on the same scale.
        """
        stage = log_name.split("_")[0]
        sl = shape_logits.detach()
        if mask3 is not None:
            sl = sl.masked_fill(~mask3, float("-inf"))
        logp = torch.log_softmax(sl, dim=-1)
        if mask3 is not None:
            logp = logp.masked_fill(~mask3, 0.0)
        totals = y.detach().sum(dim=-1).clamp(min=1.0)
        nll = -(y.detach() * logp).sum(dim=-1) / totals
        self.log(f"{stage}_multinomial_nll", nll.mean(), sync_dist=True)

    @torch.no_grad()
    def _log_per_track(self, log_name, shape_logits, log_disp, y, mask3):
        """Per-track mean NLL, computed by slicing channels and re-calling the
        SAME frozen loss module (no statistics reimplemented)."""
        stage = log_name.split("_")[0]  # "train" | "val"
        sl_logits = shape_logits.detach()
        sl_y = y.detach()
        for c, name in enumerate(self.output_tracks):
            ch = slice(c, c + 1)
            if self.hparams.loss == "multinomial":
                lt = self.loss_fn(sl_logits[:, ch], sl_y[:, ch], mask3)
            else:
                lt = self.loss_fn(
                    sl_logits[:, ch], log_disp.detach()[:, ch], sl_y[:, ch], mask3
                )
            self.log(f"{stage}_track/{name}", lt, sync_dist=True)

    @torch.no_grad()
    def _log_dispersion_trajectory(self, log_name, log_disp):
        """mean / p10 / p90 of pooled log_dispersion (no-op for multinomial)."""
        if log_disp is None:
            return
        stage = log_name.split("_")[0]
        flat = log_disp.detach().reshape(-1).float()
        self.log(f"{stage}_logdisp/mean", flat.mean(), sync_dist=True)
        self.log(f"{stage}_logdisp/p10", torch.quantile(flat, 0.10), sync_dist=True)
        self.log(f"{stage}_logdisp/p90", torch.quantile(flat, 0.90), sync_dist=True)

    def on_before_optimizer_step(self, optimizer):
        # Pre-clip 2-norm (Trainer applies gradient_clip_val=1.0 afterwards);
        # logging it shows how often clipping binds.
        norms = grad_norm(self, norm_type=2)
        total = norms.get("grad_2.0_norm_total")
        if total is not None:
            self.log("grad_2norm", total, prog_bar=False)


class _EmbeddingModelInstrumentation:
    """Per-track loss, dispersion-trajectory, multinomial-NLL and grad-norm
    logging for the embedding-based architectures (KEN and Hybrid).

    Mirrors InstrumentedBackgroundModel.  These methods touch only
    ``self.hparams.loss``, ``self.loss_fn``, ``self.output_tracks`` and
    ``self._pooled_log_dispersion``, so they are architecture-agnostic — mix
    into any model exposing that interface.
    """

    def _step(self, batch, log_name):
        if len(batch) == 4:
            x, y, mask, fl_fracs = batch
        else:
            x, y, mask = batch
            fl_fracs = None
        mask3 = _prepare_mask(mask, y)
        shape_logits, dispersion_bp = self(x)

        if self.hparams.loss == "multinomial":
            log_disp = None
            loss = self.loss_fn(shape_logits, y, mask3)
        else:
            log_disp = self._pooled_log_dispersion(dispersion_bp, mask3)
            loss = self.loss_fn(shape_logits, log_disp, y, mask3)

        if fl_fracs is not None:
            loss = _apply_fl_reweighting(loss, fl_fracs, shape_logits, y, mask3,
                                         self.loss_fn, log_disp, self.hparams.loss)

        self.log(log_name, loss, prog_bar=True, sync_dist=True)
        self._log_per_track(log_name, shape_logits, log_disp, y, mask3)
        self._log_dispersion_trajectory(log_name, log_disp)
        self._log_multinomial_nll(log_name, shape_logits, y, mask3)
        return loss

    @torch.no_grad()
    def _log_multinomial_nll(self, log_name, shape_logits, y, mask3):
        stage = log_name.split("_")[0]
        sl = shape_logits.detach()
        if mask3 is not None:
            sl = sl.masked_fill(~mask3, float("-inf"))
        logp = torch.log_softmax(sl, dim=-1)
        if mask3 is not None:
            logp = logp.masked_fill(~mask3, 0.0)
        totals = y.detach().sum(dim=-1).clamp(min=1.0)
        nll = -(y.detach() * logp).sum(dim=-1) / totals
        self.log(f"{stage}_multinomial_nll", nll.mean(), sync_dist=True)

    @torch.no_grad()
    def _log_per_track(self, log_name, shape_logits, log_disp, y, mask3):
        stage = log_name.split("_")[0]
        sl_logits = shape_logits.detach()
        sl_y = y.detach()
        for c, name in enumerate(self.output_tracks):
            ch = slice(c, c + 1)
            if self.hparams.loss == "multinomial":
                lt = self.loss_fn(sl_logits[:, ch], sl_y[:, ch], mask3)
            else:
                lt = self.loss_fn(
                    sl_logits[:, ch], log_disp.detach()[:, ch], sl_y[:, ch], mask3
                )
            self.log(f"{stage}_track/{name}", lt, sync_dist=True)

    @torch.no_grad()
    def _log_dispersion_trajectory(self, log_name, log_disp):
        if log_disp is None:
            return
        stage = log_name.split("_")[0]
        flat = log_disp.detach().reshape(-1).float()
        self.log(f"{stage}_logdisp/mean", flat.mean(), sync_dist=True)
        self.log(f"{stage}_logdisp/p10", torch.quantile(flat, 0.10), sync_dist=True)
        self.log(f"{stage}_logdisp/p90", torch.quantile(flat, 0.90), sync_dist=True)

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        total = norms.get("grad_2.0_norm_total")
        if total is not None:
            self.log("grad_2norm", total, prog_bar=False)


class InstrumentedBackgroundModelKEN(
    _EmbeddingModelInstrumentation, BackgroundModelKEN
):
    """BackgroundModelKEN + training instrumentation."""


class InstrumentedBackgroundModelHybrid(
    _EmbeddingModelInstrumentation, BackgroundModelHybrid
):
    """BackgroundModelHybrid + training instrumentation."""


# --------------------------------------------------------------------------
# Divergence guard
# --------------------------------------------------------------------------


class DivergenceStop(Callback):
    """Stop training on divergence or metric stall.

    Two independent guards, each independently disableable:

    **Divergence** (``factor``): stop when val_loss exceeds ``factor`` x its
    own best.  ``EarlyStopping(check_finite=True)`` only catches NaN/inf;
    observed divergences stayed finite (8.73, 383, 8.4e8) and ran to the
    epoch cap.  Set ``factor=0`` to disable the divergence-ratio check
    (non-finite detection is always active regardless of ``factor``).

    Divergence threshold rationale, measured on the v3 runs:

    ======================================  =================
    run                                     max val/best
    ======================================  =================
    KEN, CNN multinomial, CNN frozen-NB     1.0001 - 1.0002
    CNN multinomial lr=1e-2 (diverged)      380
    Hybrid lr=5e-3 (diverged)               1.9e9
    ======================================  =================

    The default 1.10 sits ~500x above the healthy noise floor and ~3400x
    below the smallest real divergence, so it cannot plausibly fire on a
    healthy run.

    **Stall** (``stall_patience``): stop when ``stall_patience`` consecutive
    validation epochs produce a *bitwise identical* metric value — the
    signature of a collapsed model whose output is constant and whose
    per-epoch loss depends only on batch composition.  This is deliberately
    exact equality, not an epsilon test: an epsilon-based "no improvement"
    check is what ``EarlyStopping(patience=...)`` already does, and
    duplicating it would risk firing on slow convergence.
    Set ``stall_patience=0`` to disable.

    Stall threshold rationale, measured across 15 v3 simulation runs:

    ======================================  ==========  ====================
    run                                     epochs      longest identical run
    ======================================  ==========  ====================
    all 13 healthy runs                     16-124      1 (two runs reach 2)
    lrsweep_ken_lr2e-3                      32          2
    lrsweep_ken_lr1e-3                      37          2
    **lrsweep_ken_lr2e-2 (dead)**           16          **16**
    ======================================  ==========  ====================

    The default 5 sits ~2.5x above the healthy ceiling of 2 and far below
    the observed failure of 16.
    """

    def __init__(self, factor: float = 1.10, monitor: str = "val_loss",
                 stall_patience: int = 5):
        self.factor = factor
        self.monitor = monitor
        self.stall_patience = stall_patience
        self.best = None
        self._last_val = None
        self._stall_count = 0
        # Populated when a guard fires; read after trainer.fit() to
        # determine why the run stopped.  Keys: reason, epoch, plus
        # guard-specific detail (value, best, factor, consecutive_epochs).
        self.stop_reason = None

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return
        current = float(current)

        # --- Non-finite detection (unconditional, checked first) ---
        # Must precede the stall check: inf == inf is True, so without
        # this guard a repeated-inf trace with factor=0 would be
        # mislabelled as "stalled" instead of "non-finite".
        if not math.isfinite(current):
            self.stop_reason = {
                "reason": "non_finite",
                "epoch": int(trainer.current_epoch),
                "value": str(current),  # inf/nan are not JSON-serialisable
            }
            trainer.should_stop = True
            rank_zero_info(
                f"[DivergenceStop] {self.monitor}={current} is not finite "
                f"-- stopping at epoch {trainer.current_epoch}."
            )
            return

        # --- Stall detection (exact equality) ---
        if self.stall_patience > 0:
            if self._last_val is not None and current == self._last_val:
                self._stall_count += 1
            else:
                self._stall_count = 1
            self._last_val = current
            if self._stall_count >= self.stall_patience:
                self.stop_reason = {
                    "reason": "stalled",
                    "epoch": int(trainer.current_epoch),
                    "value": current,
                    "consecutive_epochs": self._stall_count,
                }
                trainer.should_stop = True
                rank_zero_info(
                    f"[DivergenceStop] {self.monitor}={current:.10f} frozen for "
                    f"{self._stall_count} consecutive epochs -- stalled, stopping "
                    f"at epoch {trainer.current_epoch}."
                )
                return

        # --- Divergence detection ---
        if self.factor <= 0:
            return
        if self.best is None or current < self.best:
            self.best = current
            return
        if current > self.best * self.factor:
            self.stop_reason = {
                "reason": "diverged",
                "epoch": int(trainer.current_epoch),
                "value": current,
                "best": self.best,
                "factor": self.factor,
            }
            trainer.should_stop = True
            rank_zero_info(
                f"[DivergenceStop] {self.monitor}={current:.4f} exceeds "
                f"{self.factor:.2f}x best ({self.best:.4f}) -- diverged, stopping "
                f"at epoch {trainer.current_epoch}."
            )


# --------------------------------------------------------------------------
# Post-resume LR reduction (Phase 3 — divergence recovery)
# --------------------------------------------------------------------------


class ApplyLRReduction(Callback):
    """Apply a multiplicative LR reduction after Lightning restores optimizer state.

    Lightning's ``trainer.fit(ckpt_path=...)`` restores ``param_groups[i]["lr"]``
    from the checkpoint (verified: ``test_lightning_resume_restores_param_group_lr``).
    Any LR change made *before* the restore is silently overwritten.

    This callback fires on ``on_train_start`` — after the optimizer state is
    restored — and multiplies each param group's LR by ``factor``, then clamps
    each group's LR to the ``ReduceLROnPlateau`` scheduler's per-group
    ``min_lrs`` floor (if a plateau scheduler is configured).  Without the
    clamp, repeated recoveries can drive the LR below the floor, making the
    scheduler permanently inert (see ``_with_lr_schedule``'s docstring).

    When the clamp binds for any group, ``self.clamped_groups`` records
    per-group detail so the caller can include it in ``summary.json``.
    """

    def __init__(self, factor: float):
        self.factor = factor
        self.clamped_groups = None  # populated if any group is clamped

    def on_train_start(self, trainer, pl_module):
        # Read the floor from the live ReduceLROnPlateau, if present.
        min_lrs = self._get_min_lrs(trainer)

        clamped = []
        for opt in trainer.optimizers:
            for i, pg in enumerate(opt.param_groups):
                requested = pg["lr"] * self.factor
                if min_lrs is not None and i < len(min_lrs):
                    floor = min_lrs[i]
                    if requested < floor:
                        pg["lr"] = floor
                        clamped.append({
                            "group": i,
                            "requested_lr": requested,
                            "clamped_lr": floor,
                        })
                        rank_zero_info(
                            f"[ApplyLRReduction] group {i}: requested LR "
                            f"{requested:.4e} < floor {floor:.4e} — clamped "
                            f"to floor (recovery budget effectively spent)"
                        )
                        continue
                pg["lr"] = requested

        if clamped:
            self.clamped_groups = clamped

    @staticmethod
    def _get_min_lrs(trainer):
        """Read per-group min_lrs from the trainer's ReduceLROnPlateau, if any."""
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        for sched_cfg in trainer.lr_scheduler_configs:
            scheduler = sched_cfg.scheduler
            if isinstance(scheduler, ReduceLROnPlateau):
                return scheduler.min_lrs
        return None


# --------------------------------------------------------------------------
# Throughput callback
# --------------------------------------------------------------------------


class ThroughputCallback(Callback):
    """Log steps/s and samples/s over training (windowed)."""

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self._t0 = None
        self._n = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = time.perf_counter()
        self._n = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._n += 1
        dt = time.perf_counter() - self._t0
        if dt > 0:
            pl_module.log("steps_per_s", self._n / dt, prog_bar=True)
            pl_module.log("samples_per_s", self._n * self.batch_size / dt)


# --------------------------------------------------------------------------
# Dataset / model / trainer builders (importable for the ladder script)
# --------------------------------------------------------------------------


@dataclass
class TrainConfig:
    loss: str
    run_name: str
    max_epochs: int
    batch_size: int
    lr: float
    limit_batches: float | None
    num_workers: int
    seed: int
    patience: int
    store: str
    runs_root: str
    resume_from: str | None
    n_kernels: int = 512
    num_residual_layers: int = 2
    dropout: float = 0.15
    precision: str = "32"
    min_N: int = 50
    auto_lr: bool = False
    freeze_dispersion: bool = False
    dispersion_lr_scale: float = 1.0
    dispersion_window_size: int = 256
    model: str = "cnn"
    k: int = 6
    d_embed: int = 64
    d_context: int = 128
    n_context_layers: int = 2
    context_kernel_size: int = 15
    weight_decay: float = 0.0
    fl_dist_npz: str | None = None
    divergence_factor: float = 1.10
    stall_patience: int = 5
    lr_patience: int = 4
    max_lr_reductions: int = 3
    lr_factor: float = 0.5
    max_recoveries: int = 3
    recovery_factor: float = 0.5

    def __post_init__(self):
        if self.lr_patience < 1:
            raise ValueError(
                "lr_patience must be >= 1 (0 would make LR reduction and "
                "early-stopping detection ambiguous)"
            )
        if self.max_lr_reductions < 0:
            raise ValueError(
                "max_lr_reductions must be >= 0 (-1 would set min_lr above "
                "the initial LR, making the schedule silently inert)"
            )
        if self.patience < 1:
            raise ValueError(
                "patience must be >= 1 (derived from lr_patience and "
                "max_lr_reductions as (lr_patience+1)*max_lr_reductions "
                "+ lr_patience; 0 would make early-stopping detection "
                "ambiguous)"
            )
        if self.max_recoveries < 0:
            raise ValueError(
                "max_recoveries must be >= 0 (0 disables divergence recovery)"
            )
        if not (0 < self.recovery_factor <= 1):
            raise ValueError(
                "recovery_factor must be in (0, 1] "
                "(0 would zero every LR; >1 raises LR on recovery)"
            )
        if self.stall_patience == 1 or self.stall_patience < 0:
            raise ValueError(
                "stall_patience must be 0 (disabled) or >= 2 "
                "(1 would stop on the first validation epoch)"
            )


def _load_fl_band_fracs(fl_dist_npz: str, store_path: str) -> np.ndarray:
    """Load per-sample FL band fractions from an NPZ file.

    Returns (n_samples_in_store, n_bands) float32 array where each row sums
    to ~1 (the fraction of fragments in each FL band for that sample).
    Samples are matched by index order in the store.
    """
    from background_model_core import FL_BANDS

    data = np.load(fl_dist_npz)
    fl_counts = data["counts"]        # (n_fl_samples, n_lengths)
    fl_lengths = data["fragment_length"]  # (n_lengths,)

    # For each FL sample, compute fraction in each band
    n_fl = fl_counts.shape[0]
    n_bands = len(FL_BANDS)
    band_fracs = np.zeros((n_fl, n_bands), dtype=np.float32)
    totals = fl_counts.sum(axis=1, keepdims=True).astype(np.float64)
    totals = np.maximum(totals, 1.0)
    for b, (lo, hi) in enumerate(FL_BANDS):
        mask = (fl_lengths >= lo) & (fl_lengths < hi)
        band_fracs[:, b] = fl_counts[:, mask].sum(axis=1) / totals.ravel()

    # Match to store samples: the simulation assigns FL distributions to
    # samples in order (sample 0 gets fl_dist 0, etc.). For real data,
    # the mapping would come from the sample sheet.
    import zarr
    root = zarr.open_group(store_path, mode="r")
    n_store_samples = root["samples/role"].shape[0]

    if n_fl >= n_store_samples:
        return band_fracs[:n_store_samples]
    else:
        # Fewer FL samples than store samples — tile cyclically
        reps = (n_store_samples + n_fl - 1) // n_fl
        return np.tile(band_fracs, (reps, 1))[:n_store_samples]


def build_model(cfg: TrainConfig) -> L.LightningModule:
    lr_kw = dict(
        lr_patience=cfg.lr_patience,
        max_lr_reductions=cfg.max_lr_reductions,
        lr_factor=cfg.lr_factor,
    )
    if cfg.model == "ken":
        return InstrumentedBackgroundModelKEN(
            k=cfg.k, d_embed=cfg.d_embed, d_context=cfg.d_context,
            n_context_layers=cfg.n_context_layers,
            context_kernel_size=cfg.context_kernel_size,
            loss=cfg.loss, learning_rate=cfg.lr, dropout=cfg.dropout,
            weight_decay=cfg.weight_decay,
            freeze_dispersion=cfg.freeze_dispersion,
            dispersion_lr_scale=cfg.dispersion_lr_scale,
            dispersion_window_size=cfg.dispersion_window_size,
            **lr_kw,
        )
    if cfg.model == "hybrid":
        return InstrumentedBackgroundModelHybrid(
            k=cfg.k, d_embed=cfg.d_embed, n_kernels=cfg.n_kernels,
            num_residual_layers=cfg.num_residual_layers,
            loss=cfg.loss, learning_rate=cfg.lr, dropout=cfg.dropout,
            weight_decay=cfg.weight_decay,
            freeze_dispersion=cfg.freeze_dispersion,
            dispersion_lr_scale=cfg.dispersion_lr_scale,
            dispersion_window_size=cfg.dispersion_window_size,
            **lr_kw,
        )
    return InstrumentedBackgroundModel(
        loss=cfg.loss, learning_rate=cfg.lr, n_kernels=cfg.n_kernels,
        num_residual_layers=cfg.num_residual_layers, dropout=cfg.dropout,
        freeze_dispersion=cfg.freeze_dispersion,
        dispersion_lr_scale=cfg.dispersion_lr_scale,
        dispersion_window_size=cfg.dispersion_window_size,
        **lr_kw,
    )


def build_datasets(store: str, model: L.LightningModule, min_N: int = 50, seed: int = 1337,
                    fl_band_fracs=None):
    """train (jitter+RC ON) and val (center, no RC) datasets on the real store."""
    # Read tile_size from the store's own config so the harness works with any
    # tile size (production 16384 or simulation 2048).
    import zarr as _zarr
    _root = _zarr.open_group(store, mode="r")
    _tile_size = PlumbingConfig.from_json(_root.attrs["config_json"]).tile_size
    model_input_size = model.calc_input_region_size(_tile_size)
    train_ds = BackgroundTileDataset(
        store_path=store,
        model_input_size=model_input_size,
        split="train",
        sample_role="train",
        min_N=min_N,
        train_mode=True,   # jitter + RC ON
        seed=seed,
        fl_band_fracs=fl_band_fracs,
    )
    val_ds = BackgroundTileDataset(
        store_path=store,
        model_input_size=model_input_size,
        split="val",
        sample_role="train",
        min_N=min_N,
        train_mode=False,  # center crop, no RC
        seed=seed,
        fl_band_fracs=fl_band_fracs,
    )
    return train_ds, val_ds


def build_loaders(train_ds, val_ds, batch_size: int, num_workers: int):
    pin = torch.cuda.is_available()
    common = dict(
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **common
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False, **common
    )
    return train_loader, val_loader


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _write_run_meta(run_dir: str, cfg: TrainConfig, train_ds, val_ds):
    os.makedirs(run_dir, exist_ok=True)
    meta = {
        "run_name": cfg.run_name,
        "git_sha": _git_sha(),
        "config_hash": train_ds.config_hash,
        "split_version": int(train_ds.split_version),
        "store": cfg.store,
        "model": cfg.model,
        "loss": cfg.loss,
        "n_kernels": cfg.n_kernels,
        "num_residual_layers": cfg.num_residual_layers,
        "dropout": cfg.dropout,
        "auto_lr": cfg.auto_lr,
        "freeze_dispersion": cfg.freeze_dispersion,
        "dispersion_lr_scale": cfg.dispersion_lr_scale,
        "precision": cfg.precision,
        "max_epochs": cfg.max_epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "limit_batches": cfg.limit_batches,
        "num_workers": cfg.num_workers,
        "seed": cfg.seed,
        "patience": cfg.patience,
        "divergence_factor": cfg.divergence_factor,
        "stall_patience": cfg.stall_patience,
        "lr_patience": cfg.lr_patience,
        "max_lr_reductions": cfg.max_lr_reductions,
        "lr_factor": cfg.lr_factor,
        "max_recoveries": cfg.max_recoveries,
        "recovery_factor": cfg.recovery_factor,
        "n_train_pairs": len(train_ds),
        "n_val_pairs": len(val_ds),
        "plumbing_config": json.loads(train_ds.config.full_config_json()),
    }
    if cfg.model == "ken":
        meta.update({
            "k": cfg.k,
            "d_embed": cfg.d_embed,
            "d_context": cfg.d_context,
            "n_context_layers": cfg.n_context_layers,
            "context_kernel_size": cfg.context_kernel_size,
            "weight_decay": cfg.weight_decay,
        })
    elif cfg.model == "hybrid":
        meta.update({
            "k": cfg.k,
            "d_embed": cfg.d_embed,
            "weight_decay": cfg.weight_decay,
        })
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def build_trainer(cfg: TrainConfig, run_dir: str) -> L.Trainer:
    ckpt = ModelCheckpoint(
        dirpath=os.path.join(run_dir, "checkpoints"),
        monitor="val_loss",
        mode="min",
        save_top_k=-1,
        save_last=True,
        filename="{epoch}-{step}-{val_loss:.4f}",
    )
    early = EarlyStopping(
        monitor="val_loss", mode="min", patience=cfg.patience, min_delta=0.0
    )
    csv_logger = CSVLogger(save_dir=cfg.runs_root, name="", version=cfg.run_name)
    limit = cfg.limit_batches
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        precision=cfg.precision,
        max_epochs=cfg.max_epochs,
        gradient_clip_val=1.0,
        deterministic=True,
        default_root_dir=run_dir,
        logger=csv_logger,
        callbacks=[ckpt, early,
                   DivergenceStop(cfg.divergence_factor,
                                  stall_patience=cfg.stall_patience),
                   LearningRateMonitor(logging_interval="epoch"),
                   ThroughputCallback(cfg.batch_size),
                   DeviceStatsMonitor(cpu_stats=False)],
        limit_train_batches=limit if limit is not None else 1.0,
        limit_val_batches=limit if limit is not None else 1.0,
        log_every_n_steps=1,
        enable_progress_bar=False,
    )
    return trainer


def _determine_stop_reason(trainer):
    """Inspect trainer callbacks to determine why the run stopped.

    Returns a dict with at least ``{"reason": <str>, "epoch": <int>}``
    plus guard-specific detail.  Priority:

    1. Our ``DivergenceStop`` — it records its own reason directly.
    2. Lightning's ``EarlyStopping`` — ``stopped_epoch`` is 0 by default
       and set to ``trainer.current_epoch`` when it fires.  With
       ``patience >= 1`` the earliest possible firing epoch is
       ``patience``, so ``stopped_epoch > 0`` is a reliable "it fired"
       signal.  (Patience 0 would fire at epoch 0, making the check
       ambiguous; ``TrainConfig.__post_init__`` enforces patience >= 1.)
    3. Otherwise the run completed normally (hit ``max_epochs``).
    """
    for cb in trainer.callbacks:
        if isinstance(cb, DivergenceStop) and cb.stop_reason is not None:
            return cb.stop_reason
    for cb in trainer.callbacks:
        if isinstance(cb, EarlyStopping) and cb.stopped_epoch > 0:
            bs = cb.best_score
            if bs is None:
                best_score_val = None
            else:
                bs_float = float(bs)
                best_score_val = bs_float if math.isfinite(bs_float) else str(bs_float)
            return {
                "reason": "early_stopped",
                "epoch": int(cb.stopped_epoch),
                "patience": cb.patience,
                "best_score": best_score_val,
            }
    return {"reason": "completed", "epoch": int(trainer.current_epoch)}


class _RecoveryEpochTracker(Callback):
    """Track whether any training epochs started during a fit."""

    def __init__(self):
        self.epochs_started = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.epochs_started += 1


def run_recovery_loop(build_and_fit, stop_reason, global_best_score,
                       global_best_path, max_recoveries, recovery_factor):
    """Divergence recovery outer loop (§4).

    Retries training after divergence, restoring the best checkpoint and
    reducing the LR by *recovery_factor* for each attempt.  The target LR
    for the Nth recovery is ``recovery_factor ** N`` times the original
    learning rate, but two mechanisms may cause the actual LR to differ:

    1. Plateau reductions inside an improving attempt are preserved in the
       checkpoint and not compensated — the stored LR reflects them.
    2. Each group's LR is clamped to the ``ReduceLROnPlateau`` scheduler's
       per-group ``min_lrs`` floor.  When the clamp binds, the recovery
       event records the per-group detail in ``"floor_clamped_groups"``.

    Parameters
    ----------
    build_and_fit : callable(ckpt_path, extra_callbacks) -> Trainer
        Builds a fresh Trainer (with fresh callbacks), calls
        ``trainer.fit(model, ..., ckpt_path=ckpt_path)`` with the given
        extra callbacks appended, and returns the fitted Trainer.
    stop_reason : dict
        Result of ``_determine_stop_reason`` from the preceding fit.
    global_best_score : float | None
        Best val_loss across all fits so far.
    global_best_path : str
        Path to the checkpoint with *global_best_score*.
    max_recoveries : int
        Maximum number of recovery attempts (0 disables recovery).
    recovery_factor : float
        Multiplicative LR reduction per recovery (e.g. 0.5 → halve).

    Returns
    -------
    tuple of (stop_reason, global_best_score, global_best_path, recoveries)
    """
    recoveries = []
    recovery_count = 0
    # Track how many recovery reductions are baked into the current
    # best checkpoint's stored LR so the applied factor always targets
    # recovery_factor**N × original_lr.  The actual LR may be higher
    # (clamped to the scheduler floor) or lower (plateau reductions
    # inside an improving attempt baked into the checkpoint).
    best_recovery_level = 0

    while (
        stop_reason.get("reason") == "diverged"
        and recovery_count < max_recoveries
    ):
        resume_ckpt = global_best_path
        if not resume_ckpt:
            rank_zero_info(
                "[Recovery] No checkpoint available for recovery — "
                "stopping with diverged_unrecovered."
            )
            stop_reason = {
                "reason": "diverged_unrecovered",
                "epoch": stop_reason["epoch"],
                "recoveries_attempted": recovery_count,
                "detail": "no checkpoint available",
            }
            break

        recovery_count += 1
        # Factor relative to the resume checkpoint's stored LR so
        # that the effective LR = recovery_factor**N × original_lr.
        applied_factor = recovery_factor ** (recovery_count - best_recovery_level)

        rank_zero_info(
            f"[Recovery] Attempt {recovery_count}/{max_recoveries}: "
            f"restoring {resume_ckpt}, LR *= {applied_factor:.4g} "
            f"(target {recovery_factor ** recovery_count:.4g}x original)"
        )

        recovery_event = {
            "attempt": recovery_count,
            "epoch": stop_reason["epoch"],
            "pre_divergence_best": stop_reason.get("best"),
            "diverged_value": stop_reason.get("value"),
            "recovery_lr_factor": applied_factor,
            "checkpoint": resume_ckpt,
        }

        lr_cb = ApplyLRReduction(applied_factor)
        epoch_tracker = _RecoveryEpochTracker()
        trainer = build_and_fit(
            resume_ckpt, [lr_cb, epoch_tracker]
        )

        # Record floor-clamp detail if any group was clamped
        if lr_cb.clamped_groups is not None:
            recovery_event["floor_clamped_groups"] = lr_cb.clamped_groups
        stop_reason = _determine_stop_reason(trainer)

        # A recovery that trains zero epochs (max_epochs already reached
        # from the checkpoint's epoch) must not be reported as completed.
        if epoch_tracker.epochs_started == 0:
            recovery_event["zero_epochs"] = True
            recoveries.append(recovery_event)
            stop_reason = {
                "reason": "diverged_unrecovered",
                "epoch": recovery_event["epoch"],
                "recoveries_attempted": recovery_count,
                "detail": "recovery trained zero epochs (max_epochs reached)",
            }
            break

        recoveries.append(recovery_event)

        # Update global best if this attempt improved on it.
        attempt_score = trainer.checkpoint_callback.best_model_score
        if attempt_score is not None:
            attempt_score_f = float(attempt_score)
            if global_best_score is None or attempt_score_f < global_best_score:
                global_best_score = attempt_score_f
                global_best_path = trainer.checkpoint_callback.best_model_path
                best_recovery_level = recovery_count

    # If we exhausted recoveries and still diverged, mark it.
    if (
        stop_reason.get("reason") == "diverged"
        and recovery_count > 0
        and recovery_count >= max_recoveries
    ):
        stop_reason = {
            "reason": "diverged_unrecovered",
            "epoch": stop_reason["epoch"],
            "recoveries_attempted": recovery_count,
            "final_value": stop_reason.get("value"),
            "best_before_final_divergence": stop_reason.get("best"),
        }

    return stop_reason, global_best_score, global_best_path, recoveries


def run_training(cfg: TrainConfig):
    L.seed_everything(cfg.seed, workers=True)
    run_dir = os.path.join(cfg.runs_root, cfg.run_name)
    model = build_model(cfg)

    # ── optional FL-conditioned loss reweighting ──────────────────────
    fl_band_fracs = None
    if cfg.fl_dist_npz is not None:
        fl_band_fracs = _load_fl_band_fracs(cfg.fl_dist_npz, cfg.store)

    train_ds, val_ds = build_datasets(
        cfg.store, model, min_N=cfg.min_N, seed=cfg.seed,
        fl_band_fracs=fl_band_fracs,
    )
    meta = _write_run_meta(run_dir, cfg, train_ds, val_ds)
    train_loader, val_loader = build_loaders(
        train_ds, val_ds, cfg.batch_size, cfg.num_workers
    )
    trainer = build_trainer(cfg, run_dir)
    if cfg.auto_lr:
        tuner = L.pytorch.tuner.Tuner(trainer)
        lr_result = tuner.lr_find(model, train_loader, val_loader)
        suggested = lr_result.suggestion()
        print(f"[auto-lr] suggested LR: {suggested:.6e}", flush=True)
        model.hparams.learning_rate = suggested
        model.learning_rate = suggested
        meta["lr"] = suggested
        meta["auto_lr_suggestion"] = suggested
        with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # ── initial fit ───────────────────────────────────────────────────
    ckpt_path = cfg.resume_from
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    stop_reason = _determine_stop_reason(trainer)

    global_best_score = (
        float(trainer.checkpoint_callback.best_model_score)
        if trainer.checkpoint_callback.best_model_score is not None
        else None
    )
    global_best_path = trainer.checkpoint_callback.best_model_path

    # ── divergence recovery outer loop (§4) ───────────────────────────
    def _build_and_fit(ckpt_path, extra_callbacks):
        t = build_trainer(cfg, run_dir)
        t.callbacks.extend(extra_callbacks)
        t.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
        return t

    stop_reason, global_best_score, global_best_path, recoveries = (
        run_recovery_loop(
            build_and_fit=_build_and_fit,
            stop_reason=stop_reason,
            global_best_score=global_best_score,
            global_best_path=global_best_path,
            max_recoveries=cfg.max_recoveries,
            recovery_factor=cfg.recovery_factor,
        )
    )

    # ── persist final metrics summary ─────────────────────────────────
    summary = {
        "best_model_path": global_best_path,
        "best_val_loss": global_best_score,
        "global_step": int(trainer.global_step),
        "current_epoch": int(trainer.current_epoch),
        "stop_reason": stop_reason,
    }
    if recoveries:
        summary["recoveries"] = recoveries
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump({**meta, **summary}, f, indent=2)
    return trainer, model


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Background model training driver")
    p.add_argument("--loss", choices=LOSSES, required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--max-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--limit-batches",
        type=float,
        default=None,
        help="limit_train_batches/limit_val_batches (int count or float frac) for smokes",
    )
    p.add_argument("--n-kernels", type=int, default=512, help="trunk width (smokes use small)")
    p.add_argument("--num-residual-layers", type=int, default=2, help="number of residual blocks")
    p.add_argument("--dropout", type=float, default=0.15, help="spatial dropout rate (0 to disable)")
    p.add_argument("--precision", default="32",
                   choices=["32", "16-mixed", "bf16-mixed"],
                   help="training precision (bf16-mixed for ~2x speedup on A10G)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--lr-patience", type=int, default=4,
                   help="epochs without val_loss improvement before cutting LR "
                        "(ReduceLROnPlateau patience)")
    p.add_argument("--max-lr-reductions", type=int, default=3,
                   help="max number of LR halvings; EarlyStopping patience is "
                        "derived as (lr_patience+1)*max_lr_reductions + lr_patience")
    p.add_argument("--divergence-factor", type=float, default=1.10,
                   help="stop if val_loss exceeds this multiple of its own "
                        "best (0 disables the ratio check; non-finite "
                        "detection is always active). Healthy v3 runs peak "
                        "at 1.0002x; diverged ones reach 380x+.")
    p.add_argument("--stall-patience", type=int, default=5,
                   help="stop if val_loss is bitwise identical for this many "
                        "consecutive epochs (0 disables). Healthy v3 runs "
                        "repeat at most 2; collapsed runs repeat indefinitely.")
    p.add_argument("--max-recoveries", type=int, default=3,
                   help="max divergence recovery attempts (0 disables recovery; "
                        "each recovery restores the best checkpoint and reduces LR)")
    p.add_argument("--recovery-factor", type=float, default=0.5,
                   help="multiplicative LR reduction per recovery attempt "
                        "(e.g. 0.5 = halve; Nth recovery trains at factor^N × original LR)")
    p.add_argument("--min-N", type=int, default=50,
                   help="min per-track count to include a (sample, tile) pair")
    p.add_argument("--store", default=DEFAULT_STORE)
    p.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    p.add_argument("--resume-from", default=None)
    p.add_argument("--auto-lr", action="store_true",
                   help="run Lightning LR finder before training")
    p.add_argument("--freeze-dispersion", action="store_true",
                   help="freeze dispersion head (NB/DM runs at init, ~multinomial)")
    p.add_argument("--dispersion-lr-scale", type=float, default=1.0,
                   help="relative LR for dispersion head (e.g. 0.1 = 10x slower)")
    p.add_argument("--dispersion-window-size", type=int, default=256,
                   help="dispersion pooling window in bp (1 = per-base, 256 = default)")
    # KEN model selection and hyperparameters
    p.add_argument("--model", choices=["cnn", "ken", "hybrid"], default="cnn")
    p.add_argument("--k", type=int, default=6,
                   help="k-mer size (KEN and hybrid only)")
    p.add_argument("--d-embed", type=int, default=64, help="embedding dimension (KEN only)")
    p.add_argument("--d-context", type=int, default=128, help="context conv channels (KEN only)")
    p.add_argument("--n-context-layers", type=int, default=2, help="number of context conv layers (KEN only)")
    p.add_argument("--context-kernel-size", type=int, default=15, help="context conv kernel size (KEN only)")
    p.add_argument("--weight-decay", type=float, default=0.0, help="L2 on embedding table (KEN only)")
    p.add_argument("--fl-dist-npz", default=None,
                   help="path to per-sample FL distribution NPZ (gw_fldist_*.npz); "
                        "enables FL-conditioned loss reweighting per track")
    return p


def cfg_from_args(args) -> TrainConfig:
    limit = args.limit_batches
    if limit is not None and float(limit).is_integer() and limit >= 1:
        limit = int(limit)
    lr_patience = args.lr_patience
    max_lr_reductions = args.max_lr_reductions
    # EarlyStopping patience derived from LR schedule.  ReduceLROnPlateau
    # reduces when num_bad_epochs > patience (strict >), so each reduction
    # takes lr_patience + 1 bad epochs.  EarlyStopping stops when
    # wait_count >= patience, i.e. after exactly `patience` bad epochs.
    # For the final LR level to train lr_patience epochs before stopping:
    patience = (lr_patience + 1) * max_lr_reductions + lr_patience
    try:
        return TrainConfig(
            loss=args.loss,
            run_name=args.run_name,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            limit_batches=limit,
            num_workers=args.num_workers,
            seed=args.seed,
            patience=patience,
            divergence_factor=args.divergence_factor,
            stall_patience=args.stall_patience,
            store=args.store,
            runs_root=args.runs_root,
            resume_from=args.resume_from,
            n_kernels=args.n_kernels,
            num_residual_layers=args.num_residual_layers,
            precision=args.precision,
            min_N=args.min_N,
            dropout=args.dropout,
            auto_lr=args.auto_lr,
            freeze_dispersion=args.freeze_dispersion,
            dispersion_lr_scale=args.dispersion_lr_scale,
            dispersion_window_size=args.dispersion_window_size,
            model=args.model,
            k=args.k,
            d_embed=args.d_embed,
            d_context=args.d_context,
            n_context_layers=args.n_context_layers,
            context_kernel_size=args.context_kernel_size,
            weight_decay=args.weight_decay,
            fl_dist_npz=args.fl_dist_npz,
            lr_patience=lr_patience,
            max_lr_reductions=max_lr_reductions,
            max_recoveries=args.max_recoveries,
            recovery_factor=args.recovery_factor,
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    cfg = cfg_from_args(args)
    run_training(cfg)


if __name__ == "__main__":
    main()
