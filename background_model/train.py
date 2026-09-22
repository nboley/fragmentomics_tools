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
--lr --limit-batches`` plus ``--num-workers --seed --patience --store
--runs-root --resume-from`` conveniences.

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
import os
import subprocess
import time
from dataclasses import dataclass

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    Callback,
    DeviceStatsMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities import grad_norm
from torch.utils.data import DataLoader

from background_model.config import TILE, PlumbingConfig
from background_model.dataset import BackgroundTileDataset
from background_model_core import LOSSES, BackgroundModel, _prepare_mask

DEFAULT_STORE = (
    "/efs/analytics/nathanboley/background_model/stores/bg_store_b67d7c95.zarr"
)
DEFAULT_RUNS_ROOT = "/efs/analytics/nathanboley/background_model/runs"


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
        x, y, mask = batch
        mask3 = _prepare_mask(mask, y)
        shape_logits, dispersion_bp = self(x)

        if self.hparams.loss == "multinomial":
            log_disp = None
            loss = self.loss_fn(shape_logits, y, mask3)
        else:
            log_disp = self._pooled_log_dispersion(dispersion_bp, mask3)
            loss = self.loss_fn(shape_logits, log_disp, y, mask3)

        self.log(log_name, loss, prog_bar=True, sync_dist=True)
        self._log_per_track(log_name, shape_logits, log_disp, y, mask3)
        self._log_dispersion_trajectory(log_name, log_disp)
        return loss

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


def build_model(loss: str, lr: float, n_kernels: int = 512,
                num_residual_layers: int = 2,
                dropout: float = 0.15) -> InstrumentedBackgroundModel:
    return InstrumentedBackgroundModel(
        loss=loss, learning_rate=lr, n_kernels=n_kernels,
        num_residual_layers=num_residual_layers, dropout=dropout,
    )


def build_datasets(store: str, model: BackgroundModel, min_N: int = 50, seed: int = 1337):
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
    )
    val_ds = BackgroundTileDataset(
        store_path=store,
        model_input_size=model_input_size,
        split="val",
        sample_role="train",
        min_N=min_N,
        train_mode=False,  # center crop, no RC
        seed=seed,
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
        "loss": cfg.loss,
        "n_kernels": cfg.n_kernels,
        "num_residual_layers": cfg.num_residual_layers,
        "dropout": cfg.dropout,
        "auto_lr": cfg.auto_lr,
        "precision": cfg.precision,
        "max_epochs": cfg.max_epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "limit_batches": cfg.limit_batches,
        "num_workers": cfg.num_workers,
        "seed": cfg.seed,
        "patience": cfg.patience,
        "n_train_pairs": len(train_ds),
        "n_val_pairs": len(val_ds),
        "plumbing_config": json.loads(train_ds.config.full_config_json()),
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def build_trainer(cfg: TrainConfig, run_dir: str) -> L.Trainer:
    ckpt = ModelCheckpoint(
        dirpath=os.path.join(run_dir, "checkpoints"),
        monitor="val_loss",
        mode="min",
        save_top_k=2,
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
        callbacks=[ckpt, early, ThroughputCallback(cfg.batch_size),
                   DeviceStatsMonitor(cpu_stats=False)],
        limit_train_batches=limit if limit is not None else 1.0,
        limit_val_batches=limit if limit is not None else 1.0,
        log_every_n_steps=1,
        enable_progress_bar=False,
    )
    return trainer


def run_training(cfg: TrainConfig):
    L.seed_everything(cfg.seed, workers=True)
    run_dir = os.path.join(cfg.runs_root, cfg.run_name)
    model = build_model(cfg.loss, cfg.lr, cfg.n_kernels, cfg.num_residual_layers,
                        cfg.dropout)
    train_ds, val_ds = build_datasets(cfg.store, model, min_N=cfg.min_N, seed=cfg.seed)
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
        # Update meta with the discovered LR
        meta["lr"] = suggested
        meta["auto_lr_suggestion"] = suggested
        with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    trainer.fit(
        model, train_loader, val_loader, ckpt_path=cfg.resume_from
    )
    # persist final metrics summary
    summary = {
        "best_model_path": trainer.checkpoint_callback.best_model_path,
        "best_val_loss": float(trainer.checkpoint_callback.best_model_score)
        if trainer.checkpoint_callback.best_model_score is not None
        else None,
        "global_step": int(trainer.global_step),
        "current_epoch": int(trainer.current_epoch),
    }
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
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-N", type=int, default=50,
                   help="min per-track count to include a (sample, tile) pair")
    p.add_argument("--store", default=DEFAULT_STORE)
    p.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    p.add_argument("--resume-from", default=None)
    p.add_argument("--auto-lr", action="store_true",
                   help="run Lightning LR finder before training")
    return p


def cfg_from_args(args) -> TrainConfig:
    limit = args.limit_batches
    if limit is not None and float(limit).is_integer() and limit >= 1:
        limit = int(limit)
    return TrainConfig(
        loss=args.loss,
        run_name=args.run_name,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        limit_batches=limit,
        num_workers=args.num_workers,
        seed=args.seed,
        patience=args.patience,
        store=args.store,
        runs_root=args.runs_root,
        resume_from=args.resume_from,
        n_kernels=args.n_kernels,
        num_residual_layers=args.num_residual_layers,
        precision=args.precision,
        min_N=args.min_N,
        dropout=args.dropout,
        auto_lr=args.auto_lr,
    )


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    cfg = cfg_from_args(args)
    run_training(cfg)


if __name__ == "__main__":
    main()
