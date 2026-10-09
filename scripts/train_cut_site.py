#!/usr/bin/env python
"""Train the cut-site fragment model."""
import argparse, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader
from background_model.cut_site_model import CutSiteFragmentModel
from background_model.cut_site_dataset import CutSiteFragmentDataset, cut_site_collate_fn


def _nvidia_smi_sample():
    """(utilisation %, VRAM used GiB) for GPU 0, or (None, None) if unavailable.

    Returns None rather than raising: this is telemetry, and a missing or
    broken nvidia-smi must never take down a 4-hour training run.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0]
        util, mib = (float(v) for v in out.split(","))
        return util, mib / 1024.0
    except Exception:
        return None, None


class GPUStatsCallback(L.Callback):
    """Per-epoch peak VRAM and mean SM utilisation.

    Deliberately NOT lightning's DeviceStatsMonitor, for two reasons: it reports
    torch.cuda memory only and so has no SM utilisation, which is half of what
    we need; and it logs per batch, which on a full-data run here would be ~31k
    rows. pynvml is absent from both envs, so utilisation comes from nvidia-smi.

    Utilisation is SAMPLED during the epoch and averaged, not read once at the
    epoch boundary. A single end-of-epoch reading lands at a point where the
    loader is draining and the GPU is quiet, which would understate it. Peak
    memory needs no sampling -- torch.cuda tracks the true high-water mark.
    """

    def __init__(self, sample_every: int = 50):
        super().__init__()
        self.sample_every = sample_every
        self._utils: list[float] = []
        self._mem: list[float] = []

    def on_train_epoch_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._utils.clear()
        self._mem.clear()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.sample_every:
            return
        util, used = _nvidia_smi_sample()
        if util is not None:
            self._utils.append(util)
            self._mem.append(used)

    def on_train_epoch_end(self, trainer, pl_module):
        if not torch.cuda.is_available():
            return
        stats = {
            "gpu_peak_alloc_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
            "gpu_peak_reserved_gb": torch.cuda.max_memory_reserved() / 2 ** 30,
        }
        if self._utils:
            stats["gpu_util_pct_mean"] = sum(self._utils) / len(self._utils)
            stats["gpu_util_pct_min"] = min(self._utils)
            stats["gpu_mem_used_gb_max"] = max(self._mem)
        pl_module.log_dict(stats, sync_dist=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tile-size", type=int, default=2048)
    p.add_argument("--L-min", type=int, default=25)
    p.add_argument("--L-max", type=int, default=180)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--nb-w0", type=float, default=1.0)
    p.add_argument("--nb-T-anneal", type=int, default=5)
    p.add_argument("--mult-weight", type=float, default=1.0,
                   help="Weight on multinomial loss (0=NB-only)")
    p.add_argument("--precision", default="bf16-mixed",
                   choices=["32", "bf16-mixed", "16-mixed"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--min-N", type=int, default=10)
    p.add_argument("--hex-tables", default=None,
                   help="Path to Stage 1 hex tables (.npz) for two-stage training")
    p.add_argument("--freeze-hex", action="store_true",
                   help="Freeze hex lookup tables (don't update during training)")
    p.add_argument("--limit-batches", type=float, default=1.0,
                   help="Fraction of train batches per epoch (0.1 = 10%%)")
    p.add_argument("--val-check-interval", type=float, default=1.0,
                   help="Validate this often, in epochs. <1.0 gives sub-epoch "
                        "checks. Needed when one epoch is large: on the 10%% sim "
                        "the optimum fell at epoch 6 of 5,210 train regions "
                        "(~31k tile-visits), which on a 53,320-region set is "
                        "epoch 0.58 -- i.e. BEFORE the first per-epoch "
                        "validation. Costs ~3.8%% of an epoch per extra check.")
    p.add_argument("--validate-only", action="store_true",
                   help="Run validation only, no training. On a freshly built "
                        "model this measures the model AT INIT.")
    p.add_argument("--zero-correction", action="store_true",
                   help="Zero w_out and len_bias so the correction f_theta is "
                        "exactly 0 and the logits are exactly log_p0. With "
                        "--validate-only, val_mult_nll IS the hex baseline, "
                        "measured on the model's own fragment population, "
                        "estimator and normalizer. Use this rather than a "
                        "standalone evaluator -- a second implementation of the "
                        "fragment rule silently drifted on four axes.")
    args = p.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)

    model = CutSiteFragmentModel(
        L_min=args.L_min, L_max=args.L_max, tile_size=args.tile_size,
        N1=128, K=128, N2=32, H=32, L_chunk=32,
        dispersion_window_size=256,
        learning_rate=args.lr, nb_w0=args.nb_w0, nb_T_anneal=args.nb_T_anneal,
        mult_weight=args.mult_weight,
        lr_patience=4, max_lr_reductions=3, lr_factor=0.5,
        hex_tables_path=args.hex_tables,
        freeze_hex=args.freeze_hex,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    if args.zero_correction:
        if args.hex_tables is None:
            p.error("--zero-correction requires --hex-tables: with no baseline "
                    "the logits would be identically zero and the NLL uniform")
        # logit = (hid * w_out).sum(-1) + len_bias, with no other additive term,
        # so zeroing both makes the correction identically zero regardless of
        # what the stem computes. The logits are then exactly log_p0.
        with torch.no_grad():
            model.w_out.zero_()
            model.len_bias.zero_()
        assert model.w_out.abs().max().item() == 0.0
        assert model.len_bias.abs().max().item() == 0.0
        print("zero-correction: f_theta == 0 (w_out=0, len_bias=0)")

    train_ds = CutSiteFragmentDataset(
        store_path=args.store, model_input_size=model.model_input_size,
        model_L_min=args.L_min, model_L_max=args.L_max,
        split="train", min_N=args.min_N, train_mode=True, preload=True,
    )
    val_ds = CutSiteFragmentDataset(
        store_path=args.store, model_input_size=model.model_input_size,
        model_L_min=args.L_min, model_L_max=args.L_max,
        split="val", min_N=args.min_N, train_mode=False, preload=True,
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=cut_site_collate_fn,
        persistent_workers=args.workers > 0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.workers // 2), collate_fn=cut_site_collate_fn,
        persistent_workers=True,
    )

    logger = CSVLogger(save_dir=args.run_dir, name="logs")
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto", devices=1,
        precision=args.precision,
        logger=logger,
        enable_checkpointing=True,
        default_root_dir=args.run_dir,
        log_every_n_steps=10,
        val_check_interval=args.val_check_interval,
        limit_train_batches=args.limit_batches,
        callbacks=[
            GPUStatsCallback(),
            # Keep the BEST epoch by validation multinomial NLL, not just the
            # last one.  Lightning's default ModelCheckpoint monitors nothing
            # and so persists only the final epoch; on the sim sample
            # (5,333 train tiles) val_mult_nll bottomed at epoch 5 and then
            # rose monotonically to epoch 19, so the only saved checkpoint was
            # 0.0182 nats WORSE than the best.  That went unnoticed until a
            # consumer needed to score fragments FROM the best model rather
            # than just report its NLL out of metrics.csv.
            # save_last keeps the final epoch too, so nothing is lost.
            ModelCheckpoint(
                monitor="val_mult_nll",
                mode="min",
                save_top_k=1,
                save_last=True,
                # step is in the name because sub-epoch validation
                # (--val-check-interval < 1) produces several candidates per
                # epoch, and {epoch} alone would not distinguish them.
                filename="best-{epoch:02d}-{step:06d}-{val_mult_nll:.6f}",
            ),
        ],
    )
    if args.validate_only:
        trainer.validate(model, val_dl)
    else:
        trainer.fit(model, train_dl, val_dl)

    metrics = trainer.callback_metrics
    for k in sorted(metrics):
        print(f"  {k}: {metrics[k]:.4f}")


if __name__ == "__main__":
    main()
