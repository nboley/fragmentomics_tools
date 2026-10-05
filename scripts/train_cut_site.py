#!/usr/bin/env python
"""Train the cut-site fragment model."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightning as L
import torch
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader
from background_model.cut_site_model import CutSiteFragmentModel
from background_model.cut_site_dataset import CutSiteFragmentDataset, cut_site_collate_fn


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
    p.add_argument("--baseline-only", action="store_true",
                   help="Zero the neural correction and run validation only, so "
                        "val_mult_nll IS the hex-table baseline measured on the "
                        "model's own fragment population, estimator and "
                        "normalizer. Use this instead of a standalone baseline "
                        "evaluator -- a second implementation of the fragment "
                        "filter silently drifted from the dataset on four axes.")
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

    if args.baseline_only:
        if args.hex_tables is None:
            p.error("--baseline-only requires --hex-tables")
        # logit = (hid * w_out).sum(-1) + len_bias, with no other additive term,
        # so zeroing both makes the correction identically zero regardless of
        # what the stem computes. The logits are then exactly log_p0.
        with torch.no_grad():
            model.w_out.zero_()
            model.len_bias.zero_()
        assert model.w_out.abs().max().item() == 0.0
        assert model.len_bias.abs().max().item() == 0.0
        print("baseline-only: correction zeroed (w_out=0, len_bias=0)")

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
        limit_train_batches=args.limit_batches,
    )
    if args.baseline_only:
        trainer.validate(model, val_dl)
    else:
        trainer.fit(model, train_dl, val_dl)

    metrics = trainer.callback_metrics
    for k in sorted(metrics):
        print(f"  {k}: {metrics[k]:.4f}")


if __name__ == "__main__":
    main()
