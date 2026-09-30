#!/usr/bin/env python
"""Train 12-track models (KEN / Hybrid / CNN) on tile1536 quiescent data.

Thin wrapper around ``background_model.train`` that provides defaults
matched to the cut-site fragment model comparison:

  - tile1536 store (if one exists; must be built separately if not)
  - multinomial loss (baseline for fragment NLL comparison)
  - 30 epochs, batch_size 64, bf16-mixed

After training, evaluate the best checkpoint with::

    python scripts/eval_fragment_nll.py \\
        --checkpoint <run_dir>/checkpoints/best.ckpt \\
        --model ken \\
        --store /efs/.../quiet_tile1536.zarr

Usage::

    # Train KEN on the tile1536 store
    python scripts/train_12track.py \\
        --model ken --run-name ken_tile1536_mult \\
        --store /efs/.../tile1536_bg_store.zarr

    # Train Hybrid
    python scripts/train_12track.py \\
        --model hybrid --run-name hybrid_tile1536_mult \\
        --store /efs/.../tile1536_bg_store.zarr

    # Optionally override any background_model.train argument:
    python scripts/train_12track.py \\
        --model ken --run-name ken_lr_sweep \\
        --store /efs/.../tile1536_bg_store.zarr \\
        --lr 5e-4 --max-epochs 50
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from background_model.train import build_arg_parser, cfg_from_args, run_training

# Default runs root for the comparison runs
DEFAULT_RUNS_ROOT = "/efs/analytics/nathanboley/background_model/runs"


def main():
    parser = build_arg_parser()

    # Make --loss optional (defaults to multinomial for this comparison)
    for action in parser._actions:
        if hasattr(action, "dest") and action.dest == "loss":
            action.required = False

    # Override defaults for the tile1536 comparison
    parser.set_defaults(
        loss="multinomial",
        max_epochs=30,
        batch_size=64,
        precision="bf16-mixed",
        lr=1e-3,
        min_N=10,
        runs_root=DEFAULT_RUNS_ROOT,
        num_workers=4,
    )

    args = parser.parse_args()

    # Sanity check: store must exist
    if not os.path.exists(args.store):
        print(f"ERROR: store not found at {args.store}")
        print()
        print("A 12-track store for the tile1536 regions must be built first.")
        print("See background_model/preprocess.py and scripts/batch_preprocess/")
        print("for the build pipeline.")
        sys.exit(1)

    cfg = cfg_from_args(args)
    print(f"Training {cfg.model} on {cfg.store}")
    print(f"  loss={cfg.loss}  lr={cfg.lr}  epochs={cfg.max_epochs}  "
          f"batch={cfg.batch_size}  precision={cfg.precision}")
    print(f"  run_dir={os.path.join(cfg.runs_root, cfg.run_name)}")
    print()

    run_training(cfg)


if __name__ == "__main__":
    main()
