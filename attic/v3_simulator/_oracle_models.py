"""Dataset and model loading/scoring helpers for the oracle scripts.

Extracted verbatim from the former ``scripts/nb_oracle_v2.py`` when that
monolith was deleted in favour of the compute/render split (Phase 2 of
docs/pending/oracle_compute_render_split.md).  The monolith had become both a
library that ``nb_oracle_v2_compute.py`` imported from AND a competing
executable writing the same published artifact; these functions are the
library half.

This is plumbing, not statistical specification.  ``score_model_nb`` consumes
the frozen-core loss via ``_oracle_scoring.make_oracle_loss_fn`` rather than
constructing one.
"""

from __future__ import annotations

import json

import numpy as np
import torch


def load_checkpoint_path(summary_path):
    with open(summary_path) as f:
        s = json.load(f)
    return s["best_model_path"], s


def build_dataset(store_path, model_input_size):
    from background_model.dataset import BackgroundTileDataset
    return BackgroundTileDataset(
        store_path=store_path,
        model_input_size=model_input_size,
        split="val",
        sample_role="train",
        min_N=0,
        train_mode=False,
        seed=1337,
    )


def load_model(ckpt_path, model_type):
    from background_model.train import (
        InstrumentedBackgroundModelKEN,
        InstrumentedBackgroundModelHybrid,
    )
    cls = {
        "ken": InstrumentedBackgroundModelKEN,
        "hybrid": InstrumentedBackgroundModelHybrid,
    }[model_type]
    model = cls.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    return model


def create_untrained_model(model_type, summary):
    from background_model_core import BackgroundModelKEN, BackgroundModelHybrid
    from scripts._oracle_scoring import ORACLE_DISPERSION_WINDOW_SIZE
    if model_type == "ken":
        model = BackgroundModelKEN(
            k=summary.get("k", 6),
            d_embed=summary.get("d_embed", 64),
            d_context=summary.get("d_context", 128),
            n_context_layers=summary.get("n_context_layers", 2),
            context_kernel_size=summary.get("context_kernel_size", 15),
            dropout=summary.get("dropout", 0.0),
            loss="nb_offset",
            dispersion_window_size=ORACLE_DISPERSION_WINDOW_SIZE,
            freeze_dispersion=True,
        )
    elif model_type == "hybrid":
        model = BackgroundModelHybrid(
            k=summary.get("k", 6),
            d_embed=summary.get("d_embed", 64),
            n_kernels=summary.get("n_kernels", 128),
            num_residual_layers=summary.get("num_residual_layers", 3),
            dropout=summary.get("dropout", 0.0),
            loss="nb_offset",
            dispersion_window_size=ORACLE_DISPERSION_WINDOW_SIZE,
            freeze_dispersion=True,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    model.eval()
    return model


def score_model_nb(model, ds, device="cpu"):
    """Score a model under NB-offset loss with frozen dispersion (r from model).

    The model's forward returns (shape_logits, dispersion_bp) where
    dispersion_bp is the RAW per-position delta. The training loop applies
    _pooled_log_dispersion which: (1) mean-pools to the window level, and
    (2) adds log_dispersion_init. We call that method directly.

    Deliberately NOT unified with ``_oracle_scoring.eval_loss_at_log_r``: that
    one broadcasts a scalar log_r, this one uses the model's own pooled
    dispersion. Same loss object, different dispersion source.
    """
    from background_model_core import _prepare_mask
    from scripts._oracle_scoring import make_oracle_loss_fn
    loss_fn = make_oracle_loss_fn()
    model = model.to(device)
    nlls = []
    with torch.no_grad():
        for i in range(len(ds)):
            x, y, mask = ds[i]
            x = x.unsqueeze(0).to(device)
            shape_logits, dispersion_bp = model(x)
            shape_logits = shape_logits.cpu()
            dispersion_bp = dispersion_bp.cpu()
            y_t = y.unsqueeze(0)
            mask3 = _prepare_mask(mask.unsqueeze(0), y_t)

            log_disp = model._pooled_log_dispersion(dispersion_bp, mask3)

            nll = loss_fn(shape_logits, log_disp, y_t, mask3).item()
            nlls.append(nll)
    return np.array(nlls)
