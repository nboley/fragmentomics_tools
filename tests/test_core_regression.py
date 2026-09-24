"""Golden-output regression tests for background_model_core.py refactoring.

Fixtures are generated from the pre-refactor code and saved to
tests/data/core_regression_golden.pt.  On subsequent runs, values are
recomputed and compared for EXACT equality — any drift means the refactor
changed semantics and must be investigated, not accommodated.
"""
import pytest
import torch
import numpy as np
from pathlib import Path

from background_model_core import (
    BackgroundModel,
    BackgroundModelKEN,
    BackgroundModelHybrid,
    _prepare_mask,
    LOSSES,
)

GOLDEN_PATH = Path(__file__).parent / "data" / "core_regression_golden.pt"
OUT_LEN = 64
DISP_WIN = 32
SEED = 12345
TRACKS = [
    "strand_+__fl_40_65__coverage_first",
    "strand_+__fl_40_65__coverage_last",
    "strand_-__fl_40_65__coverage_first",
    "strand_-__fl_40_65__coverage_last",
]

_MODELS = {
    "BackgroundModel": (
        BackgroundModel,
        dict(
            output_tracks=TRACKS, n_kernels=16, kernel_size=8,
            num_residual_layers=1, dropout=0.0, learning_rate=1e-3,
            dispersion_window_size=DISP_WIN, log_dispersion_init=7.0,
            max_dispersion_ratio=2.0, clamp_margin=1.0,
            freeze_dispersion=False, dispersion_lr_scale=0.5,
        ),
    ),
    "BackgroundModelKEN": (
        BackgroundModelKEN,
        dict(
            output_tracks=TRACKS, k=4, d_embed=16, d_context=16,
            n_context_layers=1, context_kernel_size=5, dropout=0.0,
            learning_rate=1e-3, dispersion_window_size=DISP_WIN,
            log_dispersion_init=7.0, max_dispersion_ratio=2.0,
            clamp_margin=1.0, freeze_dispersion=False,
            dispersion_lr_scale=0.5, weight_decay=0.01,
        ),
    ),
    "BackgroundModelHybrid": (
        BackgroundModelHybrid,
        dict(
            output_tracks=TRACKS, k=4, d_embed=16, n_kernels=16,
            kernel_size=8, num_residual_layers=1, dropout=0.0,
            learning_rate=1e-3, dispersion_window_size=DISP_WIN,
            log_dispersion_init=7.0, max_dispersion_ratio=2.0,
            clamp_margin=1.0, freeze_dispersion=False,
            dispersion_lr_scale=0.5, weight_decay=0.01,
        ),
    ),
}


def _build(model_name, loss):
    """Create model and deterministic synthetic inputs."""
    cls, base_kw = _MODELS[model_name]
    torch.manual_seed(SEED)
    model = cls(**{**base_kw, "loss": loss})
    model.eval()

    gen = torch.Generator().manual_seed(SEED + 1)
    L_in = model.calc_input_region_size(OUT_LEN)

    idx = torch.randint(0, 4, (L_in,), generator=gen)
    x = torch.zeros(4, L_in)
    x.scatter_(0, idx.unsqueeze(0), 1.0)

    y = torch.randint(0, 20, (len(TRACKS), OUT_LEN), generator=gen).float()
    mask = torch.ones(OUT_LEN, dtype=torch.bool)
    mask[:4] = False
    y[:, ~mask] = 0.0
    return model, x, y, mask


def _capture(model, x, y, mask):
    """Compute every output that must survive the refactor unchanged."""
    with torch.no_grad():
        sl, dbp = model(x[None])
        m3 = _prepare_mask(mask[None], y[None])
        if model.hparams.loss == "multinomial":
            loss_val = model.loss_fn(sl, y[None], m3)
            pld = None
        else:
            pld = model._pooled_log_dispersion(dbp, m3)
            loss_val = model.loss_fn(sl, pld, y[None], m3)

    prof = model.predict_profile(x.numpy(), mask.numpy())

    opt_result = model.configure_optimizers()
    optim = opt_result["optimizer"]
    sched = opt_result["lr_scheduler"]["scheduler"]

    return {
        "shape_logits": sl[0].clone(),
        "dispersion_bp": dbp[0].clone() if dbp is not None else None,
        "pooled_log_disp": pld[0].clone() if pld is not None else None,
        "loss": loss_val.item(),
        "probs": torch.from_numpy(prof["probs"].copy()),
        "log_disp": (
            torch.from_numpy(prof["log_dispersion"].copy())
            if prof["log_dispersion"] is not None
            else None
        ),
        "opt": {
            "n_groups": len(optim.param_groups),
            "lrs": [g["lr"] for g in optim.param_groups],
            "wds": [g.get("weight_decay", 0.0) for g in optim.param_groups],
            "sched_type": type(sched).__name__,
            "min_lrs": list(sched.min_lrs),
            "monitor": opt_result["lr_scheduler"]["monitor"],
            "factor": sched.factor,
            "patience": sched.patience,
            "mode": sched.mode,
        },
    }


def _load_or_generate_golden():
    if GOLDEN_PATH.exists():
        return torch.load(GOLDEN_PATH, weights_only=False)
    data = {}
    for name in _MODELS:
        for loss in LOSSES:
            m, x, y, mask = _build(name, loss)
            data[(name, loss)] = _capture(m, x, y, mask)
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, GOLDEN_PATH)
    return data


@pytest.fixture(scope="module")
def golden():
    return _load_or_generate_golden()


COMBOS = [(n, l) for n in _MODELS for l in LOSSES]


@pytest.mark.parametrize(
    "name,loss", COMBOS, ids=[f"{n}-{l}" for n, l in COMBOS]
)
def test_golden_regression(golden, name, loss):
    """All outputs must exactly match pre-refactor golden values."""
    model, x, y, mask = _build(name, loss)
    act = _capture(model, x, y, mask)
    ref = golden[(name, loss)]

    # Forward outputs
    assert torch.equal(act["shape_logits"], ref["shape_logits"]), \
        "shape_logits changed"
    if ref["dispersion_bp"] is not None:
        assert torch.equal(act["dispersion_bp"], ref["dispersion_bp"]), \
            "dispersion_bp changed"
    else:
        assert act["dispersion_bp"] is None

    # Pooled log-dispersion
    if ref["pooled_log_disp"] is not None:
        assert torch.equal(act["pooled_log_disp"], ref["pooled_log_disp"]), \
            "pooled_log_disp changed"
    else:
        assert act["pooled_log_disp"] is None

    # Loss (exact float equality)
    assert act["loss"] == ref["loss"], \
        f"loss changed: {act['loss']} != {ref['loss']}"

    # predict_profile
    np.testing.assert_array_equal(act["probs"].numpy(), ref["probs"].numpy())
    if ref["log_disp"] is not None:
        np.testing.assert_array_equal(
            act["log_disp"].numpy(), ref["log_disp"].numpy()
        )
    else:
        assert act["log_disp"] is None

    # Optimizer structure
    for key in ("n_groups", "lrs", "wds", "sched_type", "min_lrs",
                "monitor", "factor", "patience", "mode"):
        assert act["opt"][key] == ref["opt"][key], \
            f"opt.{key}: {act['opt'][key]} != {ref['opt'][key]}"


# -- eval-mode leak (Task 3) ----------------------------------------------

@pytest.mark.parametrize("name", list(_MODELS.keys()))
def test_predict_profile_restores_train_mode(name):
    """A model in train() mode must still be in train() mode afterward."""
    model, x, _, mask = _build(name, "multinomial")
    model.train()
    assert model.training
    model.predict_profile(x.numpy(), mask.numpy())
    assert model.training, (
        f"{name}.predict_profile leaked eval mode"
    )


@pytest.mark.parametrize("name", list(_MODELS.keys()))
def test_predict_profile_preserves_eval_mode(name):
    """A model in eval() mode must still be in eval() mode afterward."""
    model, x, _, mask = _build(name, "multinomial")
    model.eval()
    assert not model.training
    model.predict_profile(x.numpy(), mask.numpy())
    assert not model.training


if __name__ == "__main__":
    if GOLDEN_PATH.exists():
        GOLDEN_PATH.unlink()
    _load_or_generate_golden()
    print(f"Golden fixture regenerated: {GOLDEN_PATH}")
