"""Tests for the training driver (background_model/train.py).

The key contract: InstrumentedBackgroundModel adds only detached diagnostic
logging; its training loss must be byte-for-byte the frozen
BackgroundModel._step loss.  Also covers CLI arg parsing.
"""
import torch
import pytest

from background_model_core import BackgroundModel
from background_model.train import (
    InstrumentedBackgroundModel,
    build_arg_parser,
    cfg_from_args,
)

L_OUT = 512  # divisible by the default nb dispersion_window_size (256)


def _batch(model, B=2, n_tracks=12):
    L_in = model.calc_input_region_size(L_OUT)
    x = torch.randn(B, 4, L_in)
    y = torch.randint(0, 5, (B, n_tracks, L_OUT)).float()
    mask = torch.ones(B, L_OUT, dtype=torch.bool)
    return x, y, mask


@pytest.mark.parametrize("loss", ["multinomial", "dirichlet_multinomial", "nb_offset"])
def test_instrumented_loss_matches_frozen(loss):
    torch.manual_seed(0)
    model = InstrumentedBackgroundModel(loss=loss, n_kernels=8)
    model.eval()  # deterministic (dropout off)
    batch = _batch(model)

    child = model._step(batch, "train_loss")
    parent = BackgroundModel._step(model, batch, "train_loss")

    assert torch.equal(child.detach(), parent.detach())
    assert torch.isfinite(child)


def test_arg_parser_limit_batches_int_vs_float():
    p = build_arg_parser()
    args = p.parse_args(
        ["--loss", "multinomial", "--run-name", "r", "--limit-batches", "3"]
    )
    cfg = cfg_from_args(args)
    assert cfg.limit_batches == 3 and isinstance(cfg.limit_batches, int)

    args = p.parse_args(
        ["--loss", "nb_offset", "--run-name", "r", "--limit-batches", "0.25"]
    )
    cfg = cfg_from_args(args)
    assert cfg.limit_batches == 0.25 and isinstance(cfg.limit_batches, float)


def test_arg_parser_requires_loss_and_run_name():
    p = build_arg_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["--run-name", "r"])  # missing --loss


# --------------------------------------------------------------------------
# DivergenceStop
#
# EarlyStopping(check_finite=True) only catches NaN/inf.  Real divergences on
# the v3 sim stayed finite (8.73, 383, 8.4e8) and ran to the epoch cap.  The
# thresholds below are taken from those runs: healthy peaks at 1.0002x best,
# diverged reaches 380x.
# --------------------------------------------------------------------------


class _FakeTrainer:
    def __init__(self):
        self.should_stop = False
        self.callback_metrics = {}
        self.current_epoch = 0
        self.sanity_checking = False


def _replay(values, factor=1.10):
    """Feed a val_loss sequence to the guard; return the stop epoch or None."""
    from background_model.train import DivergenceStop

    cb = DivergenceStop(factor)
    tr = _FakeTrainer()
    for epoch, v in enumerate(values):
        tr.current_epoch = epoch
        tr.callback_metrics = {"val_loss": torch.tensor(float(v))}
        cb.on_validation_end(tr, None)
        if tr.should_stop:
            return epoch
    return None


def test_divergence_stop_ignores_healthy_descent():
    assert _replay([7.60, 7.58, 7.56, 7.55, 7.55]) is None


def test_divergence_stop_tolerates_healthy_noise():
    """Healthy v3 runs peaked at 1.0002x best; the guard must not fire."""
    assert _replay([7.6016, 7.5665, 7.5668, 7.5666, 7.5687, 7.5665]) is None


def test_divergence_stop_fires_on_real_hybrid_trace():
    """sim_v3_A_hybrid_k6: clean to ep 8, then 7.5645 -> 8.7314."""
    trace = [7.6042, 7.6021, 7.6008, 7.5984, 7.5783,
             7.5649, 7.5648, 7.5646, 7.5645, 8.7314]
    assert _replay(trace) == 9


def test_divergence_stop_fires_on_non_finite():
    assert _replay([7.6, 7.5, float("inf")]) == 2
    assert _replay([7.6, 7.5, float("nan")]) == 2


def test_divergence_stop_disabled_by_zero_factor():
    assert _replay([7.6, 1e9], factor=0) is None


def test_divergence_stop_skips_sanity_check():
    from background_model.train import DivergenceStop

    cb = DivergenceStop(1.10)
    tr = _FakeTrainer()
    tr.sanity_checking = True
    tr.callback_metrics = {"val_loss": torch.tensor(1e9)}
    cb.on_validation_end(tr, None)
    assert not tr.should_stop


def test_divergence_factor_cli_default_and_override():
    p = build_arg_parser()
    required = ["--loss", "multinomial", "--run-name", "t"]
    assert cfg_from_args(p.parse_args(required)).divergence_factor == 1.10
    cfg = cfg_from_args(p.parse_args(required + ["--divergence-factor", "2.5"]))
    assert cfg.divergence_factor == 2.5
