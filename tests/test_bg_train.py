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


def _replay(values, factor=1.10, stall_patience=5):
    """Feed a val_loss sequence to the guard; return the stop epoch or None."""
    from background_model.train import DivergenceStop

    cb = DivergenceStop(factor, stall_patience=stall_patience)
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


# --------------------------------------------------------------------------
# Stall detection
#
# A collapsed model produces bitwise identical val_loss every epoch (the
# per-batch loss depends only on which tiles are in the batch, and the
# epoch-level mean stabilises quickly).  Healthy runs never repeat more than
# twice; the dead run (lrsweep_ken_lr2e-2) repeated 16 times.
# --------------------------------------------------------------------------


def test_stall_stop_fires_on_dead_trace():
    """lrsweep_ken_lr2e-2: val_loss frozen at 7.6127061844 for all 16 epochs."""
    dead = [7.6127061844] * 16
    # patience=5 → epochs 0-4 are 5 identical values → stop at epoch 4
    assert _replay(dead, stall_patience=5) == 4


def test_stall_stop_tolerates_healthy_consecutive():
    """Healthy runs (e.g. lrsweep_ken_lr2e-3) have at most 2 consecutive identical values."""
    trace = [7.60, 7.58, 7.57, 7.57, 7.56, 7.55, 7.55, 7.54]
    assert _replay(trace, stall_patience=5) is None


def test_stall_counter_resets_on_change():
    """Stall counter must reset when the metric changes."""
    # 4 identical, then a change, then 4 identical — never hits patience=5
    # factor=0 disables divergence so the 1.0→2.0 jump doesn't interfere
    trace = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
    assert _replay(trace, factor=0, stall_patience=5) is None


def test_stall_stop_disabled_by_zero_patience():
    """stall_patience=0 disables the stall check entirely."""
    dead = [7.6127061844] * 16
    assert _replay(dead, stall_patience=0) is None


def test_stall_patience_cli_default_and_override():
    p = build_arg_parser()
    required = ["--loss", "multinomial", "--run-name", "t"]
    assert cfg_from_args(p.parse_args(required)).stall_patience == 5
    cfg = cfg_from_args(p.parse_args(required + ["--stall-patience", "10"]))
    assert cfg.stall_patience == 10


# --------------------------------------------------------------------------
# stop_reason recording
#
# DivergenceStop.stop_reason is populated when a guard fires, so that
# summary.json can report *why* the run stopped without parsing stdout.
# --------------------------------------------------------------------------


def _replay_reason(values, factor=1.10, stall_patience=5):
    """Like _replay but return (stop_epoch, stop_reason_dict) or (None, None)."""
    from background_model.train import DivergenceStop

    cb = DivergenceStop(factor, stall_patience=stall_patience)
    tr = _FakeTrainer()
    for epoch, v in enumerate(values):
        tr.current_epoch = epoch
        tr.callback_metrics = {"val_loss": torch.tensor(float(v))}
        cb.on_validation_end(tr, None)
        if tr.should_stop:
            return epoch, cb.stop_reason
    return None, None


def test_stop_reason_diverged():
    trace = [7.6042, 7.6021, 7.6008, 7.5984, 7.5783,
             7.5649, 7.5648, 7.5646, 7.5645, 8.7314]
    epoch, reason = _replay_reason(trace)
    assert epoch == 9
    assert reason["reason"] == "diverged"
    assert reason["epoch"] == 9
    assert reason["value"] == pytest.approx(8.7314, rel=1e-5)
    assert reason["best"] == pytest.approx(7.5645, rel=1e-5)
    assert reason["factor"] == 1.10


def test_stop_reason_stalled():
    dead = [7.6127061844] * 16
    epoch, reason = _replay_reason(dead, stall_patience=5)
    assert epoch == 4
    assert reason["reason"] == "stalled"
    assert reason["epoch"] == 4
    assert reason["value"] == pytest.approx(7.6127061844, rel=1e-5)
    assert reason["consecutive_epochs"] == 5


def test_stop_reason_non_finite_inf():
    epoch, reason = _replay_reason([7.6, 7.5, float("inf")])
    assert epoch == 2
    assert reason["reason"] == "non_finite"
    assert reason["epoch"] == 2
    assert reason["value"] == "inf"


def test_stop_reason_non_finite_nan():
    epoch, reason = _replay_reason([7.6, 7.5, float("nan")])
    assert epoch == 2
    assert reason["reason"] == "non_finite"
    assert reason["epoch"] == 2
    assert reason["value"] == "nan"


def test_stop_reason_none_when_healthy():
    epoch, reason = _replay_reason([7.60, 7.58, 7.56, 7.55, 7.55])
    assert epoch is None
    assert reason is None


def test_nonfinite_not_mislabeled_as_stall():
    """Repeated inf with factor=0 must be diagnosed as non_finite, not stall.

    Regression test for the ordering defect in db45699: the stall check ran
    before the non-finite check, and inf == inf is True, so repeated inf
    incremented the stall counter.  With factor=0 (divergence disabled) the
    non-finite check was inside the divergence block and never reached.
    """
    trace = [float("inf")] * 6
    epoch, reason = _replay_reason(trace, factor=0, stall_patience=5)
    assert epoch == 0  # fires on the very first inf
    assert reason["reason"] == "non_finite"
