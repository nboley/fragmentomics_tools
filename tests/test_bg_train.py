"""Tests for the training driver (background_model/train.py).

The key contract: InstrumentedBackgroundModel adds only detached diagnostic
logging; its training loss must be byte-for-byte the frozen
BackgroundModel._step loss.  Also covers CLI arg parsing.
"""
import json
import os

import lightning as L
import torch
import pytest
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from background_model_core import BackgroundModel, _with_lr_schedule
from background_model.train import (
    ApplyLRReduction,
    DivergenceStop,
    InstrumentedBackgroundModel,
    TrainConfig,
    _determine_stop_reason,
    build_arg_parser,
    build_trainer,
    cfg_from_args,
    run_recovery_loop,
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
    # 1.005, lowered from 1.10 on 2026-09-24: 1.10 was tuned against
    # explosions (380x, 1.9e9x) and could not fire on the basin-loss
    # degradation actually observed in phase4 (1.0079x).
    assert cfg_from_args(p.parse_args(required)).divergence_factor == 1.005
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


# --------------------------------------------------------------------------
# _determine_stop_reason
#
# The function inspects trainer callbacks after fit() to classify why the run
# stopped.  Three branches: DivergenceStop fired, EarlyStopping fired, or
# the run completed normally.
# --------------------------------------------------------------------------


def _make_early_stopping(stopped_epoch=0, patience=5, best_score=None):
    """Build a real EarlyStopping with attributes pre-set for testing."""
    from lightning.pytorch.callbacks import EarlyStopping as _ES

    es = _ES(monitor="val_loss", patience=patience)
    es.stopped_epoch = stopped_epoch
    if best_score is not None:
        es.best_score = best_score
    return es


def _make_trainer_with_callbacks(callbacks, current_epoch=0):
    """Build a _FakeTrainer with an explicit callbacks list."""
    tr = _FakeTrainer()
    tr.current_epoch = current_epoch
    tr.callbacks = callbacks
    return tr


def test_determine_stop_reason_divergence_stop():
    """DivergenceStop.stop_reason takes priority over EarlyStopping."""
    div_cb = DivergenceStop(factor=1.10)
    div_cb.stop_reason = {
        "reason": "diverged",
        "epoch": 9,
        "value": 8.73,
        "best": 7.56,
        "factor": 1.10,
    }
    # Even with an EarlyStopping that also fired, DivergenceStop wins
    es_cb = _make_early_stopping(stopped_epoch=8, patience=5, best_score=torch.tensor(7.56))
    tr = _make_trainer_with_callbacks([div_cb, es_cb], current_epoch=9)
    result = _determine_stop_reason(tr)
    assert result["reason"] == "diverged"
    assert result["epoch"] == 9


def test_determine_stop_reason_early_stopped():
    """EarlyStopping branch fires when stopped_epoch > 0."""
    div_cb = DivergenceStop(factor=1.10)  # not fired
    es_cb = _make_early_stopping(stopped_epoch=12, patience=5, best_score=torch.tensor(7.55))
    tr = _make_trainer_with_callbacks([div_cb, es_cb], current_epoch=12)
    result = _determine_stop_reason(tr)
    assert result["reason"] == "early_stopped"
    assert result["epoch"] == 12
    assert result["patience"] == 5
    assert result["best_score"] == pytest.approx(7.55)


def test_determine_stop_reason_completed():
    """Run completed normally (hit max_epochs)."""
    div_cb = DivergenceStop(factor=1.10)  # not fired
    es_cb = _make_early_stopping(stopped_epoch=0, patience=5, best_score=torch.tensor(7.55))
    tr = _make_trainer_with_callbacks([div_cb, es_cb], current_epoch=99)
    result = _determine_stop_reason(tr)
    assert result["reason"] == "completed"
    assert result["epoch"] == 99


def test_determine_stop_reason_early_stopped_inf_best_score():
    """best_score=inf (Lightning's mode='min' init) is stringified for JSON safety.

    This path is currently unreachable in practice because the unconditional
    non-finite guard in DivergenceStop fires first and takes priority.  The
    guard is defensive — if EarlyStopping's internals or ordering ever changed,
    the JSON output would still be valid.
    """
    import json
    import math

    es_cb = _make_early_stopping(
        stopped_epoch=1, patience=5, best_score=torch.tensor(float("inf"))
    )
    tr = _make_trainer_with_callbacks([es_cb], current_epoch=1)
    result = _determine_stop_reason(tr)
    assert result["reason"] == "early_stopped"
    assert result["best_score"] == "inf"
    # Verify it round-trips through json.dumps without bare Infinity
    serialised = json.dumps(result)
    assert "Infinity" not in serialised
    assert '"inf"' in serialised


# --------------------------------------------------------------------------
# CLI validation (findings 1 and 5)
# --------------------------------------------------------------------------


def test_lr_patience_cli_default_and_override():
    p = build_arg_parser()
    required = ["--loss", "multinomial", "--run-name", "t"]
    cfg = cfg_from_args(p.parse_args(required))
    assert cfg.lr_patience == 4
    assert cfg.max_lr_reductions == 3
    # EarlyStopping patience derived: (4+1)*3 + 4 = 19
    assert cfg.patience == 19

    cfg2 = cfg_from_args(p.parse_args(
        required + ["--lr-patience", "3", "--max-lr-reductions", "1"]
    ))
    assert cfg2.lr_patience == 3
    assert cfg2.max_lr_reductions == 1
    # EarlyStopping patience derived: (3+1)*1 + 3 = 7
    assert cfg2.patience == 7


def test_stall_patience_rejects_one():
    p = build_arg_parser()
    args = p.parse_args(
        ["--loss", "multinomial", "--run-name", "t", "--stall-patience", "1"]
    )
    with pytest.raises(SystemExit):
        cfg_from_args(args)


def test_stall_patience_rejects_negative():
    p = build_arg_parser()
    args = p.parse_args(
        ["--loss", "multinomial", "--run-name", "t", "--stall-patience", "-1"]
    )
    with pytest.raises(SystemExit):
        cfg_from_args(args)


def test_stall_patience_zero_accepted():
    """stall_patience=0 is the documented disable path."""
    p = build_arg_parser()
    args = p.parse_args(
        ["--loss", "multinomial", "--run-name", "t", "--stall-patience", "0"]
    )
    cfg = cfg_from_args(args)
    assert cfg.stall_patience == 0


# --------------------------------------------------------------------------
# Direct-construction validation (TrainConfig.__post_init__)
#
# The invariants must hold regardless of how TrainConfig is built —
# not just through the CLI path.
# --------------------------------------------------------------------------

# Minimal required fields for direct TrainConfig construction.
# `patience` is DERIVED from the LR schedule, so pin it to the same expression
# cfg_from_args uses rather than a literal — a bare 19 silently drifts out of
# agreement the moment lr_patience or max_lr_reductions changes.
_LR_PATIENCE = TrainConfig.lr_patience
_MAX_LR_REDUCTIONS = TrainConfig.max_lr_reductions
_DERIVED_PATIENCE = (_LR_PATIENCE + 1) * _MAX_LR_REDUCTIONS + _LR_PATIENCE

_REQUIRED = dict(
    loss="multinomial", run_name="t", max_epochs=10, batch_size=8,
    lr=1e-4, limit_batches=None, num_workers=0, seed=1337,
    patience=_DERIVED_PATIENCE, store="/tmp/fake.zarr", runs_root="/tmp/runs",
    resume_from=None,
)


def test_trainconfig_rejects_patience_zero():
    with pytest.raises(ValueError, match="patience must be >= 1"):
        TrainConfig(**{**_REQUIRED, "patience": 0})


def test_trainconfig_rejects_patience_negative():
    with pytest.raises(ValueError, match="patience must be >= 1"):
        TrainConfig(**{**_REQUIRED, "patience": -1})


def test_trainconfig_rejects_lr_patience_zero():
    with pytest.raises(ValueError, match="lr_patience must be >= 1"):
        TrainConfig(**{**_REQUIRED, "lr_patience": 0})


def test_trainconfig_rejects_max_lr_reductions_negative():
    with pytest.raises(ValueError, match="max_lr_reductions must be >= 0"):
        TrainConfig(**{**_REQUIRED, "max_lr_reductions": -1})


def test_trainconfig_rejects_stall_patience_one():
    with pytest.raises(ValueError, match="stall_patience must be 0.*or >= 2"):
        TrainConfig(**{**_REQUIRED, "stall_patience": 1})


def test_trainconfig_rejects_stall_patience_negative():
    with pytest.raises(ValueError, match="stall_patience must be 0.*or >= 2"):
        TrainConfig(**{**_REQUIRED, "stall_patience": -1})


def test_trainconfig_accepts_valid_defaults():
    """Default patience=19 (derived) and stall_patience=5 must construct cleanly."""
    cfg = TrainConfig(**_REQUIRED)
    assert cfg.patience == 19
    assert cfg.stall_patience == 5


def test_trainconfig_accepts_stall_patience_zero():
    """stall_patience=0 is the documented disable path."""
    cfg = TrainConfig(**{**_REQUIRED, "stall_patience": 0})
    assert cfg.stall_patience == 0


def test_trainconfig_accepts_stall_patience_two():
    """stall_patience=2 is the minimum enabled value."""
    cfg = TrainConfig(**{**_REQUIRED, "stall_patience": 2})
    assert cfg.stall_patience == 2


# --------------------------------------------------------------------------
# Phase 1: Empirical verification of Lightning checkpoint/LR behavior
#
# Design §8 inferred (but did not verify) that:
#   (a) Lightning checkpoints include optimizer_states
#   (b) trainer.fit(ckpt_path=...) restores param_groups[i]["lr"]
# These tests confirm both empirically with a tiny synthetic model.
# --------------------------------------------------------------------------


class _TinyModel(L.LightningModule):
    """Minimal multi-param-group model for checkpoint tests."""

    def __init__(self, lr=1e-2, secondary_lr_scale=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.layer1 = torch.nn.Linear(4, 4)
        self.layer2 = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        scale = self.hparams.secondary_lr_scale
        return torch.optim.Adam([
            {"params": list(self.layer1.parameters()), "lr": lr},
            {"params": list(self.layer2.parameters()), "lr": lr * scale},
        ])


def _tiny_dataloader(n=16, in_dim=4, out_dim=2):
    x = torch.randn(n, in_dim)
    y = torch.randn(n, out_dim)
    ds = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(ds, batch_size=8)


def test_lightning_checkpoint_contains_optimizer_states(tmp_path):
    """Design §8 verification (a): checkpoints include optimizer_states."""
    model = _TinyModel(lr=1e-2)
    ckpt_cb = ModelCheckpoint(dirpath=str(tmp_path), save_top_k=-1,
                              filename="{epoch}")
    trainer = L.Trainer(
        max_epochs=2, default_root_dir=str(tmp_path),
        enable_progress_bar=False, enable_model_summary=False,
        callbacks=[ckpt_cb], accelerator="cpu",
    )
    dl = _tiny_dataloader()
    trainer.fit(model, dl, dl)

    # Load the epoch-1 checkpoint and inspect
    ckpt_path = ckpt_cb.best_model_path
    assert ckpt_path, "no checkpoint written"
    ckpt = torch.load(ckpt_path, weights_only=False)
    assert "optimizer_states" in ckpt, (
        "Lightning checkpoint does NOT contain optimizer_states — "
        "design §4.2 assumption is WRONG"
    )
    opt_state = ckpt["optimizer_states"]
    assert len(opt_state) >= 1
    # Verify param_groups with LR are recorded
    param_groups = opt_state[0]["param_groups"]
    assert len(param_groups) == 2
    assert "lr" in param_groups[0]
    assert "lr" in param_groups[1]


def test_lightning_resume_restores_param_group_lr(tmp_path):
    """Design §8 verification (b): ckpt_path restores param_groups[i]["lr"].

    Train 2 epochs at lr=1e-2 (secondary=1e-3), save checkpoint.
    Build a NEW model with lr=5e-5 (secondary=5e-6), resume from checkpoint.
    After resume, optimizer LRs should match the CHECKPOINT (1e-2 / 1e-3),
    not the freshly-constructed model (5e-5 / 5e-6).
    """
    # Phase 1: train and checkpoint
    model1 = _TinyModel(lr=1e-2, secondary_lr_scale=0.1)
    ckpt_cb = ModelCheckpoint(dirpath=str(tmp_path / "ckpts"), save_last=True)
    trainer1 = L.Trainer(
        max_epochs=2, default_root_dir=str(tmp_path / "run1"),
        enable_progress_bar=False, enable_model_summary=False,
        callbacks=[ckpt_cb], accelerator="cpu",
    )
    dl = _tiny_dataloader()
    trainer1.fit(model1, dl, dl)
    ckpt_path = ckpt_cb.last_model_path
    assert ckpt_path

    # Phase 2: new model with DIFFERENT LR, resume from checkpoint
    model2 = _TinyModel(lr=5e-5, secondary_lr_scale=0.1)
    trainer2 = L.Trainer(
        max_epochs=4, default_root_dir=str(tmp_path / "run2"),
        enable_progress_bar=False, enable_model_summary=False,
        accelerator="cpu",
    )
    trainer2.fit(model2, dl, dl, ckpt_path=str(ckpt_path))

    # Check what LR the optimizer actually has
    opt = trainer2.optimizers[0]
    restored_lr0 = opt.param_groups[0]["lr"]
    restored_lr1 = opt.param_groups[1]["lr"]

    # If Lightning restores LR from checkpoint: 1e-2 and 1e-3
    # If it uses the model's LR: 5e-5 and 5e-6
    assert restored_lr0 == pytest.approx(1e-2, rel=1e-4), (
        f"LR group 0 = {restored_lr0}; expected 1e-2 from checkpoint. "
        f"Design §4.3 assumption may be WRONG."
    )
    assert restored_lr1 == pytest.approx(1e-3, rel=1e-4), (
        f"LR group 1 = {restored_lr1}; expected 1e-3 from checkpoint. "
        f"Design §4.3 assumption may be WRONG."
    )


# --------------------------------------------------------------------------
# Phase 1: Checkpoint retention and LR logging
# --------------------------------------------------------------------------


def test_build_trainer_saves_all_epochs(tmp_path):
    """save_top_k=-1 retains every epoch's checkpoint."""
    cfg = TrainConfig(**{**_REQUIRED, "runs_root": str(tmp_path)})
    run_dir = os.path.join(str(tmp_path), cfg.run_name)
    trainer = build_trainer(cfg, run_dir)
    ckpt_cb = trainer.checkpoint_callback
    assert ckpt_cb.save_top_k == -1, f"expected save_top_k=-1, got {ckpt_cb.save_top_k}"
    assert ckpt_cb.save_last is True


def test_build_trainer_includes_lr_monitor(tmp_path):
    """LearningRateMonitor is in the trainer's callback list."""
    from lightning.pytorch.callbacks import LearningRateMonitor

    cfg = TrainConfig(**{**_REQUIRED, "runs_root": str(tmp_path)})
    run_dir = os.path.join(str(tmp_path), cfg.run_name)
    trainer = build_trainer(cfg, run_dir)
    lr_monitors = [cb for cb in trainer.callbacks
                   if isinstance(cb, LearningRateMonitor)]
    assert len(lr_monitors) == 1, "expected exactly one LearningRateMonitor"
    assert lr_monitors[0].logging_interval == "epoch"


# --------------------------------------------------------------------------
# Phase 2: ReduceLROnPlateau — per-group LR ratios, min_lr floor, derived patience
# --------------------------------------------------------------------------


class _FakeHparams:
    """Minimal hparams namespace for _with_lr_schedule tests."""
    def __init__(self, lr_patience=4, max_lr_reductions=3, lr_factor=0.5):
        self.lr_patience = lr_patience
        self.max_lr_reductions = max_lr_reductions
        self.lr_factor = lr_factor


def _make_multi_group_optimizer(lr=1e-2, scale=0.1):
    """3-group optimizer mimicking hybrid: embed(lr,wd), main(lr,wd=0), disp(lr*scale,wd=0)."""
    layers = [torch.nn.Linear(4, 4) for _ in range(3)]
    optimizer = torch.optim.Adam([
        {"params": list(layers[0].parameters()), "lr": lr, "weight_decay": 0.01},
        {"params": list(layers[1].parameters()), "lr": lr, "weight_decay": 0.0},
        {"params": list(layers[2].parameters()), "lr": lr * scale, "weight_decay": 0.0},
    ])
    return optimizer, layers


def _simulate_plateau_reductions(optimizer, scheduler, n_bad_epochs):
    """Feed n_bad_epochs of worsening val_loss to trigger reductions."""
    for i in range(n_bad_epochs):
        scheduler.step(100.0 + i)  # always worse


def test_lr_ratios_survive_one_reduction():
    """Per-group LR ratios are preserved after one ReduceLROnPlateau step."""
    opt, _ = _make_multi_group_optimizer(lr=1e-2, scale=0.1)
    hp = _FakeHparams(lr_patience=2, max_lr_reductions=3, lr_factor=0.5)
    result = _with_lr_schedule(opt, hp)
    scheduler = result["lr_scheduler"]["scheduler"]

    # Record initial ratios
    initial_lrs = [g["lr"] for g in opt.param_groups]
    assert initial_lrs[0] == pytest.approx(1e-2)
    assert initial_lrs[1] == pytest.approx(1e-2)
    assert initial_lrs[2] == pytest.approx(1e-3)

    # Trigger one reduction: patience=2 → step 0 sets best, steps 1-2
    # are bad (patience countdown), step 3 triggers the reduction.
    _simulate_plateau_reductions(opt, scheduler, 4)

    reduced_lrs = [g["lr"] for g in opt.param_groups]
    # Each group halved
    assert reduced_lrs[0] == pytest.approx(5e-3)
    assert reduced_lrs[1] == pytest.approx(5e-3)
    assert reduced_lrs[2] == pytest.approx(5e-4)
    # Ratios preserved
    assert reduced_lrs[2] / reduced_lrs[0] == pytest.approx(0.1)


def test_lr_ratios_hold_at_min_lr_floor():
    """Per-group LR ratios are preserved when LR hits the min_lr floor.

    This is the specific failure mode a scalar min_lr would cause: all
    groups would converge to the same floor, destroying dispersion_lr_scale.
    """
    opt, _ = _make_multi_group_optimizer(lr=1e-2, scale=0.1)
    hp = _FakeHparams(lr_patience=1, max_lr_reductions=2, lr_factor=0.5)
    result = _with_lr_schedule(opt, hp)
    scheduler = result["lr_scheduler"]["scheduler"]

    # Drive LR to the floor: max_lr_reductions=2 → 2 halvings allowed
    # patience=1 → 2 bad epochs per reduction
    _simulate_plateau_reductions(opt, scheduler, 20)

    floor_lrs = [g["lr"] for g in opt.param_groups]
    # Floor = initial * factor^max_lr_reductions
    assert floor_lrs[0] == pytest.approx(1e-2 * 0.25)  # 2.5e-3
    assert floor_lrs[1] == pytest.approx(1e-2 * 0.25)
    assert floor_lrs[2] == pytest.approx(1e-3 * 0.25)  # 2.5e-4
    # Ratio preserved at the floor
    assert floor_lrs[2] / floor_lrs[0] == pytest.approx(0.1)


def test_reduction_cap_binds():
    """LR does not go below initial * factor^max_lr_reductions."""
    opt, _ = _make_multi_group_optimizer(lr=1e-2, scale=0.1)
    hp = _FakeHparams(lr_patience=1, max_lr_reductions=3, lr_factor=0.5)
    result = _with_lr_schedule(opt, hp)
    scheduler = result["lr_scheduler"]["scheduler"]

    # Feed many bad epochs — more than enough to exhaust all reductions
    _simulate_plateau_reductions(opt, scheduler, 50)

    lrs = [g["lr"] for g in opt.param_groups]
    # 3 halvings: floor = initial * 0.5^3 = initial / 8
    assert lrs[0] == pytest.approx(1e-2 / 8)
    assert lrs[2] == pytest.approx(1e-3 / 8)


def test_derived_early_stopping_patience_max_reductions_3():
    """EarlyStopping patience = (lr_patience+1)*max_lr_reductions + lr_patience = 5*3+4 = 19."""
    p = build_arg_parser()
    cfg = cfg_from_args(p.parse_args(
        ["--loss", "multinomial", "--run-name", "t",
         "--lr-patience", "4", "--max-lr-reductions", "3"]
    ))
    assert cfg.patience == 19


def test_derived_early_stopping_patience_max_reductions_1():
    """EarlyStopping patience = (lr_patience+1)*max_lr_reductions + lr_patience = 5*1+4 = 9."""
    p = build_arg_parser()
    cfg = cfg_from_args(p.parse_args(
        ["--loss", "multinomial", "--run-name", "t",
         "--lr-patience", "4", "--max-lr-reductions", "1"]
    ))
    assert cfg.patience == 9


def test_derived_early_stopping_patience_max_reductions_0():
    """max_lr_reductions=0: patience = lr_patience = 4 (no LR reductions, just early stop)."""
    p = build_arg_parser()
    cfg = cfg_from_args(p.parse_args(
        ["--loss", "multinomial", "--run-name", "t",
         "--lr-patience", "4", "--max-lr-reductions", "0"]
    ))
    assert cfg.patience == 4


def test_configure_optimizers_returns_scheduler():
    """All three model classes return a scheduler dict from configure_optimizers."""
    from background_model_core import BackgroundModelKEN, BackgroundModelHybrid

    for cls, kw in [
        (BackgroundModel, {"n_kernels": 8}),
        (BackgroundModelKEN, {"d_embed": 8, "d_context": 8}),
        (BackgroundModelHybrid, {"n_kernels": 8, "d_embed": 8}),
    ]:
        model = cls(loss="multinomial", learning_rate=1e-3,
                    lr_patience=4, max_lr_reductions=3, lr_factor=0.5, **kw)
        result = model.configure_optimizers()
        assert isinstance(result, dict), f"{cls.__name__} should return a dict"
        assert "optimizer" in result
        assert "lr_scheduler" in result
        sched_cfg = result["lr_scheduler"]
        assert sched_cfg["monitor"] == "val_loss"
        scheduler = sched_cfg["scheduler"]
        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


# --------------------------------------------------------------------------
# Finding 4 — CLI bounds validation for --lr-patience and --max-lr-reductions
# --------------------------------------------------------------------------


def test_lr_patience_rejects_zero_via_cli():
    """--lr-patience 0 triggers the validation error through cfg_from_args."""
    p = build_arg_parser()
    args = p.parse_args(
        ["--loss", "multinomial", "--run-name", "t", "--lr-patience", "0"]
    )
    with pytest.raises(SystemExit):
        cfg_from_args(args)


def test_max_lr_reductions_rejects_negative_via_cli():
    """--max-lr-reductions -1 triggers the validation error through cfg_from_args."""
    p = build_arg_parser()
    args = p.parse_args(
        ["--loss", "multinomial", "--run-name", "t", "--max-lr-reductions", "-1"]
    )
    with pytest.raises(SystemExit):
        cfg_from_args(args)


def test_max_lr_reductions_zero_accepted():
    """max_lr_reductions=0 is valid: no LR schedule, pure early stopping."""
    p = build_arg_parser()
    cfg = cfg_from_args(p.parse_args(
        ["--loss", "multinomial", "--run-name", "t", "--max-lr-reductions", "0"]
    ))
    assert cfg.max_lr_reductions == 0
    # patience = (4+1)*0 + 4 = 4
    assert cfg.patience == 4


# --------------------------------------------------------------------------
# Behavioural test: the derived patience formula produces the intended
# training policy in a real Lightning loop.
#
# This is the test that would have caught the off-by-one in the original
# formula.  It runs an actual Trainer with ReduceLROnPlateau and
# EarlyStopping both active, on a synthetic model that never improves,
# and asserts that the final LR level receives lr_patience training epochs
# before EarlyStopping fires.
# --------------------------------------------------------------------------


class _NeverImprovingModel(L.LightningModule):
    """Model that returns a constant, worsening val_loss per epoch.

    Each epoch's val_loss = 100 + epoch, so the metric never improves.
    This drives both ReduceLROnPlateau and EarlyStopping on their worst-case
    (no improvement) path.
    """

    def __init__(self, lr=1e-2, lr_patience=2, max_lr_reductions=2, lr_factor=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.layer = torch.nn.Linear(4, 2)
        self.lr_history = []  # (epoch, lr) recorded at each validation

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self(x), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Constant worsening loss so the metric never improves
        loss = 100.0 + self.current_epoch
        self.log("val_loss", float(loss))
        return torch.tensor(loss)

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.lr_history.append((self.current_epoch, lr))

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return _with_lr_schedule(opt, self.hparams)


def test_final_lr_level_gets_lr_patience_epochs(tmp_path):
    """The final LR level must receive lr_patience training epochs before stopping.

    This is the behavioural assertion that the derived EarlyStopping patience
    formula is correct.  With lr_patience=2, max_lr_reductions=2, factor=0.5:

    - ReduceLROnPlateau fires after lr_patience+1 = 3 bad epochs (strict >)
    - 2 reductions happen, then lr_patience=2 more bad epochs, then stop
    - Derived patience = (2+1)*2 + 2 = 8
    - Total epochs trained = 1 (sets best) + 8 (bad) = 9 (epochs 0-8)
    - Final LR level starts after epoch with the 2nd reduction and must
      get exactly lr_patience=2 epochs of training before EarlyStopping fires.
    """
    lr_patience = 2
    max_reductions = 2
    lr_factor = 0.5
    initial_lr = 1e-2
    derived_patience = (lr_patience + 1) * max_reductions + lr_patience  # 8

    model = _NeverImprovingModel(
        lr=initial_lr, lr_patience=lr_patience,
        max_lr_reductions=max_reductions, lr_factor=lr_factor,
    )

    early = EarlyStopping(
        monitor="val_loss", mode="min", patience=derived_patience, min_delta=0.0
    )

    trainer = L.Trainer(
        max_epochs=50,  # high cap; EarlyStopping should fire well before this
        default_root_dir=str(tmp_path),
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[early],
        accelerator="cpu",
        log_every_n_steps=1,
    )

    dl = _tiny_dataloader()
    trainer.fit(model, dl, dl)

    # EarlyStopping must have fired (not hit max_epochs)
    assert early.stopped_epoch > 0, "EarlyStopping did not fire"

    # Identify the final LR level from the recorded history
    final_lr = initial_lr * lr_factor ** max_reductions
    epochs_at_final_lr = [
        ep for ep, lr in model.lr_history
        if abs(lr - final_lr) / final_lr < 1e-6
    ]

    assert len(epochs_at_final_lr) == lr_patience, (
        f"Final LR level ({final_lr}) should get exactly {lr_patience} epochs "
        f"of training before EarlyStopping fires, but got {len(epochs_at_final_lr)}. "
        f"LR history: {model.lr_history}"
    )


# --------------------------------------------------------------------------
# Phase 3: Divergence recovery
#
# These tests exercise the recovery mechanism using real Lightning training
# loops with synthetic models that deterministically diverge.  Per the task
# instructions: "For anything you can express as a loop invariant, write a
# test that RUNS a real Lightning loop and asserts the observed behaviour."
# --------------------------------------------------------------------------


class _DivergingModel(L.LightningModule):
    """Model that produces good val_loss then diverges at controlled epochs.

    val_loss = base_loss + trend*epoch  (``trend`` defaults to -0.01, i.e.
    a slow improvement).  A POSITIVE ``trend`` makes val_loss degrade with
    epoch, which is what lets a test construct a later-attempt-is-worse
    scenario — recovery resumes further along the trend, so each attempt
    scores worse than the one before.  With the default negative trend the
    opposite holds and every later attempt is better, which silently makes
    a global-best assertion vacuous (see
    ``test_global_best_tracks_across_attempts``).
    ... EXCEPT at epochs in ``spike_epochs``, where val_loss = base_loss * spike.

    ``spike_epochs`` is a set of absolute epoch numbers.  After recovery
    (checkpoint restore + resume), the epoch counter persists, so the model
    produces the same spike at the same epoch — which is the real-world
    behaviour (the model's weights haven't changed, so it hits the same
    bad gradient region).

    If ``spike_every`` is set, the model spikes at every ``spike_every``
    epochs starting from ``first_spike``, overriding ``spike_epochs``.
    This creates a model that always re-diverges, useful for testing
    exhaustion.

    Records (epoch, lr_group0, lr_group1) at each validation for assertions.
    Uses two param groups with different LRs (scale=0.1) to verify that
    per-group ratios survive recovery.
    """

    def __init__(self, lr=1e-2, scale=0.1, spike_epochs=None,
                 first_spike=3, spike_every=None,
                 spike=2.0, base_loss=10.0, trend=-0.01,
                 lr_patience=4, max_lr_reductions=3, lr_factor=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.layer1 = torch.nn.Linear(4, 4)
        self.layer2 = torch.nn.Linear(4, 2)
        self.lr_history = []
        # Non-hparam state: build the spike set
        if spike_every is not None:
            self._spike_epochs = set(
                range(first_spike, 200, spike_every)
            )
        elif spike_epochs is not None:
            self._spike_epochs = set(spike_epochs)
        else:
            self._spike_epochs = {first_spike}

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self(x), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        ep = self.current_epoch
        base = self.hparams.base_loss
        if ep in self._spike_epochs:
            loss = base * self.hparams.spike
        else:
            loss = base + self.hparams.trend * ep
        self.log("val_loss", float(loss))
        return torch.tensor(loss)

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            lrs = [pg["lr"] for pg in self.trainer.optimizers[0].param_groups]
            self.lr_history.append((self.current_epoch, *lrs))

    def configure_optimizers(self):
        lr = self.hparams.lr
        scale = self.hparams.scale
        opt = torch.optim.Adam([
            {"params": list(self.layer1.parameters()), "lr": lr},
            {"params": list(self.layer2.parameters()), "lr": lr * scale},
        ])
        return _with_lr_schedule(opt, self.hparams)


def _run_with_recovery(model, tmp_path, max_recoveries=3, recovery_factor=0.5,
                        divergence_factor=1.10, max_epochs=30,
                        lr_patience=4, max_lr_reductions=3):
    """Run a training loop with the production recovery loop.

    Returns (final_stop_reason, global_best_score, global_best_path,
             recoveries_list, model).
    """
    dl = _tiny_dataloader()
    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir, exist_ok=True)

    derived_patience = (lr_patience + 1) * max_lr_reductions + lr_patience

    def build_and_fit(ckpt_path, extra_callbacks):
        ckpt = ModelCheckpoint(
            dirpath=os.path.join(run_dir, "checkpoints"),
            monitor="val_loss", mode="min", save_top_k=-1, save_last=True,
            filename="{epoch}-{step}-{val_loss:.4f}",
        )
        early = EarlyStopping(
            monitor="val_loss", mode="min",
            patience=derived_patience, min_delta=0.0,
        )
        div = DivergenceStop(divergence_factor)
        cbs = [ckpt, early, div] + list(extra_callbacks)
        trainer = L.Trainer(
            max_epochs=max_epochs,
            default_root_dir=run_dir,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=cbs,
            accelerator="cpu",
            log_every_n_steps=1,
        )
        trainer.fit(model, dl, dl, ckpt_path=ckpt_path)
        return trainer

    # Initial fit
    trainer = build_and_fit(None, [])
    stop_reason = _determine_stop_reason(trainer)

    global_best_score = (
        float(trainer.checkpoint_callback.best_model_score)
        if trainer.checkpoint_callback.best_model_score is not None
        else None
    )
    global_best_path = trainer.checkpoint_callback.best_model_path

    # Production recovery loop
    stop_reason, global_best_score, global_best_path, recoveries = (
        run_recovery_loop(
            build_and_fit=build_and_fit,
            stop_reason=stop_reason,
            global_best_score=global_best_score,
            global_best_path=global_best_path,
            max_recoveries=max_recoveries,
            recovery_factor=recovery_factor,
        )
    )

    return stop_reason, global_best_score, global_best_path, recoveries, model


def test_recovery_fires_on_divergence(tmp_path):
    """When a model diverges, recovery restores the best checkpoint and continues.

    The model spikes at epoch 5 only.  After recovery from the best
    checkpoint (epoch ~4), the resumed run continues past epoch 5 without
    re-spiking (epoch 5 was the only spike), so it completes normally.
    """
    model = _DivergingModel(
        lr=1e-2, spike_epochs={5}, spike=2.0,
        lr_patience=2, max_lr_reductions=1, lr_factor=0.5,
    )
    stop_reason, best_score, _, recoveries, _ = _run_with_recovery(
        model, tmp_path, max_recoveries=1, max_epochs=20,
        lr_patience=2, max_lr_reductions=1,
    )

    # Recovery must have fired
    assert len(recoveries) == 1, f"expected 1 recovery, got {len(recoveries)}"
    assert recoveries[0]["attempt"] == 1
    assert recoveries[0]["recovery_lr_factor"] == 0.5

    # The run should NOT end as diverged (epoch 5 was the only spike)
    assert stop_reason["reason"] != "diverged", (
        f"Run still diverged after recovery: {stop_reason}"
    )
    assert best_score is not None


def test_recovery_exhaustion_produces_unrecovered(tmp_path):
    """When all recovery attempts are exhausted, stop_reason is diverged_unrecovered.

    The model spikes every 3 epochs (3, 6, 9, ...) so every recovery
    attempt eventually re-diverges.  With max_recoveries=2, we expect
    exactly 2 recoveries and then diverged_unrecovered.
    """
    model = _DivergingModel(
        lr=1e-2, first_spike=3, spike_every=3, spike=5.0, base_loss=10.0,
        lr_patience=4, max_lr_reductions=1, lr_factor=0.5,
    )
    stop_reason, _, _, recoveries, _ = _run_with_recovery(
        model, tmp_path, max_recoveries=2, max_epochs=50,
        lr_patience=4, max_lr_reductions=1,
    )

    assert stop_reason["reason"] == "diverged_unrecovered"
    assert stop_reason["recoveries_attempted"] == 2
    assert len(recoveries) == 2


def test_recovery_lr_ratios_preserved(tmp_path):
    """Per-group LR ratios survive divergence recovery.

    The _DivergingModel uses two param groups (lr, lr*0.1).  After recovery,
    the ratio must still be 0.1 — a scalar LR reset would destroy it.
    """
    model = _DivergingModel(
        lr=1e-2, scale=0.1, spike_epochs={5}, spike=2.0,
        lr_patience=2, max_lr_reductions=1, lr_factor=0.5,
    )
    _, _, _, recoveries, model = _run_with_recovery(
        model, tmp_path, max_recoveries=1, max_epochs=20,
        lr_patience=2, max_lr_reductions=1,
    )

    assert len(recoveries) >= 1, "Recovery did not fire"

    # Check every LR pair recorded after the recovery epoch
    recovery_epoch = recoveries[0]["epoch"]
    post_recovery_lrs = [
        (ep, lr0, lr1) for ep, lr0, lr1 in model.lr_history
        if ep >= recovery_epoch
    ]
    assert len(post_recovery_lrs) > 0, "No LR records after recovery"
    for ep, lr0, lr1 in post_recovery_lrs:
        ratio = lr1 / lr0
        assert ratio == pytest.approx(0.1, rel=1e-6), (
            f"LR ratio at epoch {ep}: {ratio} (expected 0.1). "
            f"LRs: group0={lr0}, group1={lr1}"
        )


def test_apply_lr_reduction_scales_all_groups(tmp_path):
    """ApplyLRReduction callback multiplies every param group's LR by factor.

    This is the mechanism that applies the recovery LR reduction AFTER
    Lightning restores optimizer state from checkpoint (§4.3).
    """
    model = _TinyModel(lr=1e-2, secondary_lr_scale=0.1)
    ckpt_cb = ModelCheckpoint(dirpath=str(tmp_path / "ckpts"), save_last=True)
    trainer1 = L.Trainer(
        max_epochs=2, default_root_dir=str(tmp_path / "run1"),
        enable_progress_bar=False, enable_model_summary=False,
        callbacks=[ckpt_cb], accelerator="cpu",
    )
    dl = _tiny_dataloader()
    trainer1.fit(model, dl, dl)
    ckpt_path = ckpt_cb.last_model_path

    # Resume with ApplyLRReduction(0.5) — LR should be halved from checkpoint
    model2 = _TinyModel(lr=1e-2, secondary_lr_scale=0.1)
    lr_cb = ApplyLRReduction(0.5)
    trainer2 = L.Trainer(
        max_epochs=4, default_root_dir=str(tmp_path / "run2"),
        enable_progress_bar=False, enable_model_summary=False,
        callbacks=[lr_cb], accelerator="cpu",
    )
    trainer2.fit(model2, dl, dl, ckpt_path=str(ckpt_path))

    opt = trainer2.optimizers[0]
    # Checkpoint had LR 1e-2 and 1e-3; after 0.5x reduction: 5e-3 and 5e-4
    assert opt.param_groups[0]["lr"] == pytest.approx(5e-3, rel=1e-4), (
        f"group0 LR = {opt.param_groups[0]['lr']}, expected 5e-3"
    )
    assert opt.param_groups[1]["lr"] == pytest.approx(5e-4, rel=1e-4), (
        f"group1 LR = {opt.param_groups[1]['lr']}, expected 5e-4"
    )
    # Ratio preserved
    ratio = opt.param_groups[1]["lr"] / opt.param_groups[0]["lr"]
    assert ratio == pytest.approx(0.1, rel=1e-4)


def test_global_best_tracks_across_attempts(tmp_path):
    """best_val_loss must reflect the global best, not just the last attempt.

    §4.5: fresh callbacks per attempt mean ModelCheckpoint.best_model_score
    resets.  A later WORSE attempt must not overwrite the global best.

    CHARACTERIZATION TEST -- READ THIS BEFORE TRUSTING IT AS A GUARD.

    This test CANNOT currently fail, and that is a deliberate, documented
    state rather than an oversight.  It was rewritten 2026-09-24 after
    test-audit flagged the original as vacuous; investigating the fix showed
    the vacuity ran deeper than the audit diagnosed.

    §4.5 warns that a fresh `ModelCheckpoint` per attempt loses
    `best_model_score`, so a later WORSE attempt could be reported as the
    run's result.  **That failure mode is not reachable as configured.**
    `build_trainer` uses `dirpath=<run_dir>/checkpoints` with `run_dir`
    constant across attempts, and Lightning only declines to reload
    `best_model_score` *when dirpath changes*
    (`model_checkpoint.py`, "The dirpath has changed ... won't be
    reloaded").  Dirpath is stable here, so every fresh ModelCheckpoint
    reloads the score from the resumed checkpoint -- which is always
    `global_best_path` -- and therefore already holds the global best.

    Measured directly: with the degrading trend below, attempts 2 and 3 can
    only reach 10.1 on their own, yet each reports best_model_score = 10.0.
    So "last attempt's best" and "global best" coincide by construction, and
    no assertion can separate them without diverging from how production
    actually builds its trainers.

    The explicit global-best tracking in `run_recovery_loop` is therefore
    correct and defensive, but NOT load-bearing today.  It becomes
    load-bearing the moment dirpath varies per attempt.  This test pins the
    invariant so that change is caught; it does not prove the tracking code
    is exercised.

    Timeline (base_loss=10.0, trend=+0.1, spike at epoch 3):
      initial fit : ep0=10.0 (best), ep1=10.1, ep2=10.2, ep3=30.0 -> diverge
      recovery 1  : resumes from ep0 ckpt; own reachable best would be 10.1
      recovery 2  : same
    """
    base_loss = 10.0
    model = _DivergingModel(
        lr=1e-2, spike_epochs={3}, spike=3.0, base_loss=base_loss,
        trend=+0.1,
        lr_patience=4, max_lr_reductions=1, lr_factor=0.5,
    )
    _, global_best, _, recoveries, _ = _run_with_recovery(
        model, tmp_path, max_recoveries=2, max_epochs=30,
        lr_patience=4, max_lr_reductions=1,
    )

    assert global_best is not None
    assert len(recoveries) >= 1, f"need >= 1 recovery, got {len(recoveries)}"

    # The initial fit's epoch-0 value is the only time base_loss is seen
    # undegraded, so it is the unique global optimum.  Every recovery
    # attempt resumes at epoch >= 1, where the degrading trend has already
    # pushed the reachable minimum to base_loss + 0.1.
    assert global_best == pytest.approx(base_loss, abs=1e-6), (
        f"global_best={global_best}, expected {base_loss} (the initial "
        f"fit's epoch-0 score). A value of ~{base_loss + 0.1} would mean "
        f"the run reported a later, worse attempt's best -- the §4.5 "
        f"regression. NOTE: per this test's docstring that cannot happen "
        f"while ModelCheckpoint.dirpath is stable across attempts, so if "
        f"this fires, check whether dirpath became per-attempt."
    )


def test_clean_run_unaffected_by_recovery(tmp_path):
    """A model that never diverges should produce no recoveries and
    a clean stop_reason."""
    # No spike epochs means it never diverges
    model = _DivergingModel(
        lr=1e-2, spike_epochs=set(), spike=2.0,
        lr_patience=2, max_lr_reductions=1, lr_factor=0.5,
    )
    stop_reason, best_score, _, recoveries, _ = _run_with_recovery(
        model, tmp_path, max_recoveries=3, max_epochs=10,
        lr_patience=2, max_lr_reductions=1,
    )

    assert len(recoveries) == 0
    assert stop_reason["reason"] in ("completed", "early_stopped")
    assert best_score is not None


def test_recovery_counter_is_explicit_not_lr_based(tmp_path):
    """Recovery uses an explicit counter, not min_lr floor detection.

    §4.3/§5.1: once all groups sit at min_lr, ReduceLROnPlateau still fires
    its no-op reduction and resets num_bad_epochs, cycling forever.  The
    recovery mechanism must count attempts explicitly.

    We verify this by setting max_lr_reductions=0 (LR is already at its
    floor from the start) and confirming recovery still counts and
    terminates properly.
    """
    model = _DivergingModel(
        lr=1e-2, first_spike=3, spike_every=3, spike=5.0, base_loss=10.0,
        lr_patience=4, max_lr_reductions=0, lr_factor=0.5,
    )
    stop_reason, _, _, recoveries, _ = _run_with_recovery(
        model, tmp_path, max_recoveries=2, max_epochs=50,
        lr_patience=4, max_lr_reductions=0,
    )

    # With max_recoveries=2, exactly 2 recovery attempts should fire
    assert len(recoveries) == 2
    assert stop_reason["reason"] == "diverged_unrecovered"
    assert stop_reason["recoveries_attempted"] == 2


def test_max_recoveries_zero_disables_recovery(tmp_path):
    """max_recoveries=0 means no recovery: divergence stops the run."""
    model = _DivergingModel(
        lr=1e-2, spike_epochs={3}, spike=2.0,
        lr_patience=2, max_lr_reductions=1, lr_factor=0.5,
    )
    stop_reason, _, _, recoveries, _ = _run_with_recovery(
        model, tmp_path, max_recoveries=0, max_epochs=30,
        lr_patience=2, max_lr_reductions=1,
    )

    assert len(recoveries) == 0
    assert stop_reason["reason"] == "diverged"


def test_max_recoveries_cli_default_and_override():
    """--max-recoveries CLI arg plumbs through to TrainConfig."""
    p = build_arg_parser()
    required = ["--loss", "multinomial", "--run-name", "t"]
    cfg = cfg_from_args(p.parse_args(required))
    assert cfg.max_recoveries == 3

    cfg2 = cfg_from_args(p.parse_args(required + ["--max-recoveries", "1"]))
    assert cfg2.max_recoveries == 1

    cfg3 = cfg_from_args(p.parse_args(required + ["--max-recoveries", "0"]))
    assert cfg3.max_recoveries == 0


def test_trainconfig_rejects_max_recoveries_negative():
    with pytest.raises(ValueError, match="max_recoveries must be >= 0"):
        TrainConfig(**{**_REQUIRED, "max_recoveries": -1})


# --------------------------------------------------------------------------
# Finding 2: LR ladder correctness across recovery attempts
#
# When a recovery attempt improves the global best, the best checkpoint
# stores an already-reduced LR.  The factor applied to subsequent
# attempts must account for this to avoid double-counting.
# --------------------------------------------------------------------------


def test_recovery_lr_ladder_when_best_improves(tmp_path):
    """Nth recovery trains at recovery_factor**N × original_lr.

    Scenario: spike at epochs 5 and 15.
    - Initial fit diverges at epoch 5.  Best checkpoint at epoch 4.
    - Recovery 1 resumes from epoch 4 (stored LR = L), applies 0.5 → 0.5L.
      Improves (val_loss drops below initial best).  Diverges at epoch 15.
      Global best moves to the recovery-1 checkpoint (stored LR = 0.5L).
    - Recovery 2 resumes from recovery-1 best (stored LR = 0.5L).
      Correct: apply 0.5 → 0.25L.
      Buggy (cumulative from original): apply 0.25 → 0.5L × 0.25 = 0.125L.

    Uses max_lr_reductions=3 so the plateau floor (1.25e-3) is below the
    recovery-2 target (2.5e-3) and the clamp does not interfere.
    """
    from collections import defaultdict

    original_lr = 1e-2
    model = _DivergingModel(
        lr=original_lr, scale=0.1,
        spike_epochs={5, 15}, spike=2.0, base_loss=10.0,
        lr_patience=4, max_lr_reductions=3, lr_factor=0.5,
    )

    stop_reason, _, _, recoveries, model = _run_with_recovery(
        model, tmp_path, max_recoveries=3, max_epochs=40,
        recovery_factor=0.5, lr_patience=4, max_lr_reductions=3,
    )

    assert len(recoveries) >= 2, f"Need >= 2 recoveries, got {len(recoveries)}"

    # ── guard this test's own premise ────────────────────────────────────
    # This test discriminates 0.25L (correct) from 0.125L (double-counting
    # bug).  The floor clamp added in 0e1473c can interfere with that, so
    # assert we are still in a regime where the comparison is meaningful.
    # Without this, a future parameter tweak could slide the buggy value
    # into a range the clamp masks, silently retiring the regression guard.
    #
    # NOTE: the floor here equals the buggy value EXACTLY (both 1.25e-3),
    # so this test sits precisely on the clamp boundary.  That is benign --
    # under strict `<` the clamp does not fire, and even under `<=` the
    # clamped result is still != expected -- but it is benign by
    # coincidence rather than by construction, which is why it is asserted
    # here instead of left to a comment.
    floor = original_lr * 0.5 ** 3  # lr_factor ** max_lr_reductions
    expected_lr = original_lr * 0.25
    buggy_lr = original_lr * 0.125
    assert floor < expected_lr, (
        f"Premise broken: plateau floor {floor:.4e} is not below the "
        f"expected recovery-2 LR {expected_lr:.4e}, so the clamp would mask "
        f"the correct value and this test can no longer prove the ladder."
    )
    clamped_buggy = max(buggy_lr, floor)  # what the clamp would yield
    assert clamped_buggy != pytest.approx(expected_lr, rel=1e-4), (
        f"Premise broken: the buggy LR {buggy_lr:.4e} clamps to "
        f"{clamped_buggy:.4e}, indistinguishable from the expected "
        f"{expected_lr:.4e}. This test would PASS against the very bug it "
        f"exists to catch. Re-pick the parameters."
    )

    # Group lr_history entries by epoch; later entries for the same epoch
    # come from recovery attempts (the model object persists across fits).
    epoch_lr_records = defaultdict(list)
    for ep, lr0, lr1 in model.lr_history:
        epoch_lr_records[ep].append((lr0, lr1))

    # Recovery 1: resumes from epoch 4 checkpoint, first validation at
    # epoch 5 (second occurrence).  Expected LR = 0.5 × original.
    assert len(epoch_lr_records[5]) >= 2, "epoch 5 should appear in initial + recovery 1"
    recovery_1_lr0 = epoch_lr_records[5][-1][0]
    assert recovery_1_lr0 == pytest.approx(original_lr * 0.5, rel=1e-4), (
        f"Recovery 1 LR = {recovery_1_lr0}, expected {original_lr * 0.5}"
    )

    # Recovery 2: resumes from recovery-1 best, first validation at
    # epoch 15 (second occurrence).  Expected LR = 0.25 × original.
    assert len(epoch_lr_records[15]) >= 2, "epoch 15 should appear in recovery 1 + recovery 2"
    recovery_2_lr0 = epoch_lr_records[15][-1][0]
    assert recovery_2_lr0 == pytest.approx(original_lr * 0.25, rel=1e-4), (
        f"Recovery 2 LR = {recovery_2_lr0}, expected {original_lr * 0.25}. "
        f"If {original_lr * 0.125:.4e}, the cumulative factor is double-counting."
    )


# --------------------------------------------------------------------------
# Finding 3: Zero-epoch recovery detection
# --------------------------------------------------------------------------


def test_zero_epoch_recovery_reports_unrecovered(tmp_path):
    """A recovery that trains zero epochs must NOT be reported as completed.

    When max_epochs is low and the best checkpoint is near the cap,
    Lightning's fit() returns immediately.  Without a guard, this reads
    as a clean completion.
    """
    # Spike at epoch 3 with max_epochs=5: diverges at epoch 3.
    # Best checkpoint at epoch 2 (current_epoch=2).
    # Recovery resumes from epoch 2: Lightning trains epochs 3 and 4
    # (max_epochs=5 means epochs 0-4).
    # Spike again at epoch 3 (second occurrence), but fresh DivergenceStop
    # absorbs it.  Epoch 4 is fine.  Recovery completes.
    #
    # To force zero epochs: set max_epochs so low that after the first
    # divergence, the recovery checkpoint is at or past max_epochs.
    # spike_epochs={2} with max_epochs=3: diverges at epoch 2.
    # Best checkpoint at epoch 1.  Recovery resumes from epoch 1:
    # Lightning trains epoch 2 (the last before cap).  That's 1 epoch.
    #
    # For true zero: spike at epoch 1 with max_epochs=2.
    # Initial fit: epoch 0 (val=9.99, best), epoch 1 (spike=20, diverge).
    # Best checkpoint at epoch 0.  Recovery resumes from epoch 0:
    # Lightning trains epoch 1 (the last).  Still 1 epoch.
    #
    # True zero requires max_epochs <= checkpoint_epoch + 1.
    # spike at epoch 0 would set best=20 (no prior better), so the
    # checkpoint IS the spike checkpoint.  Not useful.
    #
    # Alternative: two-phase.  Initial fit runs 3 epochs, diverges at 2.
    # Best ckpt at epoch 1.  max_epochs=2.  Recovery from epoch 1:
    # current_epoch restored to 1, max_epochs=2 → 0 new epochs.
    model = _DivergingModel(
        lr=1e-2, spike_epochs={2}, spike=2.0, base_loss=10.0,
        lr_patience=4, max_lr_reductions=1, lr_factor=0.5,
    )
    # Initial fit with enough epochs to diverge
    dl = _tiny_dataloader()
    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir, exist_ok=True)
    derived_patience = (4 + 1) * 1 + 4  # 9

    def build_and_fit(ckpt_path, extra_callbacks, max_epochs=10):
        ckpt_cb = ModelCheckpoint(
            dirpath=os.path.join(run_dir, "checkpoints"),
            monitor="val_loss", mode="min", save_top_k=-1, save_last=True,
            filename="{epoch}-{step}-{val_loss:.4f}",
        )
        early = EarlyStopping(
            monitor="val_loss", mode="min",
            patience=derived_patience, min_delta=0.0,
        )
        div = DivergenceStop(1.10)
        cbs = [ckpt_cb, early, div] + list(extra_callbacks)
        trainer = L.Trainer(
            max_epochs=max_epochs,
            default_root_dir=run_dir,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=cbs,
            accelerator="cpu",
            log_every_n_steps=1,
        )
        trainer.fit(model, dl, dl, ckpt_path=ckpt_path)
        return trainer

    # Phase 1: initial fit with enough room to diverge at epoch 2
    trainer = build_and_fit(None, [], max_epochs=10)
    stop_reason = _determine_stop_reason(trainer)
    assert stop_reason["reason"] == "diverged", f"Expected diverged, got {stop_reason}"

    global_best_score = float(trainer.checkpoint_callback.best_model_score)
    global_best_path = trainer.checkpoint_callback.best_model_path
    assert global_best_path, "No checkpoint saved before divergence"

    # Phase 2: recovery with max_epochs=2, so the checkpoint at epoch 1
    # leaves zero room to train.
    def build_and_fit_capped(ckpt_path, extra_callbacks):
        return build_and_fit(ckpt_path, extra_callbacks, max_epochs=2)

    stop_reason, _, _, recoveries = run_recovery_loop(
        build_and_fit=build_and_fit_capped,
        stop_reason=stop_reason,
        global_best_score=global_best_score,
        global_best_path=global_best_path,
        max_recoveries=3,
        recovery_factor=0.5,
    )

    assert stop_reason["reason"] == "diverged_unrecovered", (
        f"Zero-epoch recovery should be diverged_unrecovered, got {stop_reason}"
    )
    assert len(recoveries) == 1
    assert recoveries[0].get("zero_epochs") is True


# --------------------------------------------------------------------------
# Finding 4: recovery_factor validation
# --------------------------------------------------------------------------


def test_trainconfig_rejects_recovery_factor_zero():
    with pytest.raises(ValueError, match="recovery_factor must be in"):
        TrainConfig(**{**_REQUIRED, "recovery_factor": 0.0})


def test_trainconfig_rejects_recovery_factor_negative():
    with pytest.raises(ValueError, match="recovery_factor must be in"):
        TrainConfig(**{**_REQUIRED, "recovery_factor": -0.5})


def test_trainconfig_rejects_recovery_factor_above_one():
    with pytest.raises(ValueError, match="recovery_factor must be in"):
        TrainConfig(**{**_REQUIRED, "recovery_factor": 1.5})


def test_trainconfig_accepts_recovery_factor_one():
    """recovery_factor=1.0 is a no-op (resume at same LR) — valid edge case."""
    cfg = TrainConfig(**{**_REQUIRED, "recovery_factor": 1.0})
    assert cfg.recovery_factor == 1.0


def test_recovery_factor_cli_default_and_override():
    p = build_arg_parser()
    required = ["--loss", "multinomial", "--run-name", "t"]
    cfg = cfg_from_args(p.parse_args(required))
    assert cfg.recovery_factor == 0.5

    cfg2 = cfg_from_args(p.parse_args(required + ["--recovery-factor", "0.3"]))
    assert cfg2.recovery_factor == pytest.approx(0.3)


def test_recovery_factor_zero_via_cli():
    p = build_arg_parser()
    args = p.parse_args(
        ["--loss", "multinomial", "--run-name", "t", "--recovery-factor", "0"]
    )
    with pytest.raises(SystemExit):
        cfg_from_args(args)


# --------------------------------------------------------------------------
# Finding 7: recoveries list reaches summary.json on disk
# --------------------------------------------------------------------------


def test_recoveries_written_to_summary_json(tmp_path):
    """The recoveries list must be JSON-serialisable and contain expected fields."""
    model = _DivergingModel(
        lr=1e-2, first_spike=3, spike_every=3, spike=5.0, base_loss=10.0,
        lr_patience=4, max_lr_reductions=1, lr_factor=0.5,
    )
    stop_reason, best_score, best_path, recoveries, _ = _run_with_recovery(
        model, tmp_path, max_recoveries=2, max_epochs=50,
        lr_patience=4, max_lr_reductions=1,
    )

    assert len(recoveries) >= 1

    # Write summary.json using the same logic as run_training
    run_dir = str(tmp_path / "summary_run")
    os.makedirs(run_dir, exist_ok=True)
    summary = {
        "best_model_path": best_path,
        "best_val_loss": best_score,
        "stop_reason": stop_reason,
    }
    if recoveries:
        summary["recoveries"] = recoveries
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Read back and verify structure
    with open(summary_path) as f:
        loaded = json.load(f)

    assert "recoveries" in loaded
    assert len(loaded["recoveries"]) == len(recoveries)
    # All six fields the design doc §0 specifies, asserted meaningfully
    # rather than by presence alone.  pre_divergence_best and
    # diverged_value were produced by production code but asserted nowhere
    # until test-audit flagged the gap (2026-09-24).
    spike_value = 10.0 * 5.0  # base_loss * spike
    for i, event in enumerate(loaded["recoveries"], start=1):
        assert event["attempt"] == i, (
            f"attempts must be numbered from 1 in order, got {event['attempt']}"
        )
        assert isinstance(event["epoch"], int) and event["epoch"] >= 0
        assert event["checkpoint"].endswith(".ckpt"), (
            f"checkpoint should be a .ckpt path, got {event['checkpoint']!r}"
        )
        assert 0 < event["recovery_lr_factor"] <= 1.0, (
            f"recovery_lr_factor must be a reduction in (0, 1], got "
            f"{event['recovery_lr_factor']}"
        )

        # The divergence that triggered this recovery: the value seen must
        # be the spike, and it must be strictly worse than the best that
        # preceded it -- otherwise DivergenceStop would not have fired.
        assert event["diverged_value"] == pytest.approx(spike_value, rel=1e-3), (
            f"diverged_value={event['diverged_value']}, expected the spike "
            f"{spike_value}"
        )
        assert event["pre_divergence_best"] < event["diverged_value"], (
            f"pre_divergence_best={event['pre_divergence_best']} must be "
            f"better than the diverged value {event['diverged_value']}"
        )
        # Sanity: the pre-divergence best comes from the improving trend,
        # so it sits at or just below base_loss.
        assert event["pre_divergence_best"] <= 10.0


# --------------------------------------------------------------------------
# BUG PROOF: recovery LR can go below the plateau scheduler's floor
#
# The plateau scheduler computes min_lrs = [g["lr"] * factor ** max_reductions]
# from the INITIAL lr at optimizer-build time.  The recovery ladder multiplies
# param_group["lr"] directly with no reference to that floor.  After enough
# recoveries, the LR drops below the floor, making the scheduler permanently
# inert (it clamps to min_lr on every step, but old_lr == new_lr so the
# assignment guard fires and num_bad_epochs resets silently forever).
# --------------------------------------------------------------------------


def test_recovery_lr_breaches_plateau_floor(tmp_path):
    """PROOF: recovery drives LR below the scheduler's min_lr floor.

    Setup: lr=1e-2, lr_factor=0.5, max_lr_reductions=1 → floor = 5e-3.
    recovery_factor=0.5, max_recoveries=3.
    Model spikes every 4 epochs to force repeated recovery.

    After recovery 1: LR = 5e-3 (== floor, OK)
    After recovery 2: LR = 2.5e-3 (< floor 5e-3, BUG)

    This test FAILS on f598a36 (proving the bug) and must PASS after the fix.
    """
    original_lr = 1e-2
    model = _DivergingModel(
        lr=original_lr, scale=0.1,
        first_spike=4, spike_every=4, spike=5.0, base_loss=10.0,
        lr_patience=4, max_lr_reductions=1, lr_factor=0.5,
    )

    stop_reason, _, _, recoveries, model = _run_with_recovery(
        model, tmp_path, max_recoveries=3, max_epochs=50,
        recovery_factor=0.5, lr_patience=4, max_lr_reductions=1,
    )

    assert len(recoveries) >= 2, f"Need >= 2 recoveries, got {len(recoveries)}"

    # The plateau floor for group 0: 1e-2 * 0.5^1 = 5e-3
    floor_lr0 = original_lr * 0.5 ** 1

    # Check that NO recorded LR is below the floor
    for ep, lr0, lr1 in model.lr_history:
        assert lr0 >= floor_lr0 - 1e-10, (
            f"epoch {ep}: group0 LR={lr0:.6e} < floor={floor_lr0:.6e}. "
            f"Recovery drove the LR below the scheduler's min_lr."
        )


def test_floor_clamp_reported_in_recovery_event_and_json(tmp_path):
    """When the clamp binds, floor_clamped_groups reaches summary.json.

    Same setup as test_recovery_lr_breaches_plateau_floor: the second
    recovery clamps.  Verify the event structure and JSON round-trip.
    """
    original_lr = 1e-2
    model = _DivergingModel(
        lr=original_lr, scale=0.1,
        first_spike=4, spike_every=4, spike=5.0, base_loss=10.0,
        lr_patience=4, max_lr_reductions=1, lr_factor=0.5,
    )

    _, best_score, best_path, recoveries, _ = _run_with_recovery(
        model, tmp_path, max_recoveries=3, max_epochs=50,
        recovery_factor=0.5, lr_patience=4, max_lr_reductions=1,
    )

    assert len(recoveries) >= 2

    # At least one recovery should have floor_clamped_groups
    clamped_events = [r for r in recoveries if "floor_clamped_groups" in r]
    assert len(clamped_events) >= 1, (
        f"Expected at least one clamped recovery, got none. "
        f"Events: {recoveries}"
    )

    # Check structure of the clamp report
    for event in clamped_events:
        for entry in event["floor_clamped_groups"]:
            assert "group" in entry
            assert "requested_lr" in entry
            assert "clamped_lr" in entry
            assert entry["clamped_lr"] >= entry["requested_lr"]

    # JSON round-trip
    run_dir = str(tmp_path / "summary_run")
    os.makedirs(run_dir, exist_ok=True)
    summary = {
        "best_model_path": best_path,
        "best_val_loss": best_score,
        "recoveries": recoveries,
    }
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(summary_path) as f:
        loaded = json.load(f)

    loaded_clamped = [r for r in loaded["recoveries"]
                      if "floor_clamped_groups" in r]
    assert len(loaded_clamped) == len(clamped_events)
    for event in loaded_clamped:
        for entry in event["floor_clamped_groups"]:
            assert isinstance(entry["group"], int)
            assert isinstance(entry["requested_lr"], float)
            assert isinstance(entry["clamped_lr"], float)


def test_no_clamp_when_recovery_stays_above_floor(tmp_path):
    """A run that never reaches the floor produces no floor_clamped_groups.

    Uses max_lr_reductions=3 → floor = lr * 0.5^3 = 1.25e-3.
    Single recovery at factor=0.5 → LR = 5e-3, well above the floor.
    """
    model = _DivergingModel(
        lr=1e-2, scale=0.1, spike_epochs={5}, spike=2.0, base_loss=10.0,
        lr_patience=4, max_lr_reductions=3, lr_factor=0.5,
    )
    _, _, _, recoveries, _ = _run_with_recovery(
        model, tmp_path, max_recoveries=1, max_epochs=20,
        recovery_factor=0.5, lr_patience=4, max_lr_reductions=3,
    )

    assert len(recoveries) == 1
    assert "floor_clamped_groups" not in recoveries[0], (
        f"No clamp expected when LR stays above floor, "
        f"but got: {recoveries[0].get('floor_clamped_groups')}"
    )


def test_multi_group_floor_clamp_different_floors(tmp_path):
    """Groups with different initial LRs have different floors; both are clamped.

    _DivergingModel uses scale=0.1, so:
      group 0: lr=1e-2, floor = 1e-2 * 0.5 = 5e-3
      group 1: lr=1e-3, floor = 1e-3 * 0.5 = 5e-4

    After recovery 1: group 0 = 5e-3, group 1 = 5e-4 (both at floor).
    After recovery 2: both requested below floor → both clamped.
    """
    original_lr = 1e-2
    scale = 0.1
    model = _DivergingModel(
        lr=original_lr, scale=scale,
        first_spike=4, spike_every=4, spike=5.0, base_loss=10.0,
        lr_patience=4, max_lr_reductions=1, lr_factor=0.5,
    )

    _, _, _, recoveries, model = _run_with_recovery(
        model, tmp_path, max_recoveries=3, max_epochs=50,
        recovery_factor=0.5, lr_patience=4, max_lr_reductions=1,
    )

    assert len(recoveries) >= 2

    # Find a recovery where both groups were clamped
    multi_clamped = [
        r for r in recoveries
        if "floor_clamped_groups" in r
        and len(r["floor_clamped_groups"]) == 2
    ]
    assert len(multi_clamped) >= 1, (
        f"Expected at least one recovery clamping both groups. "
        f"Events: {recoveries}"
    )

    # Verify per-group floors
    floor_g0 = original_lr * 0.5
    floor_g1 = original_lr * scale * 0.5
    for event in multi_clamped:
        groups = {e["group"]: e for e in event["floor_clamped_groups"]}
        assert 0 in groups and 1 in groups
        assert groups[0]["clamped_lr"] == pytest.approx(floor_g0, rel=1e-6)
        assert groups[1]["clamped_lr"] == pytest.approx(floor_g1, rel=1e-6)
        # Floors differ by the scale factor
        assert groups[1]["clamped_lr"] / groups[0]["clamped_lr"] == pytest.approx(
            scale, rel=1e-6
        )

    # LR history must respect both floors
    for ep, lr0, lr1 in model.lr_history:
        assert lr0 >= floor_g0 - 1e-10, (
            f"epoch {ep}: group0 LR={lr0:.6e} < floor={floor_g0:.6e}"
        )
        assert lr1 >= floor_g1 - 1e-10, (
            f"epoch {ep}: group1 LR={lr1:.6e} < floor={floor_g1:.6e}"
        )
