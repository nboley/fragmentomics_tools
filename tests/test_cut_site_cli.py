"""CLI-level tests for the cut-site scripts.

Covers two seams that unit tests of the library cannot see:

- ``--tile-size`` is required by ``train_cut_site.py`` and ``gpu_benchmark.py``
  and must match the store's ``config_json`` (owner decision 176).  The dataset
  reads the tile size from the store, so a CLI value that silently disagreed
  would build the model for one geometry and feed it another.
- The ``.npz`` written by ``stage1_hex_baseline.py --output`` is what
  ``CutSiteFragmentModel(hex_tables_path=...)`` reads.  The writer and reader
  name the arrays independently, so the key names are checked end to end.
"""

import sys

import numpy as np
import pytest

from background_model.cut_site_model import CutSiteFragmentModel
from scripts import gpu_benchmark, stage1_hex_baseline, train_cut_site
from tests.test_cut_site_store_dataset import _create_synthetic_store

_SCRIPTS = {"train_cut_site": train_cut_site, "gpu_benchmark": gpu_benchmark}


def _run_main(monkeypatch, module, argv):
    monkeypatch.setattr(sys, "argv", [module.__name__] + argv)
    module.main()


def _script_argv(name, store_path, tmp_path):
    argv = ["--store", store_path]
    if name == "train_cut_site":
        argv += ["--run-dir", str(tmp_path / "run")]
    return argv


@pytest.mark.parametrize("name", sorted(_SCRIPTS))
def test_tile_size_is_required(name, tmp_path, monkeypatch, capsys):
    store_path, _ = _create_synthetic_store(tmp_path, tile_size=256)
    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, _SCRIPTS[name],
                  _script_argv(name, store_path, tmp_path))
    assert exc.value.code == 2
    assert "--tile-size" in capsys.readouterr().err


@pytest.mark.parametrize("name", sorted(_SCRIPTS))
def test_tile_size_mismatch_with_store_is_rejected(name, tmp_path, monkeypatch,
                                                   capsys):
    store_path, cfg = _create_synthetic_store(tmp_path, tile_size=256)
    argv = _script_argv(name, store_path, tmp_path) + ["--tile-size", "512"]
    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, _SCRIPTS[name], argv)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert f"--tile-size 512 != store tile_size {cfg.tile_size}" in err
    # The check fires before anything is written.
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("freeze_hex", [True, False])
def test_stage1_npz_loads_into_model(freeze_hex, tmp_path, monkeypatch):
    store_path, cfg = _create_synthetic_store(
        tmp_path, tile_size=256, n_frags_per_tile=200)
    npz_path = str(tmp_path / "hex_tables.npz")
    _run_main(monkeypatch, stage1_hex_baseline,
              ["--store", store_path, "--output", npz_path])

    model = CutSiteFragmentModel(
        tile_size=cfg.tile_size, N1=8, K=8, N2=4, H=4, L_chunk=32,
        dispersion_window_size=cfg.tile_size,
        hex_tables_path=npz_path, freeze_hex=freeze_hex,
    )
    saved = np.load(npz_path)
    for key in ["start_fwd", "stop_fwd", "start_rev", "stop_rev"]:
        table = saved[key]
        # Non-vacuous: the synthetic fragments must have produced a real table.
        assert np.ptp(table) > 0, key
        loaded = getattr(model, f"hex_log_{key}").detach().numpy()
        np.testing.assert_allclose(
            loaded, np.log(np.maximum(table, 1e-30)).astype(np.float32))
    n_L = model.hparams.L_max - model.hparams.L_min + 1
    assert model._log_pL.shape == (n_L,)
    np.testing.assert_allclose(
        model._log_pL.numpy(),
        np.log(np.maximum(saved["p_L"], 1e-30)).astype(np.float32))
    assert int(saved["L_min"]) == model.hparams.L_min
    assert int(saved["L_max"]) == model.hparams.L_max
