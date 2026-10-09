"""Tests for scripts/score_cut_site_fragments.py (CPU only).

The scorer must agree with the training loss, not with a re-derivation of
it: the central test reproduces `_step`'s multinomial NLL from the scorer's
per-fragment output on the same regions.
"""

import types

import lightning as L
import numpy as np
import pandas as pd
import pytest
import torch
import zarr

from background_model.cut_site_dataset import (
    CutSiteFragmentDataset,
    cut_site_collate_fn,
)
from background_model.cut_site_model import (
    HEX_TABLE_ENCODING,
    CutSiteFragmentModel,
    compute_log_p0,
)
from background_model_core import calculate_start_and_stop_from_jitter
from scripts.score_cut_site_fragments import (
    COLUMNS,
    SPLITS,
    build_table,
    load_model,
    region_weighted_mult_nll,
    score_store,
    write_table,
)
from tests.test_cut_site_store_dataset import _create_synthetic_store

N_TILES = 4
TILE_SPLITS = np.array([0, 1, 1, 2], dtype=np.uint8)   # train, val, val, test


@pytest.fixture
def setup(tmp_path):
    store_path, cfg = _create_synthetic_store(
        tmp_path, tile_size=256, jitter=32, n_tiles=N_TILES,
        n_frags_per_tile=60, seed=7,
    )
    zarr.open_group(store_path, mode="r+")["tiles/split"][:] = TILE_SPLITS

    rng = np.random.default_rng(3)
    n_L = cfg.L_max - cfg.L_min + 1
    hex_path = str(tmp_path / "hex.npz")
    np.savez(
        hex_path,
        **{k: rng.dirichlet(np.ones(4096)) for k in
           ("start_fwd", "stop_fwd", "start_rev", "stop_rev")},
        p_L=rng.dirichlet(np.ones(n_L)),
        hex_encoding=HEX_TABLE_ENCODING,
    )

    torch.manual_seed(0)
    model = CutSiteFragmentModel(
        L_min=cfg.L_min, L_max=cfg.L_max, tile_size=cfg.tile_size,
        N1=8, K=8, N2=4, H=4, L_chunk=32,
        dispersion_window_size=cfg.tile_size, nb_w0=0.0,
        hex_tables_path=hex_path, freeze_hex=True,
    )
    with torch.no_grad():       # a live correction, so f_theta != 0
        model.w_out.normal_()
        model.len_bias.normal_()
    return store_path, cfg, hex_path, model.eval()


def _dataset(store_path, cfg, model, split, min_N):
    return CutSiteFragmentDataset(
        store_path=store_path, model_input_size=model.model_input_size,
        model_L_min=cfg.L_min, model_L_max=cfg.L_max,
        split=split, min_N=min_N, train_mode=False,
    )


def test_reproduces_step_mult_nll(setup):
    """Region-weighted NLL from the scorer rows == _step's L_mult."""
    store_path, cfg, _, model = setup
    result = score_store(model, store_path)

    val_ds = _dataset(store_path, cfg, model, "val", min_N=10)
    assert len(val_ds) == 2
    batch = cut_site_collate_fn([val_ds[i] for i in range(len(val_ds))])
    with torch.no_grad():
        step_nll = model._step(batch, "val").item()   # nb_w0=0 -> loss == L_mult

    indptr = zarr.open_group(store_path, mode="r")["fragments/indptr"][:]
    got = region_weighted_mult_nll(result["log_prob"], result["scored"],
                                   indptr, val_ds)
    assert got["n_regions"] == 2
    assert got["region_weighted_nll"] == pytest.approx(step_nll, abs=1e-5)


def test_every_region_is_normalised(setup):
    store_path, _, _, model = setup
    result = score_store(model, store_path)
    assert result["max_abs_logsumexp"] < 1e-5


def test_coverage_counts_and_nan_policy(setup):
    store_path, cfg, _, model = setup
    result = score_store(model, store_path)
    nnz = len(result["scored"])

    dataset_total = sum(
        int(ds[i]["frag_N"])
        for name in SPLITS
        for ds in [_dataset(store_path, cfg, model, SPLITS[name][1], min_N=1)]
        for i in range(len(ds))
    )
    scored = result["scored"]
    assert scored.sum() == dataset_total
    assert 0 < scored.sum() < nnz          # the fixture has unscored rows
    assert result["processed"].all()
    assert np.isnan(result["log_prob"][~scored]).all()
    assert np.isnan(result["log_prob_ml"][~scored]).all()
    lp, ml = result["log_prob"][scored], result["log_prob_ml"][scored]
    assert np.isfinite(lp).all() and (lp < 0).all()
    assert (ml >= lp).all()

    # log_prob_ml is the logsumexp of BOTH strand cells.  Note ml - lp is
    # NOT bounded by ln 2: it is log(1 + p_other / p_observed).
    ds = _dataset(store_path, cfg, model, "train", min_N=1)
    item = ds[0]
    with torch.no_grad():
        lp_dom = model.domain_log_probs(item["x"][None])[0][0].double()
    pair = lp_dom[item["frag_m"], item["frag_L_idx"]].numpy()      # (n, 2)
    idx = item["frag_store_idx"].numpy()
    np.testing.assert_allclose(
        result["log_prob"][idx], pair[np.arange(len(idx)), item["frag_s"].numpy()],
        rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        result["log_prob_ml"][idx], np.logaddexp(pair[:, 0], pair[:, 1]),
        rtol=0, atol=1e-12)


def test_unscored_rows_are_exactly_those_outside_the_centred_crop(setup):
    store_path, cfg, _, model = setup
    result = score_store(model, store_path)
    root = zarr.open_group(store_path, mode="r")
    crop_start, _ = calculate_start_and_stop_from_jitter(
        cfg.l_target, cfg.tile_size, 0)
    p = root["fragments/start"][:].astype(np.int64) - crop_start
    inside = (p >= 0) & (p < cfg.tile_size)
    np.testing.assert_array_equal(result["scored"], inside)


def test_table_coordinates_and_order(setup):
    store_path, _, _, model = setup
    result = score_store(model, store_path)
    df = build_table(store_path, result)
    root = zarr.open_group(store_path, mode="r")
    indptr = root["fragments/indptr"][:]
    tile = np.repeat(np.arange(N_TILES), np.diff(indptr))

    assert list(df.columns) == list(COLUMNS)
    assert len(df) == indptr[-1]
    exp_start = root["tiles/start"][:][tile] + root["fragments/start"][:].astype(np.int64)
    np.testing.assert_array_equal(df["start"], exp_start)
    np.testing.assert_array_equal(
        df["stop"] - df["start"], root["fragments/length"][:].astype(np.int64))
    np.testing.assert_array_equal(
        df["strand"], np.where(root["fragments/strand"][:] == 0, "+", "-"))
    names = np.array(["train", "val", "test"])
    np.testing.assert_array_equal(df["split"], names[TILE_SPLITS[tile]])


def test_limit_regions_and_split_selection(setup):
    store_path, _, _, model = setup
    indptr = zarr.open_group(store_path, mode="r")["fragments/indptr"][:]

    result = score_store(model, store_path, limit_regions=2)
    rows = np.zeros(indptr[-1], dtype=bool)
    rows[: indptr[2]] = True                   # tiles 0 and 1, store order
    np.testing.assert_array_equal(result["processed"], rows)
    assert not result["scored"][~rows].any()
    assert len(build_table(store_path, result)) == indptr[2]

    val_only = score_store(model, store_path, splits=["val"])
    rows = np.zeros(indptr[-1], dtype=bool)
    rows[indptr[1]: indptr[3]] = True          # tiles 1 and 2
    np.testing.assert_array_equal(val_only["processed"], rows)
    # Same tile, same crop, same model -> same values regardless of selection.
    full = score_store(model, store_path)
    np.testing.assert_array_equal(val_only["log_prob"][rows], full["log_prob"][rows])


def test_zero_correction_is_exactly_the_prior(setup):
    store_path, cfg, hex_path, _ = setup
    model = load_model(cfg, hex_path, zero_correction=True)
    ds = _dataset(store_path, cfg, model, "train", min_N=1)
    x = ds[0]["x"][None]
    with torch.no_grad():
        got, _, _ = model.domain_log_probs(x)
        log_p0 = compute_log_p0(
            x, model.hex_log_start_fwd, model.hex_log_stop_fwd,
            model.hex_log_start_rev, model.hex_log_stop_rev, model._log_pL,
            cfg.tile_size, cfg.L_min, cfg.L_max)
        want = torch.log_softmax(log_p0.reshape(1, -1), -1).reshape_as(got)
    torch.testing.assert_close(got, want, rtol=0, atol=1e-6)


def test_checkpoint_round_trip(setup, tmp_path):
    store_path, cfg, hex_path, model = setup
    ds = _dataset(store_path, cfg, model, "train", min_N=1)
    # A real Lightning checkpoint.  validate, not fit: fit would need the
    # val_loss that ReduceLROnPlateau monitors.
    trainer = L.Trainer(limit_val_batches=1, logger=False,
                        enable_checkpointing=False, accelerator="cpu",
                        enable_progress_bar=False, enable_model_summary=False)
    trainer.validate(model, torch.utils.data.DataLoader(
        ds, batch_size=1, collate_fn=cut_site_collate_fn), verbose=False)
    ckpt = str(tmp_path / "m.ckpt")
    trainer.save_checkpoint(ckpt, weights_only=True)

    loaded = load_model(cfg, hex_path, checkpoint=ckpt)
    model.eval()
    x = ds[0]["x"][None]
    with torch.no_grad():
        torch.testing.assert_close(loaded.domain_log_probs(x)[0],
                                   model.domain_log_probs(x)[0], rtol=0, atol=0)


def test_load_model_rejects_both_or_neither(setup):
    _, cfg, hex_path, _ = setup
    with pytest.raises(ValueError):
        load_model(cfg, hex_path)
    with pytest.raises(ValueError):
        load_model(cfg, hex_path, checkpoint="x.ckpt", zero_correction=True)


def test_region_weighted_estimator_semantics():
    """Per-region mean, then mean over regions; an empty crop counts as 0."""
    ds = types.SimpleNamespace(index=[(0, 0), (0, 1), (0, 2)], n_tiles=3)
    indptr = np.array([0, 2, 3, 6])
    log_prob = np.array([-1.0, -3.0, np.nan, -6.0, -6.0, -6.0])
    scored = np.array([True, True, False, True, True, True])
    got = region_weighted_mult_nll(log_prob, scored, indptr, ds)
    # (2 + 0 + 6) / 3; the fragment-weighted total/total would be 22/5.
    assert got["region_weighted_nll"] == pytest.approx(8 / 3)
    assert got["n_regions_no_scored_fragment"] == 1


def test_write_table_format(setup, tmp_path):
    store_path, _, _, model = setup
    df = build_table(store_path, score_store(model, store_path, limit_regions=1))
    path = str(tmp_path / "out.tsv")
    write_table(df, path)
    # dtype=str: pandas would otherwise parse true/false into bools.
    back = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    assert list(back.columns) == list(COLUMNS)
    assert set(back["scored"]) <= {"true", "false"}
    assert (back.loc[back["scored"] == "false", "log_prob"] == "NaN").all()
    sc = back["scored"] == "true"
    np.testing.assert_array_equal(
        back.loc[sc, "log_prob"].astype(float).to_numpy(),
        df.loc[df["scored"], "log_prob"].to_numpy())
