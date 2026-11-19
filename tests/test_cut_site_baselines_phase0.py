"""Phase-0 baseline tooling: reference .npz emitter, S2 reducer, G0 closed form.

Every NLL here comes from the sanctioned path (score_fragments CLI ->
reduce_split_nll -> score_fragments.region_weighted_mult_nll).  The tests pin
that the reducer adds nothing to that path, and that the two reference
priors score to their analytic values.
"""

import json

import numpy as np
import pandas as pd
import pytest
import zarr

from scripts.cut_site_model.baselines.emit_reference_npz import TABLES, reference_tables
from scripts.cut_site_model.baselines.phase0_gate import closed_form_unif_pL
from scripts.cut_site_model.baselines.reduce_split_nll import (
    read_per_fragment,
    reduce_splits,
)
from scripts.cut_site_model.score_fragments import (
    SPLITS,
    load_model,
    main as score_main,
    region_weighted_mult_nll,
    score_store,
)
from background_model.cut_site_model.dataset import CutSiteFragmentDataset
from background_model.cut_site_model.model import (
    HEX_REGISTRATION_KEY,
    hex_registration_record,
)
from tests.test_cut_site_store_dataset import _create_synthetic_store

TILE_SPLITS = np.array([0, 1, 1, 2, 2], dtype=np.uint8)   # train, val, val, test, test
MIN_N = 10


@pytest.fixture
def store(tmp_path):
    store_path, cfg = _create_synthetic_store(
        tmp_path, tile_size=256, jitter=32, n_tiles=len(TILE_SPLITS),
        n_frags_per_tile=60, seed=7,
    )
    zarr.open_group(store_path, mode="r+")["tiles/split"][:] = TILE_SPLITS
    rng = np.random.default_rng(3)
    n_L = cfg.L_max - cfg.L_min + 1
    source = {t: rng.dirichlet(np.ones(4096)) for t in TABLES}
    source.update(p_L=rng.dirichlet(np.ones(n_L)), k=np.int64(6),
                  L_min=np.int64(cfg.L_min), L_max=np.int64(cfg.L_max),
                  **hex_registration_record())
    hex_path = str(tmp_path / "hex.npz")
    np.savez(hex_path, **source)
    return store_path, cfg, hex_path, tmp_path


def _emit(store, kind):
    _, _, hex_path, tmp_path = store
    path = str(tmp_path / f"{kind}.npz")
    np.savez(path, **reference_tables(np.load(hex_path), kind))
    return path


def _score(store, npz, name, extra=()):
    store_path, _, _, tmp_path = store
    tsv = str(tmp_path / f"{name}.tsv")
    score_main(["--store", store_path, "--hex-tables", npz, "--zero-correction",
                "--output", tsv, "--splits", "val", "test", *extra])
    return tsv


def _ln_D(cfg):
    return np.log(float(cfg.tile_size * (cfg.L_max - cfg.L_min + 1) * 2))


def test_emitter_schema(store):
    _, cfg, hex_path, _ = store
    src = np.load(hex_path)
    n_L = cfg.L_max - cfg.L_min + 1
    for kind in ("uniform", "unif_pL"):
        d = np.load(_emit(store, kind))
        assert set(d.files) == (set(TABLES)
                                | {"p_L", "k", "L_min", "L_max", HEX_REGISTRATION_KEY})
        for t in TABLES:
            assert d[t].shape == (4096,)
            assert np.all(d[t] == d[t][0])
            assert d[t].sum(dtype=np.float64) == pytest.approx(1.0, abs=1e-12)
        assert d["p_L"].shape == (n_L,)
        assert (int(d["k"]), int(d["L_min"]), int(d["L_max"])) == (6, cfg.L_min, cfg.L_max)
    assert np.array_equal(np.load(_emit(store, "unif_pL"))["p_L"], src["p_L"])
    assert np.all(np.load(_emit(store, "uniform"))["p_L"] == 1.0 / n_L)


def test_emitter_rejects_wrong_p_L_shape(store):
    _, _, hex_path, _ = store
    src = dict(np.load(hex_path))
    src["p_L"] = src["p_L"][:-1]
    with pytest.raises(ValueError, match="p_L shape"):
        reference_tables(src, "unif_pL")


def test_s2_equals_the_sanctioned_reduction_exactly(store):
    """T3: the TSV round trip and re-join add nothing to region_weighted_mult_nll."""
    store_path, cfg, hex_path, _ = store
    tsv = _score(store, hex_path, "prior")
    got = reduce_splits(tsv, store_path, ["val", "test"], MIN_N)

    model = load_model(cfg, hex_path, zero_correction=True)
    result = score_store(model, store_path, splits=("val", "test"))
    indptr = zarr.open_group(store_path, mode="r")["fragments/indptr"][:]
    for name in ("val", "test"):
        ds = CutSiteFragmentDataset(
            store_path=store_path, model_input_size=model.model_input_size,
            model_L_min=cfg.L_min, model_L_max=cfg.L_max, split=SPLITS[name][1],
            min_N=MIN_N, train_mode=False,
        )
        want = region_weighted_mult_nll(result["log_prob"], result["scored"], indptr, ds)
        assert got[name]["n_regions"] == want["n_regions"] > 0
        assert abs(got[name]["region_weighted_nll"] - want["region_weighted_nll"]) < 1e-12


def test_s2_val_matches_the_scorers_own_val_reduction(store):
    store_path, _, hex_path, tmp_path = store
    tsv = _score(store, hex_path, "prior", ["--reproduce-val-mult-nll"])
    with open(tsv + ".summary.json") as f:
        scorer_val = json.load(f)["val_mult_nll"]["region_weighted_nll"]
    got = reduce_splits(tsv, store_path, ["val"], MIN_N)["val"]["region_weighted_nll"]
    assert got == scorer_val


def test_uniform_scores_ln_D_on_every_fragment(store):
    """T1: constant tables and constant p_L give ln|D| everywhere."""
    store_path, cfg, _, _ = store
    tsv = _score(store, _emit(store, "uniform"), "uniform")
    pf = read_per_fragment(tsv, store_path)
    lp = pf["log_prob"][pf["scored"]]
    assert len(lp) > 0
    assert np.max(np.abs(lp + _ln_D(cfg))) < 1e-5
    for name, r in reduce_splits(tsv, store_path, ["val", "test"], MIN_N).items():
        assert r["region_weighted_nll"] == pytest.approx(_ln_D(cfg), abs=1e-5)


def test_unif_pL_matches_its_closed_form(store):
    """T2: S2 on unif+p_L == -mean(log p_L[L]) + ln(tile_size * 2) per region."""
    store_path, cfg, _, _ = store
    npz = _emit(store, "unif_pL")
    tsv = _score(store, npz, "unif_pL")
    got = reduce_splits(tsv, store_path, ["val", "test"], MIN_N)
    for name in ("val", "test"):
        cf, n_reg, n_frags = closed_form_unif_pL(store_path, npz, name, MIN_N)
        assert n_reg == got[name]["n_regions"]
        assert n_frags > 0
        assert got[name]["region_weighted_nll"] == pytest.approx(cf, abs=1e-5)
    # Per fragment too: log p = log p_L(L) - ln(2 * tile_size).
    # Parse `scored` as a STRING and compare explicitly, matching
    # `read_per_fragment`.  Letting pandas infer it would lean on its
    # auto-conversion of "true"/"false" to bool, which is a different rule
    # from the production reader's -- two parsers for one column is how they
    # drift.  (Reviewer F-01.)
    df = pd.read_csv(tsv, sep="\t", dtype={"scored": str})
    df = df[df["scored"] == "true"]
    log_p_L = np.log(np.load(npz)["p_L"])
    want = log_p_L[(df["stop"] - df["start"] - cfg.L_min).to_numpy()] - np.log(2.0 * cfg.tile_size)
    assert np.max(np.abs(df["log_prob"].to_numpy() - want)) < 1e-5


def test_s2_refuses_a_partially_scored_split(store):
    store_path, _, hex_path, _ = store
    tsv = _score(store, hex_path, "partial", ["--limit-regions", "1"])
    with pytest.raises(ValueError, match="not every tile was scored"):
        reduce_splits(tsv, store_path, ["val"], MIN_N)


def test_s2_refuses_a_tsv_that_does_not_match_the_store(store):
    store_path, _, hex_path, _ = store
    tsv = _score(store, hex_path, "tampered")
    df = pd.read_csv(tsv, sep="\t", dtype=str)
    df.loc[0, "start"] = str(int(df.loc[0, "start"]) + 1)
    df.to_csv(tsv, sep="\t", index=False)
    with pytest.raises(ValueError, match="do not match the store rows"):
        read_per_fragment(tsv, store_path)
