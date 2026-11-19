"""Phase-0b.2/0b.3 tooling: fold-restricted counting, scoring and reduction,
D4 ``return_terms``, the CV selection of N0's ``w`` and the paired delta.

Every NLL still comes from the one sanctioned reduction,
``score_fragments.region_weighted_mult_nll``; these tests pin that the new
keywords and the fold restriction add nothing to it.
"""

import json
import shutil

import numpy as np
import pytest
import zarr

from scripts.cut_site_model.baselines.count_tables import (
    TABLES,
    count_split,
    main as count_main,
)
from scripts.cut_site_model.baselines.cv_select_w import cv_curve, make_folds, w_tag
from scripts.cut_site_model.baselines.paired_delta import paired_delta
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
from background_model.cut_site_model.model import HEX_REGISTRATION_KEY
from tests.test_cut_site_store_dataset import _create_synthetic_store

TILE_SPLITS = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2, 2], dtype=np.uint8)
TRAIN_TILES = np.nonzero(TILE_SPLITS == 0)[0]
MIN_N = 10


@pytest.fixture
def store(tmp_path):
    store_path, cfg = _create_synthetic_store(
        tmp_path, tile_size=256, jitter=32, n_tiles=len(TILE_SPLITS),
        n_frags_per_tile=80, seed=11,
    )
    zarr.open_group(store_path, mode="r+")["tiles/split"][:] = TILE_SPLITS
    return store_path, cfg, tmp_path


def _counts(store, name, tiles=None):
    store_path, _, tmp_path = store
    out = str(tmp_path / f"{name}.counts.npz")
    argv = ["count", "--store", store_path, "--split", "train", "--output", out]
    if tiles is not None:
        tiles_path = str(tmp_path / f"{name}.tiles.npy")
        np.save(tiles_path, np.asarray(tiles, dtype=np.int64))
        argv += ["--tiles-npy", tiles_path]
    count_main(argv)
    return out


def _tables(store, counts, name, *est):
    out = str(store[2] / f"{name}.npz")
    count_main(["tables", "--counts", counts, "--output", out, "--estimator", *est])
    return out


def _score(store, npz, name, splits=("val", "test"), extra=()):
    store_path, _, tmp_path = store
    tsv = str(tmp_path / f"{name}.tsv")
    score_main(["--store", store_path, "--hex-tables", npz, "--zero-correction",
                "--output", tsv, "--splits", *splits, *extra])
    return tsv


def _prior(store):
    return _tables(store, _counts(store, "all"), "prior", "shrink", "--w", "64")


def test_return_terms_is_additive_and_leaves_the_default_unchanged(store):
    store_path, cfg, _ = store
    npz = _prior(store)
    model = load_model(cfg, npz, zero_correction=True)
    result = score_store(model, store_path, splits=("val",))
    indptr = zarr.open_group(store_path, mode="r")["fragments/indptr"][:]
    ds = CutSiteFragmentDataset(
        store_path=store_path, model_input_size=model.model_input_size,
        model_L_min=cfg.L_min, model_L_max=cfg.L_max, split="val",
        min_N=MIN_N, train_mode=False,
    )
    default = region_weighted_mult_nll(result["log_prob"], result["scored"], indptr, ds)
    assert set(default) == {"region_weighted_nll", "n_regions",
                            "n_regions_no_scored_fragment"}
    full = region_weighted_mult_nll(result["log_prob"], result["scored"], indptr, ds,
                                    return_terms=True)
    assert {k: full[k] for k in default} == default
    assert full["terms"].dtype == np.float64
    assert np.array_equal(full["term_tiles"], [t for _, t in ds.index])
    assert full["terms"].mean(dtype=np.float64) == default["region_weighted_nll"]
    # Each term is the region's own -sum(lp)/N, derived here from the store rows.
    for term, t in zip(full["terms"], full["term_tiles"]):
        lp = result["log_prob"][indptr[t]:indptr[t + 1]]
        lp = lp[result["scored"][indptr[t]:indptr[t + 1]]]
        assert term == pytest.approx(-lp.sum() / len(lp), rel=1e-12)


def test_count_split_tiles_partition_is_additive(store):
    store_path, _, _ = store
    whole, meta = count_split(store_path, verbose=False)
    a, meta_a = count_split(store_path, verbose=False, tiles=TRAIN_TILES[:2])
    b, _ = count_split(store_path, verbose=False, tiles=TRAIN_TILES[2:])
    assert meta["n_tiles"] == len(TRAIN_TILES) and not meta["tile_restricted"]
    assert meta_a["n_tiles"] == 2 and meta_a["tile_restricted"]
    for key in (*TABLES, "N_start", "length_counts", "length_counts_all"):
        assert np.array_equal(a[key] + b[key], whole[key]), key
    assert a["length_counts_all"].sum() > 0 and b["length_counts_all"].sum() > 0


def test_count_split_refuses_a_tile_outside_the_split(store):
    store_path, _, _ = store
    val_tile = int(np.nonzero(TILE_SPLITS == 1)[0][0])
    with pytest.raises(ValueError, match="not in split"):
        count_split(store_path, verbose=False, tiles=[TRAIN_TILES[0], val_tile])


def test_make_folds_partitions_train(store):
    store_path, _, _ = store
    folds = make_folds(store_path, k=3, seed=5)
    held = np.concatenate([h for _, h in folds])
    assert np.array_equal(np.sort(held), TRAIN_TILES)          # disjoint, covering
    for fit, h in folds:
        assert np.array_equal(np.union1d(fit, h), TRAIN_TILES)
        assert len(np.intersect1d(fit, h)) == 0
    again = make_folds(store_path, k=3, seed=5)
    assert all(np.array_equal(h1, h2) for (_, h1), (_, h2) in zip(folds, again))


def test_fold_restricted_scoring_and_reduction_match_the_full_split(store):
    """A fold TSV reduced over its own regions gives exactly the terms the full
    TRAIN TSV gives for those regions, and the restriction is enforced."""
    store_path, _, tmp_path = store
    npz = _prior(store)
    held = TRAIN_TILES[[1, 4]]
    held_path = str(tmp_path / "held.npy")
    np.save(held_path, held)
    full_tsv = _score(store, npz, "full", splits=("train",))
    fold_tsv = _score(store, npz, "fold", splits=("train",),
                      extra=("--tiles-npy", held_path))

    pf_full = read_per_fragment(full_tsv, store_path)
    pf_fold = read_per_fragment(fold_tsv, store_path)
    assert np.array_equal(np.nonzero(pf_fold["processed_tiles"])[0], held)
    rows = pf_fold["scored"]
    assert rows.sum() > 0
    assert np.array_equal(pf_fold["log_prob"][rows], pf_full["log_prob"][rows])

    fold = reduce_splits(fold_tsv, store_path, ["train"], MIN_N, tiles=held,
                         return_terms=True)["train"]
    whole = reduce_splits(full_tsv, store_path, ["train"], MIN_N,
                          return_terms=True)["train"]
    assert np.array_equal(fold["term_tiles"], held)
    at = np.isin(whole["term_tiles"], held)
    assert np.array_equal(fold["terms"], whole["terms"][at])
    with pytest.raises(ValueError, match="not every tile was scored"):
        reduce_splits(fold_tsv, store_path, ["train"], MIN_N)


def test_s2_reduces_a_tsv_whose_tables_the_loader_now_refuses(store):
    """The Phase-0 old-prior TSV names m+L-1 tables the loader refuses.  S2
    reads the per-fragment values, never the tables, so it must still reduce."""
    store_path, cfg, tmp_path = store
    npz = _prior(store)
    tsv = _score(store, npz, "prior")
    want = reduce_splits(tsv, store_path, ["val", "test"], MIN_N)

    legacy = dict(np.load(npz))
    del legacy[HEX_REGISTRATION_KEY]
    legacy_path = str(tmp_path / "legacy.npz")
    np.savez(legacy_path, **legacy)
    with pytest.raises(ValueError, match=HEX_REGISTRATION_KEY):
        load_model(cfg, legacy_path, zero_correction=True)

    legacy_tsv = str(tmp_path / "legacy.tsv")
    shutil.copy(tsv, legacy_tsv)
    with open(tsv + ".summary.json") as f:
        summary = json.load(f)
    summary["hex_tables"] = legacy_path
    del summary["tiles_npy"]               # Phase-0 summaries predate the key
    with open(legacy_tsv + ".summary.json", "w") as f:
        json.dump(summary, f)
    assert reduce_splits(legacy_tsv, store_path, ["val", "test"], MIN_N) == want


def test_cv_curve_end_to_end_selects_the_argmin(store):
    store_path, _, tmp_path = store
    k, grid = 2, [0.0, 64.0, 4096.0]
    cv_dir = tmp_path / "cv"
    cv_dir.mkdir()
    for i, (fit, held) in enumerate(make_folds(store_path, k, seed=1)):
        np.save(cv_dir / f"fold{i}_heldout.npy", held)
        counts = _counts(store, f"fold{i}", tiles=fit)
        (cv_dir / f"fold{i}").mkdir()
        for w in grid:
            npz = _tables(store, counts, f"fold{i}_{w_tag(w)}", "shrink", "--w", str(w))
            tsv = _score(store, npz, f"fold{i}_{w_tag(w)}", splits=("train",),
                         extra=("--tiles-npy", str(cv_dir / f"fold{i}_heldout.npy")))
            for suffix in ("", ".summary.json"):
                shutil.move(tsv + suffix, str(cv_dir / f"fold{i}" / f"{w_tag(w)}.tsv") + suffix)

    out = cv_curve(store_path, str(cv_dir), grid, k, MIN_N)
    cvs = {r["w"]: r["cv_nll"] for r in out["rows"]}
    assert out["w_star"] == min(cvs, key=cvs.get)
    assert out["n_regions"] == len(TRAIN_TILES)
    # The CV score is the region-weighted NLL over every TRAIN region, each
    # under tables fitted without its fold: check against the pooled terms.
    for r in out["rows"]:
        terms = np.concatenate([
            reduce_splits(str(cv_dir / f"fold{i}" / f"{w_tag(r['w'])}.tsv"), store_path,
                          ["train"], MIN_N, tiles=np.load(cv_dir / f"fold{i}_heldout.npy"),
                          return_terms=True)["train"]["terms"]
            for i in range(k)])
        assert r["cv_nll"] == pytest.approx(terms.mean(), abs=1e-12)
        if r["w"] == out["w_star"]:
            assert r["delta_vs_w_star"] == 0 and r["paired_se_vs_w_star"] == 0


def test_paired_delta(store):
    store_path, _, _ = store
    counts = _counts(store, "all")
    a = _score(store, _tables(store, counts, "a", "shrink", "--w", "64"), "a")
    b = _score(store, _tables(store, counts, "b", "stage1_ref"), "b")
    res = paired_delta(a, b, store_path, ["val", "test"], MIN_N, n_boot=200, seed=0)
    ra = reduce_splits(a, store_path, ["val", "test"], MIN_N, return_terms=True)
    rb = reduce_splits(b, store_path, ["val", "test"], MIN_N, return_terms=True)
    for name in ("val", "test"):
        d = ra[name]["terms"] - rb[name]["terms"]
        r = res[name]
        assert r["delta"] == ra[name]["region_weighted_nll"] - rb[name]["region_weighted_nll"]
        assert r["delta"] == pytest.approx(d.mean(), abs=1e-12)
        assert r["paired_se"] == pytest.approx(d.std(ddof=1) / np.sqrt(len(d)), rel=1e-12)
        assert r["bootstrap_se"] > 0
    # Self-pairing: zero delta, zero SE.
    same = paired_delta(a, a, store_path, ["val"], MIN_N, n_boot=50)["val"]
    assert same["delta"] == 0 and same["paired_se"] == 0
