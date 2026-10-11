"""Tests for scripts/simulator/cut_site_oracle_nll.py.

Three properties, each pinned by its own test:

1. Closed form.  A uniform generator puts ``p = 1 / (2 * region_len * n_L)`` on
   every cell, so on the crop the oracle is ``log(2 * tile_size * n_L)`` -- the
   uniform baseline -- in every region and every split.
2. The restated distribution IS the generator's: ``region_oracle`` reproduces
   ``sample_region``'s ``probs`` bit for bit, and its ``Z`` equals the crop's
   share of a full enumeration that sums to 1.
3. Same population as the model: on one store, the oracle's region population
   and scored rows equal the scorer's, under the trainer's ``min_N``.
"""

import numpy as np
import pytest
import zarr

from background_model.band_model.store import _create_array
from background_model.constants import HEX_HALF, NHEX
from background_model.cut_site_model.model import hex_registration_record
from background_model.fragment_lengths import FragmentLengthDist
from background_model.simulator.draw import sample_region
from background_model.simulator.measure import TABLE_NAMES
from scripts.cut_site_model.baselines.reduce_split_nll import (
    read_per_fragment,
    reduce_splits,
)
from scripts.cut_site_model.score_fragments import SPLITS
from scripts.cut_site_model.score_fragments import main as score_main
from scripts.simulator.cut_site_oracle_nll import (
    oracle_arrays,
    region_oracle,
    store_rows,
    strand_start_probs,
    write_and_reduce,
)
from tests.test_cut_site_store_dataset import _create_synthetic_store

L_LO, L_HI = 25, 180              # the store fixture's default length range
N_L = L_HI - L_LO + 1
TILE, JITTER = 256, 32
REGION_LEN = TILE + 2 * JITTER     # l_target
CROP_START = JITTER                # crop_origin(l_target, tile, 0)
SEQ_EXTRA = 2 * HEX_HALF + L_HI    # the simulator's padded frame

TILE_SPLITS = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2, 2], dtype=np.uint8)
SHRUNK = {7: 5, 9: 5}              # tile -> stored rows kept; below min_N=10
MIN_N = 10


def _uniform():
    r = {k: np.ones(NHEX, dtype=np.float64) for k in TABLE_NAMES}
    return r, FragmentLengthDist(np.ones(N_L, dtype=np.int64), L_LO)


def _seq(rng):
    return rng.choice(list(b"ACGT"), size=REGION_LEN + SEQ_EXTRA).astype(np.uint8).tobytes()


def test_uniform_generator_matches_the_closed_form():
    r, fl = _uniform()
    rng = np.random.default_rng(0)
    starts = rng.integers(0, REGION_LEN, 50)
    lengths = rng.integers(L_LO, L_HI + 1, 50)
    is_plus = rng.random(50) < 0.5
    out = region_oracle(_seq(rng), REGION_LEN, starts, lengths, is_plus,
                        r=r, fl=fl, p_plus=0.5, crop_start=CROP_START,
                        tile_size=TILE)
    np.testing.assert_allclose(out["Z"], TILE / REGION_LEN, rtol=1e-13)
    np.testing.assert_allclose(out["p"], 1 / (2 * REGION_LEN * N_L), rtol=1e-13)
    np.testing.assert_allclose(out["p_other"], out["p"], rtol=1e-13)
    np.testing.assert_allclose(-np.log(out["p"] / out["Z"]),
                               np.log(2 * TILE * N_L), rtol=1e-14)


def test_restatement_reproduces_sample_region_bit_for_bit():
    rs = np.random.default_rng(1)
    r = {k: rs.uniform(0.1, 3.0, NHEX) for k in TABLE_NAMES}
    fl = FragmentLengthDist(rs.integers(1, 50, N_L), L_LO)
    seq = bytearray(_seq(rs))
    seq[100:104] = b"NNNN"         # invalid windows: dead starts and ends
    seq = bytes(seq)
    p_plus = 0.5

    starts, lengths, is_plus, probs = sample_region(
        seq, REGION_LEN, 200, r=r, fl=fl, p_plus=p_plus,
        rng=np.random.default_rng(5))
    out = region_oracle(seq, REGION_LEN, starts, lengths, is_plus, r=r, fl=fl,
                        p_plus=p_plus, crop_start=CROP_START, tile_size=TILE)
    assert np.array_equal(out["p"], probs)

    # Z is the crop's share of a full enumeration that sums to 1.
    dist = strand_start_probs(seq, REGION_LEN, r=r, fl=fl)
    total = crop = 0.0
    for s, ps in ((True, p_plus), (False, 1 - p_plus)):
        start_probs, W_s, t_s = dist[s]
        cell = ps * start_probs[:, None] * np.divide(
            W_s, t_s[:, None], out=np.zeros_like(W_s), where=t_s[:, None] > 0)
        total += cell.sum(dtype=np.float64)
        crop += cell[CROP_START:CROP_START + TILE].sum(dtype=np.float64)
        # p_other is this strand's cell for the other strand's fragments.
        other = is_plus != s
        assert np.allclose(out["p_other"][other],
                           cell[starts[other], lengths[other] - L_LO], rtol=1e-13)
    assert abs(total - 1.0) < 1e-12
    assert abs(out["Z"] - crop) < 1e-12
    assert 0.6 < out["Z"] < 0.95   # non-uniform, so not the geometric 0.8


@pytest.fixture
def store(tmp_path):
    """Ten tiles, two of them shrunk below MIN_N, starts over all of l_target."""
    store_path, cfg = _create_synthetic_store(
        tmp_path, tile_size=TILE, jitter=JITTER, n_tiles=len(TILE_SPLITS),
        n_frags_per_tile=80, seed=11, full_start_range=True)
    assert (cfg.l_target, cfg.L_min, cfg.L_max) == (REGION_LEN, L_LO, L_HI)
    root = zarr.open_group(store_path, mode="r+")
    root["tiles/split"][:] = TILE_SPLITS
    frags = root["fragments"]
    indptr = frags["indptr"][:]
    keep = np.ones(indptr[-1], dtype=bool)
    for t, k in SHRUNK.items():
        keep[indptr[t] + k:indptr[t + 1]] = False
    cols = {c: frags[c][:][keep] for c in ("start", "length", "strand")}
    tile_of_row = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    cols["indptr"] = np.concatenate(
        [[0], np.cumsum(np.bincount(tile_of_row[keep], minlength=len(indptr) - 1))]
    ).astype(indptr.dtype)
    for c, v in cols.items():
        del frags[c]
        _create_array(frags, c, data=v)
    return store_path, cfg, tmp_path


def _oracle(store_path):
    st = store_rows(store_path)
    rng = np.random.default_rng(2)
    seqs = [_seq(rng) for _ in st["tile_start"]]
    r, fl = _uniform()
    p_side = np.full(len(st["start"]), 1 / (2 * REGION_LEN * N_L))
    return st, oracle_arrays(st, seqs, p_side, r=r, fl=fl, p_plus=0.5,
                             n_workers=1)


def test_store_oracle_is_the_closed_form_on_the_models_population(store):
    store_path, _, tmp_path = store
    st, arrays = _oracle(store_path)
    exact = write_and_reduce(store_path, arrays, "exact", str(tmp_path), MIN_N)
    expected_regions = {"train": 6, "val": 1, "test": 1}
    for s in SPLITS:
        assert exact[s]["n_regions"] == expected_regions[s]
        np.testing.assert_allclose(exact[s]["region_weighted_nll"],
                                   np.log(2 * TILE * N_L), rtol=1e-13)

    # The published estimator: log Z_hat_r = log(n_scored / n_stored) per region.
    ratio = write_and_reduce(store_path, arrays, "count_ratio", str(tmp_path),
                             MIN_N)
    log_zhat = np.log(arrays["n_scored"] / arrays["n_stored"])
    for s in SPLITS:
        tiles = ratio[s]["term_tiles"]
        want = np.log(2 * REGION_LEN * N_L) + log_zhat[tiles]
        np.testing.assert_allclose(ratio[s]["terms"], want, rtol=1e-13)


def test_oracle_population_equals_the_scorers(store):
    store_path, cfg, tmp_path = store
    _, arrays = _oracle(store_path)
    exact = write_and_reduce(store_path, arrays, "exact", str(tmp_path), MIN_N)

    rng = np.random.default_rng(3)
    hex_path = str(tmp_path / "hex.npz")
    np.savez(hex_path,
             **{k: rng.dirichlet(np.ones(NHEX)) for k in
                ("start_fwd", "stop_fwd", "start_rev", "stop_rev")},
             p_L=rng.dirichlet(np.ones(N_L)), **hex_registration_record())
    tsv = str(tmp_path / "scorer.tsv")
    score_main(["--store", store_path, "--hex-tables", hex_path,
                "--zero-correction", "--output", tsv])
    model = reduce_splits(tsv, store_path, list(SPLITS), MIN_N, return_terms=True)

    assert np.array_equal(read_per_fragment(tsv, store_path)["scored"],
                          arrays["scored"])
    for s in SPLITS:
        assert np.array_equal(model[s]["term_tiles"], exact[s]["term_tiles"]), s
    # Non-vacuity: min_N removed the shrunk tiles, and some rows are unscored.
    assert not set(SHRUNK) & set(np.concatenate(
        [exact[s]["term_tiles"] for s in SPLITS]).tolist())
    assert 0 < arrays["scored"].sum() < len(arrays["scored"])
