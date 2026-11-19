"""Tests for scripts/simulator/cut_site_oracle_nll.py.

What the tests pin:

1. Closed form.  A uniform generator puts ``p = 1 / (2 * region_len * n_L)`` on
   every cell, so on the crop the oracle is ``log(2 * tile_size * n_L)`` -- the
   uniform baseline -- per region and, on a store, in every split; the
   count-ratio normaliser gives ``log Z_hat_r = log(n_scored / n_stored)``.
2. The scored distribution IS the generator's: ``region_oracle`` reproduces
   ``sample_region``'s ``probs`` bit for bit, and its ``Z`` equals the crop's
   share of a full enumeration that sums to 1.  A second case puts an N block
   in the sequence so the live-start gate excludes starts, and uses an
   asymmetric ``p_plus`` so the strand weighting is exercised.
3. The run's gates on its inputs: ``read_sidecar`` parses ``p`` exactly (the
   default pandas parser does not), and ``check_p_reproduces`` accepts bitwise
   equal ``p`` and refuses a one-ulp difference, a NaN and a row-count mismatch.
4. Same population as the model: on one store, the oracle's region population
   and scored rows equal the scorer's, under the trainer's ``min_N``.
5. ``log_prob_ml`` is conditioned on the crop as ``log_prob`` is.
"""

import gzip

import numpy as np
import pandas as pd
import pytest
import zarr

from background_model.band_model.store import _create_array
from background_model.constants import HEX_HALF, NHEX
from background_model.cut_site_model.model import hex_registration_record
from background_model.fragment_lengths import FragmentLengthDist
from background_model.hexamers import hexamer_indices
from background_model.simulator.draw import sample_region, start_distribution
from background_model.simulator.measure import TABLE_NAMES
from scripts.cut_site_model.baselines.reduce_split_nll import (
    read_per_fragment,
    reduce_splits,
)
from scripts.cut_site_model.score_fragments import SPLITS
from scripts.cut_site_model.score_fragments import main as score_main
from scripts.simulator.cut_site_oracle_nll import (
    check_p_reproduces,
    oracle_arrays,
    read_sidecar,
    region_oracle,
    run_regions,
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


def test_cell_p_matches_sample_region_bit_for_bit():
    """Guard the oracle's per-fragment product, NOT the start distributions.

    Both sides now read ``draw.start_distribution``, so this no longer compares
    two independent derivations of the distributions -- that guard lives in
    tests/test_simulator_draw.py.  What it still checks is real: the oracle's
    ``cell_p`` formula against ``sample_region``'s own product (draw.py:233-237).
    """
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


DEAD_LO, DEAD_HI = 250, 450   # N block (sequence/track coordinates)


def test_cell_p_live_gate_and_asymmetric_strands():
    rs = np.random.default_rng(6)
    r = {k: rs.uniform(0.1, 3.0, NHEX) for k in TABLE_NAMES}
    fl = FragmentLengthDist(rs.integers(1, 50, N_L), L_LO)
    seq = bytearray(_seq(rs))
    seq[DEAD_LO:DEAD_HI] = b"N" * (DEAD_HI - DEAD_LO)
    seq = bytes(seq)
    p_plus = 0.3

    dist = strand_start_probs(seq, REGION_LEN, r=r, fl=fl)
    _, _, valid = hexamer_indices(seq)
    pos = np.arange(REGION_LEN)
    # Non-vacuity: some start has a valid start window but every end window
    # ([i + L_LO, i + L_HI]) invalid, so the live gate actually excludes it.
    fired = any(bool((valid[pos] & (dist[s][2] == 0)).any()) for s in (True, False))
    assert fired, "N block did not kill any live start; adjust DEAD_LO/DEAD_HI"

    starts, lengths, is_plus, probs = sample_region(
        seq, REGION_LEN, 200, r=r, fl=fl, p_plus=p_plus,
        rng=np.random.default_rng(7))
    out = region_oracle(seq, REGION_LEN, starts, lengths, is_plus, r=r, fl=fl,
                        p_plus=p_plus, crop_start=CROP_START, tile_size=TILE)
    assert np.array_equal(out["p"], probs)

    # Z is the crop's share of a full enumeration that sums to 1, weighted by
    # the asymmetric strand split p_plus=0.3 / 0.7.
    total = crop = 0.0
    for s, ps in ((True, p_plus), (False, 1 - p_plus)):
        start_probs, W_s, t_s = dist[s]
        cell = ps * start_probs[:, None] * np.divide(
            W_s, t_s[:, None], out=np.zeros_like(W_s), where=t_s[:, None] > 0)
        total += cell.sum(dtype=np.float64)
        crop += cell[CROP_START:CROP_START + TILE].sum(dtype=np.float64)
        other = is_plus != s
        assert np.allclose(out["p_other"][other],
                           cell[starts[other], lengths[other] - L_LO], rtol=1e-13)
    assert abs(total - 1.0) < 1e-12
    assert abs(out["Z"] - crop) < 1e-12

    # The asymmetric strand weighting is actually exercised: swapping
    # p_plus and 1 - p_plus moves Z by more than numerical noise.
    out_swapped = region_oracle(seq, REGION_LEN, starts, lengths, is_plus, r=r,
                                fl=fl, p_plus=1 - p_plus, crop_start=CROP_START,
                                tile_size=TILE)
    assert abs(out["Z"] - out_swapped["Z"]) > 1e-6


@pytest.mark.parametrize("dead_plus", [True, False])
def test_dead_strand_raises_not_none(dead_plus):
    """``start_distribution`` returns None for a dead strand; the oracle must
    turn that into ValueError, never pass None on to a score."""
    rs = np.random.default_rng(8)
    r = {k: rs.uniform(0.1, 3.0, NHEX) for k in TABLE_NAMES}
    # Zero only this strand's start table: the other strand stays live.
    r["start_fwd" if dead_plus else "end_rev"][:] = 0.0
    fl = FragmentLengthDist(rs.integers(1, 50, N_L), L_LO)
    seq = _seq(rs)

    # Non-vacuity: the helper really reports exactly this strand dead.
    raw = start_distribution(seq, REGION_LEN, r=r, fl=fl)
    assert raw[dead_plus] is None and raw[not dead_plus] is not None

    sign = "+" if dead_plus else "-"
    with pytest.raises(ValueError, match=rf"strand \{sign} has no live start"):
        strand_start_probs(seq, REGION_LEN, r=r, fl=fl)
    with pytest.raises(ValueError, match=rf"strand \{sign} has no live start"):
        region_oracle(seq, REGION_LEN, [0], [L_LO], [not dead_plus], r=r, fl=fl,
                      p_plus=0.5, crop_start=CROP_START, tile_size=TILE)


def test_read_sidecar_is_exact(tmp_path):
    rng = np.random.default_rng(42)
    n = 10_000
    p = rng.uniform(1e-9, 1e-5, n).astype(np.float64)
    sidecar = tmp_path / "sim.p.tsv.gz"
    with gzip.open(sidecar, "wt") as f:
        f.write("# seed=1 sample=S\n")
        f.write("contig\tstart\tstop\tstrand\tp\n")
        for i, pv in enumerate(p):
            f.write(f"chr1\t{i}\t{i + 100}\t+\t{pv:.17g}\n")

    # Non-vacuity: pandas' default float parser misreads at least one value,
    # so the test actually exercises float_precision="round_trip".
    default_p = pd.read_csv(sidecar, sep="\t", skiprows=1)["p"].to_numpy(np.float64)
    assert not np.array_equal(default_p, p)

    header, frame = read_sidecar(str(sidecar))
    assert header == {"seed": "1", "sample": "S"}
    assert np.array_equal(frame["p"].to_numpy(np.float64), p)


def test_p_reproduction_gate_is_bitwise():
    p = np.random.default_rng(4).uniform(1e-9, 1e-5, 1000)
    check = check_p_reproduces(p.copy(), p)
    assert check == {"n": 1000, "n_bitwise_equal": 1000, "max_rel_diff": 0.0}

    # One ulp on one row is refused.
    off = p.copy()
    off[17] = np.nextafter(off[17], np.inf)
    with pytest.raises(ValueError, match="does not reproduce"):
        check_p_reproduces(off, p)

    # A NaN difference is refused, not passed by a failed comparison.
    nan = p.copy()
    nan[3] = np.nan
    with pytest.raises(ValueError, match="does not reproduce"):
        check_p_reproduces(nan, p)

    with pytest.raises(ValueError, match="rows"):
        check_p_reproduces(p[:-1], p)


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


def test_log_prob_ml_is_conditioned_on_the_crop(store):
    """log_prob_ml is the strand-marginal of the uniform conditional on the
    crop: p_other == p under the uniform generator, so p + p_other = 2p and
    the two factors of 2 against Z_exact = TILE / REGION_LEN cancel to
    -log(TILE * N_L)."""
    store_path, _, _ = store
    _, arrays = _oracle(store_path)
    scored = arrays["scored"]
    log_prob_ml = arrays["exact"]["log_prob_ml"]
    expected = -np.log(np.float64(TILE * N_L))
    np.testing.assert_allclose(log_prob_ml[scored], expected, rtol=1e-13)
    assert np.all(np.isnan(log_prob_ml[~scored]))


# ---------------------------------------------------------------------------
# run_regions: the --n-regions subset gate
# ---------------------------------------------------------------------------

def _write_bed(path, n):
    """n regions on chr1, 1kb apart, the shape run_cut_site_simulator emits."""
    with open(path, "w") as fh:
        for i in range(n):
            fh.write(f"chr1\t{1000 * i}\t{1000 * i + 500}\n")
    return str(path)


def test_run_regions_slices_to_the_runs_subset(tmp_path):
    """A --n-regions run records the FULL bed but the SLICED count.

    Loading the bed without the slice measures a different population, which
    is what made every subset run unscoreable.
    """
    bed = _write_bed(tmp_path / "full.bed", 50)
    run = {"region_bed": bed, "n_regions": 7}

    rdf = run_regions(run, "hg38")

    assert len(rdf) == 7                      # the slice happened
    # and it is the FIRST 7, the same ones the simulator drew from
    assert list(rdf["start"]) == [1000 * i for i in range(7)]


def test_run_regions_is_a_noop_for_a_full_run(tmp_path):
    bed = _write_bed(tmp_path / "full.bed", 12)
    rdf = run_regions({"region_bed": bed, "n_regions": 12}, "hg38")
    assert len(rdf) == 12


def test_run_regions_rejects_a_bed_that_is_too_short(tmp_path):
    """A shorter bed would silently yield fewer regions; iloc does not raise."""
    bed = _write_bed(tmp_path / "short.bed", 3)
    with pytest.raises(ValueError, match="fewer than the run's n_regions"):
        run_regions({"region_bed": bed, "n_regions": 9}, "hg38")
