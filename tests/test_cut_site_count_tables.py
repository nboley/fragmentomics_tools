"""The S3 table counter (scripts/cut_site_model/baselines/count_tables.py) and
the m+L registration guard (gate G0b of docs/pending/cut_site_baseline_priors_design.md).

- T4b  the counter attributes every fragment's hexamers to the cells
       ``compute_log_p0`` reads, all four tables, both strands, both ends
- conformance  the counter's C, N and raw rates reproduce
       ``simulator.measure``'s (``cut_site_hexamers``, ``uniform_hexamer_counts``,
       ``propensities``) on a shared fixture, EXHAUSTIVELY over all 4096
       hexamers -- the sole guard against a silent, shape-preserving
       endianness or reverse-complement error, so nothing is sampled
- T4   the ``stage1_ref`` estimator reproduces stage1's ``start_fwd``, the one
       table the change set leaves unchanged
- T13  the m+L loader refuses unregistered or misregistered tables and
       checkpoints, and the width guards lost their stale ``-1``
"""

import os
from types import SimpleNamespace

import lightning as L
import numpy as np
import pandas as pd
import pysam
import pytest
import torch
import zarr

from background_model.constants import HEX_HALF, NHEX
from background_model.cut_site_model.dataset import (
    CutSiteFragmentDataset,
    cut_site_collate_fn,
)
from background_model.cut_site_model.model import (
    HEX_REGISTRATION_KEY,
    STOP_CUT_BASE_OFFSET,
    WINDOW_HALF,
    CutSiteFragmentModel,
    compute_log_p0,
    hex_registration_record,
)
from background_model.cut_site_model.store import CutSiteStoreConfig
from background_model.hexamers import rc_permutation
from background_model.fragment_lengths import FragmentLengthDist
from background_model.simulator.measure import (
    counts_from_hexamers,
    cut_site_hexamers,
    propensities,
    uniform_hexamer_counts,
)
from scripts.cut_site_model.baselines.count_tables import (
    DENOMINATOR,
    TABLE_ENDS,
    TABLES,
    build_tables,
    count_split,
    p_L_from,
    rates,
)
from scripts.cut_site_model.score_fragments import load_model
from tests.test_cut_site_store_dataset import _create_synthetic_store
from tests.test_hex_registration import model_index_of_sim_index

REAL_V4_NPZ = ("/efs/analytics/nathanboley/background_model/cut_site_runs/"
               "stage1_hex_tables_v4_start.npz")
REAL_BAD981D_CKPT = (
    "/efs/analytics/nathanboley/model_output_real_rd56413_20261009/train_real/"
    "logs/version_0/checkpoints/best-epoch=07-step=012214-val_mult_nll=12.002583.ckpt")


def _store_geometry(cfg):
    """``(tile_in_seq, crop_start)``, the test's own derivation: the tile is
    the centred crop of both the stored sequence and the l_target window."""
    return (cfg.l_seq - cfg.tile_size) // 2, (cfg.l_target - cfg.tile_size) // 2


def _small_store(tmp_path, **kw):
    args = dict(tile_size=256, n_tiles=4, n_frags_per_tile=200, seed=3)
    args.update(kw)
    return _create_synthetic_store(tmp_path, **args)


# ---------------------------------------------------------------------------
# T4b: counter vs the model's own reads
# ---------------------------------------------------------------------------


def test_T4b_counter_counts_where_compute_log_p0_reads(tmp_path):
    """Every table = the bincount, over the admitted fragments of its strand,
    of the index compute_log_p0 gathers for that fragment's (m, L, s) under an
    identity table.  Exhaustive over all 4096 cells of all four tables."""
    store_path, cfg = _small_store(tmp_path)
    counts, meta = count_split(store_path, verbose=False)
    assert meta["n_dropped_invalid_or_masked"] == 0

    P, L_min, L_max = cfg.tile_size, cfg.L_min, cfg.L_max
    n_hex = 4 ** 6
    slot = {name: i for i, name in enumerate(
        ("start_fwd", "stop_fwd", "start_rev", "stop_rev"))}
    want = {name: np.zeros(NHEX, dtype=np.int64) for name in TABLES}
    ds = CutSiteFragmentDataset(
        store_path=store_path,
        model_input_size=CutSiteFragmentModel(tile_size=P).model_input_size,
        model_L_min=L_min, model_L_max=L_max, split="train", min_N=0,
        train_mode=False)
    log_pL = torch.zeros(L_max - L_min + 1, dtype=torch.float64)
    for i in range(len(ds)):
        item = ds[i]
        x = item["x"][None].double()
        m, li, s = (item[k].numpy() for k in ("frag_m", "frag_L_idx", "frag_s"))
        for name in TABLES:
            tabs = [torch.zeros(n_hex, dtype=torch.float64) for _ in range(4)]
            tabs[slot[name]] = torch.arange(n_hex, dtype=torch.float64)
            got = compute_log_p0(x, *tabs, log_pL, P, L_min, L_max)[0].numpy()
            sel = s == TABLE_ENDS[name][0]
            idx = np.rint(got[m[sel], li[sel], s[sel]]).astype(np.int64)
            want[name] += np.bincount(idx, minlength=NHEX)
    for name in TABLES:
        assert want[name].sum() > 0
        np.testing.assert_array_equal(counts[name], want[name], err_msg=name)


# ---------------------------------------------------------------------------
# Conformance: the counter reproduces simulator.measure on a shared fixture
# ---------------------------------------------------------------------------


def _simulator_side(store_path, cfg, counts, tmp_path):
    """C, N and r(h) from simulator.measure's OWN functions on the same tiles
    and fragments: sequence fetched from a FASTA of the stored tiles, regions
    = the centred tile, stops half-open at start + length (fragments_h5)."""
    root = zarr.open_group(store_path, mode="r")
    P, L_min, L_max = cfg.tile_size, cfg.L_min, cfg.L_max
    tile_in_seq, crop_start = _store_geometry(cfg)
    seq = root["tiles/seq"][:]
    tiles = np.nonzero(root["tiles/split"][:] == 0)[0]
    indptr = root["fragments/indptr"][:]
    start, length, strand = (root[f"fragments/{k}"][:].astype(np.int64)
                             for k in ("start", "length", "strand"))

    fasta = str(tmp_path / "tiles.fa")
    with open(fasta, "w") as f:
        for t in tiles:
            f.write(f">t{t}\n{seq[t].tobytes().decode()}\n")
    pysam.faidx(fasta)
    rdf = pd.DataFrame({"contig": [f"t{t}" for t in tiles],
                        "start": np.full(len(tiles), tile_in_seq),
                        "stop": np.full(len(tiles), tile_in_seq + P)})
    fl = FragmentLengthDist(counts["length_counts"], L_min)
    N, _ = uniform_hexamer_counts(rdf, fasta, fl, n_workers=1, verbose=False)

    frames = []
    for t in tiles:                 # one sample, so the flat row is t
        lo, hi = indptr[t], indptr[t + 1]
        m, Ln, s = start[lo:hi] - crop_start, length[lo:hi], strand[lo:hi]
        keep = (m >= 0) & (m < P) & (Ln >= L_min) & (Ln <= L_max)
        rfa = SimpleNamespace(n_frags=int(keep.sum()), starts_0=m[keep],
                              stops_0=m[keep] + Ln[keep],
                              fragment_strands=np.where(s[keep] == 0, "+", "-"))
        frames.append(cut_site_hexamers(rfa, seq[t][tile_in_seq - HEX_HALF:]))
    C = counts_from_hexamers(pd.concat(frames, ignore_index=True))
    return C, N, propensities(C, N)


def test_counter_matches_simulator_propensities_all_4096(tmp_path):
    store_path, cfg = _create_synthetic_store(
        tmp_path, tile_size=512, n_tiles=80, n_frags_per_tile=200, seed=11)
    counts, _ = count_split(store_path, verbose=False)
    C_sim, N_sim, r_sim = _simulator_side(store_path, cfg, counts, tmp_path)

    # Non-vacuous: nearly every hexamer is placed, and the tables are full.
    assert (N_sim["start"] > 0).mean() > 0.99
    for name in TABLES:
        assert (counts[name] > 0).sum() > 2000, name

    to_model = model_index_of_sim_index()          # model index of sim index b
    b = np.arange(NHEX)
    perm = rc_permutation()
    np.testing.assert_array_equal(counts["N_start"][to_model], N_sim["start"])
    np.testing.assert_allclose(counts["N_end"][to_model], N_sim["end"],
                               rtol=1e-12, atol=0)
    # Model tables are keyed by the FORWARD k-mer at the cut; the simulator's
    # _rev tables by its reverse complement, and its end_* are our stop_*.
    sim_cell = {"start_fwd": ("start_fwd", b), "stop_fwd": ("end_fwd", b),
                "start_rev": ("start_rev", perm[b]), "stop_rev": ("end_rev", perm[b])}
    for name in TABLES:
        sim_name, cell = sim_cell[name]
        np.testing.assert_array_equal(counts[name][to_model], C_sim[sim_name][cell],
                                      err_msg=name)
        end = TABLE_ENDS[name][1]
        r = rates(counts[name], counts[DENOMINATOR[end]], 0.0, 0.0)
        np.testing.assert_allclose(r[to_model], r_sim[sim_name][cell],
                                   rtol=1e-12, atol=0, err_msg=name)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def test_T4_stage1_ref_reproduces_stage1_left_end_tables(tmp_path, capsys):
    """N0_ref's start_fwd AND stop_rev == stage1's.  Both are read at the
    left cut m with the start-position denominator, so neither the 3'
    registration nor the denominator change reaches them.  (The design names
    start_fwd alone; stop_rev is the minus-strand table at the same end.)  The
    two right-end tables differ by design, and must."""
    from scripts.cut_site_model.stage1_hex_baseline import compute_hex_tables

    store_path, cfg = _small_store(tmp_path, n_tiles=50, n_frags_per_tile=100)
    counts, _ = count_split(store_path, verbose=False)
    tables, params = build_tables(counts, "stage1_ref")
    assert all(p["w"] == NHEX and p["q"] * NHEX == 1.0 for p in params.values())

    root = zarr.open_group(store_path, mode="r")
    s1, s1_p_L, _ = compute_hex_tables(
        root, CutSiteStoreConfig.from_json(root.attrs["config_json"]), split="train")
    for name in ("start_fwd", "stop_rev"):
        np.testing.assert_allclose(tables[name], s1[name], rtol=1e-12, atol=0,
                                   err_msg=name)
    # Every window is ACGT here, so no fragment is dropped and p_L agrees too.
    np.testing.assert_allclose(p_L_from(counts), s1_p_L, rtol=1e-12, atol=0)
    for name in ("stop_fwd", "start_rev"):
        assert not np.allclose(tables[name], s1[name], rtol=1e-6), name


def test_estimators(tmp_path):
    store_path, _ = _small_store(tmp_path)
    counts, _ = count_split(store_path, verbose=False)
    raw, _ = build_tables(counts, "raw")
    shrunk_far, params = build_tables(counts, "shrink", w=1e12)
    for name in TABLES:
        C = counts[name].astype(np.float64)
        N = counts[DENOMINATOR[TABLE_ENDS[name][1]]].astype(np.float64)
        assert raw[name].sum(dtype=np.float64) == pytest.approx(1.0, abs=1e-12)
        clean = np.where(N > 0, C / np.where(N > 0, N, 1), 0.0)
        np.testing.assert_allclose(raw[name], clean / clean.sum(), rtol=1e-12)
        # q defaults to the pooled cuts-per-occurrence rate, and w -> inf
        # shrinks every placed hexamer onto it: a flat table over N > 0.
        assert params[name]["q"] == pytest.approx(C.sum() / N.sum(), rel=1e-12)
        flat = shrunk_far[name][N > 0]
        assert flat.max() / flat.min() == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError):
        build_tables(counts, "stage1_ref", w=1.0)
    with pytest.raises(ValueError):
        build_tables(counts, "shrink")


@pytest.mark.parametrize("defect", ["N", "mask"])
def test_invalid_or_masked_cut_drops_the_fragment(tmp_path, defect):
    """A non-ACGT base spoils the six windows that contain it; a masked base
    the one cut site on it.  A fragment needs both cuts usable, and p_L drops
    exactly the fragments the tables drop, as simulator.measure does."""
    store_path, cfg = _small_store(tmp_path)
    clean, _ = count_split(store_path, verbose=False)
    tile_in_seq, crop_start = _store_geometry(cfg)
    q_bad = cfg.tile_size // 2
    root = zarr.open_group(store_path, mode="r+")
    if defect == "N":
        root["tiles/seq"][0, tile_in_seq + q_bad] = ord("N")
        spoiled = np.arange(q_bad - HEX_HALF + 1, q_bad + HEX_HALF + 1)
    else:
        root["tiles/mask"][0, crop_start + q_bad] = False
        spoiled = np.array([q_bad])
    counts, meta = count_split(store_path, verbose=False)

    lo, hi = root["fragments/indptr"][0:2]
    m = root["fragments/start"][lo:hi].astype(np.int64) - crop_start
    Ln = root["fragments/length"][lo:hi].astype(np.int64)
    adm = (m >= 0) & (m < cfg.tile_size) & (Ln >= cfg.L_min) & (Ln <= cfg.L_max)
    m, Ln = m[adm], Ln[adm]
    # Cut bases from the span: the base right of each junction.
    last = m + Ln - 1
    hit = np.isin(m, spoiled) | np.isin(last + 1, spoiled)
    assert hit.sum() > 0, "fixture: no fragment touches the spoiled base"

    assert meta["n_dropped_invalid_or_masked"] == hit.sum()
    np.testing.assert_array_equal(counts["length_counts_all"], clean["length_counts_all"])
    np.testing.assert_array_equal(
        counts["length_counts_all"] - counts["length_counts"],
        np.bincount(Ln[hit] - cfg.L_min, minlength=len(counts["length_counts"])))
    assert clean["N_start"].sum() - counts["N_start"].sum() == len(spoiled)
    assert sum(counts[t].sum() for t in TABLES) == 2 * (meta["n_fragments_counted"])


# ---------------------------------------------------------------------------
# T13: refuse anything counted at another registration
# ---------------------------------------------------------------------------


def _write_tables(path, n_L, **extra):
    rng = np.random.default_rng(0)
    np.savez(path, **{t: rng.dirichlet(np.ones(NHEX)) for t in TABLES},
             p_L=rng.dirichlet(np.ones(n_L)), k=np.int64(6),
             L_min=np.int64(25), L_max=np.int64(25 + n_L - 1), **extra)


@pytest.mark.parametrize("extra, match", [
    ({}, HEX_REGISTRATION_KEY),                                  # stage1 / v4 schema
    ({HEX_REGISTRATION_KEY: np.int64(-1)}, r"m\+L-1"),
])
def test_T13_loader_refuses_unregistered_tables(tmp_path, extra, match):
    path = str(tmp_path / "t.npz")
    _write_tables(path, 156, **extra)
    for freeze_hex in (True, False):
        with pytest.raises(ValueError, match=match):
            CutSiteFragmentModel(tile_size=256, N1=4, K=4, N2=4, H=4,
                                 hex_tables_path=path, freeze_hex=freeze_hex)


def _checkpoint(tmp_path, cfg, store_path, hex_path):
    torch.manual_seed(0)
    model = CutSiteFragmentModel(
        L_min=cfg.L_min, L_max=cfg.L_max, tile_size=cfg.tile_size,
        N1=8, K=8, N2=4, H=4, L_chunk=32, dispersion_window_size=cfg.tile_size,
        nb_w0=0.0, hex_tables_path=hex_path, freeze_hex=True)
    ds = CutSiteFragmentDataset(
        store_path=store_path, model_input_size=model.model_input_size,
        model_L_min=cfg.L_min, model_L_max=cfg.L_max, split="train", min_N=1,
        train_mode=False)
    trainer = L.Trainer(limit_val_batches=1, logger=False, enable_checkpointing=False,
                        accelerator="cpu", enable_progress_bar=False,
                        enable_model_summary=False)
    trainer.validate(model, torch.utils.data.DataLoader(
        ds, batch_size=1, collate_fn=cut_site_collate_fn), verbose=False)
    ckpt = str(tmp_path / "m.ckpt")
    trainer.save_checkpoint(ckpt, weights_only=True)
    return model, ckpt


@pytest.mark.parametrize("edit, match", [
    ("drop", "predates the m\\+L registration"),       # bad981d-format state dict
    ("minus_one", r"m\+L-1"),
])
def test_T13_checkpoint_at_another_registration_is_refused(tmp_path, edit, match):
    store_path, cfg = _small_store(tmp_path)
    hex_path = str(tmp_path / "hex.npz")
    _write_tables(hex_path, cfg.L_max - cfg.L_min + 1, **hex_registration_record())
    model, ckpt = _checkpoint(tmp_path, cfg, store_path, hex_path)

    # The positive path first: a checkpoint written under m+L loads.
    load_model(cfg, hex_path, checkpoint=ckpt)

    blob = torch.load(ckpt, weights_only=False)
    key = "_hex_stop_cut_base_offset"
    assert int(blob["state_dict"][key]) == STOP_CUT_BASE_OFFSET
    if edit == "drop":
        del blob["state_dict"][key]
    else:
        blob["state_dict"][key] = torch.tensor(-1)
    bad = str(tmp_path / "bad.ckpt")
    torch.save(blob, bad)
    with pytest.raises(RuntimeError, match=match):
        load_model(cfg, hex_path, checkpoint=bad)
    with pytest.raises(RuntimeError, match=match):
        model.load_state_dict(blob["state_dict"])


def test_T13_width_guards_lost_their_stale_minus_one():
    """L_max = WINDOW_HALF + 1 passed the old ``WINDOW_HALF < L_max - 1``
    check; under m+L its far 3' cut lies past L2a's h and must be refused.
    Likewise compute_log_p0 must refuse an x one base too narrow for it."""
    with pytest.raises(ValueError, match="cannot reach every"):
        CutSiteFragmentModel(tile_size=64, L_min=25, L_max=WINDOW_HALF + 1)
    CutSiteFragmentModel(tile_size=64, L_min=25, L_max=WINDOW_HALF)

    P, L_min, L_max = 64, 25, 180
    n_hex, n_L = 4 ** 6, L_max - L_min + 1
    tabs = [torch.zeros(n_hex) for _ in range(4)]
    # hex_idx width = x width - 5.  The far 3' cut (m = P-1, L = L_max) sits
    # at hex position WINDOW_HALF + (P-1) + L_max, so the width that just
    # reaches it is WINDOW_HALF + P + L_max.
    need = WINDOW_HALF + P + L_max
    ok_x = torch.zeros(1, 4, need + 5)
    compute_log_p0(ok_x, *tabs, torch.zeros(n_L), P, L_min, L_max)
    with pytest.raises(ValueError, match="too narrow"):
        compute_log_p0(ok_x[..., :-1], *tabs, torch.zeros(n_L), P, L_min, L_max)


@pytest.mark.skipif(not (os.path.exists(REAL_V4_NPZ) and os.path.exists(REAL_BAD981D_CKPT)),
                    reason="real-store artifacts are on EFS")
def test_T13_real_v4_tables_and_bad981d_checkpoint_are_refused(tmp_path):
    """The two real artifacts the guard exists for."""
    with pytest.raises(ValueError, match=HEX_REGISTRATION_KEY):
        CutSiteFragmentModel(tile_size=1024, hex_tables_path=REAL_V4_NPZ, freeze_hex=True)
    hex_path = str(tmp_path / "hex.npz")
    _write_tables(hex_path, 156, **hex_registration_record())
    with pytest.raises(RuntimeError, match="predates the m\\+L registration"):
        load_model(CutSiteStoreConfig(tile_size=1024), hex_path,
                   checkpoint=REAL_BAD981D_CKPT)
