"""End-to-end test for the real cut-site store builder.

Calls ``build_store`` / ``_load_regions`` / ``_extract_worker`` from
``scripts/cut_site_model/build_store.py`` directly -- unlike
``tests/test_build_cut_site_store.py``, which copies the builder's logic by
hand and so cannot catch a drift between the copy and the real code (e.g. the
``samples/role`` removal, checked there only by reading).

Coordinate convention (established from fragments_h5.fetch_array's docstring
and background_model.cut_site_model.store.recompute_endpoints_from_fragments,
both read before writing this file): the h5 stores fragments as
``[start, stop)`` -- stop EXCLUSIVE -- so ``length = stop - start`` and the
inclusive last base is ``start + length - 1``. ``fetch_array`` returns every
fragment that OVERLAPS the half-open query region ``[region_start,
region_stop)``; a fragment whose absolute start equals ``region_stop`` does
not overlap and is never handed to the builder's own start-based filter.

Geometry: tile_size=256, jitter=32, rf_budget=186 => l_target=320, l_seq=692
(l_seq = tile_size + 2*(jitter+rf_budget) = 256 + 2*218 = 692).

Region layout (BED row centres, toy genome, contig "chrT"):
    180   -> dropped: r.start=20, seq_start=20-186=-146 < 0
    520   -> region A: r.start=360, r.stop=680 (fragment fixtures below)
    1220  -> region B: r.start=1060, r.stop=1380 (1 fragment)
    1920  -> region C: r.start=1760, r.stop=2080 (1 fragment)
    2620  -> region D: r.start=2460, r.stop=2780 (1 fragment)
All seq windows stay inside the toy genome's de Bruijn core (bases
0..4100), so sequence content is plain ACGT with no N's or soft-masking.

Region A fragment fixtures (abs coords), chosen to hit every admission
boundary against region A's l_target window [360, 680):
    (359, 384, '+')  region-local start -1            -> DROPPED (start < 0)
    (360, 385, '-')  region-local start 0,  len 25     -> KEPT (strand 1)
    (679, 704, '+')  region-local start 319, len 25    -> KEPT (strand 0);
                     stop=704 runs past reg_stop=680 -- end-past-region case
    (680, 705, '+')  region-local start 320            -> ABSENT (no overlap
                     with [360,680) at all -- fetch_array never returns it)
    (460, 484, '+')  region-local start 100, len L_MIN-1 -> DROPPED (too short)
    (460, 485, '-')  region-local start 100, len L_MIN   -> KEPT (strand 1)
    (460, 640, '+')  region-local start 100, len L_MAX   -> KEPT (strand 0)
    (460, 641, '-')  region-local start 100, len L_MAX+1 -> DROPPED (too long)
    (560, 585, '+')  region-local start 200, len 25 x2   -> exact duplicate;
                     dedup keeps 1, --no-dedup keeps both
"""

import argparse
import os
import sys

import numpy as np
import pytest
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.cut_site_model.build_store import build_store
from background_model.constants import L_MAX, L_MIN
from background_model.cut_site_model.dataset import CutSiteFragmentDataset
from background_model.cut_site_model.model import CutSiteFragmentModel
from background_model.cut_site_model.store import (
    CutSiteStoreConfig,
    densify_endpoints,
    fragment_slice,
    recompute_endpoints_from_fragments,
)

from cut_site_helpers import _build_h5

TILE_SIZE = 256
JITTER = 32
RF_BUDGET = 186
L_TARGET = TILE_SIZE + 2 * JITTER          # 320
MARGIN = (L_TARGET - TILE_SIZE) // 2        # 32 -- centred-crop origin


def _region_bed_row(centre):
    """(start, stop) BED fields whose integer midpoint is exactly ``centre``."""
    return centre - 5, centre + 5


# Region centres: first is dropped (too close to contig start), rest valid.
DROPPED_CENTRE = 180
REGION_CENTRES = {"A": 520, "B": 1220, "C": 1920, "D": 2620}


def _region_bounds(centre):
    return centre - L_TARGET // 2, centre + L_TARGET // 2


REGION_BOUNDS = {name: _region_bounds(c) for name, c in REGION_CENTRES.items()}


@pytest.fixture(scope="module")
def regions_bed(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("regions")
    bed_path = str(tmpdir / "regions.bed")
    rows = [("chrT", *_region_bed_row(DROPPED_CENTRE))]
    rows += [("chrT", *_region_bed_row(c)) for c in REGION_CENTRES.values()]
    with open(bed_path, "w") as f:
        for contig, start, stop in rows:
            f.write(f"{contig}\t{start}\t{stop}\n")
    return bed_path


@pytest.fixture(scope="module")
def fragment_h5(toy_fasta, tmp_path_factory):
    a0 = REGION_BOUNDS["A"][0]  # 360
    bed_rows = [
        # --- region A: every admission boundary ---
        ("chrT", a0 - 1, a0 - 1 + L_MIN, "+", 40, 40),       # relstart -1: DROP
        ("chrT", a0, a0 + L_MIN, "-", 40, 40),               # relstart 0: KEEP
        ("chrT", a0 + 319, a0 + 319 + L_MIN, "+", 40, 40),   # relstart 319: KEEP, end past region
        ("chrT", a0 + 320, a0 + 320 + L_MIN, "+", 40, 40),   # relstart 320: ABSENT (no overlap)
        ("chrT", a0 + 100, a0 + 100 + (L_MIN - 1), "+", 40, 40),  # len L_MIN-1: DROP
        ("chrT", a0 + 100, a0 + 100 + L_MIN, "-", 40, 40),        # len L_MIN: KEEP
        ("chrT", a0 + 100, a0 + 100 + L_MAX, "+", 40, 40),        # len L_MAX: KEEP
        ("chrT", a0 + 100, a0 + 100 + (L_MAX + 1), "-", 40, 40),  # len L_MAX+1: DROP
        ("chrT", a0 + 200, a0 + 200 + L_MIN, "+", 40, 40),   # duplicate #1
        ("chrT", a0 + 200, a0 + 200 + L_MIN, "+", 40, 40),   # duplicate #2 (exact dup)
        # --- regions B, C, D: one fragment each ---
        ("chrT", REGION_BOUNDS["B"][0] + 50, REGION_BOUNDS["B"][0] + 50 + L_MIN, "+", 40, 40),
        ("chrT", REGION_BOUNDS["C"][0] + 50, REGION_BOUNDS["C"][0] + 50 + 100, "-", 40, 40),
        ("chrT", REGION_BOUNDS["D"][0] + 50, REGION_BOUNDS["D"][0] + 50 + L_MAX, "+", 40, 40),
    ]
    tmpdir = str(tmp_path_factory.mktemp("h5"))
    return _build_h5(bed_rows, toy_fasta, tmpdir)


def _build(regions_bed, fragment_h5, toy_fasta, output, no_dedup=False):
    args = argparse.Namespace(
        regions=regions_bed,
        reference=toy_fasta,
        sample=fragment_h5,
        output=output,
        tile_size=TILE_SIZE,
        jitter=JITTER,
        rf_budget=RF_BUDGET,
        L_min=L_MIN,
        L_max=L_MAX,
        workers=1,
        no_dedup=no_dedup,
        split_seed=42,
        val_frac=0.25,
        test_frac=0.25,
    )
    build_store(args)
    return output


@pytest.fixture(scope="module")
def store_path(regions_bed, fragment_h5, toy_fasta, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("store") / "store.zarr")
    return _build(regions_bed, fragment_h5, toy_fasta, out, no_dedup=False)


@pytest.fixture(scope="module")
def store_root(store_path):
    return zarr.open_group(store_path, mode="r")


@pytest.fixture(scope="module")
def region_index(store_root):
    """Map region name -> tile index, located by the region's stored start."""
    starts = np.asarray(store_root["tiles/start"][:])
    idx = {}
    for name, (r_start, _) in REGION_BOUNDS.items():
        hits = np.nonzero(starts == r_start)[0]
        assert len(hits) == 1, f"region {name} (start={r_start}) not found exactly once"
        idx[name] = int(hits[0])
    return idx


def test_store_schema(store_root):
    """Schema matches the docstring: no samples/role, split codes in range,
    config round-trips through CutSiteStoreConfig."""
    assert "library" in store_root["samples"]
    assert "total_fragments" in store_root["samples"]
    assert "role" not in store_root["samples"]

    split = np.asarray(store_root["tiles/split"][:])
    assert set(np.unique(split)).issubset({0, 1, 2})

    assert "config_json" in store_root.attrs
    assert "config_hash" in store_root.attrs
    cfg = CutSiteStoreConfig.from_json(store_root.attrs["config_json"])
    assert cfg.tile_size == TILE_SIZE
    assert cfg.jitter == JITTER
    assert cfg.rf_budget == RF_BUDGET
    assert cfg.L_min == L_MIN
    assert cfg.L_max == L_MAX
    assert cfg.l_target == L_TARGET
    assert cfg.config_hash() == store_root.attrs["config_hash"]


def test_dropped_region_absent(store_root):
    """The near-contig-start region (seq_start < 0) is dropped; T == 4."""
    contigs = store_root["tiles/contig"][:]
    starts = np.asarray(store_root["tiles/start"][:])
    assert len(contigs) == 4
    dropped_start, _ = _region_bounds(DROPPED_CENTRE)
    assert dropped_start not in starts


def test_region_a_fragment_content(store_root, region_index):
    """Region A's stored fragments match the hand-derived expectation from
    the BED rows: dedup keeps one copy of the duplicate pair, boundary
    lengths L_MIN/L_MAX are kept, L_MIN-1/L_MAX+1 are dropped, the
    region-local-start -1 and 320 fragments are absent."""
    T = len(store_root["tiles/contig"])
    idx = region_index["A"]
    start, length, strand = fragment_slice(store_root, 0, idx, T)

    got = sorted(zip(start.tolist(), length.tolist(), strand.tolist()))
    expected = sorted([
        (0, L_MIN, 1),
        (100, L_MIN, 1),
        (100, L_MAX, 0),
        (200, L_MIN, 0),
        (319, L_MIN, 0),
    ])
    assert got == expected


def test_no_dedup_keeps_duplicate(regions_bed, fragment_h5, toy_fasta, tmp_path_factory, region_index):
    """--no-dedup (args.no_dedup=True) keeps both copies of the exact
    duplicate; dedup (default, tested above) keeps one."""
    out = str(tmp_path_factory.mktemp("store_nodedup") / "store.zarr")
    path = _build(regions_bed, fragment_h5, toy_fasta, out, no_dedup=True)
    root = zarr.open_group(path, mode="r")

    starts = np.asarray(root["tiles/start"][:])
    idx_a = int(np.nonzero(starts == REGION_BOUNDS["A"][0])[0][0])
    T = len(root["tiles/contig"])
    start, length, strand = fragment_slice(root, 0, idx_a, T)
    got = sorted(zip(start.tolist(), length.tolist(), strand.tolist()))
    expected = sorted([
        (0, L_MIN, 1),
        (100, L_MIN, 1),
        (100, L_MAX, 0),
        (200, L_MIN, 0),
        (200, L_MIN, 0),  # duplicate, kept twice
        (319, L_MIN, 0),
    ])
    assert got == expected
    assert int(root["samples/total_fragments"][0]) == 9  # 6 (A) + 1 + 1 + 1


def test_endpoints_consistency(store_root, region_index):
    """Dense endpoints recomputed from stored fragments match stored
    endpoints; totals/N matches the centre-tile sum of that recomputation."""
    T = len(store_root["tiles/contig"])
    frag_indptr = np.asarray(store_root["fragments/indptr"][:])
    ep_indptr = np.asarray(store_root["endpoints/indptr"][:])
    N_totals = np.asarray(store_root["totals/N"][:])

    for t in range(T):
        flo, fhi = int(frag_indptr[t]), int(frag_indptr[t + 1])
        fs = np.asarray(store_root["fragments/start"][flo:fhi])
        fl = np.asarray(store_root["fragments/length"][flo:fhi])
        recomputed = recompute_endpoints_from_fragments(fs, fl, L_TARGET)

        elo, ehi = int(ep_indptr[t]), int(ep_indptr[t + 1])
        stored = densify_endpoints(
            np.asarray(store_root["endpoints/pos"][elo:ehi]),
            np.asarray(store_root["endpoints/track"][elo:ehi]),
            np.asarray(store_root["endpoints/data"][elo:ehi]),
            L_TARGET,
        )
        np.testing.assert_array_equal(stored, recomputed)

        for track in range(2):
            expected_total = recomputed[track, MARGIN:MARGIN + TILE_SIZE].sum()
            assert N_totals[0, t, track] == expected_total


def test_dataset_loads_each_split(store_root, store_path):
    """CutSiteFragmentDataset constructs for every split present and each
    admitted fragment's frag_store_idx row matches the store's raw arrays."""
    model = CutSiteFragmentModel(tile_size=TILE_SIZE, N1=4, K=4, N2=2, H=2, L_chunk=32)
    model_input_size = model.model_input_size

    frag_start_all = np.asarray(store_root["fragments/start"][:])
    frag_length_all = np.asarray(store_root["fragments/length"][:])
    frag_strand_all = np.asarray(store_root["fragments/strand"][:])

    split_counts = {"train": 2, "val": 1, "heldout_inactive": 1}
    for split, expected_len in split_counts.items():
        ds = CutSiteFragmentDataset(
            store_path=store_path,
            model_input_size=model_input_size,
            model_L_min=L_MIN,
            model_L_max=L_MAX,
            split=split,
            min_N=0,
            train_mode=False,
        )
        assert len(ds) == expected_len, (split, len(ds))

        item = ds[0]
        n_frag = item["frag_N"].item()
        assert n_frag == item["frag_m"].shape[0]
        store_idx = item["frag_store_idx"].numpy()
        assert ((store_idx >= 0) & (store_idx < len(frag_start_all))).all()

        # Cross-check: crop coords stored on the item match re-deriving them
        # from the store's raw (region-local) arrays at frag_store_idx.
        expect_m = frag_start_all[store_idx] - MARGIN
        expect_L_idx = frag_length_all[store_idx].astype(np.int64) - L_MIN
        expect_s = frag_strand_all[store_idx].astype(np.int64)
        np.testing.assert_array_equal(item["frag_m"].numpy(), expect_m)
        np.testing.assert_array_equal(item["frag_L_idx"].numpy(), expect_L_idx)
        np.testing.assert_array_equal(item["frag_s"].numpy(), expect_s)
