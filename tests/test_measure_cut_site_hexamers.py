"""Functional test for scripts/measure_cut_site_hexamers.py.

Self-contained: uses no fixture from tests/conftest.py or helper from
tests/cut_site_helpers.py, only the committed golden chr6 fixtures.

The script and the driver both call ``simulator.measure.measure_sample``, so
the npz is checked two ways: against ``measure_sample``'s own return (the
script wrote what was measured), and against invariants that hold whatever
the sequence computes (counts agree with each other, the uniform null covers
every position once).  The second set does not repeat the measure sequence.
"""

import hashlib
import importlib.util
import json
import os

import numpy as np
import pytest

from fragmentomics_tools.dataframe import RegionDataFrame

from background_model import constants
from background_model.simulator.measure import (
    DEFAULT_MIN_EXPECTED,
    TABLE_NAMES,
    measure_sample,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")
_FRAGMENTS_H5 = os.path.join(_DATA, "golden.small.chr6.frag.h5")
_FASTA = os.path.join(_DATA, "GRCh38.p12.genome.chr6_99110000_99130000.fa.gz")

_SCRIPT_PATH = os.path.join(_HERE, "..", "scripts", "measure_cut_site_hexamers.py")
_spec = importlib.util.spec_from_file_location("measure_cut_site_hexamers", _SCRIPT_PATH)
mch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mch)


def _write_bed(path, tiles):
    with open(path, "w") as fh:
        for contig, start, stop in tiles:
            fh.write(f"{contig}\t{start}\t{stop}\n")


# Four contiguous 2000 bp tiles inside the fixture's populated chr6 window
# (99,110,000-99,130,000, confirmed to carry fragments by
# test_golden_h5_matches_bruteforce in tests/test_simulator_measure.py).
# Measured directly against the golden h5 (2026-10-09): at the default
# min_mapq 10 all 7 admitted fragments start in the FIRST tile, so
# --n-regions 2 still exercises real counting. At min_mapq 50 the count falls
# to 5, and at 61 nothing survives (count_srdf raises).
_TILES = [
    ("chr6", 99_118_000, 99_120_000),
    ("chr6", 99_120_000, 99_122_000),
    ("chr6", 99_122_000, 99_124_000),
    ("chr6", 99_124_000, 99_126_000),
]
_TILE_LEN = 2000


@pytest.fixture
def region_bed(tmp_path):
    bed = tmp_path / "regions.bed"
    _write_bed(str(bed), _TILES)
    return str(bed)


def _run_main(out_dir, region_bed, *extra):
    return mch.main([
        "--sample-id", "test-sample",
        "--fragments-h5", _FRAGMENTS_H5,
        "--region-bed", region_bed,
        "--fasta", _FASTA,
        "--out-dir", str(out_dir),
        "--n-workers", "1",
        *extra,
    ])


def _load_json(out_dir):
    with open(out_dir / "test-sample.cut_site_measure.json") as fh:
        return json.load(fh)


def test_main_returns_0_and_writes_both_files_with_no_leftovers(tmp_path, region_bed):
    out_dir = tmp_path / "out"
    assert _run_main(out_dir, region_bed) == 0

    npz_path = out_dir / "test-sample.cut_site_measure.npz"
    json_path = out_dir / "test-sample.cut_site_measure.json"
    assert npz_path.exists()
    assert json_path.exists()

    leftovers = [p for p in os.listdir(out_dir) if p.endswith(".tmp")]
    assert leftovers == []


def test_npz_arrays_match_measure_sample(tmp_path, region_bed):
    """The script writes exactly what ``measure_sample`` measured, in the
    declared dtypes.  Same call, so this checks the WRITER, not the measure
    sequence -- ``test_npz_invariants`` covers that independently."""
    out_dir = tmp_path / "out"
    assert _run_main(out_dir, region_bed) == 0

    rdf = RegionDataFrame.from_bed(region_bed, ref="hg38")
    m = measure_sample(rdf, "test-sample", _FRAGMENTS_H5, _FASTA,
                       min_mapq=10, n_workers=1, verbose=False)

    npz = np.load(out_dir / "test-sample.cut_site_measure.npz")
    for table in TABLE_NAMES:
        got = npz[f"C_{table}"]
        assert got.dtype == np.int64
        np.testing.assert_array_equal(got, m.C[table].astype(np.int64))

        got_r = npz[f"r_{table}"]
        assert got_r.dtype == np.float64
        np.testing.assert_array_equal(got_r, m.r[table].astype(np.float64))

    assert npz["N_start"].dtype == np.int64
    np.testing.assert_array_equal(npz["N_start"], m.N["start"].astype(np.int64))
    assert npz["N_end"].dtype == np.float64
    np.testing.assert_array_equal(npz["N_end"], m.N["end"].astype(np.float64))

    np.testing.assert_array_equal(npz["fl_counts"], m.fl.counts.astype(np.int64))
    assert int(npz["fl_min_fl"]) == m.fl.min_fl

    np.testing.assert_array_equal(npz["region_counts"],
                                  m.region_counts.astype(np.int64))
    np.testing.assert_array_equal(
        npz["region_index"], np.asarray(m.srdf["region_index"], dtype=np.int64))


def test_npz_invariants(tmp_path, region_bed):
    """Hand-checkable relations between the written arrays, independent of
    how the measure sequence computes them."""
    out_dir = tmp_path / "out"
    assert _run_main(out_dir, region_bed) == 0
    npz = np.load(out_dir / "test-sample.cut_site_measure.npz")
    meta = _load_json(out_dir)
    stats, n_meta = meta["count_stats"], meta["n_meta"]
    n_regions = len(_TILES)

    np.testing.assert_array_equal(npz["region_index"], np.arange(n_regions))

    # One admitted population: per-region counts, the stats and f(L) all
    # count the same fragments.
    n_admitted = int(npz["region_counts"].sum())
    assert n_admitted == stats["n_after_filters"] > 0
    assert int(npz["fl_counts"].sum()) == n_admitted
    # f(L) is stored over its observed support, inside the admitted bounds.
    fl_lo = int(npz["fl_min_fl"])
    fl_hi = fl_lo + len(npz["fl_counts"]) - 1
    assert constants.L_MIN <= fl_lo <= fl_hi <= constants.L_MAX

    # Each counted fragment adds one 5' and one 3' cut site, on one strand.
    n_counted = stats["n_counted"]
    assert n_counted <= n_admitted
    assert int(npz["C_start_fwd"].sum()) == int(npz["C_end_fwd"].sum()) == stats["n_plus"]
    assert int(npz["C_start_rev"].sum()) == int(npz["C_end_rev"].sum()) == stats["n_minus"]
    assert stats["n_plus"] + stats["n_minus"] == n_counted

    # The uniform null places one start at every position of every tile. The
    # fixture window holds no N, so every position is a valid hexamer.
    assert n_meta["n_start_invalid"] == 0
    assert int(npz["N_start"].sum()) == n_regions * _TILE_LEN
    # Each start's end weight is a normalised f(L), so the end mass matches.
    np.testing.assert_allclose(npz["N_end"].sum(), n_regions * _TILE_LEN,
                               rtol=1e-12)

    # r = C / N on the table that pairs with N_start directly; 0 where N = 0.
    n = npz["N_start"].astype(np.float64)
    want = np.zeros_like(n)
    want[n > 0] = npz["C_start_fwd"][n > 0] / n[n > 0]
    np.testing.assert_array_equal(npz["r_start_fwd"], want)


def test_n_regions_flag_limits_the_region_set(tmp_path, region_bed):
    out_dir = tmp_path / "out"
    assert _run_main(out_dir, region_bed, "--n-regions", "2") == 0

    npz = np.load(out_dir / "test-sample.cut_site_measure.npz")
    assert len(npz["region_index"]) == 2
    np.testing.assert_array_equal(npz["region_index"], np.arange(2))
    assert int(npz["N_start"].sum()) == 2 * _TILE_LEN

    meta = _load_json(out_dir)
    assert meta["n_regions"] == 2
    assert meta["n_regions_limit"] == 2


def test_min_mapq_flag_reaches_the_fetch(tmp_path, region_bed):
    """``--min-mapq`` must reach ``count_sample``.  On the golden fixture,
    MAPQ 50 drops 2 of the 7 fragments MAPQ 10 admits (measured
    2026-10-09), so a script that ignored the flag -- or hardcoded the
    default -- would write 7 both times."""
    out_lo, out_hi = tmp_path / "lo", tmp_path / "hi"
    assert _run_main(out_lo, region_bed, "--min-mapq", "10") == 0
    assert _run_main(out_hi, region_bed, "--min-mapq", "50") == 0
    lo, hi = _load_json(out_lo), _load_json(out_hi)

    assert lo["min_mapq"] == 10 and hi["min_mapq"] == 50
    assert lo["count_stats"]["n_after_filters"] == 7
    assert hi["count_stats"]["n_after_filters"] == 5
    npz_hi = np.load(out_hi / "test-sample.cut_site_measure.npz")
    assert int(npz_hi["region_counts"].sum()) == 5


def test_json_metadata_fields(tmp_path, region_bed):
    out_dir = tmp_path / "out"
    assert _run_main(out_dir, region_bed) == 0
    meta = _load_json(out_dir)

    assert meta["sample_id"] == "test-sample"
    assert meta["admission"] == "start-in-region"
    assert meta["min_expected"] == DEFAULT_MIN_EXPECTED

    st = os.stat(_FRAGMENTS_H5)
    assert meta["fragments_h5_size"] == st.st_size
    assert meta["fragments_h5_mtime"] == st.st_mtime

    want_sha = hashlib.sha256(open(region_bed, "rb").read()).hexdigest()
    assert meta["region_bed_sha256"] == want_sha

    assert meta["constants"]["L_MIN"] == constants.L_MIN
    assert meta["constants"]["L_MAX"] == constants.L_MAX
    assert meta["constants"]["N_LENGTHS"] == constants.N_LENGTHS
    assert meta["constants"]["KMER"] == constants.KMER
    assert meta["constants"]["HEX_HALF"] == constants.HEX_HALF
    assert meta["constants"]["NHEX"] == constants.NHEX

    npz = np.load(out_dir / "test-sample.cut_site_measure.npz")
    assert meta["count_stats"]["n_after_filters"] == int(npz["region_counts"].sum())
