"""Functional test for scripts/measure_cut_site_hexamers.py.

Self-contained: uses no fixture from tests/conftest.py or helper from
tests/cut_site_helpers.py, only the committed golden chr6 fixtures.
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
    FragmentLengthDist,
    TABLE_NAMES,
    count_sample,
    propensities,
    uniform_hexamer_counts,
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
# Measured directly against the golden h5: fragments cluster in
# [99118000,99122000); the first two tiles here are the non-empty ones so
# that --n-regions 2 still exercises real counting.
_TILES = [
    ("chr6", 99_118_000, 99_120_000),
    ("chr6", 99_120_000, 99_122_000),
    ("chr6", 99_122_000, 99_124_000),
    ("chr6", 99_124_000, 99_126_000),
]


@pytest.fixture
def region_bed(tmp_path):
    bed = tmp_path / "regions.bed"
    _write_bed(str(bed), _TILES)
    return str(bed)


def _direct_measure(region_bed, n_regions=None):
    rdf = RegionDataFrame.from_bed(region_bed, ref="hg38")
    if n_regions is not None:
        rdf = rdf.iloc[:n_regions]
    C, region_counts, stats, srdf = count_sample(
        rdf, "test-sample", _FRAGMENTS_H5, _FASTA,
        min_mapq=10, n_workers=1, verbose=False,
    )
    fl = FragmentLengthDist.from_srdf(srdf)
    N, n_meta = uniform_hexamer_counts(rdf, _FASTA, fl, n_workers=1, verbose=False)
    r = propensities(C, N)
    return rdf, C, region_counts, stats, srdf, fl, N, n_meta, r


def test_main_returns_0_and_writes_both_files_with_no_leftovers(tmp_path, region_bed):
    out_dir = tmp_path / "out"
    rc = mch.main([
        "--sample-id", "test-sample",
        "--fragments-h5", _FRAGMENTS_H5,
        "--region-bed", region_bed,
        "--fasta", _FASTA,
        "--out-dir", str(out_dir),
        "--n-workers", "1",
    ])
    assert rc == 0

    npz_path = out_dir / "test-sample.cut_site_measure.npz"
    json_path = out_dir / "test-sample.cut_site_measure.json"
    assert npz_path.exists()
    assert json_path.exists()

    leftovers = [p for p in os.listdir(out_dir) if p.endswith(".tmp")]
    assert leftovers == []


def test_npz_arrays_match_calling_measure_directly(tmp_path, region_bed):
    out_dir = tmp_path / "out"
    rc = mch.main([
        "--sample-id", "test-sample",
        "--fragments-h5", _FRAGMENTS_H5,
        "--region-bed", region_bed,
        "--fasta", _FASTA,
        "--out-dir", str(out_dir),
        "--n-workers", "1",
    ])
    assert rc == 0

    rdf, C, region_counts, stats, srdf, fl, N, n_meta, r = _direct_measure(region_bed)
    region_index = np.asarray(srdf["region_index"], dtype=np.int64)

    npz = np.load(out_dir / "test-sample.cut_site_measure.npz")
    for table in TABLE_NAMES:
        got = npz[f"C_{table}"]
        assert got.dtype == np.int64
        np.testing.assert_array_equal(got, C[table].astype(np.int64))

        got_r = npz[f"r_{table}"]
        assert got_r.dtype == np.float64
        np.testing.assert_array_equal(got_r, r[table].astype(np.float64))

    assert npz["N_start"].dtype == np.int64
    np.testing.assert_array_equal(npz["N_start"], N["start"].astype(np.int64))
    assert npz["N_end"].dtype == np.float64
    np.testing.assert_array_equal(npz["N_end"], N["end"].astype(np.float64))

    np.testing.assert_array_equal(npz["fl_counts"], fl.counts.astype(np.int64))
    assert int(npz["fl_min_fl"]) == fl.min_fl

    np.testing.assert_array_equal(npz["region_counts"], region_counts.astype(np.int64))
    np.testing.assert_array_equal(npz["region_index"], region_index)
    np.testing.assert_array_equal(npz["region_index"], np.arange(len(_TILES)))


def test_n_regions_flag_limits_the_region_set(tmp_path, region_bed):
    out_dir = tmp_path / "out"
    rc = mch.main([
        "--sample-id", "test-sample",
        "--fragments-h5", _FRAGMENTS_H5,
        "--region-bed", region_bed,
        "--fasta", _FASTA,
        "--out-dir", str(out_dir),
        "--n-regions", "2",
        "--n-workers", "1",
    ])
    assert rc == 0

    npz = np.load(out_dir / "test-sample.cut_site_measure.npz")
    assert len(npz["region_index"]) == 2
    np.testing.assert_array_equal(npz["region_index"], np.arange(2))

    with open(out_dir / "test-sample.cut_site_measure.json") as fh:
        meta = json.load(fh)
    assert meta["n_regions"] == 2
    assert meta["n_regions_limit"] == 2


def test_json_metadata_fields(tmp_path, region_bed):
    out_dir = tmp_path / "out"
    rc = mch.main([
        "--sample-id", "test-sample",
        "--fragments-h5", _FRAGMENTS_H5,
        "--region-bed", region_bed,
        "--fasta", _FASTA,
        "--out-dir", str(out_dir),
        "--n-workers", "1",
    ])
    assert rc == 0

    with open(out_dir / "test-sample.cut_site_measure.json") as fh:
        meta = json.load(fh)

    assert meta["sample_id"] == "test-sample"
    assert meta["admission"] == "start-in-region"

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
