"""Session fixtures for the cut-site simulator tests.

Shared by ``test_hexamers.py``, ``test_simulator_measure.py``,
``test_simulator_draw.py`` and ``test_cut_site_hygiene.py``, which were one file
until owner decision 188.  Moved verbatim from
``tests/test_cut_site_simulator.py``; the helpers they build on are in
``tests/cut_site_helpers.py``.

``make test`` passes ``--ignore=tests/conftest.py``: that stops this file being
COLLECTED for doctests (see the Makefile), not loaded -- pytest still loads it
as a plugin, so these fixtures are available to every module in ``tests/``.
"""

import numpy as np
import pandas as pd
import pytest

from cut_site_helpers import (
    DB_CORE_LEN,
    TANDEM_HEX,
    TANDEM_REPEATS,
    _build_h5,
    _build_toy_genome,
    _write_fasta,
)

from background_model.constants import L_MAX, L_MIN, N_LENGTHS
from background_model.simulator.measure import FragmentLengthDist
from fragmentomics_tools.dataframe import RegionDataFrame


@pytest.fixture(scope="session")
def toy_genome():
    return _build_toy_genome()


@pytest.fixture(scope="session")
def toy_dir(tmp_path_factory, toy_genome):
    tmpdir = str(tmp_path_factory.mktemp("toy"))
    fa_path = _write_fasta(toy_genome, tmpdir)
    return {"dir": tmpdir, "fasta": fa_path, "genome": toy_genome}


@pytest.fixture(scope="session")
def toy_fasta(toy_dir):
    return toy_dir["fasta"]


@pytest.fixture(scope="session")
def toy_regions(toy_genome):
    """Contiguous 1000-bp tiles starting at offset 3 (the minimum left_pad)."""
    starts = list(range(3, 5003, 1000))
    return [(s, s + 1000) for s in starts]


@pytest.fixture(scope="session")
def toy_rdf(toy_regions):
    return RegionDataFrame(pd.DataFrame({
        "contig": "chrT",
        "start": [s for s, _ in toy_regions],
        "stop": [e for _, e in toy_regions],
    }), ref="hg38")


@pytest.fixture(scope="session")
def admission_h5(toy_dir, toy_genome):
    """h5 with planted boundary fragments for T2 tests.

    Includes enough valid fragments on both strands to pass the strand
    balance check when counting across both tiles.
    """
    g0 = 3  # first tile start
    R = 1000
    g1 = g0 + R

    frags = [
        # MAPQ boundary: min(60,9)=9 < 10 drop; min(10,10)=10 keep
        ("chrT", g0 + 100, g0 + 200, "+", 60, 9),
        ("chrT", g0 + 150, g0 + 250, "+", 10, 10),
        # Dedup: same (s,e), first mapq 5 (drop by MAPQ), second mapq 30 (keep)
        ("chrT", g0 + 300, g0 + 400, "+", 5, 5),
        ("chrT", g0 + 300, g0 + 400, "-", 30, 30),
        # Length bounds: 24 drop, 25 keep, 180 keep, 181 drop
        ("chrT", g0 + 500, g0 + 524, "+", 30, 30),  # L=24
        ("chrT", g0 + 500, g0 + 525, "+", 30, 30),  # L=25
        ("chrT", g0 + 500, g0 + 680, "-", 30, 30),  # L=180
        ("chrT", g0 + 500, g0 + 681, "-", 30, 30),  # L=181
        # Start admission: just inside at g0, just outside at g0+R
        ("chrT", g0, g0 + 100, "+", 30, 30),
        ("chrT", g0 + R - 1, g0 + R + 79, "-", 30, 30),
        ("chrT", g0 + R, g0 + R + 100, "+", 30, 30),  # goes to next tile
        # Dedup key omits strand: same (s,e) different strands
        ("chrT", g0 + 700, g0 + 800, "+", 30, 30),
        ("chrT", g0 + 700, g0 + 800, "-", 30, 30),
        # Max overhang: start at g0+R-1, L=180
        ("chrT", g0 + R - 1, g0 + R - 1 + 180, "+", 30, 30),
    ]
    # Add enough valid fragments on both strands for strand balance
    rng = np.random.RandomState(111)
    for i in range(40):
        s = g0 + rng.randint(50, 900)
        L = rng.randint(L_MIN, L_MAX + 1)
        strand = "+" if i % 2 == 0 else "-"
        frags.append(("chrT", s, s + L, strand, 30, 30))

    h5 = _build_h5(frags, toy_dir["fasta"], toy_dir["dir"], name="admission")
    return h5


@pytest.fixture(scope="session")
def bruteforce_h5(toy_dir, toy_genome):
    """h5 with many deterministic fragments for T3 brute-force comparison."""
    g0 = 3
    R = 1000
    frags = []
    rng = np.random.RandomState(123)
    for i in range(500):
        start = g0 + rng.randint(0, R)
        length = rng.randint(L_MIN, L_MAX + 1)
        strand = "+" if rng.random() < 0.5 else "-"
        frags.append(("chrT", start, start + length, strand, 30, 30))

    # Fragments with an N in a cut-site window, one per side.
    #
    # The window for a cut site at genomic `gc` is genome[gc-HEX_HALF : gc+HEX_HALF],
    # so it contains the N at `n_pos` iff gc - 3 <= n_pos < gc + 3, i.e.
    # gc in [n_pos-2, n_pos+3]. This was `n_pos - 3`, whose window is
    # [n_pos-6, n_pos) and EXCLUDES the N -- off by one, so the fixture planted
    # nothing and `test_n_window_fragment_dropped_from_tables` had nothing to
    # detect. That, not a hexamer collision, is why mutation M7 (dropping the
    # validity gate) went uncaught by all 47 tests.
    n_block_start = DB_CORE_LEN + TANDEM_REPEATS * len(TANDEM_HEX)
    n_pos = n_block_start + 6  # the single-N position in the n_block
    # invalid START window, valid end (the end lands in the soft-masked copy,
    # which is ACGT and case-folded, so it stays valid)
    frags.append(("chrT", n_pos, n_pos + 50, "+", 30, 30))
    # invalid END window, valid start (the start sits in the tandem repeat,
    # pure ACGT) -- covers the `e_ok` half of `ok = s_ok & e_ok` separately,
    # so a mutation dropping only one side is still caught
    frags.append(("chrT", n_pos - 50, n_pos, "+", 30, 30))

    h5 = _build_h5(frags, toy_dir["fasta"], toy_dir["dir"], name="bruteforce")
    return h5


@pytest.fixture(scope="session")
def simple_fl():
    """A simple fragment-length distribution for testing."""
    counts = np.zeros(N_LENGTHS, dtype=np.int64)
    counts[0] = 10    # L=25
    counts[25] = 30   # L=50
    counts[75] = 50   # L=100
    counts[155] = 10  # L=180
    return FragmentLengthDist(counts, L_MIN)
