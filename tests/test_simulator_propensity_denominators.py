"""Regression test: ``propensities`` must pair each table with the expectation
for the positions it actually tallies.

This pins the defect found on 2026-10-07, where the two ``_rev`` tables divided
by each other's expectation. It was silent: every total stayed plausible, and
the error lived entirely in ``fl_end_weight``'s edge ramp -- 11.7% of a 1536 bp
tile, up to 156x at offset 25.

The test is DETERMINISTIC rather than statistical. The closed-loop version
(draw from the uniform null, require ``r`` constant) is what originally found
the bug, but it needs ~6e6 fragments to separate the correct pairing (CV 0.041)
from the swapped one (CV 0.064), which is too slow for the default suite. Here
the two expectations are made trivially distinguishable instead, so the pairing
is read off exactly and the assertion cannot be satisfied by luck.

Scoped narrowly on purpose: it asserts the PAIRING, not the numerics of
``uniform_hexamer_counts``.
"""

import numpy as np
import pytest

from background_model.simulator.measure import TABLE_NAMES, propensities
from background_model.constants import NHEX
from background_model.hexamers import rc_permutation


@pytest.fixture
def distinguishable_expectations():
    """``N_start`` and ``N_end`` that no permutation or mix-up can confuse.

    Powers of two, disjoint in magnitude: every sum of a subset is unique, so a
    wrong denominator cannot coincidentally produce the right quotient.
    """
    n_start = (1.0 + np.arange(NHEX)).astype(np.float64)
    n_end = 1024.0 * (1.0 + np.arange(NHEX)).astype(np.float64)
    return {"start": n_start, "end": n_end}


def test_each_table_divides_by_the_expectation_for_the_positions_it_tallies(
    distinguishable_expectations,
):
    """The whole point of the fix, stated as the four pairings.

    ``start``/``end`` name the MOLECULE's 5'/3' cut site, but a minus-strand
    fragment's 5' cut site is at its GENOMIC STOP. ``counts_from_hexamers``
    tallies genomic stops into ``start_rev`` and genomic starts into
    ``end_rev``, so those are the expectations they need -- crossed relative to
    their names.
    """
    expected = distinguishable_expectations
    n_start, n_end = expected["start"], expected["end"]
    perm = rc_permutation()
    counts = {t: np.ones(NHEX, dtype=np.int64) for t in TABLE_NAMES}

    r = propensities(counts, expected)

    want = {
        "start_fwd": 1.0 / n_start,          # genomic starts
        "end_fwd": 1.0 / n_end,              # genomic stops
        "start_rev": 1.0 / n_end[perm],      # genomic STOPS, relabelled
        "end_rev": 1.0 / n_start[perm],      # genomic STARTS, relabelled
    }
    for table in TABLE_NAMES:
        np.testing.assert_allclose(
            r[table], want[table], rtol=0, atol=0,
            err_msg=(
                f"{table} is not dividing by the expectation for the positions "
                f"it tallies. Table names are molecule-relative; denominators "
                f"must be position-relative."
            ),
        )


def test_the_rev_tables_are_not_swapped(distinguishable_expectations):
    """Name the failure, not just the invariant.

    The previous code was self-consistent and plausible, so an invariant that
    merely says "r is finite and positive" passes under the bug. This asserts
    the specific wrong answer is NOT produced.
    """
    expected = distinguishable_expectations
    n_start, n_end = expected["start"], expected["end"]
    perm = rc_permutation()
    counts = {t: np.ones(NHEX, dtype=np.int64) for t in TABLE_NAMES}

    r = propensities(counts, expected)

    swapped_start_rev = 1.0 / n_start[perm]
    swapped_end_rev = 1.0 / n_end[perm]
    assert not np.allclose(r["start_rev"], swapped_start_rev), (
        "start_rev is dividing by N_start[perm] -- the 2026-10-07 bug. It "
        "tallies genomic STOPS, so it needs N_end[perm]."
    )
    assert not np.allclose(r["end_rev"], swapped_end_rev), (
        "end_rev is dividing by N_end[perm] -- the 2026-10-07 bug. It tallies "
        "genomic STARTS, so it needs N_start[perm]."
    )


def test_the_two_expectations_are_actually_different():
    """Guard the guard.

    If ``N_start`` and ``N_end`` ever became equal, both tests above would pass
    under the bug. On real data they differ because ``N_end`` carries
    ``fl_end_weight``'s edge ramp, but a fixture that lost that property would
    make this file silently vacuous.
    """
    n_start = (1.0 + np.arange(NHEX)).astype(np.float64)
    n_end = 1024.0 * (1.0 + np.arange(NHEX)).astype(np.float64)
    assert not np.allclose(n_start, n_end)
    perm = rc_permutation()
    assert not np.allclose(n_start[perm], n_end[perm])
    # and the permutation must not be the identity, or _fwd/_rev collapse
    assert not np.array_equal(perm, np.arange(NHEX))


def test_permutation_is_an_involution():
    """``rc_permutation`` applied twice is the identity.

    Both ``_rev`` denominators index through it once. If it were not an
    involution, "the reverse complement of the reverse complement" would not
    return the original hexamer and the relabelling argument would not hold.
    """
    perm = rc_permutation()
    np.testing.assert_array_equal(perm[perm], np.arange(NHEX))
