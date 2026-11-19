"""scripts/eval_fragment_nll.py after owner decision 185 (L_MAX 179 -> 180).

L_MAX now comes from background_model.constants and is INCLUSIVE, but the FL
bands are half-open, so L = 180 is admitted yet falls in no band.  The owner's
rule, for this evaluator only: an L = 180 fragment is scored with the TOP
band's tracks, and log Z includes that cell.  Any fragment still in no
(strand, band) group must raise: the scorer fills log_frag_p from np.empty,
so it would otherwise contribute uninitialised memory.
"""

import numpy as np
import pytest

import scripts.eval_fragment_nll as ev
from background_model import constants
from background_model.tracks import FL_BANDS
from background_model_core import calculate_start_and_stop_from_jitter

P, JITTER = 512, 32
L_TARGET = P + 2 * JITTER
N_L = ev.L_MAX - ev.L_MIN + 1
TOP = max(range(len(FL_BANDS)), key=lambda b: FL_BANDS[b][1])


def _region(lengths, p=10, strand=0, probs=None, p_marginal=None):
    crop_start, _ = calculate_start_and_stop_from_jitter(L_TARGET, P, 0)
    n = len(lengths)
    if probs is None:
        probs = np.full((12, P), 1.0 / P)
    if p_marginal is None:
        p_marginal = np.full(N_L, 1.0 / N_L)
    return ev._eval_region(
        probs, np.ones(P, dtype=bool),
        np.full(n, crop_start + p), np.asarray(lengths), np.full(n, strand),
        P, L_TARGET, p_marginal,
    )


def _random_probs(seed):
    probs = np.random.default_rng(seed).random((12, P)) + 0.1
    return probs / probs.sum(axis=1, keepdims=True)


def test_bounds_are_the_shared_inclusive_ones():
    assert (ev.L_MIN, ev.L_MAX) == (constants.L_MIN, constants.L_MAX) == (25, 180)
    # 180 is the top band's half-open hi, so it is in no half-open band.
    assert not any(lo <= ev.L_MAX < hi for lo, hi in FL_BANDS)
    assert FL_BANDS[TOP][1] == ev.L_MAX


def test_band_index_only_exception_is_L_MAX():
    """Every length keeps its half-open band; only L_MAX joins the top band."""
    Ls = np.arange(ev.L_MIN, ev.L_MAX + 1)
    got = ev._band_index(Ls)
    for L, b in zip(Ls.tolist(), got.tolist()):
        if L == ev.L_MAX:
            assert b == TOP
        else:
            lo, hi = FL_BANDS[b]
            assert lo <= L < hi
    # Outside the admitted range nothing is assigned.
    assert ev._band_index([ev.L_MIN - 1, ev.L_MAX + 1]).tolist() == [-1, -1]


def test_in_band_fragments_score_finite():
    nll, n = _region([25, 100, 110, 179])
    assert n == 4 and np.isfinite(nll)


@pytest.mark.parametrize("strand", [0, 1])
def test_L_MAX_fragment_scores_with_top_band_tracks(strand):
    """An L = 180 fragment's score is the top band's first/last tracks at its
    endpoints, plus log p(180), minus log Z."""
    probs = _random_probs(185 + strand)
    p_marginal = np.random.default_rng(7).random(N_L) + 0.1
    p_marginal /= p_marginal.sum()
    start = 10
    end = start + ev.L_MAX - 1

    nll, n = _region([ev.L_MAX], p=start, strand=strand,
                     probs=probs, p_marginal=p_marginal)
    assert n == 1

    def score(band):
        first_t, last_t = ev.ENDPOINT_MAP[(strand, band)]
        f_pos, l_pos = (start, end) if strand == 0 else (end, start)
        return -(np.log(probs[first_t, f_pos]) + np.log(probs[last_t, l_pos])
                 + np.log(p_marginal[ev.L_MAX - ev.L_MIN])
                 - ev._compute_log_Z(probs, p_marginal, P))

    np.testing.assert_allclose(nll, score(TOP), rtol=0, atol=1e-12)
    # The band choice is visible: any other band's tracks give another score.
    for b in range(len(FL_BANDS)):
        if b != TOP:
            assert abs(nll - score(b)) > 1e-6


def test_log_Z_includes_the_L_MAX_cell():
    """With all p(L) mass on L = 180, log Z is the top band's sum alone."""
    probs = _random_probs(186)
    p_marginal = np.zeros(N_L)
    p_marginal[-1] = 1.0
    L = ev.L_MAX
    Z = 0.0
    for s in (0, 1):
        first_t, last_t = ev.ENDPOINT_MAP[(s, TOP)]
        n_start = P - L + 1
        if s == 0:
            Z += np.dot(probs[first_t, :n_start], probs[last_t, L - 1:L - 1 + n_start])
        else:
            Z += np.dot(probs[first_t, L - 1:L - 1 + n_start], probs[last_t, :n_start])
    np.testing.assert_allclose(ev._compute_log_Z(probs, p_marginal, P), np.log(Z),
                               rtol=1e-12)


def test_fragment_in_no_strand_band_group_is_refused():
    """A strand outside {0, 1} joins no group; it must raise, not score np.empty."""
    with pytest.raises(ValueError, match=r"no \(strand, band\) group"):
        _region([100, 150], strand=2)
