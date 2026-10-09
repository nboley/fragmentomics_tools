"""scripts/eval_fragment_nll.py after owner decision 185 (L_MAX 179 -> 180).

L_MAX now comes from background_model.constants and is INCLUSIVE, but the FL
bands are half-open, so L = 180 is admitted yet falls in no band.  The scorer
fills log_frag_p from np.empty one band at a time, so an unbanded fragment
would contribute uninitialised memory.  Until the owner decides how a 12-track
model should score it, the evaluator must refuse rather than guess.
"""

import numpy as np
import pytest

import scripts.eval_fragment_nll as ev
from background_model import constants
from background_model.tracks import FL_BANDS
from background_model_core import calculate_start_and_stop_from_jitter

P, JITTER = 512, 32
L_TARGET = P + 2 * JITTER


def _region(lengths, p=10, strand=0):
    crop_start, _ = calculate_start_and_stop_from_jitter(L_TARGET, P, 0)
    n = len(lengths)
    probs = np.full((12, P), 1.0 / P)
    p_marginal = np.full(ev.L_MAX - ev.L_MIN + 1, 1.0 / (ev.L_MAX - ev.L_MIN + 1))
    return ev._eval_region(
        probs, np.ones(P, dtype=bool),
        np.full(n, crop_start + p), np.asarray(lengths), np.full(n, strand),
        P, L_TARGET, p_marginal,
    )


def test_bounds_are_the_shared_inclusive_ones():
    assert (ev.L_MIN, ev.L_MAX) == (constants.L_MIN, constants.L_MAX) == (25, 180)
    # 180 is the top band's half-open hi, so it is in no band.
    assert not any(lo <= ev.L_MAX < hi for lo, hi in FL_BANDS)


def test_in_band_fragments_score_finite():
    nll, n = _region([25, 100, 110, 179])
    assert n == 4 and np.isfinite(nll)


@pytest.mark.parametrize("strand", [0, 1])
def test_unbanded_L_MAX_fragment_is_refused(strand):
    with pytest.raises(ValueError, match="fall in no FL band"):
        _region([100, ev.L_MAX], strand=strand)
