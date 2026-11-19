# attic/twelve_track_comparison — REFERENCE ONLY, NOT LIVE CODE

The two scripts that existed solely to compare the 12-track band model against
the cut-site fragment model on a common per-fragment NLL metric, plus the test
for one of them. Retired 2026-10-10 by owner ruling. **Do not import from here,
and do not treat anything here as describing current behaviour.**

`make test` collects `test/ tests/ fragmentomics_tools/`, so nothing under
`attic/` runs. Moving `tests/test_eval_fragment_nll.py` took **7 tests** out of
the default suite.

Nothing here runs as written: both scripts reach the repo root with a
two-level `dirname` walk that assumed `scripts/`, and
`eval_fragment_nll.py`'s test imported it as `scripts.eval_fragment_nll`.
Those paths no longer exist.

## Why it was retired

**The comparison itself was dropped, not the code.** The 12-track model needs
~16 kb tiles at this depth, and the quiet region set is 60,613 isolated
1,536 bp islands — only 4 runs of adjacent regions are long enough for a single
16 kb tile. `background_model/band_model/dataset.py` gates on `N.min(axis=2)`
across all 12 tracks, and at tile 1024 this sample gives a median of 2
counts/track with 22,910 tiles having a track at zero, so `num_samples=0` and
the loader raises. Incompatible at any tile size on that region set.

Revive when masking is handled — large tiles that *contain* repeat and
blacklist positions, with those positions excluded from the loss via the mask,
rather than requiring fully clean regions. Then 16 kb tiles become available
and the comparison is worth rebuilding.

## These scripts are correct, not broken

Worth stating plainly, because retirement usually implies the opposite.

`eval_fragment_nll.py` was retired **immediately after a real bug in it was
fixed**. It previously used `L_MAX = max(hi) - 1 = 179`, the half-open reading
of `FL_BANDS`. A length-180 fragment is admitted by the cut-site store and
model but falls in no track band, so had one reached the scoring path it would
have been scored from uninitialised `np.empty` memory.

Owner decision 185 settled it: in this script only, a length-180 fragment is
scored with the top band's tracks `(110, 180)`, `log Z` includes that cell, and
a guard raises if any fragment lands in no `(strand, band)` group. `FL_BANDS`
and `tracks.py` were untouched. That fix is present in the copy here.

So anyone reviving this should start from the assumption that the L=180
handling is sound, and should read `background_model/constants.py` for why 180
is inclusive.

## Numbers these scripts produced, and why not to quote them

`docs/pending/training_analysis.md` section 10 reports `12.325` for the old CNN
from `eval_fragment_nll.py`. **That is a training-set score**: the 12-track
store it was measured on shared 100% of its regions with the cut-site
validation split. Leakage flatters the 12-track model, and it still lost, so
the direction held — but the margin is understated and the figure is not
comparable with the cut-site validation numbers beside it.

## Contents

    scripts/eval_fragment_nll.py   12-track -> per-fragment NLL bridge
    scripts/train_12track.py       training entry point for the band model
    tests/test_eval_fragment_nll.py  7 tests, decision-185 behaviour
