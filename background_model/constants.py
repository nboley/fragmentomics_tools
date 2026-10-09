"""Shared cut-site definitions -- the one home for them (owner decisions 174-177).

Definitions only: the fragment length bounds and the k-mer geometry that the
simulator and the cut-site store/model must agree on.  The functions built on
them (encoder, vocabulary, reverse-complement permutation) stay in
``background_model.hexamers``, which imports its constants from here.
Simulator-internal values (``UNIFORM_BLOCK_SIZE``, ``TABLE_NAMES``, the
encoder's lookup tables) stay with the code that owns them.

Imports only ``background_model.tracks``, which is stdlib-only, so this sits
BELOW ``hexamers`` in the layering and anything can import it without pulling
in numpy, pandas or torch.

**The length bounds are derived from ``tracks.FL_BANDS``, not restated.**
``FL_BANDS`` is locked by the owner and is the single source: ``L_MIN`` is the
lowest band's ``lo`` and ``L_MAX`` the highest band's ``hi``.  ``L_MAX`` is
INCLUSIVE -- a fragment of exactly ``L_MAX`` is admitted -- even though each
band is half-open ``[lo, hi)``, so a length-``L_MAX`` fragment is admitted by
the cut-site path but falls in no track band.  Inclusive 180 is the meaning
here and in the cut-site store and model on the ``worktree-cut-site-model``
branch (``cut_site_store.py``, ``cut_site_model.py``).  One evaluator on that
branch, ``scripts/eval_fragment_nll.py``, uses ``max(hi) - 1`` = 179, the
half-open reading; reconciling it is that branch's decision.  Do not "align"
``L_MAX`` here to the bands without an owner decision, since it changes which
fragments are counted.

>>> L_MIN, L_MAX, N_LENGTHS
(25, 180, 156)
>>> KMER, HEX_HALF, NHEX
(6, 3, 4096)
"""

from background_model.tracks import FL_BANDS

# ── fragment length bounds, INCLUSIVE, from FL_BANDS ──────────────────────
#
# The library's subset_fragment_lengths is half-open, so callers pass
# L_MAX + 1 to it; see cut_site_stats.filter_fragments.

L_MIN: int = min(lo for lo, _hi in FL_BANDS)          # 25
L_MAX: int = max(hi for _lo, hi in FL_BANDS)          # 180, inclusive
N_LENGTHS: int = L_MAX - L_MIN + 1                    # 156

# ── k-mer geometry around a cut site ──────────────────────────────────────

KMER: int = 6
HEX_HALF: int = KMER // 2    # 3 bases in, 3 out, around a cut site
NHEX: int = 4 ** KMER        # 4096
