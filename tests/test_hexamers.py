"""Tests for ``background_model/hexamers.py`` (the encoder) and
``background_model/constants.py`` (the shared cut-site definitions).

Split out of ``tests/test_cut_site_simulator.py`` by owner decision 188; the
test bodies are unchanged.  Shared fixtures are in ``tests/conftest.py`` and
shared helpers in ``tests/cut_site_helpers.py``.  The cross-module hygiene
checks (layering, doctests, oracle independence) are in
``tests/test_cut_site_hygiene.py``.

Every oracle imports NOTHING from the module under test.  The AST check
``test_oracle_is_independent`` enforces this.

Mutations each test must catch are documented in-line as comments.
"""

import ast
import glob
import os

import numpy as np

import cut_site_oracle as oracle
from cut_site_helpers import DB_CORE_LEN, TANDEM_HEX, TANDEM_REPEATS

from background_model.constants import (
    HEX_HALF,
    KMER,
    L_MAX,
    L_MIN,
    N_LENGTHS,
    NHEX,
)
from background_model.hexamers import (
    hexamer_indices,
    hexamer_vocabulary,
    rc_permutation,
)


# ── T0: Encoder ─────────────────────────────────────────────────────────────

class TestT0Encoder:
    """Pure encoder tests, no h5 needed."""

    def test_encoder_matches_oracle_all_4096(self):
        """M1 (swap G↔T), M2 (little-endian), M33 (valid always True)."""
        for i in range(NHEX):
            h = oracle.DECODE(i)
            fwd, rc, valid = hexamer_indices(h)
            assert int(fwd[0]) == oracle.IDX(h), f"fwd mismatch at {h}"
            assert int(rc[0]) == oracle.IDX(oracle.RC(h)), f"rc mismatch at {h}"
            assert bool(valid[0]) is True, f"valid hexamer marked invalid: {h}"

        db = oracle.de_bruijn(4, 6)
        fwd, rc, valid = hexamer_indices(db)
        assert valid.all(), "de Bruijn core has no invalid windows"
        for pos in range(len(db) - 5):
            h = db[pos:pos + 6]
            assert int(fwd[pos]) == oracle.IDX(h)

    def test_rc_permutation_matches_oracle(self):
        """M4 (complement without reverse)."""
        perm = rc_permutation()
        for i in range(NHEX):
            h = oracle.DECODE(i)
            assert perm[i] == oracle.IDX(oracle.RC(h))
        np.testing.assert_array_equal(perm[perm], np.arange(NHEX))
        palindromes = oracle.all_palindromes()
        fixed = set(np.where(perm == np.arange(NHEX))[0])
        assert len(fixed) == 64
        assert fixed == {oracle.IDX(h) for h in palindromes}

    def test_vocabulary_matches_oracle(self):
        """M1 (swap G↔T)."""
        vocab = hexamer_vocabulary()
        for i in range(NHEX):
            assert vocab[i].decode() == oracle.DECODE(i)

    def test_lowercase_folds(self, toy_genome):
        """M3 (drop lowercase from LUT)."""
        lc_start = DB_CORE_LEN + TANDEM_REPEATS * len(TANDEM_HEX) + len("ACGTAC") + 1 + len("ACGTAC") + 10
        lc_seg = toy_genome[lc_start:lc_start + 50]
        uc_seg = lc_seg.upper()
        assert lc_seg != uc_seg, "fixture must have lowercase"
        fwd_lc, rc_lc, valid_lc = hexamer_indices(lc_seg)
        fwd_uc, rc_uc, valid_uc = hexamer_indices(uc_seg)
        np.testing.assert_array_equal(fwd_lc, fwd_uc)
        np.testing.assert_array_equal(rc_lc, rc_uc)
        np.testing.assert_array_equal(valid_lc, valid_uc)

    def test_single_n_invalid_at_every_offset(self):
        """M33 (valid always True)."""
        for offset in range(KMER):
            bases = list("ACGTAC")
            bases[offset] = "N"
            seq = "".join(bases)
            _fwd, _rc, valid = hexamer_indices(seq)
            assert len(valid) == 1
            assert not valid[0], f"N at offset {offset} should be invalid"

            bases[offset] = "n"
            seq = "".join(bases)
            _fwd, _rc, valid = hexamer_indices(seq)
            assert not valid[0], f"n at offset {offset} should be invalid"


# ── Constants ───────────────────────────────────────────────────────────────

class TestConstants:
    """The one home of the shared cut-site definitions.

    Moved from ``TestT7Hygiene`` by owner decision 188.
    """

    def test_constants_single_source(self):
        """Owner decisions 174-177: the shared cut-site definitions live ONLY
        in ``background_model/constants.py``, and the length bounds are
        derived from the locked ``tracks.FL_BANDS`` rather than restated.
        Guards mutations C1 (``L_MAX`` off by one), C2 (a module restates
        ``L_MAX`` locally -- same value, so only the AST check can see it) and
        C3 (the same restatement in tuple form, ``L_MIN, L_MAX = 25, 180``).
        """
        import background_model.constants as const_mod
        from background_model.tracks import FL_BANDS

        # The values the owner fixed, and their derivation from FL_BANDS.
        assert (L_MIN, L_MAX, N_LENGTHS) == (25, 180, 156)
        assert (KMER, HEX_HALF, NHEX) == (6, 3, 4096)
        assert L_MIN == min(lo for lo, _hi in FL_BANDS)
        assert L_MAX == max(hi for _lo, hi in FL_BANDS)  # INCLUSIVE

        shared = {"L_MIN", "L_MAX", "N_LENGTHS", "KMER", "HEX_HALF", "NHEX"}

        def bound_names(tree):
            """Every name a module binds, at any depth and in any form:
            a Store-context Name covers plain, tuple, starred, augmented and
            annotated assignment, ``for``/``with`` targets, comprehensions
            and the walrus; parameters, ``import ... as`` and
            ``except ... as`` bind without a Name node, so take them too."""
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    yield node.id, node.lineno
                elif isinstance(node, ast.arg):
                    yield node.arg, node.lineno
                elif isinstance(node, ast.alias) and node.asname:
                    yield node.asname, 0
                elif isinstance(node, ast.ExceptHandler) and node.name:
                    yield node.name, node.lineno

        # Non-vacuity of the walker itself: every binding form must be seen.
        probe = ast.parse(
            "L_MIN, (L_MAX, *N_LENGTHS) = 1, (2, 3)\nKMER += 1\nHEX_HALF: int = 3\n"
            "for NHEX in (): pass\nwith f() as KMER: pass\n(HEX_HALF := 3)\n"
            "def g(L_MAX): pass\nimport x as NHEX\n"
        )
        assert {n for n, _ in bound_names(probe)} == shared

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        excluded = {
            # The one home: scanned below for non-vacuity, not as a restatement.
            const_mod.__file__,
            # A self-contained GPU cost skeleton; its L range (25..256) is
            # deliberately not the cut-site one and never meets the store.
            os.path.join(repo_root, "scripts", "bench_fragment_logit_sweep.py"),
        }
        paths = sorted(
            p for p in (
                glob.glob(os.path.join(repo_root, "background_model", "**", "*.py"),
                          recursive=True)
                + glob.glob(os.path.join(repo_root, "scripts", "*.py"))
            )
            # attic/ is superseded code kept for reference, not maintained.
            if p not in excluded and "attic" not in p.split(os.sep)
        )
        # Non-vacuity of the glob: the known importers must be in the scan.
        rel = {os.path.relpath(p, repo_root) for p in paths}
        for known in ("background_model/hexamers.py",
                      "background_model/simulator/measure.py",
                      "background_model/simulator/draw.py",
                      "scripts/run_cut_site_simulator.py"):
            assert known in rel, known

        def parse(path):
            with open(path) as f:
                return ast.parse(f.read(), path)

        offenders = [
            f"{os.path.relpath(path, repo_root)}:{line} binds {name}"
            for path in paths
            for name, line in bound_names(parse(path))
            if name in shared
        ]
        assert not offenders, (
            "these must import from background_model.constants instead:\n"
            + "\n".join(offenders)
        )
        # Non-vacuity: the walker must find the real definitions.
        seen_in_constants = {
            n for n, _ in bound_names(parse(const_mod.__file__)) if n in shared
        }
        assert seen_in_constants == shared, shared - seen_in_constants
