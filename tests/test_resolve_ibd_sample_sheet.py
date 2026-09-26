"""Tests for scripts/resolve_ibd_sample_sheet.py.

Exercises the flat-stem derivation, the layout preference rule (ibd/frag_h5s
beats the flat cache root when a library resolves both ways), and the
fail-loud-on-ambiguity contract.  All tests build a synthetic cache under
tmp_path and touch no real EFS data.
"""
import sys, os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from resolve_ibd_sample_sheet import (
    flat_stem,
    find_ibd,
    find_flat,
    resolve_library,
    resolve_sheet,
    AmbiguousMatch,
)


def _make_ibd(cache_root, seqrun, md5, library):
    d = cache_root / "ibd" / "frag_h5s" / seqrun
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{md5}-{library}.hg38.fragments.h5"
    p.touch()
    return p


def _make_flat(cache_root, md5, n, stem):
    p = cache_root / f"{md5}-{n}-{stem}.fragments.h5"
    p.touch()
    return p


# ── flat_stem ──────────────────────────────────────────────────────────────

class TestFlatStem:
    def test_strips_lib_suffix(self):
        assert flat_stem("RD-50442-Lib1") == "RD-50442"

    def test_strips_lib2(self):
        assert flat_stem("RD-56928-Lib2") == "RD-56928"

    def test_passthrough_without_suffix(self):
        assert flat_stem("RD-50442") == "RD-50442"

    def test_only_trailing_suffix_stripped(self):
        # An internal Lib token that is not the trailing suffix is preserved.
        assert flat_stem("BDS-1-Lib3_prod-Lib1") == "BDS-1-Lib3_prod"


# ── preference: ibd wins over flat ──────────────────────────────────────────

class TestPreference:
    def test_prefers_ibd_when_both_present(self, tmp_path):
        _make_ibd(tmp_path, "NC-1", "aaaa", "RD-1-Lib1")
        _make_flat(tmp_path, "bbbb", "68", "RD-1")
        path, layout = resolve_library(tmp_path, "NC-1", "RD-1-Lib1")
        assert layout == "ibd"
        assert path.endswith("aaaa-RD-1-Lib1.hg38.fragments.h5")

    def test_flat_used_when_no_ibd(self, tmp_path):
        _make_flat(tmp_path, "bbbb", "68", "RD-1")
        path, layout = resolve_library(tmp_path, "NC-1", "RD-1-Lib1")
        assert layout == "flat"
        assert path.endswith("bbbb-68-RD-1.fragments.h5")

    def test_unresolved_when_neither(self, tmp_path):
        (tmp_path / "ibd" / "frag_h5s" / "NC-1").mkdir(parents=True)
        path, layout = resolve_library(tmp_path, "NC-1", "RD-1-Lib1")
        assert path is None and layout is None


# ── ambiguity: >1 match in a layout must fail loudly ─────────────────────────

class TestAmbiguity:
    def test_ibd_ambiguous_raises(self, tmp_path):
        _make_ibd(tmp_path, "NC-1", "aaaa", "RD-1-Lib1")
        _make_ibd(tmp_path, "NC-1", "cccc", "RD-1-Lib1")  # duplicate stem
        with pytest.raises(AmbiguousMatch, match="ibd/frag_h5s"):
            find_ibd(tmp_path, "NC-1", "RD-1-Lib1")

    def test_flat_ambiguous_raises(self, tmp_path):
        _make_flat(tmp_path, "aaaa", "68", "RD-1")
        _make_flat(tmp_path, "cccc", "70", "RD-1")  # duplicate stem
        with pytest.raises(AmbiguousMatch, match="flat"):
            find_flat(tmp_path, "RD-1-Lib1")

    def test_ambiguity_propagates_through_resolve(self, tmp_path):
        _make_ibd(tmp_path, "NC-1", "aaaa", "RD-1-Lib1")
        _make_ibd(tmp_path, "NC-1", "cccc", "RD-1-Lib1")
        with pytest.raises(AmbiguousMatch):
            resolve_library(tmp_path, "NC-1", "RD-1-Lib1")

    def test_single_match_not_ambiguous(self, tmp_path):
        _make_ibd(tmp_path, "NC-1", "aaaa", "RD-1-Lib1")
        assert find_ibd(tmp_path, "NC-1", "RD-1-Lib1") is not None


# ── end-to-end sheet resolution ──────────────────────────────────────────────

class TestResolveSheet:
    def test_mixed_sheet(self, tmp_path):
        _make_ibd(tmp_path, "NC-1", "aaaa", "RD-1-Lib1")   # ibd
        _make_flat(tmp_path, "bbbb", "68", "RD-2")          # flat
        # RD-3 has neither -> unresolved
        sheet = tmp_path / "in.tsv"
        sheet.write_text(
            "library\tseqrun\n"
            "RD-1-Lib1\tNC-1\n"
            "RD-2-Lib1\tNC-1\n"
            "RD-3-Lib1\tNC-1\n"
        )
        resolved, unresolved = resolve_sheet(str(sheet), tmp_path)
        by_name = {r["sample_name"]: r for r in resolved}
        assert set(by_name) == {"RD-1-Lib1", "RD-2-Lib1"}
        assert by_name["RD-1-Lib1"]["layout"] == "ibd"
        assert by_name["RD-2-Lib1"]["layout"] == "flat"
        assert unresolved == ["RD-3-Lib1"]

    def test_seqrun_scoping(self, tmp_path):
        """An ibd file under a DIFFERENT seqrun must not resolve the library."""
        _make_ibd(tmp_path, "NC-1", "aaaa", "RD-1-Lib1")  # lives under NC-1
        sheet = tmp_path / "in.tsv"
        sheet.write_text("library\tseqrun\nRD-1-Lib1\tNC-2\n")  # sheet says NC-2
        resolved, unresolved = resolve_sheet(str(sheet), tmp_path)
        assert resolved == []
        assert unresolved == ["RD-1-Lib1"]
