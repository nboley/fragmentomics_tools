"""Assert the committed orientation manifest still describes reality.

Fragment orientation — how ``from_fragments_h5`` flips coordinates and swaps
strands for minus-strand regions — is the silent-failure mode the layering
refactor is most likely to break.  Phases 3-4 move the four SRDF geometry
overrides that keep fragment arrays coupled to region coordinates, and getting
the flip wrong silently destroys strand asymmetry rather than raising.

This test regenerates the orientation manifest from real data (one deep h5,
20 CTCF regions, three strand queries per region, SRDF resize and binning
coupling) and compares it to the committed copy.  A movement fails the suite
on the commit that causes it.

It reuses ``scripts/capture_orientation_fixtures.py`` rather than
reimplementing the digest logic, for the same reason the interval test does:
a second implementation of the digest can drift from the one that produced
the committed file.

Marked ``requires_real_data``: the inputs live on EFS.  Skips when absent
under ``make test``; ``make test-realdata`` refuses to skip.
"""

import importlib.util
import os
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "test" / "fixtures" / "orientation_manifest.tsv"
CAPTURE = REPO / "scripts" / "capture_orientation_fixtures.py"

EFS_INPUTS = [
    Path("/efs/analytics/nathanboley/biomarker-projects/data_cache/DC4-16709"
         "/764d5ce67737e478b927fc0ef17f2df1-86-AC-124104-Lib1_DC4-16709_S22.hg38.fragments.h5"),
    Path("/efs/analytics/lily/ssDNA/top_1000_TF/CTCF.hg38.bed"),
]


def _load_capture_module():
    """Import the capture script by path — ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("_capture_orientation", CAPTURE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _missing_inputs():
    return [str(p) for p in EFS_INPUTS if not p.exists()]


def _require_real_data(request):
    missing = _missing_inputs()
    if not missing:
        return
    if request.config.getoption("--realdata", default=False):
        pytest.fail(
            "--realdata was requested but these inputs are missing:\n  "
            + "\n  ".join(missing)
            + "\nThis target exists so a real-data check cannot silently skip."
        )
    pytest.skip(f"real-data inputs unavailable: {missing[0]} (and possibly others)")


@pytest.mark.requires_real_data
class TestOrientationManifestStillDescribesReality:
    def test_manifest_file_is_committed_and_parsable(self):
        assert MANIFEST.exists(), f"{MANIFEST} is missing"
        df = pd.read_csv(MANIFEST, sep="\t")
        assert list(df.columns) == ["op", "dataset", "n", "sha256_16", "note"]
        assert len(df) > 0

    def test_regenerated_manifest_matches_the_committed_one(self, request):
        """The actual regression net.

        Compares op/dataset/n/sha256_16. The ``note`` column is excluded: it
        carries human-readable context that legitimately changes (paths, etc.).
        """
        _require_real_data(request)

        cap = _load_capture_module()
        regenerated = pd.concat(
            [pd.DataFrame(cap.input_provenance()), cap.capture()],
            ignore_index=True,
        )

        committed = pd.read_csv(MANIFEST, sep="\t")
        keys = ["op", "dataset", "n", "sha256_16"]

        got = regenerated[keys].sort_values(keys).reset_index(drop=True)
        want = committed[keys].sort_values(keys).reset_index(drop=True)

        if not got.equals(want):
            merged = want.merge(
                got, on=["op", "dataset"], how="outer", suffixes=("_committed", "_now")
            )
            moved = merged[
                (merged["sha256_16_committed"] != merged["sha256_16_now"])
                | (merged["n_committed"] != merged["n_now"])
            ]
            pytest.fail(
                "The committed orientation manifest no longer describes real-data output.\n"
                "If this change is intended, regenerate with\n"
                "  PYTHONPATH=<worktree> python scripts/capture_orientation_fixtures.py "
                "--out test/fixtures/orientation_manifest.tsv\n"
                "and record WHY the movement is correct.\n"
                "A movement recorded without investigation is how a "
                "592-interval regression got rationalised once already.\n\n"
                + moved.to_string(index=False)
            )


@pytest.mark.requires_real_data
class TestStrandSymmetryInvariant:
    """Verify the strand-flip invariant holds on real data.

    For any genomic interval, loading with strand="-" must produce the
    same fragments as loading strandless and then calling reverse_strand.
    This is the invariant that correction.py depends on: it queries
    strandless, corrects, then orients at the aggregation layer.
    """

    @pytest.mark.xfail(
        reason=(
            "PRODUCTION DEFECT: from_fragments_h5 does not reverse "
            "fragment_strands when flipping for minus-strand regions "
            "(fragment_array.py:1793). Coordinates are reversed ([::-1]) "
            "but strands are only swapped, not reordered — so strand[j] "
            "refers to a different fragment than starts_0[j] after the flip. "
            "The methyl arrays and gc ARE reversed (lines 1797-1802); only "
            "strands are missed. The TODO at line 1786 confirms this was "
            "meant to be refactored into reverse_strand(), which does "
            "reverse strands correctly (line 747). "
            "This is a FINDING — do not patch it away."
        ),
        strict=True,
    )
    def test_symmetry_rows_all_pass(self, request):
        _require_real_data(request)

        cap = _load_capture_module()
        manifest = pd.concat(
            [pd.DataFrame(cap.input_provenance()), cap.capture()],
            ignore_index=True,
        )

        sym_rows = manifest[manifest["op"] == "symmetry_reverse_strandless_vs_minus"]
        assert len(sym_rows) > 0, "No symmetry rows found"

        failures = sym_rows[sym_rows["n"] == 0]
        if len(failures) > 0:
            pytest.fail(
                f"Strand symmetry broken for {len(failures)} regions:\n"
                + failures[["dataset", "note"]].to_string(index=False)
            )


@pytest.mark.requires_real_data
class TestOrientationCrossProcessDeterminism:
    """Orientation digests must be stable across processes.

    Same pattern as the interval cross-process test: two subprocesses,
    no PYTHONHASHSEED pinning, compare outputs.
    """

    def test_two_subprocesses_agree(self, request):
        _require_real_data(request)

        import subprocess
        import sys

        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "import importlib.util\n"
            "spec = importlib.util.spec_from_file_location('c', %r)\n"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            "print(m.capture()[['op','dataset','n','sha256_16']].to_csv(index=False))\n"
        ) % (str(REPO), str(CAPTURE))

        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO)
        env.pop("PYTHONHASHSEED", None)

        outs = []
        for _ in range(2):
            r = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                env=env,
                timeout=900,
            )
            assert r.returncode == 0, f"subprocess failed:\n{r.stderr[-2000:]}"
            outs.append(r.stdout)

        assert outs[0] == outs[1], (
            "Orientation digests differ between two processes on identical input. "
            "Something iterates a set or dict ordered by hash randomisation. "
            "Check that fragment arrays are sorted deterministically."
        )
