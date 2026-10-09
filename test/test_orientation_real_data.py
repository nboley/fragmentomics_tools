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


VALID_STATUS_VALUES = {"correct"}


class TestManifestFileIntegrity:
    """Structural checks on the committed manifest — no real data needed."""

    def test_manifest_file_is_committed_and_parsable(self):
        assert MANIFEST.exists(), f"{MANIFEST} is missing"
        df = pd.read_csv(MANIFEST, sep="\t")
        assert list(df.columns) == [
            "op", "dataset", "n", "sha256_16", "status", "note",
        ]
        assert len(df) > 0

    def test_status_values_are_from_allowed_set(self):
        df = pd.read_csv(MANIFEST, sep="\t")
        bad = set(df["status"].unique()) - VALID_STATUS_VALUES
        assert not bad, f"Unknown status values in manifest: {bad}"

    def test_no_pinned_broken_rows_remain(self):
        """All construction-time defects are fixed — no pinned_broken rows
        should remain in the manifest."""
        df = pd.read_csv(MANIFEST, sep="\t")
        broken = df[df["status"].str.startswith("pinned_broken")]
        assert len(broken) == 0, (
            f"{len(broken)} pinned_broken rows remain in manifest after "
            "autoflip removal — regenerate the manifest."
        )


@pytest.mark.requires_real_data
class TestOrientationManifestStillDescribesReality:
    def test_regenerated_manifest_matches_the_committed_one(self, request):
        """The actual regression net.

        Compares op/dataset/n/sha256_16/status. The ``note`` column is
        excluded: it carries human-readable context that legitimately changes.
        """
        _require_real_data(request)

        cap = _load_capture_module()
        regenerated = pd.concat(
            [pd.DataFrame(cap.input_provenance()), cap.capture()],
            ignore_index=True,
        )

        committed = pd.read_csv(MANIFEST, sep="\t")
        keys = ["op", "dataset", "n", "sha256_16", "status"]

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
    """Verify the no-autoflip invariant holds on real data.

    After removing construction-time autoflip, loading with strand="-"
    must produce the same array data as loading strandless — both return
    unflipped genomic-order data.  Orientation is deferred to the consumer
    layer via reverse_strand() or make_data_direction_match_strand().
    """

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
class TestWeightsSymmetryInvariant:
    """Verify the weights no-autoflip invariant on real data.

    After removing construction-time autoflip, weights from a minus-strand
    load must be identical to weights from a strandless load — both in
    original genomic order.
    """

    def test_weights_symmetry_rows_all_pass(self, request):
        _require_real_data(request)

        cap = _load_capture_module()
        manifest = pd.concat(
            [pd.DataFrame(cap.input_provenance()), cap.capture()],
            ignore_index=True,
        )

        sym_rows = manifest[
            manifest["op"] == "symmetry_weights_strandless_vs_minus"
        ]
        assert len(sym_rows) > 0, "No weights symmetry rows found"

        failures = sym_rows[sym_rows["n"] == 0]
        if len(failures) > 0:
            pytest.fail(
                f"Weights symmetry broken for {len(failures)} regions:\n"
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
