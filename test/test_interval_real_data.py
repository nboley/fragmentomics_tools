"""Assert the committed interval manifest still describes reality.

Phase 0 captured digests of every interval operation on real data —
964,593 CTCF sites and the hg38 blacklist — and committed them to
``test/fixtures/interval_manifest.tsv``. **Nothing read that file.** It was
checked only when someone remembered to regenerate it and eyeball a diff,
which is a ritual rather than a test.

Two real regressions lived in the repo as a direct consequence:

* The bioframe migration changed ``merge`` output by **592 intervals** on the
  CTCF set. It survived multiple commits and two implementation reviews, and
  was recorded in the design doc as a *deliberate* fixture movement.
* Pair ordering was non-deterministic across processes. It was stable *within*
  a process, so no in-process test could see it.

Neither was visible to ``make test``, by construction. This module closes that
gap: it regenerates the manifest and compares it to the committed copy, so a
real-data movement fails the suite on the commit that causes it instead of
being discovered later and rationalised.

It reuses ``scripts/capture_interval_fixtures.py`` rather than reimplementing
the capture. A second implementation of the digest logic could drift from the
one that produced the committed file, which would make this test assert
agreement between two things that are both wrong.

Marked ``requires_real_data``: the inputs live on EFS and the CTCF load is
~30s. It **skips** when they are absent so an ordinary ``make test`` still
works off-EFS, and ``make test-realdata`` refuses to skip — a silently skipped
regression net is how the 592-interval movement got missed in the first place.
"""

import importlib.util
import os
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "test" / "fixtures" / "interval_manifest.tsv"
CAPTURE = REPO / "scripts" / "capture_interval_fixtures.py"

# The real inputs. Checked by path rather than by trying and catching, because
# /efs is slow and a failed open can block for a long time.
EFS_INPUTS = [
    Path("/efs/analytics/nathanboley/data_resources/genome/hg38-blacklist.v2.bed.gz"),
    Path("/efs/analytics/lily/ssDNA/top_1000_TF/CTCF.hg38.bed"),
]


def _load_capture_module():
    """Import the capture script by path — ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("_capture_fixtures", CAPTURE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _missing_inputs():
    return [str(p) for p in EFS_INPUTS if not p.exists()]


# `--realdata` is supplied by `make test-realdata`; without it, absent inputs
# skip. With it, absent inputs are an error, because the whole point of that
# target is that it cannot silently pass.
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
class TestManifestStillDescribesReality:
    def test_manifest_file_is_committed_and_parsable(self):
        assert MANIFEST.exists(), f"{MANIFEST} is missing"
        df = pd.read_csv(MANIFEST, sep="\t")
        assert list(df.columns) == ["op", "dataset", "n", "sha256_16", "note"]
        assert len(df) > 0

    def test_regenerated_manifest_matches_the_committed_one(self, request):
        """The actual regression net.

        Compares op/dataset/n/sha256_16. The ``note`` column is excluded: it
        carries input paths, which legitimately differ by machine and say
        nothing about whether the computation changed.
        """
        _require_real_data(request)

        cap = _load_capture_module()
        datasets = cap.build_inputs()
        regenerated = pd.concat(
            [pd.DataFrame(cap.input_provenance(datasets)), cap.capture(datasets)],
            ignore_index=True,
        )

        committed = pd.read_csv(MANIFEST, sep="\t")
        keys = ["op", "dataset", "n", "sha256_16"]

        got = regenerated[keys].sort_values(keys).reset_index(drop=True)
        want = committed[keys].sort_values(keys).reset_index(drop=True)

        if not got.equals(want):
            # Report the specific rows, not just "frames differ" -- the whole
            # value of this test is naming which operation moved.
            merged = want.merge(
                got, on=["op", "dataset"], how="outer", suffixes=("_committed", "_now")
            )
            moved = merged[
                (merged["sha256_16_committed"] != merged["sha256_16_now"])
                | (merged["n_committed"] != merged["n_now"])
            ]
            pytest.fail(
                "The committed manifest no longer describes real-data output.\n"
                "If this change is intended, regenerate with\n"
                "  python scripts/capture_interval_fixtures.py "
                "--out test/fixtures/interval_manifest.tsv\n"
                "and record WHY in the fixture-movements table of "
                "docs/pending/interval_api_design.md.\n"
                "A movement recorded without investigation is how a "
                "592-interval regression got rationalised once already.\n\n"
                + moved.to_string(index=False)
            )


@pytest.mark.requires_real_data
class TestCrossProcessDeterminism:
    """Pair ordering must be stable across processes, not just within one.

    This is checked here rather than in the synthetic suite because that is
    where it failed: identical code on identical input produced a different
    digest in every process, while pinning PYTHONHASHSEED made them agree.
    An in-process test cannot see it.
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
            "d = m.build_inputs()\n"
            "print(m.capture(d)[['op','dataset','n','sha256_16']].to_csv(index=False))\n"
        ) % (str(REPO), str(CAPTURE))

        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO)
        # Deliberately NOT pinning PYTHONHASHSEED -- that would mask exactly
        # the instability this test exists to catch.
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
            "Real-data digests differ between two processes on identical input. "
            "Something iterates a set or dict ordered by hash randomisation. "
            "Check that pair results are sorted before being returned."
        )
