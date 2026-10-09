"""Test provenance guards: dirty-script prevention AND inherit-vs-rederive.

The dirty-script guard lives in
``scripts.build_hexamer_prior.verify_script_provenance`` and guarantees that
every ``commit_sha`` + ``script_sha256`` recorded in an artifact can be
resolved by checking out that commit.  It scopes to the script file itself —
unrelated dirty files elsewhere in the worktree are tolerated, which is
necessary because this worktree is shared with another active stream.

The inherit-vs-rederive guard (``TestInheritedProvenance``) verifies that a
downstream consumer copies its input's recorded provenance forward verbatim
and adds only its own step's sha.  It must never state a sha for a step it
did not run and cannot witness.
"""
import hashlib
import os
import subprocess
import tempfile

import pytest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_hexamer_prior import verify_script_provenance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _head_sha(cwd: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL,
    ).decode().strip()


def _script_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------------------
# Tests against the REAL repo (fast, no setup)
# ---------------------------------------------------------------------------

class TestVerifyScriptProvenance:
    """Test verify_script_provenance against the live worktree."""

    # The build script itself is committed and clean — the guard should pass.
    SCRIPT = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "build_hexamer_prior.py",
    )

    def test_clean_script_passes(self):
        """A committed, unmodified script passes the guard."""
        commit_sha, script_sha = verify_script_provenance(self.SCRIPT)
        # commit_sha must be a 40-char hex string
        assert len(commit_sha) == 40
        assert all(c in "0123456789abcdef" for c in commit_sha)
        # script_sha must equal the on-disk sha256
        assert script_sha == _script_sha256(self.SCRIPT)

    def test_returned_sha_matches_committed_blob(self):
        """The returned script_sha matches ``git show HEAD:<script>``."""
        commit_sha, script_sha = verify_script_provenance(self.SCRIPT)
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=os.path.dirname(self.SCRIPT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        rel = os.path.relpath(self.SCRIPT, repo_root)
        blob = subprocess.check_output(
            ["git", "show", f"HEAD:{rel}"],
            cwd=repo_root, stderr=subprocess.DEVNULL,
        )
        assert script_sha == hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Tests using a throwaway git repo (slower, but tests failure paths)
# ---------------------------------------------------------------------------

class TestVerifyScriptProvenanceIsolated:
    """Failure-path tests using a temporary git repo."""

    @pytest.fixture(autouse=True)
    def _setup_temp_repo(self, tmp_path):
        """Create a tiny git repo with one committed script."""
        self.repo = str(tmp_path / "repo")
        os.makedirs(self.repo)
        subprocess.run(["git", "init"], cwd=self.repo,
                        capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "test@test"],
                        cwd=self.repo, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "test"],
                        cwd=self.repo, capture_output=True, check=True)

        self.script = os.path.join(self.repo, "my_script.py")
        with open(self.script, "w") as f:
            f.write("print('hello')\n")

        subprocess.run(["git", "add", "my_script.py"], cwd=self.repo,
                        capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=self.repo,
                        capture_output=True, check=True)

    def test_clean_passes(self):
        commit_sha, sha = verify_script_provenance(self.script)
        assert len(commit_sha) == 40
        assert sha == _script_sha256(self.script)

    def test_dirty_script_raises(self):
        # Modify the committed script without committing.
        with open(self.script, "a") as f:
            f.write("print('dirty')\n")
        with pytest.raises(RuntimeError, match="differs from HEAD"):
            verify_script_provenance(self.script)

    def test_untracked_script_raises(self):
        new_script = os.path.join(self.repo, "new_script.py")
        with open(new_script, "w") as f:
            f.write("print('new')\n")
        with pytest.raises(RuntimeError, match="not tracked"):
            verify_script_provenance(new_script)

    def test_staged_but_uncommitted_raises(self):
        # Modify and stage, but do not commit.
        with open(self.script, "a") as f:
            f.write("print('staged')\n")
        subprocess.run(["git", "add", "my_script.py"], cwd=self.repo,
                        capture_output=True, check=True)
        with pytest.raises(RuntimeError, match="differs from HEAD"):
            verify_script_provenance(self.script)

    def test_unrelated_dirty_file_tolerated(self):
        # Create an unrelated dirty file — guard should still pass for
        # the committed script.
        other = os.path.join(self.repo, "unrelated.txt")
        with open(other, "w") as f:
            f.write("this is dirty\n")
        # Guard passes because the SCRIPT is clean.
        commit_sha, sha = verify_script_provenance(self.script)
        assert sha == _script_sha256(self.script)


# ---------------------------------------------------------------------------
# Inherit-vs-rederive rule
# ---------------------------------------------------------------------------

class TestInheritedProvenance:
    """Verify that _extract_artifact_provenance inherits upstream provenance
    from the artifact rather than re-deriving it from the current tree.

    The rule: a consumer copies its input's recorded provenance forward
    verbatim and adds only its own step's sha.  Re-reading HEAD for an
    upstream step is a guess, and it is wrong precisely when someone has
    since edited that upstream script.
    """

    def test_old_format_scalar_sha(self):
        """Old-format artifacts (scalar ``script_sha256``) are normalised."""
        from scripts.measure_hexamer_disattenuation import _extract_artifact_provenance

        artifact = {
            "provenance": {
                "commit_sha": "abc123def456",
                "script_sha256": "deadbeef0123",
            }
        }
        prov = _extract_artifact_provenance(artifact)
        assert prov["commit_sha"] == "abc123def456"
        assert prov["script_shas"] == {
            "scripts/build_hexamer_prior.py": "deadbeef0123",
        }

    def test_new_format_dict_shas(self):
        """New-format artifacts (dict ``script_shas``) are passed through."""
        from scripts.measure_hexamer_disattenuation import _extract_artifact_provenance

        artifact = {
            "provenance": {
                "commit_sha": "abc123def456",
                "script_shas": {
                    "scripts/build_hexamer_prior.py": "sha_build",
                    "scripts/count_cut_site_hexamers.py": "sha_count",
                },
            }
        }
        prov = _extract_artifact_provenance(artifact)
        assert prov["commit_sha"] == "abc123def456"
        assert prov["script_shas"]["scripts/build_hexamer_prior.py"] == "sha_build"
        assert prov["script_shas"]["scripts/count_cut_site_hexamers.py"] == "sha_count"

    def test_missing_commit_sha_raises(self):
        """Artifacts without ``provenance.commit_sha`` raise ValueError."""
        from scripts.measure_hexamer_disattenuation import _extract_artifact_provenance

        with pytest.raises(ValueError, match="commit_sha"):
            _extract_artifact_provenance({"provenance": {}})

    def test_missing_script_sha_raises(self):
        """Artifacts with commit but no script sha raise ValueError."""
        from scripts.measure_hexamer_disattenuation import _extract_artifact_provenance

        with pytest.raises(ValueError, match="script_sha"):
            _extract_artifact_provenance({"provenance": {"commit_sha": "abc"}})

    def test_extraction_uses_artifact_values_not_git(self):
        """Extraction reads the artifact dict — no git calls.

        A fake commit/sha that does not exist in any repo must pass through
        without error.  If someone refactors to call git (re-derive), this
        test will fail because the fake values won't resolve.
        """
        from scripts.measure_hexamer_disattenuation import _extract_artifact_provenance

        artifact = {
            "provenance": {
                "commit_sha": "0000000000000000000000000000000000000000",
                "script_sha256": "ffffffffffffffffffffffffffffffff",
            }
        }
        prov = _extract_artifact_provenance(artifact)
        assert prov["commit_sha"] == "0000000000000000000000000000000000000000"
        assert prov["script_shas"]["scripts/build_hexamer_prior.py"] == (
            "ffffffffffffffffffffffffffffffff"
        )
