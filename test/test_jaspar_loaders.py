"""Tests for PFM loaders in fragmentomics_tools.public_data_resources.jaspar.

Uses small fixture PFM files checked in under test/data/.
"""
from pathlib import Path

import numpy as np
import pytest

from fragmentomics_tools.public_data_resources.jaspar import (
    DEFAULT_JASPAR_CORE_PATH,
    DEFAULT_HOCOMOCO_PATH,
    get_redundant_core_pfms,
    get_all_human_hocomoco11_pfms,
    get_all_manual_override_pfms,
    pfm_to_logo,
)

FIXTURE_DIR = Path(__file__).parent / "data"
JASPAR_FIXTURE = str(FIXTURE_DIR / "test_jaspar_core.txt")
HOCOMOCO_FIXTURE = str(FIXTURE_DIR / "test_hocomoco.txt")


class TestConstants:
    def test_jaspar_constant_is_string(self):
        assert isinstance(DEFAULT_JASPAR_CORE_PATH, str)

    def test_hocomoco_constant_is_string(self):
        assert isinstance(DEFAULT_HOCOMOCO_PATH, str)


class TestGetRedundantCorePfms:
    def test_loads_from_fixture(self):
        pfms = get_redundant_core_pfms(path=JASPAR_FIXTURE)
        assert len(pfms) == 3

    def test_pfm_names_uppercased(self):
        pfms = get_redundant_core_pfms(path=JASPAR_FIXTURE)
        names = {p["pfm_name"] for p in pfms}
        assert names == {"RUNX1", "TFAP2A", "CTCF"}

    def test_pfm_ids(self):
        pfms = get_redundant_core_pfms(path=JASPAR_FIXTURE)
        ids = {p["pfm_id"] for p in pfms}
        assert ids == {"MA0002.1", "MA0003.1", "MA0139.1"}

    def test_pfm_shapes(self):
        pfms = get_redundant_core_pfms(path=JASPAR_FIXTURE)
        for p in pfms:
            assert p["pfm"].ndim == 2
            assert p["pfm"].shape[1] == 4  # A, C, G, T

    def test_ctcf_motif_width(self):
        pfms = get_redundant_core_pfms(path=JASPAR_FIXTURE)
        ctcf = [p for p in pfms if p["pfm_name"] == "CTCF"][0]
        assert ctcf["pfm"].shape[0] == 19

    def test_calculate_logo(self):
        pfms = get_redundant_core_pfms(path=JASPAR_FIXTURE, calculate_logo=True)
        for p in pfms:
            assert "logo" in p
            assert p["logo"].shape == p["pfm"].shape

    def test_no_logo_by_default(self):
        pfms = get_redundant_core_pfms(path=JASPAR_FIXTURE)
        for p in pfms:
            assert "logo" not in p

    def test_missing_path_raises(self):
        with pytest.raises(FileNotFoundError):
            get_redundant_core_pfms(path="/nonexistent/path.txt")


class TestGetAllHumanHocomoco11Pfms:
    def test_loads_from_fixture(self):
        pfms = get_all_human_hocomoco11_pfms(path=HOCOMOCO_FIXTURE)
        assert len(pfms) == 3

    def test_pfm_names(self):
        pfms = get_all_human_hocomoco11_pfms(path=HOCOMOCO_FIXTURE)
        names = {p["pfm_name"] for p in pfms}
        assert names == {"CTCF", "FOXA1", "GATA3"}

    def test_pfm_ids(self):
        pfms = get_all_human_hocomoco11_pfms(path=HOCOMOCO_FIXTURE)
        ids = {p["pfm_id"] for p in pfms}
        # All three fixture entries share the same ID suffix
        assert ids == {"HUMAN.H11MO.0.A"}

    def test_pfm_shapes(self):
        pfms = get_all_human_hocomoco11_pfms(path=HOCOMOCO_FIXTURE)
        for p in pfms:
            assert p["pfm"].ndim == 2
            assert p["pfm"].shape[1] == 4

    def test_calculate_logo(self):
        pfms = get_all_human_hocomoco11_pfms(path=HOCOMOCO_FIXTURE, calculate_logo=True)
        for p in pfms:
            assert "logo" in p

    def test_missing_path_raises(self):
        with pytest.raises(FileNotFoundError):
            get_all_human_hocomoco11_pfms(path="/nonexistent/path.txt")


class TestGetAllManualOverridePfms:
    def test_returns_nonempty(self):
        pfms = get_all_manual_override_pfms()
        assert len(pfms) > 0

    def test_ctcf_present(self):
        pfms = get_all_manual_override_pfms()
        names = {p["pfm_name"] for p in pfms}
        assert "CTCF" in names

    def test_calculate_logo(self):
        pfms = get_all_manual_override_pfms(calculate_logo=True)
        for p in pfms:
            assert "logo" in p


class TestPfmToLogo:
    def test_output_shape(self):
        pfm = np.array([[100, 0, 0, 0], [0, 100, 0, 0], [0, 0, 100, 0]])
        logo = pfm_to_logo(pfm)
        assert logo.shape == pfm.shape

    def test_high_info_position(self):
        # A position with all counts in one base should have high information
        pfm = np.array([[1000, 0, 0, 0]])
        logo = pfm_to_logo(pfm)
        assert logo[0, 0] > 1.5  # close to 2 bits


class TestOffByOneRegression:
    """Regression tests for the parser off-by-one fix.

    The bug was `while i < len(rows) - 5` which skips the last motif when
    the file has no trailing blank line (exactly N*5 lines). The fix is
    `while i + 4 < len(rows)`.
    """

    def test_jaspar_no_trailing_newline(self, tmp_path):
        """JASPAR file with exactly 10 lines (2 motifs, no trailing blank)."""
        content = (
            ">MA0002.1\tRUNX1\n"
            "A  [    10     12      4 ]\n"
            "C  [     2      2      7 ]\n"
            "G  [     3      1      1 ]\n"
            "T  [    11     11     14 ]\n"
            ">MA0003.1\tTFAP2A\n"
            "A  [     0      0      0 ]\n"
            "C  [     0    185    185 ]\n"
            "G  [   185      0      0 ]\n"
            "T  [     0      0      0 ]\n"
        )
        f = tmp_path / "no_trailing.txt"
        f.write_text(content)
        pfms = get_redundant_core_pfms(path=str(f))
        assert len(pfms) == 2, f"Expected 2 motifs, got {len(pfms)} (last motif dropped?)"

    def test_hocomoco_no_trailing_newline(self, tmp_path):
        """HOCOMOCO file with exactly 10 lines (2 motifs, no trailing blank)."""
        content = (
            ">CTCF_HUMAN.H11MO.0.A\n"
            "87\t167\t281\n"
            "291\t145\t49\n"
            "76\t414\t449\n"
            "459\t187\t134\n"
            ">FOXA1_HUMAN.H11MO.0.A\n"
            "8\t0\t0\n"
            "2\t0\t28\n"
            "4\t399\t4\n"
            "232\t10\t8\n"
        )
        f = tmp_path / "no_trailing.txt"
        f.write_text(content)
        pfms = get_all_human_hocomoco11_pfms(path=str(f))
        assert len(pfms) == 2, f"Expected 2 motifs, got {len(pfms)} (last motif dropped?)"
