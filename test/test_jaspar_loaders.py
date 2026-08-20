"""Tests for PFM loaders in fragmentomics_tools.public_data_resources.jaspar.

Uses small fixture PFM files checked in under test/data/.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from fragmentomics_tools.public_data_resources.jaspar import (
    DEFAULT_JASPAR_CORE_PATH,
    DEFAULT_HOCOMOCO_PATH,
    MissingPFMError,
    get_redundant_core_pfms,
    get_all_human_hocomoco11_pfms,
    get_all_manual_override_pfms,
    get_all_human_pfms,
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
        assert ids == {"HUMAN.H11MO.0.A", "HUMAN.H11MO.0.A", "HUMAN.H11MO.0.A"}

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


class TestDefaultPathNotRequired:
    """Verify that callers CAN pass a path, bypassing the default."""

    def test_jaspar_with_explicit_path(self):
        pfms = get_redundant_core_pfms(path=JASPAR_FIXTURE)
        assert len(pfms) == 3

    def test_hocomoco_with_explicit_path(self):
        pfms = get_all_human_hocomoco11_pfms(path=HOCOMOCO_FIXTURE)
        assert len(pfms) == 3
