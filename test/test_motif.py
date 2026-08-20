"""Tests for fragmentomics_tools.motif — get_pfms and TFConv1D.

Uses small fixture PFM files and the caller-supplied pfms= path
to avoid depending on the hardcoded analytics-host paths.
"""
from pathlib import Path

import numpy as np
import pytest

from fragmentomics_tools.public_data_resources.jaspar import (
    BindingModel,
    MissingPFMError,
    get_redundant_core_pfms,
)
from fragmentomics_tools.motif import get_pfms, TFConv1D

FIXTURE_DIR = Path(__file__).parent / "data"
JASPAR_FIXTURE = str(FIXTURE_DIR / "test_jaspar_core.txt")
HOCOMOCO_FIXTURE = str(FIXTURE_DIR / "test_hocomoco.txt")


class TestGetPfms:
    def test_with_caller_supplied_pfms(self):
        """When all_pfms is passed, no file I/O happens."""
        fixture_pfms = get_redundant_core_pfms(path=JASPAR_FIXTURE)
        result = get_pfms({"CTCF"}, all_pfms=fixture_pfms)
        assert len(result) == 1
        assert result[0].name == "CTCF"

    def test_returns_binding_models(self):
        fixture_pfms = get_redundant_core_pfms(path=JASPAR_FIXTURE)
        result = get_pfms({"CTCF", "RUNX1"}, all_pfms=fixture_pfms)
        for bm in result:
            assert isinstance(bm, BindingModel)

    def test_sorted_by_name(self):
        fixture_pfms = get_redundant_core_pfms(path=JASPAR_FIXTURE)
        result = get_pfms({"TFAP2A", "CTCF", "RUNX1"}, all_pfms=fixture_pfms)
        names = [bm.name for bm in result]
        assert names == sorted(names)

    def test_missing_tf_raises(self):
        """When a TF is absent from both primary and fallback, MissingPFMError is raised."""
        fixture_pfms = get_redundant_core_pfms(path=JASPAR_FIXTURE)
        with pytest.raises(MissingPFMError, match="NONEXISTENT"):
            get_pfms({"NONEXISTENT"}, all_pfms=fixture_pfms)

    def test_missing_tf_raises_when_fallback_unavailable(self, monkeypatch):
        """In containers the fallback DB file doesn't exist — get_pfms must
        catch FileNotFoundError and raise MissingPFMError, not propagate
        the FileNotFoundError."""
        from fragmentomics_tools.public_data_resources import jaspar as jaspar_mod
        fixture_pfms = get_redundant_core_pfms(path=JASPAR_FIXTURE)
        # Make the fallback DB unreachable (simulates container)
        monkeypatch.setattr(jaspar_mod, "DEFAULT_JASPAR_CORE_PATH", "/nonexistent/jaspar.txt")
        monkeypatch.setattr(jaspar_mod, "DEFAULT_HOCOMOCO_PATH", "/nonexistent/hocomoco.txt")
        with pytest.raises(MissingPFMError, match="NONEXISTENT"):
            get_pfms({"NONEXISTENT"}, all_pfms=fixture_pfms)


class TestTFConv1D:
    def test_with_pfms_arg(self):
        """TFConv1D(pfms=...) bypasses file-based lookup entirely."""
        fixture_pfms = get_redundant_core_pfms(path=JASPAR_FIXTURE, calculate_logo=True)
        bms = [
            BindingModel(p["pfm"], id=p["pfm_id"], name=p["pfm_name"])
            for p in fixture_pfms
            if p["pfm_name"] == "CTCF"
        ]
        model = TFConv1D(pfms=bms, tf_width=11)
        assert model.tf_names == ["CTCF"]

    def test_with_pfms_no_file_access(self):
        """When pfms= is supplied, the hardcoded default paths are never touched."""
        # Create a BindingModel from raw data — no file I/O
        pfm_data = np.ones((11, 4))
        bm = BindingModel(pfm_data, id="TEST.1", name="FAKE_TF")
        model = TFConv1D(pfms=[bm], tf_width=11)
        assert len(model.pfms) == 1
        assert model.pfms[0].name == "FAKE_TF"

    def test_empty_pfms_raises(self):
        with pytest.raises(ValueError, match="at least one PFM"):
            TFConv1D(pfms=[], tf_width=11)

    def test_neither_pfms_nor_tfs_raises(self):
        with pytest.raises(ValueError, match="Either"):
            TFConv1D(tf_width=11)
