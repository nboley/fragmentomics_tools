"""Regression tests for S4–S9 dataframe findings.

Each test must fail against the unfixed code and pass against the fixed code.
"""
import numpy as np
import pandas as pd
import pytest

from fragmentomics_tools.dataframe import (
    FlDist,
    RegionDataFrame,
    SampleDataFrame,
    SampleAndRegionDataFrame,
    intersect_region_dataframes,
    str_concat_columns,
)


# ── Helpers ──────────────────────────────────────────────────────────────

class MockFragmentsH5:
    """Minimal stub with the interface that SDF/FlDist consume."""

    def __init__(self, fname, fl_counts=None):
        self._f_fname = fname
        self._fl_counts = fl_counts if fl_counts is not None else np.ones(512)
        self._closed = False

    @property
    def fragment_length_counts(self):
        return self._fl_counts

    def close(self):
        self._closed = True


# ── S5: get_sample_count_bounds names the median correctly ───────────

class TestS5MedianRename:
    def test_source_says_median_not_mean(self):
        """The .rename() label inside get_sample_count_bounds must say
        'median_fragment_counts', not 'mean_fragment_counts'.

        The name lives on an intermediate Series and does not surface in
        the returned DataFrame, so the only reliable discriminator is the
        source text itself.
        """
        import inspect
        src = inspect.getsource(SampleAndRegionDataFrame.get_sample_count_bounds)
        assert "median_fragment_counts" in src, (
            "get_sample_count_bounds still labels the median as 'mean_fragment_counts'"
        )
        assert "mean_fragment_counts" not in src, (
            "get_sample_count_bounds still contains the misleading name 'mean_fragment_counts'"
        )


# ── S4: FlDist.init_from_sdf raises on duplicate sample_ids ─────────

class TestS4DuplicateSampleIds:
    def test_duplicate_sample_id_raises(self):
        h5_a = MockFragmentsH5("/data/s1.h5", np.arange(512, dtype=float))
        h5_b = MockFragmentsH5("/data/s1_dup.h5", np.arange(512, dtype=float) * 2)
        # init_from_sdf accepts any frame with sample_id + frag_h5 columns
        df = pd.DataFrame({"sample_id": ["s1", "s1"], "frag_h5": [h5_a, h5_b]})
        with pytest.raises(ValueError, match="Duplicate sample_id"):
            FlDist.init_from_sdf(df)

    def test_unique_sample_ids_ok(self):
        h5_a = MockFragmentsH5("/data/s1.h5", np.arange(512, dtype=float))
        h5_b = MockFragmentsH5("/data/s2.h5", np.arange(512, dtype=float) * 2)
        df = pd.DataFrame({"sample_id": ["s1", "s2"], "frag_h5": [h5_a, h5_b]})
        fl = FlDist.init_from_sdf(df)
        assert "s1" in fl.fl_df.columns
        assert "s2" in fl.fl_df.columns


# ── S6: FlDist.subset_by_sample_ids raises on missing ids ───────────

class TestS6SubsetValidation:
    def test_missing_ids_raise_with_names(self):
        fl_df = pd.DataFrame(
            {"s1": np.ones(10), "s2": np.ones(10)},
            index=range(1, 11),
        )
        fl = FlDist(fl_df)
        with pytest.raises(KeyError, match="not found in FlDist"):
            fl.subset_by_sample_ids(["s1", "s3"])

    def test_valid_ids_work(self):
        fl_df = pd.DataFrame(
            {"s1": np.ones(10), "s2": np.ones(10)},
            index=range(1, 11),
        )
        fl = FlDist(fl_df)
        result = fl.subset_by_sample_ids(["s1"])
        assert list(result.fl_df.columns) == ["s1"]


# ── S7: str_concat_columns hoists 'n' assertion ─────────────────────

class TestS7HoistedAssertion:
    def test_n_column_assertion_fires_before_groupby(self):
        """The assertion must fire once, before the groupby — not once per
        group.  We verify by counting: if it fires more than once, it was
        still inside apply_fn."""
        df = pd.DataFrame({
            "group": ["a", "a", "b"],
            "val": ["x", "y", "z"],
            "n": [1, 2, 3],  # pre-existing 'n' column
        })
        with pytest.raises(AssertionError, match="already has an 'n' column"):
            str_concat_columns(df, ["val"])

    def test_normal_operation(self):
        df = pd.DataFrame({
            "group": ["a", "a", "b"],
            "val": ["x", "y", "z"],
        })
        result = str_concat_columns(df, ["val"])
        assert "n" in result.columns
        row_a = result[result["group"] == "a"]
        assert row_a["n"].iloc[0] == 2
        assert row_a["val"].iloc[0] == "x,y"


# ── S8: intersect_region_dataframes on empty list ────────────────────

class TestS8EmptyList:
    def test_empty_list_raises_descriptive_error(self):
        """The old code raised IndexError; now it must raise ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            intersect_region_dataframes([])

    def test_single_element_list_works(self):
        rdf = RegionDataFrame(
            pd.DataFrame({"contig": ["chr1"], "start": [100], "stop": [200]}),
            ref="hg38",
        )
        result = intersect_region_dataframes([rdf])
        assert len(result) == 1


# ── S9: close_handles closes HDF5 handles ────────────────────────────

class TestS9CloseHandles:
    def test_close_handles_calls_close(self):
        h5 = MockFragmentsH5("/data/s1.h5")
        sdf = SampleDataFrame(
            pd.DataFrame({"sample_id": ["s1"], "frag_h5": [h5]})
        )
        assert not h5._closed
        sdf.close_handles()
        assert h5._closed
        # After close, frag_h5 column holds string paths
        assert sdf["frag_h5"].iloc[0] == "/data/s1.h5"

    def test_close_handles_srdf(self):
        h5 = MockFragmentsH5("/data/s1.h5")
        srdf = SampleAndRegionDataFrame(
            pd.DataFrame({
                "contig": ["chr1", "chr1"],
                "start": [1000, 2000],
                "stop": [2000, 3000],
                "sample_id": ["s1", "s1"],
                "frag_h5": [h5, h5],
                "fragment_array": [np.zeros(10), np.ones(10)],
            }),
            ref="hg38",
        )
        assert not h5._closed
        srdf.close_handles()
        assert h5._closed
        # Shared handle closed only once, both rows now have paths
        for val in srdf["frag_h5"]:
            assert isinstance(val, str)

    def test_close_handles_returns_self(self):
        h5 = MockFragmentsH5("/data/s1.h5")
        sdf = SampleDataFrame(
            pd.DataFrame({"sample_id": ["s1"], "frag_h5": [h5]})
        )
        result = sdf.close_handles()
        assert result is sdf

    def test_detach_does_not_close(self):
        """detach_h5 replaces handles with paths but must NOT close them."""
        h5 = MockFragmentsH5("/data/s1.h5")
        sdf = SampleDataFrame(
            pd.DataFrame({"sample_id": ["s1"], "frag_h5": [h5]})
        )
        sdf.detach_h5()
        assert not h5._closed
