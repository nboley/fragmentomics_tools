import pickle
import numpy as np
import pandas as pd
import pytest

from fragmentomics_tools.dataframe import (
    SampleAndRegionDataFrame,
    SampleDataFrame,
    FlDist,
)


class MockFragmentsH5:
    """Minimal stub with the interface that SDF/SRDF consume."""

    def __init__(self, fname, fl_counts=None):
        self._f_fname = fname
        self._fl_counts = fl_counts if fl_counts is not None else np.ones(512)

    @property
    def fragment_length_counts(self):
        return self._fl_counts


@pytest.fixture
def mock_h5():
    return MockFragmentsH5("/data/sample1.fragments.h5", np.arange(512, dtype=float))


@pytest.fixture
def mock_h5_b():
    return MockFragmentsH5("/data/sample2.fragments.h5", np.arange(512, dtype=float) * 2)


@pytest.fixture
def sdf(mock_h5):
    return SampleDataFrame(
        pd.DataFrame({"sample_id": ["s1"], "frag_h5": [mock_h5]})
    )


@pytest.fixture
def sdf_two(mock_h5, mock_h5_b):
    return SampleDataFrame(
        pd.DataFrame({"sample_id": ["s1", "s2"], "frag_h5": [mock_h5, mock_h5_b]})
    )


@pytest.fixture
def srdf(mock_h5):
    return SampleAndRegionDataFrame(
        pd.DataFrame(
            {
                "contig": ["chr1", "chr1"],
                "start": [1000, 2000],
                "stop": [2000, 3000],
                "sample_id": ["s1", "s1"],
                "frag_h5": [mock_h5, mock_h5],
                "fragment_array": [np.zeros(100), np.ones(100)],
            }
        ),
        ref="hg38",
    )


# --- SampleDataFrame tests ---


class TestSampleDataFrameDetach:
    def test_fl_dist_built_with_live_handle(self, sdf):
        """FlDist is built when frag_h5 contains live handles."""
        assert sdf._fl_dist is not None
        # Should not raise
        _ = sdf.fl_dist

    def test_detach_replaces_handle_with_path(self, sdf, mock_h5):
        sdf.detach_h5()
        assert sdf["frag_h5"].iloc[0] == mock_h5._f_fname
        assert isinstance(sdf["frag_h5"].iloc[0], str)

    def test_fl_dist_survives_detach(self, sdf):
        """fl_dist was built before detach → still accessible after."""
        fl_before = sdf.fl_dist
        sdf.detach_h5()
        fl_after = sdf.fl_dist
        assert (fl_before.fl_df.values == fl_after.fl_df.values).all()

    def test_pickle_roundtrip_after_detach(self, sdf):
        sdf.detach_h5()
        data = pickle.dumps(sdf)
        sdf2 = pickle.loads(data)
        assert isinstance(sdf2, SampleDataFrame)
        assert list(sdf2["sample_id"]) == ["s1"]
        assert sdf2["frag_h5"].iloc[0] == "/data/sample1.fragments.h5"

    def test_pickle_roundtrip_preserves_fl_dist(self, sdf):
        fl_before = sdf.fl_dist
        sdf.detach_h5()
        sdf2 = pickle.loads(pickle.dumps(sdf))
        fl_after = sdf2.fl_dist
        assert (fl_before.fl_df.values == fl_after.fl_df.values).all()

    def test_fl_dist_none_when_paths_only(self):
        """SDF constructed with string paths → _fl_dist is None."""
        sdf = SampleDataFrame(
            pd.DataFrame({"sample_id": ["s1"], "frag_h5": ["/some/path.h5"]})
        )
        assert sdf._fl_dist is None

    def test_fl_dist_raises_when_none(self):
        """Accessing fl_dist when _fl_dist is None raises RuntimeError."""
        sdf = SampleDataFrame(
            pd.DataFrame({"sample_id": ["s1"], "frag_h5": ["/some/path.h5"]})
        )
        with pytest.raises(RuntimeError, match="fl_dist unavailable"):
            _ = sdf.fl_dist

    def test_detach_idempotent(self, sdf, mock_h5):
        sdf.detach_h5()
        path = sdf["frag_h5"].iloc[0]
        sdf.detach_h5()  # second call should be safe
        assert sdf["frag_h5"].iloc[0] == path

    def test_two_samples_fl_dist(self, sdf_two):
        """FlDist is built correctly with multiple samples."""
        assert sdf_two._fl_dist is not None
        fl = sdf_two.fl_dist
        assert "s1" in fl.fl_df.columns
        assert "s2" in fl.fl_df.columns


# --- SampleAndRegionDataFrame tests ---


class TestSRDFDetach:
    def test_detach_replaces_handles_with_paths(self, srdf, mock_h5):
        srdf.detach_h5()
        for val in srdf["frag_h5"]:
            assert isinstance(val, str)
            assert val == mock_h5._f_fname

    def test_fragment_array_survives_detach(self, srdf):
        fa_before = list(srdf["fragment_array"])
        srdf.detach_h5()
        fa_after = list(srdf["fragment_array"])
        for a, b in zip(fa_before, fa_after):
            assert (a == b).all()

    def test_pickle_roundtrip_after_detach(self, srdf):
        srdf.detach_h5()
        data = pickle.dumps(srdf)
        srdf2 = pickle.loads(data)
        assert isinstance(srdf2, SampleAndRegionDataFrame)
        assert list(srdf2["sample_id"]) == ["s1", "s1"]
        assert list(srdf2["contig"]) == ["chr1", "chr1"]
        assert list(srdf2["start"]) == [1000, 2000]

    def test_pickle_roundtrip_preserves_fragment_array(self, srdf):
        fa_before = [x.copy() for x in srdf["fragment_array"]]
        srdf.detach_h5()
        srdf2 = pickle.loads(pickle.dumps(srdf))
        fa_after = list(srdf2["fragment_array"])
        for a, b in zip(fa_before, fa_after):
            assert (a == b).all()

    def test_pickle_roundtrip_preserves_ref(self, srdf):
        srdf.detach_h5()
        srdf2 = pickle.loads(pickle.dumps(srdf))
        assert srdf2.ref == "hg38"

    def test_detach_returns_self(self, srdf):
        result = srdf.detach_h5()
        assert result is srdf

    def test_detach_idempotent(self, srdf, mock_h5):
        srdf.detach_h5()
        path = srdf["frag_h5"].iloc[0]
        srdf.detach_h5()
        assert srdf["frag_h5"].iloc[0] == path

    def test_srdf_from_path_strings(self):
        """SRDF can be constructed with path strings in frag_h5."""
        srdf = SampleAndRegionDataFrame(
            pd.DataFrame(
                {
                    "contig": ["chr1"],
                    "start": [1000],
                    "stop": [2000],
                    "sample_id": ["s1"],
                    "frag_h5": ["/some/path.h5"],
                    "fragment_array": [np.zeros(100)],
                }
            ),
            ref="hg38",
        )
        assert srdf["frag_h5"].iloc[0] == "/some/path.h5"
