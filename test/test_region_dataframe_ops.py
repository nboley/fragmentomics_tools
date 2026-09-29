"""Regression tests for RegionDataFrame operations that failed silently.

Both bugs covered here returned plausible-looking results rather than raising,
so nothing downstream could have noticed. They use synthetic in-memory data on
purpose, so they do not depend on any external fixture.
"""

import os
import tempfile

import pandas as pd
import pytest

from fragmentomics_tools.dataframe import RegionDataFrame
from fragmentomics_tools.intervals import merge, overlap_indices, overlaps


@pytest.fixture
def blacklist_bed():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "blacklist.bed")
        with open(path, "w") as fh:
            fh.write("chr1\t150\t180\n")
            fh.write("chr1\t500\t520\n")
        yield path


class TestMerge:
    """Test the new intervals.merge function."""

    def test_merge_does_not_produce_all_nan_columns(self):
        rdf = RegionDataFrame(
            pd.DataFrame(
                {
                    "contig": ["chr1", "chr1", "chr2"],
                    "start": [100, 150, 500],
                    "stop": [200, 300, 600],
                    "strand": ["+", "+", "-"],
                    "name": ["a", "b", "c"],
                    "score": [1.5, 2.5, 3.5],
                }
            ),
            ref="hg38",
        )
        merged = merge(rdf)
        all_nan = [c for c in merged.columns if merged[c].isna().all()]
        assert all_nan == [], f"columns became all-NaN after merge: {all_nan}"

    def test_merge_collapses_overlapping_regions(self):
        rdf = RegionDataFrame(
            pd.DataFrame(
                {
                    "contig": ["chr1", "chr1", "chr2"],
                    "start": [100, 150, 500],
                    "stop": [200, 300, 600],
                }
            ),
            ref="hg38",
        )
        merged = merge(rdf)

        # chr1:100-200 and chr1:150-300 overlap and collapse into one interval
        assert len(merged) == 2
        chr1 = merged[merged.contig == "chr1"]
        assert len(chr1) == 1
        assert int(chr1.iloc[0]["start"]) == 100
        assert int(chr1.iloc[0]["stop"]) == 300


class TestSplitOnColumn:
    def test_splits_on_the_named_column(self):
        rdf = RegionDataFrame(
            pd.DataFrame(
                {
                    "contig": ["chr1", "chr2", "chr3"],
                    "start": [100, 200, 300],
                    "stop": [150, 250, 350],
                    "group": ["a", "b", "c"],
                }
            ),
            ref="hg38",
        )
        g1, g2 = rdf.split_on_column("group", [["a"], ["b", "c"]])

        assert sorted(g1["group"]) == ["a"]
        assert sorted(g2["group"]) == ["b", "c"]


class TestAttachBlacklistRegions:
    def test_attaches_blacklist_coordinates_not_query_coordinates(
        self, blacklist_bed
    ):
        rdf = RegionDataFrame(
            pd.DataFrame({"contig": ["chr1"], "start": [100], "stop": [200]}),
            ref="hg38",
        )
        out = rdf.attach_blacklist_regions(blacklist_bed)

        attached = list(out["blacklist_regions"])[0]
        assert len(attached) == 1
        region = attached[0]
        assert (region.start, region.stop) == (150, 180), (
            f"expected the blacklist interval chr1:150-180, got "
            f"chr1:{region.start}-{region.stop}"
        )

    def test_non_overlapping_region_gets_empty(self, blacklist_bed):
        rdf = RegionDataFrame(
            pd.DataFrame(
                {"contig": ["chr1", "chr1"], "start": [100, 900], "stop": [200, 950]}
            ),
            ref="hg38",
        )
        out = rdf.attach_blacklist_regions(blacklist_bed)

        by_start = {int(r["start"]): r["blacklist_regions"] for _, r in out.iterrows()}
        # chr1:100-200 overlaps chr1:150-180
        assert len(by_start[100]) == 1
        # chr1:900-950 overlaps nothing
        assert by_start[900] == ""

    def test_multiple_blacklist_hits_are_all_attached(self, blacklist_bed):
        rdf = RegionDataFrame(
            pd.DataFrame({"contig": ["chr1"], "start": [100], "stop": [600]}),
            ref="hg38",
        )
        out = rdf.attach_blacklist_regions(blacklist_bed)

        attached = list(out["blacklist_regions"])[0]
        spans = sorted((r.start, r.stop) for r in attached)
        assert spans == [(150, 180), (500, 520)]

    def test_does_not_mutate_self(self, blacklist_bed):
        """attach_blacklist_regions must always return a new frame."""
        rdf = RegionDataFrame(
            pd.DataFrame({"contig": ["chr1"], "start": [100], "stop": [200]}),
            ref="hg38",
        )
        assert "blacklist_regions" not in rdf.columns
        out = rdf.attach_blacklist_regions(blacklist_bed)
        assert "blacklist_regions" in out.columns
        assert "blacklist_regions" not in rdf.columns

    def test_empty_blacklist(self):
        """No overlaps: every row gets empty string."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "empty.bed")
            with open(path, "w") as fh:
                fh.write("chrX\t9000\t9999\n")
            rdf = RegionDataFrame(
                pd.DataFrame({"contig": ["chr1"], "start": [100], "stop": [200]}),
                ref="hg38",
            )
            out = rdf.attach_blacklist_regions(path)
            assert out["blacklist_regions"].iloc[0] == ""


class TestEqSemantics:
    """I10: __eq__ used to return a scalar bool, breaking the pandas contract
    for element-wise comparison and making instances unhashable."""

    def test_eq_returns_elementwise(self):
        rdf = RegionDataFrame(
            pd.DataFrame({"contig": ["chr1"], "start": [100], "stop": [200]}),
            ref="hg38",
        )
        result = rdf == rdf
        assert isinstance(result, pd.DataFrame), (
            f"expected DataFrame from ==, got {type(result).__name__}"
        )

    def test_equals_rdf_compares_regions(self):
        rdf1 = RegionDataFrame(
            pd.DataFrame({"contig": ["chr1"], "start": [100], "stop": [200]}),
            ref="hg38",
        )
        rdf2 = RegionDataFrame(
            pd.DataFrame({"contig": ["chr1"], "start": [100], "stop": [200]}),
            ref="hg38",
        )
        rdf3 = RegionDataFrame(
            pd.DataFrame({"contig": ["chr1"], "start": [100], "stop": [300]}),
            ref="hg38",
        )
        assert rdf1.equals_rdf(rdf2)
        assert not rdf1.equals_rdf(rdf3)


class TestCenterOnSummit:
    def test_summit_at_stop_is_rejected(self):
        rdf = RegionDataFrame(
            pd.DataFrame({
                "contig": ["chr1"], "start": [100], "stop": [200], "summit": [200]
            }),
            ref="hg38",
        )
        with pytest.raises(ValueError, match="summits must either be within"):
            rdf.center_on_summit()

    def test_summit_inside_region_works(self):
        rdf = RegionDataFrame(
            pd.DataFrame({
                "contig": ["chr1"], "start": [100], "stop": [200], "summit": [150]
            }),
            ref="hg38",
        )
        result = rdf.center_on_summit()
        assert "summit" not in result.columns
        assert int(result.stop.iloc[0]) - int(result.start.iloc[0]) == 100

    def test_does_not_mutate_self_by_default(self):
        rdf = RegionDataFrame(
            pd.DataFrame({
                "contig": ["chr1"], "start": [100], "stop": [200], "summit": [120]
            }),
            ref="hg38",
        )
        original_start = int(rdf.start.iloc[0])
        result = rdf.center_on_summit()
        assert int(rdf.start.iloc[0]) == original_start
        assert int(result.start.iloc[0]) != original_start


class TestOverlapsWithMissingContig:
    """Replacing the old TestOverlapsRdf with the new API."""

    def test_missing_contig_returns_false(self):
        rdf = RegionDataFrame(
            pd.DataFrame({
                "contig": ["chr1", "chr2"],
                "start": [100, 200],
                "stop": [200, 300],
            }),
            ref="hg38",
        )
        query = RegionDataFrame(
            pd.DataFrame({"contig": ["chr1"], "start": [150], "stop": [250]}),
            ref="hg38",
        )
        result = overlaps(rdf, query)
        assert list(result) == [True, False]


class TestAndAndConcat:
    def test_concat_empty_raises_valueerror(self):
        with pytest.raises(ValueError, match="at least one"):
            RegionDataFrame.concat([])

    def test_concat_preserves_data(self):
        rdf1 = RegionDataFrame(
            pd.DataFrame({"contig": ["chr1"], "start": [100], "stop": [200]}),
            ref="hg38",
        )
        rdf2 = RegionDataFrame(
            pd.DataFrame({"contig": ["chr2"], "start": [300], "stop": [400]}),
            ref="hg38",
        )
        result = RegionDataFrame.concat([rdf1, rdf2])
        assert len(result) == 2
        assert isinstance(result, RegionDataFrame)


class TestLiftOver:
    def test_empty_rdf_does_not_crash(self):
        rdf = RegionDataFrame(
            pd.DataFrame({"contig": [], "start": [], "stop": []}),
            ref="hg38",
        )
        result = rdf.lift_over("hg19")
        assert len(result) == 0
        assert isinstance(result, RegionDataFrame)
        assert result.ref == "hg19"

    def test_failed_liftover_uses_na_not_minus_one(self):
        rdf = RegionDataFrame(
            pd.DataFrame({
                "contig": ["chrX_FAKE"],
                "start": [100],
                "stop": [200],
                "strand": ["."],
            }),
            ref="hg38",
        )
        result = rdf.lift_over("hg19", remove_non_liftoverable_regions=False)
        assert pd.isna(result.start.iloc[0])


class TestFromBed:
    def test_empty_bed_returns_empty_rdf(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "empty.bed")
            with open(path, "w") as fh:
                pass
            rdf = RegionDataFrame.from_bed(path, ref="hg38")
            assert len(rdf) == 0
            assert isinstance(rdf, RegionDataFrame)


class TestUniqueRegions:
    def test_output_is_ascending_by_default(self):
        rdf = RegionDataFrame(
            pd.DataFrame({
                "contig": ["chr2", "chr1", "chr1"],
                "start": [200, 300, 100],
                "stop": [250, 350, 150],
            }),
            ref="hg38",
        )
        result = rdf.unique_regions()
        assert list(result.contig) == ["chr1", "chr1", "chr2"]
        assert list(result.start) == [100, 300, 200]


class TestAntiJoinReplacement:
    """Replaces TestDropOverlappingRegions — uses intervals.overlap_indices."""

    def test_works_with_non_hg38_reference(self):
        rdf = RegionDataFrame(
            pd.DataFrame({
                "contig": ["chr1", "chr1"],
                "start": [100, 500],
                "stop": [200, 600],
            }),
            ref="hg19",
        )
        blacklist = RegionDataFrame(
            pd.DataFrame({"contig": ["chr1"], "start": [150], "stop": [250]}),
            ref="hg19",
        )
        idx = overlap_indices(rdf, blacklist, how="anti")
        kept = rdf.iloc[idx["a_index"].values.astype(int)]
        assert len(kept) == 1
        assert int(kept.start.iloc[0]) == 500


class TestResizeBoundaryConditions:
    def test_discard_does_not_corrupt_self(self):
        rdf = RegionDataFrame(
            pd.DataFrame({
                "contig": ["chr1", "chr1"],
                "start": [10, 1000],
                "stop": [100, 2000],
            }),
            ref="hg38",
        )
        original_starts = list(rdf.start)
        rdf._resize_region_boundaries(
            left=-50, inplace=True, discard_invalid_resizes=True
        )
        assert list(rdf.start) == original_starts

    def test_no_warning_when_zero_discarded(self, caplog):
        import logging
        rdf = RegionDataFrame(
            pd.DataFrame({"contig": ["chr1"], "start": [1000], "stop": [2000]}),
            ref="hg38",
        )
        with caplog.at_level(logging.WARNING, logger="fragmentomics_tools.dataframe"):
            rdf.resize_regions(500, discard_invalid_resizes=True)
        discard_msgs = [r for r in caplog.records if "Discarded" in r.message]
        assert len(discard_msgs) == 0

    def test_truncate_beyond_region_length_raises(self):
        rdf = RegionDataFrame(
            pd.DataFrame({"contig": ["chr1"], "start": [1000], "stop": [1100]}),
            ref="hg38",
        )
        with pytest.raises(ValueError, match="truncation amounts exceed"):
            rdf.truncate_regions(left_amt=60, right_amt=60)


class TestBinRegionsIntoWindows:
    def test_short_region_valid_mode_gives_clear_error(self):
        rdf = RegionDataFrame(
            pd.DataFrame({"contig": ["chr1"], "start": [1000], "stop": [1020]}),
            ref="hg38",
        )
        with pytest.raises(ValueError, match="shorter than window_size"):
            rdf.bin_regions_into_windows(window_size=100, mode="valid", stride=50)

    @pytest.mark.parametrize("window_size,stride,expected_n", [
        (100, 100, 10),
        (200, 100, 9),
        (400, 100, 7),
        (1000, 100, 1),
    ])
    def test_valid_mode_windows_stay_within_region(
        self, window_size, stride, expected_n
    ):
        rdf = RegionDataFrame(
            pd.DataFrame({"contig": ["chr1"], "start": [1000], "stop": [2000]}),
            ref="hg38",
        )
        result = rdf.bin_regions_into_windows(
            window_size=window_size, mode="valid", stride=stride,
        )
        assert len(result) == expected_n
        assert result["start"].min() >= 1000
        assert result["stop"].max() <= 2000

    @pytest.mark.parametrize("region_len,window_size,expected", [
        (1000, 300, [(1050, 1350), (1350, 1650), (1650, 1950)]),
        (999, 100, [(1049, 1149), (1149, 1249), (1249, 1349), (1349, 1449),
                    (1449, 1549), (1549, 1649), (1649, 1749), (1749, 1849),
                    (1849, 1949)]),
        (1000, 250, [(1000, 1250), (1250, 1500), (1500, 1750), (1750, 2000)]),
    ])
    def test_valid_mode_default_stride_stays_centred(
        self, region_len, window_size, expected
    ):
        rdf = RegionDataFrame(
            pd.DataFrame(
                {"contig": ["chr1"], "start": [1000], "stop": [1000 + region_len]}
            ),
            ref="hg38",
        )
        result = rdf.bin_regions_into_windows(window_size=window_size, mode="valid")
        got = [(int(a), int(b)) for a, b in zip(result["start"], result["stop"])]
        assert got == expected
