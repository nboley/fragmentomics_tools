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


@pytest.fixture
def blacklist_bed():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "blacklist.bed")
        with open(path, "w") as fh:
            fh.write("chr1\t150\t180\n")
            fh.write("chr1\t500\t520\n")
        yield path


class TestMergeRegions:
    """bedtools merge emits BED3 unless -c/-o aggregation is requested."""

    def test_merge_does_not_produce_all_nan_columns(self):
        # Regression: merge_regions passed names=list(self.columns) to
        # to_dataframe(). bedtools merge only emits 3 columns, so every extra
        # name became an all-NaN column -- silent data corruption rather than
        # an error.
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
        merged = rdf.merge_regions()

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
        merged = rdf.merge_regions()

        # chr1:100-200 and chr1:150-300 overlap and collapse into one interval
        assert len(merged) == 2
        chr1 = merged[merged.contig == "chr1"]
        assert len(chr1) == 1
        assert int(chr1.iloc[0]["start"]) == 100
        assert int(chr1.iloc[0]["stop"]) == 300

    def test_merge_keeps_aggregated_columns(self):
        rdf = RegionDataFrame(
            pd.DataFrame(
                {
                    "contig": ["chr1", "chr1"],
                    "start": [100, 150],
                    "stop": [200, 300],
                    "strand": [".", "."],
                    "name": ["a", "b"],
                }
            ),
            ref="hg38",
        )
        # -c/-o makes bedtools emit a 4th column; it must survive.
        merged = rdf.merge_regions(c=5, o="collapse")
        assert len(merged.columns) >= 4
        assert not merged[merged.columns[3]].isna().all()


class TestAttachBlacklistRegions:
    def test_attaches_blacklist_coordinates_not_query_coordinates(
        self, blacklist_bed
    ):
        # Regression: this used iter_regions(), which reads contig/start/stop.
        # On the intersection result those columns belong to *self*, so every
        # query region was annotated with itself instead of the overlapping
        # blacklist interval.
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
        # a query spanning both blacklist intervals should collect both
        rdf = RegionDataFrame(
            pd.DataFrame({"contig": ["chr1"], "start": [100], "stop": [600]}),
            ref="hg38",
        )
        out = rdf.attach_blacklist_regions(blacklist_bed)

        attached = list(out["blacklist_regions"])[0]
        spans = sorted((r.start, r.stop) for r in attached)
        assert spans == [(150, 180), (500, 520)]
