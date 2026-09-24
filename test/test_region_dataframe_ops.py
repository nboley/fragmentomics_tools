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


class TestGetOverlappingBaseCounts:
    """This method raised TypeError on every call before the fix.

    It passes wao=True through intersect_with_bed to intersect_with_rdf, whose
    signature did not accept **intersect_kwargs even though its docstring
    documented them.
    """

    @pytest.fixture
    def anno_bed(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "anno.bed")
            with open(path, "w") as fh:
                fh.write("chr1\t150\t180\n")  # 30bp inside chr1:100-200
                fh.write("chr1\t190\t260\n")  # 10bp inside chr1:100-200
                fh.write("chr2\t500\t600\n")  # covers chr2:500-600 entirely
            yield path

    def test_counts_are_correct(self, anno_bed):
        rdf = RegionDataFrame(
            pd.DataFrame(
                {
                    "contig": ["chr1", "chr1", "chr2"],
                    "start": [100, 700, 500],
                    "stop": [200, 800, 600],
                }
            ),
            ref="hg38",
        )
        res = rdf.get_overlapping_base_counts(anno_bed)

        # chr1:100-200 overlaps 30bp + 10bp; chr1:700-800 overlaps nothing;
        # chr2:500-600 is fully covered.
        assert list(res["counts"]) == [40, 0, 100]
        # the longest single overlapping interval per region
        assert list(res["max_counts"]) == [30, 0, 100]


class TestSplitOnColumn:
    def test_splits_on_the_named_column(self):
        # Regression: the query string was the literal "column_name in @values",
        # so it looked for a column actually called "column_name" and ignored
        # the parameter entirely. The method could never have worked.
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


class TestGetFragmentCoverageSum:
    """R3: _get_fragment_coverage_sum returned a length-0 array on an empty
    BED intersection, where a length-len(self) zero array was expected."""

    @pytest.fixture
    def empty_bed(self):
        """A BED file with one region that does not overlap any test query."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "empty.bed")
            with open(path, "w") as fh:
                fh.write("chrX\t9000\t9999\n")
            yield path

    def test_empty_intersection_returns_correct_shape(self, empty_bed):
        rdf = RegionDataFrame(
            pd.DataFrame(
                {
                    "contig": ["chr1", "chr2"],
                    "start": [100, 200],
                    "stop": [150, 250],
                    "id": ["a", "b"],
                }
            ),
            ref="hg38",
        )
        result = rdf._get_fragment_coverage_sum(empty_bed)
        assert len(result) == len(rdf), (
            f"expected length {len(rdf)}, got {len(result)}"
        )
        assert (result == 0).all()


class TestIntersectWithRdfReturnType:
    """I4: intersect_with_rdf returned a plain DataFrame when the intersection
    was empty, but a RegionDataFrame otherwise. Callers chaining RDF methods
    hit AttributeError only on the empty path."""

    def test_empty_intersection_returns_subclass(self):
        rdf1 = RegionDataFrame(
            pd.DataFrame({"contig": ["chr1"], "start": [100], "stop": [200]}),
            ref="hg38",
        )
        rdf2 = RegionDataFrame(
            pd.DataFrame({"contig": ["chr2"], "start": [500], "stop": [600]}),
            ref="hg38",
        )
        result = rdf1.intersect_with_rdf(rdf2)
        assert isinstance(result, RegionDataFrame), (
            f"expected RegionDataFrame, got {type(result).__name__}"
        )

    def test_empty_and_nonempty_have_same_columns(self):
        rdf1 = RegionDataFrame(
            pd.DataFrame({"contig": ["chr1"], "start": [100], "stop": [200]}),
            ref="hg38",
        )
        # non-overlapping
        rdf_far = RegionDataFrame(
            pd.DataFrame({"contig": ["chr2"], "start": [500], "stop": [600]}),
            ref="hg38",
        )
        empty = rdf1.intersect_with_rdf(rdf_far)

        # overlapping
        rdf_near = RegionDataFrame(
            pd.DataFrame({"contig": ["chr1"], "start": [150], "stop": [250]}),
            ref="hg38",
        )
        nonempty = rdf1.intersect_with_rdf(rdf_near)

        assert set(empty.columns) == set(nonempty.columns)


class TestEqSemantics:
    """I10: __eq__ used to return a scalar bool, breaking the pandas contract
    for element-wise comparison and making instances unhashable."""

    def test_eq_returns_elementwise(self):
        rdf = RegionDataFrame(
            pd.DataFrame({"contig": ["chr1"], "start": [100], "stop": [200]}),
            ref="hg38",
        )
        # pandas __eq__ should return a DataFrame of booleans, not a scalar
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
    """I11: center_on_summit used summit <= stop, but half-open [start, stop)
    means a summit equal to stop is outside the region."""

    def test_summit_at_stop_is_rejected(self):
        rdf = RegionDataFrame(
            pd.DataFrame({
                "contig": ["chr1"], "start": [100], "stop": [200], "summit": [200]
            }),
            ref="hg38",
        )
        # summit == stop is outside the half-open interval
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
        # region length preserved
        assert int(result.stop.iloc[0]) - int(result.start.iloc[0]) == 100

    def test_does_not_mutate_self_by_default(self):
        """I21: center_on_summit mutated self in place with no inplace param."""
        rdf = RegionDataFrame(
            pd.DataFrame({
                "contig": ["chr1"], "start": [100], "stop": [200], "summit": [120]
            }),
            ref="hg38",
        )
        original_start = int(rdf.start.iloc[0])
        result = rdf.center_on_summit()
        assert int(rdf.start.iloc[0]) == original_start, (
            "center_on_summit mutated self without inplace=True"
        )
        # result should be different
        assert int(result.start.iloc[0]) != original_start


class TestOverlapsRdf:
    """I16: overlaps_rdf raised KeyError when self had a contig absent from
    the query's interval dict."""

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
        result = rdf.overlaps_rdf(query)
        # chr1 overlaps, chr2 does not (contig absent from query)
        assert list(result) == [True, False]


class TestAndAndConcat:
    """I17: __and__ hardcoded RegionDataFrame, dropping subclass identity.
    I19: concat([]) crashed with IndexError instead of a useful error."""

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
    """I18: lift_over crashed on empty RDF (zip(*[]) can't unpack).
    I20: failed liftover regions got -1 coordinates instead of NA."""

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
        assert pd.isna(result.start.iloc[0]), (
            f"expected NA for failed liftover, got {result.start.iloc[0]}"
        )


class TestFromBed:
    """I6: from_bed crashed with IndexError on an empty file.
    I22: dead has_header computation removed."""

    def test_empty_bed_returns_empty_rdf(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "empty.bed")
            with open(path, "w") as fh:
                pass  # empty file
            rdf = RegionDataFrame.from_bed(path, ref="hg38")
            assert len(rdf) == 0
            assert isinstance(rdf, RegionDataFrame)


class TestUniqueRegions:
    """I23: unique_regions sorted coordinates descending, which is unusual
    and changes which duplicate survives in edge cases."""

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
        # should be ascending coordinate order
        assert list(result.contig) == ["chr1", "chr1", "chr2"]
        assert list(result.start) == [100, 300, 200]


class TestGetIntervalDict:
    """I5: get_interval_dict rejected unstranded regions with expand."""

    def test_unstranded_with_expand_does_not_crash(self):
        rdf = RegionDataFrame(
            pd.DataFrame({
                "contig": ["chr1"],
                "start": [100],
                "stop": [200],
                "strand": ["."],
            }),
            ref="hg38",
        )
        d = rdf.get_interval_dict(
            data_cols=None, expand_upstream=10, expand_downstream=10
        )
        tree = d["chr1"]
        # unstranded: symmetric expand by max(10, 10) = 10 in both directions
        assert len(tree[90:210]) > 0
        assert len(tree[89:90]) == 0  # just outside


class TestDropOverlappingRegions:
    """I7: drop_overlapping_regions hardcoded assert ref == 'hg38'."""

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
        result = rdf.drop_overlapping_regions(blacklist)
        # chr1:100-200 overlaps blacklist, chr1:500-600 does not
        assert len(result) == 1
        assert int(result.start.iloc[0]) == 500
