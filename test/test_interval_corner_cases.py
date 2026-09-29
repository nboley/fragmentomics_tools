"""Phase 0 — synthetic corner-case tests for interval operations.

These tests capture the CURRENT behaviour of the production code so that
Phase 1 (restructuring) and Phase 2 (bioframe swap) can be validated
against a recorded baseline.

Each edge case is written deliberately — real data will not contain a
zero-length interval or a book-ended pair often enough to catch a
regression. The eight cases from the design doc's bedtools-vs-bioframe
differential (book-ended, 1bp gap, 1bp shared, nested, duplicates,
single-base, zero-length, two contigs) are tested for every relevant
operation, plus strand-aware variants and the `.`-vs-`.` pinning.

If a test fails against production code, it is a FINDING TO REPORT,
not something to patch away.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from fragmentomics_tools.dataframe import RegionDataFrame


def _rdf(rows, ref="hg38"):
    """Convenience: build a RegionDataFrame from a dict of columns."""
    return RegionDataFrame(pd.DataFrame(rows), ref=ref)


# ═══════════════════════════════════════════════════════════════════════
# merge_regions — bedtools merge (sorts internally, then merges)
# ═══════════════════════════════════════════════════════════════════════

class TestMergeCornerCases:
    """Pin merge_regions on the eight differential-test cases.

    bedtools merge considers book-ended (touching) intervals as adjacent
    and merges them by default.  Coordinates are half-open [start, stop).
    """

    def test_book_ended_are_merged(self):
        """[0,10) + [10,20) share a boundary — merged to [0,20)."""
        rdf = _rdf({"contig": ["chr1", "chr1"], "start": [0, 10], "stop": [10, 20]})
        merged = rdf.merge_regions()
        assert len(merged) == 1
        assert int(merged.start.iloc[0]) == 0
        assert int(merged.stop.iloc[0]) == 20

    def test_one_bp_gap_stays_separate(self):
        """[0,10) + [11,20) — 1bp gap at position 10, no merge."""
        rdf = _rdf({"contig": ["chr1", "chr1"], "start": [0, 11], "stop": [10, 20]})
        merged = rdf.merge_regions()
        assert len(merged) == 2

    def test_one_bp_shared_merges(self):
        """[0,10) + [9,20) share position 9."""
        rdf = _rdf({"contig": ["chr1", "chr1"], "start": [0, 9], "stop": [10, 20]})
        merged = rdf.merge_regions()
        assert len(merged) == 1
        assert int(merged.start.iloc[0]) == 0
        assert int(merged.stop.iloc[0]) == 20

    def test_nested_absorbed(self):
        """[0,100) + [10,20) — inner interval absorbed."""
        rdf = _rdf({"contig": ["chr1", "chr1"], "start": [0, 10], "stop": [100, 20]})
        merged = rdf.merge_regions()
        assert len(merged) == 1
        assert int(merged.start.iloc[0]) == 0
        assert int(merged.stop.iloc[0]) == 100

    def test_duplicates_collapse(self):
        """Two identical intervals collapse to one."""
        rdf = _rdf({"contig": ["chr1", "chr1"], "start": [10, 10], "stop": [20, 20]})
        merged = rdf.merge_regions()
        assert len(merged) == 1

    def test_single_base_survives(self):
        """[10,11) — single-base interval survives merge."""
        rdf = _rdf({"contig": ["chr1"], "start": [10], "stop": [11]})
        merged = rdf.merge_regions()
        assert len(merged) == 1
        assert int(merged.start.iloc[0]) == 10
        assert int(merged.stop.iloc[0]) == 11

    def test_zero_length_interval_survives_alone(self):
        """[10,10) is degenerate, and the current backend keeps it.

        Measured, not assumed. An earlier version of this test asserted
        ``len(merged) >= 0``, which is true of every possible result and so
        pinned nothing — the exact failure mode these fixtures exist to catch.
        """
        rdf = _rdf({"contig": ["chr1"], "start": [10], "stop": [10]})
        merged = rdf.merge_regions()
        assert len(merged) == 1
        assert int(merged.start.iloc[0]) == 10
        assert int(merged.stop.iloc[0]) == 10

    def test_zero_length_beside_a_real_interval_is_kept(self):
        """A degenerate interval disjoint from a real one survives separately."""
        rdf = _rdf({"contig": ["chr1", "chr1"], "start": [10, 100], "stop": [10, 200]})
        merged = rdf.merge_regions()
        assert len(merged) == 2
        assert list(map(int, merged.start)) == [10, 100]
        assert list(map(int, merged.stop)) == [10, 200]

    def test_zero_length_inside_a_real_interval_is_absorbed(self):
        """[150,150) inside [100,200) is absorbed, leaving one interval.

        The discriminating case of the three: it separates "degenerate
        intervals are always kept" from "kept only when disjoint". A backend
        that dropped zero-length intervals outright would pass the other two
        and fail this one.
        """
        rdf = _rdf({"contig": ["chr1", "chr1"], "start": [150, 100], "stop": [150, 200]})
        merged = rdf.merge_regions()
        assert len(merged) == 1
        assert int(merged.start.iloc[0]) == 100
        assert int(merged.stop.iloc[0]) == 200

    def test_two_contigs_stay_separate(self):
        """Intervals on different contigs never merge."""
        rdf = _rdf({
            "contig": ["chr1", "chr2", "chr1"],
            "start": [0, 0, 5],
            "stop": [10, 20, 15],
        })
        merged = rdf.merge_regions()
        assert len(merged) == 2
        chr1 = merged[merged.contig == "chr1"]
        assert len(chr1) == 1
        assert int(chr1.start.iloc[0]) == 0
        assert int(chr1.stop.iloc[0]) == 15

    def test_output_is_sorted(self):
        """Merged output is sorted by contig, start."""
        rdf = _rdf({
            "contig": ["chr2", "chr1", "chr1"],
            "start": [100, 50, 10],
            "stop": [200, 60, 20],
        })
        merged = rdf.merge_regions()
        starts = list(zip(merged.contig, merged.start))
        assert starts == sorted(starts)


# ═══════════════════════════════════════════════════════════════════════
# join_on_overlap — bedtools intersect -wa -wb (default)
# ═══════════════════════════════════════════════════════════════════════

class TestJoinOnOverlapCornerCases:
    """Pin join_on_overlap behaviour on edge cases.

    Uses bedtools intersect -wa -wb by default.  Returns whole A
    intervals (not geometric intersections) with B's columns suffixed.
    Book-ended intervals do NOT intersect (they share no bases).
    """

    def test_book_ended_do_not_intersect(self):
        """[100,200) and [200,300) share no bases — no overlap."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        b = _rdf({"contig": ["chr1"], "start": [200], "stop": [300]})
        result = a.join_on_overlap(b)
        assert len(result) == 0

    def test_one_bp_shared_intersects(self):
        """[100,200) and [199,300) share position 199."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        b = _rdf({"contig": ["chr1"], "start": [199], "stop": [300]})
        result = a.join_on_overlap(b)
        assert len(result) == 1

    def test_nested_intersects(self):
        """[0,100) contains [10,20) — overlap."""
        a = _rdf({"contig": ["chr1"], "start": [0], "stop": [100]})
        b = _rdf({"contig": ["chr1"], "start": [10], "stop": [20]})
        result = a.join_on_overlap(b)
        assert len(result) == 1
        # Returns A's full interval, not the geometric clip
        assert int(result.start.iloc[0]) == 0
        assert int(result.stop.iloc[0]) == 100

    def test_one_to_many(self):
        """One A interval overlaps two B intervals → two result rows."""
        a = _rdf({"contig": ["chr1"], "start": [0], "stop": [100]})
        b = _rdf({
            "contig": ["chr1", "chr1"],
            "start": [10, 50],
            "stop": [20, 60],
        })
        result = a.join_on_overlap(b)
        assert len(result) == 2

    def test_many_to_one(self):
        """Two A intervals overlap one B → two result rows."""
        a = _rdf({
            "contig": ["chr1", "chr1"],
            "start": [10, 50],
            "stop": [30, 70],
        })
        b = _rdf({"contig": ["chr1"], "start": [20], "stop": [60]})
        result = a.join_on_overlap(b)
        assert len(result) == 2

    def test_two_contigs_match_only_within(self):
        """Overlaps only match within the same contig."""
        a = _rdf({
            "contig": ["chr1", "chr2"],
            "start": [100, 100],
            "stop": [200, 200],
        })
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [250]})
        result = a.join_on_overlap(b)
        assert len(result) == 1

    def test_b_columns_are_suffixed(self):
        """B's columns get the rsuff to avoid name collisions."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [250]})
        result = a.join_on_overlap(b, rsuff="B")
        assert "contig_B" in result.columns
        assert "start_B" in result.columns
        assert "stop_B" in result.columns

    def test_index_is_preserved(self):
        """A's original index is carried through the join."""
        a = _rdf({
            "contig": ["chr1", "chr1"],
            "start": [100, 300],
            "stop": [200, 400],
        })
        a.index = pd.Index([10, 20])
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [250]})
        result = a.join_on_overlap(b)
        assert len(result) == 1
        assert result.index[0] == 10

    def test_duplicates(self):
        """Two identical A intervals overlapping B → two result rows."""
        a = _rdf({
            "contig": ["chr1", "chr1"],
            "start": [100, 100],
            "stop": [200, 200],
        })
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [250]})
        result = a.join_on_overlap(b)
        assert len(result) == 2


# ═══════════════════════════════════════════════════════════════════════
# drop_overlapping_regions — the anti-join
# ═══════════════════════════════════════════════════════════════════════

class TestDropOverlappingCornerCases:
    """Pin drop_overlapping_regions — keeps non-overlapping A intervals."""

    def test_overlapping_removed(self):
        a = _rdf({
            "contig": ["chr1", "chr1"],
            "start": [100, 500],
            "stop": [200, 600],
        })
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [250]})
        result = a.drop_overlapping_regions(b)
        assert len(result) == 1
        assert int(result.start.iloc[0]) == 500

    def test_nothing_overlaps_keeps_all(self):
        a = _rdf({
            "contig": ["chr1", "chr1"],
            "start": [100, 500],
            "stop": [200, 600],
        })
        b = _rdf({"contig": ["chr2"], "start": [100], "stop": [200]})
        result = a.drop_overlapping_regions(b)
        assert len(result) == 2

    def test_everything_overlaps_returns_empty(self):
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        b = _rdf({"contig": ["chr1"], "start": [0], "stop": [300]})
        result = a.drop_overlapping_regions(b)
        assert len(result) == 0
        assert isinstance(result, RegionDataFrame)

    def test_book_ended_not_removed(self):
        """Book-ended intervals don't overlap (in intersect), so kept."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        b = _rdf({"contig": ["chr1"], "start": [200], "stop": [300]})
        result = a.drop_overlapping_regions(b)
        assert len(result) == 1

    def test_nested_is_removed(self):
        """A nested inside B is still overlapping → removed."""
        a = _rdf({"contig": ["chr1"], "start": [120], "stop": [180]})
        b = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        result = a.drop_overlapping_regions(b)
        assert len(result) == 0


# ═══════════════════════════════════════════════════════════════════════
# overlaps_rdf — IntervalTree-based boolean overlap
# ═══════════════════════════════════════════════════════════════════════

class TestOverlapsRdfCornerCases:
    """Pin overlaps_rdf behaviour.

    Uses IntervalTree (not bedtools), with half-open semantics.
    Book-ended intervals do NOT overlap.  max_distance expands the
    query intervals symmetrically for unstranded features.
    """

    def test_book_ended_no_overlap(self):
        """Half-open: [100,200) and [200,300) don't share any bases."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        b = _rdf({"contig": ["chr1"], "start": [200], "stop": [300]})
        result = a.overlaps_rdf(b)
        assert list(result) == [False]

    def test_one_bp_shared(self):
        """[100,200) and [199,300) share position 199."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        b = _rdf({"contig": ["chr1"], "start": [199], "stop": [300]})
        result = a.overlaps_rdf(b)
        assert list(result) == [True]

    def test_nested(self):
        a = _rdf({"contig": ["chr1"], "start": [0], "stop": [100]})
        b = _rdf({"contig": ["chr1"], "start": [10], "stop": [20]})
        result = a.overlaps_rdf(b)
        assert list(result) == [True]

    def test_max_distance_bridges_book_ended(self):
        """max_distance=1 expands query [200,300) to [199,301), which
        overlaps with [100,200) at position 199."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        b = _rdf({"contig": ["chr1"], "start": [200], "stop": [300]})
        result = a.overlaps_rdf(b, max_distance=1)
        assert list(result) == [True]

    def test_max_distance_not_quite_enough(self):
        """[100,200) + [202,300) — edge-to-edge gap is 2bp.

        max_distance expands the query by N on each side, but IntervalTree
        uses half-open intervals, so expanded [200,302) and [100,200) are
        book-ended and still don't overlap.  max_distance=3 is the first
        value that bridges a 2bp gap (expanding to [199,303) which shares
        position 199 with A).

        This is a quirk of the implementation: the effective bridging
        distance is max_distance - 1, not max_distance.
        """
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        b = _rdf({"contig": ["chr1"], "start": [202], "stop": [300]})
        assert list(a.overlaps_rdf(b, max_distance=1)) == [False]
        assert list(a.overlaps_rdf(b, max_distance=2)) == [False]  # still book-ended
        assert list(a.overlaps_rdf(b, max_distance=3)) == [True]

    def test_two_contigs(self):
        a = _rdf({
            "contig": ["chr1", "chr2"],
            "start": [100, 100],
            "stop": [200, 200],
        })
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [250]})
        result = a.overlaps_rdf(b)
        assert list(result) == [True, False]

    def test_single_base_overlap(self):
        """[100,200) and [199,200) share exactly position 199."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        b = _rdf({"contig": ["chr1"], "start": [199], "stop": [200]})
        result = a.overlaps_rdf(b)
        assert list(result) == [True]

    def test_duplicates(self):
        """Duplicate intervals both get True."""
        a = _rdf({
            "contig": ["chr1", "chr1"],
            "start": [100, 100],
            "stop": [200, 200],
        })
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [250]})
        result = a.overlaps_rdf(b)
        assert list(result) == [True, True]


# ═══════════════════════════════════════════════════════════════════════
# get_overlapping_base_counts — known bugs included
# ═══════════════════════════════════════════════════════════════════════

class TestGetOverlappingBaseCountsCornerCases:
    """Pin get_overlapping_base_counts, including its documented bugs.

    KNOWN BUG (from design doc): this method keys aggregation on
    (contig, start, stop) with strand EXCLUDED.  Two rows sharing
    coordinates but different strands collide: the later row overwrites
    the earlier in the lookup dict, and the groupby combines both rows'
    base counts into that one entry.  Phase 1 deletes this method.
    """

    @pytest.fixture
    def anno_bed(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "anno.bed")
            with open(path, "w") as fh:
                fh.write("chr1\t150\t180\n")
                fh.write("chr1\t190\t260\n")
            yield path

    def test_book_ended_no_overlap(self):
        """Book-ended: [180,190) vs annotation at [150,180) — no shared bases."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "anno.bed")
            with open(path, "w") as fh:
                fh.write("chr1\t150\t180\n")
            rdf = _rdf({"contig": ["chr1"], "start": [180], "stop": [200]})
            res = rdf.get_overlapping_base_counts(path)
            assert list(res["counts"]) == [0]
            assert list(res["max_counts"]) == [0]

    def test_nested_gets_full_inner_length(self):
        """Query fully contains annotation → overlap = annotation length."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "anno.bed")
            with open(path, "w") as fh:
                fh.write("chr1\t120\t130\n")  # 10bp, nested inside [100,200)
            rdf = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
            res = rdf.get_overlapping_base_counts(path)
            assert list(res["counts"]) == [10]
            assert list(res["max_counts"]) == [10]

    def test_strand_collision_bug(self):
        """KNOWN BUG: two rows with same (contig, start, stop) but different
        strands collide in region2idx.

        region2idx = {(contig, start, stop): idx} — strand excluded.
        The dict comprehension's last write wins (idx=1), so the groupby
        deposits both rows' combined overlaps into counts[1] while
        counts[0] stays at zero.
        """
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "anno.bed")
            with open(path, "w") as fh:
                fh.write("chr1\t120\t130\n")  # 10bp overlap with [100,200)

            rdf = _rdf({
                "contig": ["chr1", "chr1"],
                "start": [100, 100],
                "stop": [200, 200],
                "strand": ["+", "-"],
            })
            res = rdf.get_overlapping_base_counts(path)

            # Bug: first row (idx=0) gets nothing — it was overwritten in
            # region2idx.  Second row (idx=1) gets the combined count from
            # both rows' overlaps (10 + 10 = 20).
            assert res["counts"][0] == 0, (
                "Bug behavior: first row should get 0 due to key collision"
            )
            assert res["counts"][1] == 20, (
                "Bug behavior: second row should get combined count (20)"
            )
            assert res["max_counts"][0] == 0
            assert res["max_counts"][1] == 10


# ═══════════════════════════════════════════════════════════════════════
# _get_fragment_coverage_sum — BED path (the one that changes shape
# in Phase 2 when chunking is introduced)
# ═══════════════════════════════════════════════════════════════════════

class TestFragmentCoverageSumCornerCases:
    """Pin _get_fragment_coverage_sum with the BED-file path.

    This is the method that changes from streaming (pybedtools) to
    chunked in-memory reads (bioframe) in Phase 2.  The fixture ensures
    the chunked result matches the unchunked one.
    """

    def test_basic_counting(self):
        rdf = _rdf({
            "contig": ["chr1", "chr1", "chr2"],
            "start": [100, 500, 100],
            "stop": [200, 600, 200],
            "id": ["r1", "r2", "r3"],
        })
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "fragments.bed")
            with open(path, "w") as fh:
                # 3 fragments in region r1
                fh.write("chr1\t120\t140\n")
                fh.write("chr1\t150\t160\n")
                fh.write("chr1\t180\t190\n")
                # 1 fragment in region r2
                fh.write("chr1\t510\t520\n")
                # 0 fragments in r3 (different contig)
            result = rdf._get_fragment_coverage_sum(path)
            assert len(result) == 3
            assert result[0] == 3.0
            assert result[1] == 1.0
            assert result[2] == 0.0

    def test_fragment_spanning_two_regions(self):
        """A fragment overlapping two regions is counted in both."""
        rdf = _rdf({
            "contig": ["chr1", "chr1"],
            "start": [100, 190],
            "stop": [200, 300],
            "id": ["r1", "r2"],
        })
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "fragments.bed")
            with open(path, "w") as fh:
                fh.write("chr1\t150\t250\n")  # spans both regions
            result = rdf._get_fragment_coverage_sum(path)
            assert result[0] == 1.0
            assert result[1] == 1.0

    def test_book_ended_fragment_not_counted(self):
        """A fragment book-ended with a region doesn't overlap it."""
        rdf = _rdf({
            "contig": ["chr1"],
            "start": [100],
            "stop": [200],
            "id": ["r1"],
        })
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "fragments.bed")
            with open(path, "w") as fh:
                fh.write("chr1\t200\t300\n")  # starts where region ends
            result = rdf._get_fragment_coverage_sum(path)
            assert result[0] == 0.0


# ═══════════════════════════════════════════════════════════════════════
# from_beds_merged — concat + merge
# ═══════════════════════════════════════════════════════════════════════

class TestFromBedsMergedCornerCases:
    """Pin from_beds_merged behaviour."""

    def test_single_bed(self):
        """Single-file path delegates to from_bed."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "one.bed")
            with open(path, "w") as fh:
                fh.write("chr1\t100\t200\n")
                fh.write("chr1\t300\t400\n")
            result = RegionDataFrame.from_beds_merged([path], ref="hg38")
            assert len(result) == 2

    def test_two_beds_overlapping_merge(self):
        """Two BEDs with overlapping regions get merged."""
        with tempfile.TemporaryDirectory() as d:
            p1 = os.path.join(d, "a.bed")
            p2 = os.path.join(d, "b.bed")
            with open(p1, "w") as fh:
                fh.write("chr1\t100\t200\n")
            with open(p2, "w") as fh:
                fh.write("chr1\t150\t300\n")
            result = RegionDataFrame.from_beds_merged([p1, p2], ref="hg38")
            assert len(result) == 1
            assert int(result.start.iloc[0]) == 100
            assert int(result.stop.iloc[0]) == 300

    def test_two_beds_non_overlapping(self):
        """Two BEDs with no overlap stay separate after merge."""
        with tempfile.TemporaryDirectory() as d:
            p1 = os.path.join(d, "a.bed")
            p2 = os.path.join(d, "b.bed")
            with open(p1, "w") as fh:
                fh.write("chr1\t100\t200\n")
            with open(p2, "w") as fh:
                fh.write("chr1\t500\t600\n")
            result = RegionDataFrame.from_beds_merged([p1, p2], ref="hg38")
            assert len(result) == 2

    def test_book_ended_beds_merge(self):
        """Book-ended intervals from two BEDs merge (merge considers
        them adjacent)."""
        with tempfile.TemporaryDirectory() as d:
            p1 = os.path.join(d, "a.bed")
            p2 = os.path.join(d, "b.bed")
            with open(p1, "w") as fh:
                fh.write("chr1\t0\t100\n")
            with open(p2, "w") as fh:
                fh.write("chr1\t100\t200\n")
            result = RegionDataFrame.from_beds_merged([p1, p2], ref="hg38")
            assert len(result) == 1
            assert int(result.start.iloc[0]) == 0
            assert int(result.stop.iloc[0]) == 200

    def test_chroms_filter(self):
        """chroms parameter filters to selected chromosomes."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "multi.bed")
            with open(path, "w") as fh:
                fh.write("chr1\t100\t200\n")
                fh.write("chr2\t100\t200\n")
                fh.write("chr3\t100\t200\n")
            result = RegionDataFrame.from_beds_merged(
                [path], ref="hg38", chroms=["chr1", "chr3"]
            )
            assert len(result) == 2
            assert set(result.contig) == {"chr1", "chr3"}

    def test_default_callback_is_identity(self):
        """bed_filter_callback=None produces the same result as an
        explicit identity filter.  This pins that dropping the callback
        parameter in Phase 2 changes nothing."""
        with tempfile.TemporaryDirectory() as d:
            p1 = os.path.join(d, "a.bed")
            p2 = os.path.join(d, "b.bed")
            with open(p1, "w") as fh:
                fh.write("chr1\t100\t200\n")
                fh.write("chr1\t400\t500\n")
            with open(p2, "w") as fh:
                fh.write("chr1\t300\t600\n")
            default = RegionDataFrame.from_beds_merged([p1, p2], ref="hg38")
            explicit = RegionDataFrame.from_beds_merged(
                [p1, p2], ref="hg38",
                bed_filter_callback=lambda _: True,
            )
            assert len(default) == len(explicit)
            assert list(default.contig) == list(explicit.contig)
            assert list(default.start) == list(explicit.start)
            assert list(default.stop) == list(explicit.stop)


# ═══════════════════════════════════════════════════════════════════════
# Strand behaviour — pin that strand is IGNORED in all current ops
# ═══════════════════════════════════════════════════════════════════════

class TestStrandBehavior:
    """Pin that strand is IGNORED in all current interval operations.

    No production code passes a strand flag to bedtools.  The bedtools
    path ignores strand by default; overlaps_rdf uses coordinate-only
    IntervalTree.

    Phase 1's same_strand parameter must treat "." vs "." as NO MATCH
    (the bedtools -s / bioframe on=['strand'] divergence).  The tests
    here pin the CURRENT behaviour (strand ignored → everything matches
    on coordinates alone) so that the Phase 1 change is visible as a
    diff, not an assumption.
    """

    def test_join_opposite_strands_overlap(self):
        """+ vs - on overlapping coordinates: match (strand ignored)."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200], "strand": ["+"]})
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [250], "strand": ["-"]})
        result = a.join_on_overlap(b)
        assert len(result) == 1

    def test_join_dot_vs_dot_overlap(self):
        """'.' vs '.' on overlapping coordinates: match (strand ignored).

        This is the case that will CHANGE in Phase 1: same_strand=True
        must treat '.' vs '.' as NO match.  But currently strand is not
        compared at all, so they match.
        """
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200], "strand": ["."]})
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [250], "strand": ["."]})
        result = a.join_on_overlap(b)
        assert len(result) == 1

    def test_merge_ignores_strand(self):
        """+ and - regions that overlap coordinately merge."""
        rdf = _rdf({
            "contig": ["chr1", "chr1"],
            "start": [100, 150],
            "stop": [200, 300],
            "strand": ["+", "-"],
        })
        merged = rdf.merge_regions()
        assert len(merged) == 1

    def test_overlaps_rdf_ignores_strand(self):
        """IntervalTree is coordinate-only; opposite strands still match."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200], "strand": ["+"]})
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [250], "strand": ["-"]})
        result = a.overlaps_rdf(b)
        assert list(result) == [True]

    def test_overlaps_rdf_dot_vs_dot(self):
        """'.' vs '.' currently matches (strand not compared).

        Phase 1: same_strand=True must make this NOT match.
        """
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200], "strand": ["."]})
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [250], "strand": ["."]})
        result = a.overlaps_rdf(b)
        assert list(result) == [True]

    def test_drop_overlapping_ignores_strand(self):
        """+ regions are dropped by - blacklist if coordinates overlap."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200], "strand": ["+"]})
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [250], "strand": ["-"]})
        result = a.drop_overlapping_regions(b)
        assert len(result) == 0

    def test_merge_dot_vs_dot_merges(self):
        """'.' stranded regions merge if coordinates overlap."""
        rdf = _rdf({
            "contig": ["chr1", "chr1"],
            "start": [100, 150],
            "stop": [200, 300],
            "strand": [".", "."],
        })
        merged = rdf.merge_regions()
        assert len(merged) == 1
