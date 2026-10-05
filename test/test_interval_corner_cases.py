"""Phase 0 → Phase 1 — synthetic corner-case tests for interval operations.

Updated for the intervals module API (overlap_indices, overlaps, merge, cluster).
The old bedtools-backed methods have been removed; these tests now exercise
the bioframe-backed replacements.

Fixture movements from the Phase 0 baseline are recorded in the design doc
(docs/pending/interval_api_design.md, §Fixture movements).
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from fragmentomics_tools.dataframe import RegionDataFrame
from fragmentomics_tools.intervals import (
    overlap_indices,
    overlaps,
    merge,
    cluster,
    nearest,
)


def _rdf(rows, ref="hg38"):
    """Convenience: build a RegionDataFrame from a dict of columns."""
    return RegionDataFrame(pd.DataFrame(rows), ref=ref)


# ═══════════════════════════════════════════════════════════════════════
# merge — replaces merge_regions
# ═══════════════════════════════════════════════════════════════════════

class TestMergeCornerCases:
    """Pin merge on the eight differential-test cases.

    FIXTURE MOVEMENT: merge(wiggle=0) does NOT merge book-ended intervals.
    This differs from the old merge_regions() which delegated to bedtools
    merge, which considers book-ended as adjacent.  wiggle=1 restores the
    old book-ended merging behavior.
    """

    def test_book_ended_not_merged_at_wiggle_0(self):
        """[0,10) + [10,20) share a boundary — NOT merged at wiggle=0."""
        rdf = _rdf({"contig": ["chr1", "chr1"], "start": [0, 10], "stop": [10, 20]})
        merged = merge(rdf, wiggle=0)
        assert len(merged) == 2

    def test_book_ended_merged_at_wiggle_1(self):
        """[0,10) + [10,20) — merged at wiggle=1 (gap=0 <= 1)."""
        rdf = _rdf({"contig": ["chr1", "chr1"], "start": [0, 10], "stop": [10, 20]})
        merged = merge(rdf, wiggle=1)
        assert len(merged) == 1
        assert int(merged.start.iloc[0]) == 0
        assert int(merged.stop.iloc[0]) == 20

    def test_one_bp_gap_stays_separate(self):
        """[0,10) + [11,20) — 1bp gap at position 10, no merge."""
        rdf = _rdf({"contig": ["chr1", "chr1"], "start": [0, 11], "stop": [10, 20]})
        merged = merge(rdf)
        assert len(merged) == 2

    def test_one_bp_shared_merges(self):
        """[0,10) + [9,20) share position 9."""
        rdf = _rdf({"contig": ["chr1", "chr1"], "start": [0, 9], "stop": [10, 20]})
        merged = merge(rdf)
        assert len(merged) == 1
        assert int(merged.start.iloc[0]) == 0
        assert int(merged.stop.iloc[0]) == 20

    def test_nested_absorbed(self):
        """[0,100) + [10,20) — inner interval absorbed."""
        rdf = _rdf({"contig": ["chr1", "chr1"], "start": [0, 10], "stop": [100, 20]})
        merged = merge(rdf)
        assert len(merged) == 1
        assert int(merged.start.iloc[0]) == 0
        assert int(merged.stop.iloc[0]) == 100

    def test_duplicates_collapse(self):
        """Two identical intervals collapse to one."""
        rdf = _rdf({"contig": ["chr1", "chr1"], "start": [10, 10], "stop": [20, 20]})
        merged = merge(rdf)
        assert len(merged) == 1

    def test_single_base_survives(self):
        """[10,11) — single-base interval survives merge."""
        rdf = _rdf({"contig": ["chr1"], "start": [10], "stop": [11]})
        merged = merge(rdf)
        assert len(merged) == 1
        assert int(merged.start.iloc[0]) == 10
        assert int(merged.stop.iloc[0]) == 11

    def test_two_contigs_stay_separate(self):
        """Intervals on different contigs never merge."""
        rdf = _rdf({
            "contig": ["chr1", "chr2", "chr1"],
            "start": [0, 0, 5],
            "stop": [10, 20, 15],
        })
        merged = merge(rdf)
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
        merged = merge(rdf)
        starts = list(zip(merged.contig, merged.start))
        assert starts == sorted(starts)


# ═══════════════════════════════════════════════════════════════════════
# overlap_indices — replaces join_on_overlap
# ═══════════════════════════════════════════════════════════════════════

class TestOverlapIndicesCornerCases:
    """Pin overlap_indices behaviour on edge cases."""

    def test_book_ended_do_not_intersect(self):
        """[100,200) and [200,300) share no bases — no overlap."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        b = _rdf({"contig": ["chr1"], "start": [200], "stop": [300]})
        result = overlap_indices(a, b)
        assert len(result) == 0

    def test_one_bp_shared_intersects(self):
        """[100,200) and [199,300) share position 199."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        b = _rdf({"contig": ["chr1"], "start": [199], "stop": [300]})
        result = overlap_indices(a, b)
        assert len(result) == 1
        assert int(result["overlap_bases"].iloc[0]) == 1

    def test_nested_intersects(self):
        """[0,100) contains [10,20) — overlap of 10bp."""
        a = _rdf({"contig": ["chr1"], "start": [0], "stop": [100]})
        b = _rdf({"contig": ["chr1"], "start": [10], "stop": [20]})
        result = overlap_indices(a, b)
        assert len(result) == 1
        assert int(result["overlap_bases"].iloc[0]) == 10

    def test_one_to_many(self):
        """One A interval overlaps two B intervals → two result rows."""
        a = _rdf({"contig": ["chr1"], "start": [0], "stop": [100]})
        b = _rdf({
            "contig": ["chr1", "chr1"],
            "start": [10, 50],
            "stop": [20, 60],
        })
        result = overlap_indices(a, b)
        assert len(result) == 2

    def test_many_to_one(self):
        """Two A intervals overlap one B → two result rows."""
        a = _rdf({
            "contig": ["chr1", "chr1"],
            "start": [10, 50],
            "stop": [30, 70],
        })
        b = _rdf({"contig": ["chr1"], "start": [20], "stop": [60]})
        result = overlap_indices(a, b)
        assert len(result) == 2

    def test_two_contigs_match_only_within(self):
        """Overlaps only match within the same contig."""
        a = _rdf({
            "contig": ["chr1", "chr2"],
            "start": [100, 100],
            "stop": [200, 200],
        })
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [250]})
        result = overlap_indices(a, b)
        assert len(result) == 1

    def test_how_validation(self):
        """Invalid how values raise ValueError."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [250]})
        with pytest.raises(ValueError, match="how="):
            overlap_indices(a, b, how="bogus")

    def test_anti_join(self):
        """how='anti' returns A rows with no match in B."""
        a = _rdf({
            "contig": ["chr1", "chr1"],
            "start": [100, 500],
            "stop": [200, 600],
        })
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [250]})
        result = overlap_indices(a, b, how="anti")
        assert len(result) == 1
        assert int(result["a_pos"].iloc[0]) == 1


# ═══════════════════════════════════════════════════════════════════════
# anti join — replaces drop_overlapping_regions
# ═══════════════════════════════════════════════════════════════════════

class TestAntiJoinCornerCases:
    """Pin anti join — keeps non-overlapping A intervals."""

    def test_overlapping_removed(self):
        a = _rdf({
            "contig": ["chr1", "chr1"],
            "start": [100, 500],
            "stop": [200, 600],
        })
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [250]})
        result = overlap_indices(a, b, how="anti")
        assert len(result) == 1
        assert int(result["a_pos"].iloc[0]) == 1

    def test_nothing_overlaps_keeps_all(self):
        a = _rdf({
            "contig": ["chr1", "chr1"],
            "start": [100, 500],
            "stop": [200, 600],
        })
        b = _rdf({"contig": ["chr2"], "start": [100], "stop": [200]})
        result = overlap_indices(a, b, how="anti")
        assert len(result) == 2

    def test_everything_overlaps_returns_empty(self):
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        b = _rdf({"contig": ["chr1"], "start": [0], "stop": [300]})
        result = overlap_indices(a, b, how="anti")
        assert len(result) == 0

    def test_book_ended_not_removed(self):
        """Book-ended intervals don't overlap, so kept in anti."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        b = _rdf({"contig": ["chr1"], "start": [200], "stop": [300]})
        result = overlap_indices(a, b, how="anti")
        assert len(result) == 1

    def test_nested_is_removed(self):
        """A nested inside B is still overlapping → removed."""
        a = _rdf({"contig": ["chr1"], "start": [120], "stop": [180]})
        b = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        result = overlap_indices(a, b, how="anti")
        assert len(result) == 0


# ═══════════════════════════════════════════════════════════════════════
# overlaps — replaces overlaps_rdf
# ═══════════════════════════════════════════════════════════════════════

class TestOverlapsCornerCases:
    """Pin overlaps behaviour."""

    def test_book_ended_no_overlap(self):
        """Half-open: [100,200) and [200,300) don't share any bases."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        b = _rdf({"contig": ["chr1"], "start": [200], "stop": [300]})
        result = overlaps(a, b)
        assert list(result) == [False]

    def test_one_bp_shared(self):
        """[100,200) and [199,300) share position 199."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        b = _rdf({"contig": ["chr1"], "start": [199], "stop": [300]})
        result = overlaps(a, b)
        assert list(result) == [True]

    def test_nested(self):
        a = _rdf({"contig": ["chr1"], "start": [0], "stop": [100]})
        b = _rdf({"contig": ["chr1"], "start": [10], "stop": [20]})
        result = overlaps(a, b)
        assert list(result) == [True]

    def test_wiggle_bridges_book_ended(self):
        """wiggle=1 bridges a book-ended gap (gap=0 <= 1)."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        b = _rdf({"contig": ["chr1"], "start": [200], "stop": [300]})
        result = overlaps(a, b, wiggle=1)
        assert list(result) == [True]

    def test_wiggle_boundary(self):
        """A gap of exactly G matches at wiggle=G and not at G-1.

        This is the required boundary test from the design doc.
        """
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        b = _rdf({"contig": ["chr1"], "start": [210], "stop": [300]})  # gap = 10
        assert list(overlaps(a, b, wiggle=9)) == [False]
        assert list(overlaps(a, b, wiggle=10)) == [True]

    def test_two_contigs(self):
        a = _rdf({
            "contig": ["chr1", "chr2"],
            "start": [100, 100],
            "stop": [200, 200],
        })
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [250]})
        result = overlaps(a, b)
        assert list(result) == [True, False]

    def test_single_base_overlap(self):
        """[100,200) and [199,200) share exactly position 199."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        b = _rdf({"contig": ["chr1"], "start": [199], "stop": [200]})
        result = overlaps(a, b)
        assert list(result) == [True]

    def test_duplicates(self):
        """Duplicate intervals both get True."""
        a = _rdf({
            "contig": ["chr1", "chr1"],
            "start": [100, 100],
            "stop": [200, 200],
        })
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [250]})
        result = overlaps(a, b)
        assert list(result) == [True, True]


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

    def test_book_ended_beds_not_merged(self):
        """FIXTURE MOVEMENT: book-ended intervals from two BEDs do NOT
        merge at wiggle=0 (the new default).  The old merge_regions()
        delegated to bedtools merge which merged book-ended by default."""
        with tempfile.TemporaryDirectory() as d:
            p1 = os.path.join(d, "a.bed")
            p2 = os.path.join(d, "b.bed")
            with open(p1, "w") as fh:
                fh.write("chr1\t0\t100\n")
            with open(p2, "w") as fh:
                fh.write("chr1\t100\t200\n")
            result = RegionDataFrame.from_beds_merged([p1, p2], ref="hg38")
            assert len(result) == 2

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


# ═══════════════════════════════════════════════════════════════════════
# Strand behaviour — same_strand parameter
# ═══════════════════════════════════════════════════════════════════════

class TestStrandBehavior:
    """Pin strand behaviour in the new interval API.

    Default (same_strand=False): strand is ignored, everything matches
    on coordinates alone.

    same_strand=True: only {+,+} and {-,-} match.  "." vs "." does NOT
    match (bedtools -s convention).
    """

    def test_overlap_opposite_strands_default(self):
        """Default: + vs - on overlapping coordinates match."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200], "strand": ["+"]})
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [250], "strand": ["-"]})
        assert list(overlaps(a, b)) == [True]

    def test_overlap_dot_vs_dot_default(self):
        """Default: . vs . match (strand ignored)."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200], "strand": ["."]})
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [250], "strand": ["."]})
        assert list(overlaps(a, b)) == [True]

    def test_overlap_dot_vs_dot_same_strand(self):
        """same_strand=True: . vs . does NOT match."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200], "strand": ["."]})
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [250], "strand": ["."]})
        assert list(overlaps(a, b, same_strand=True)) == [False]

    def test_overlap_same_strand_plus(self):
        """same_strand=True: + vs + matches."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200], "strand": ["+"]})
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [250], "strand": ["+"]})
        assert list(overlaps(a, b, same_strand=True)) == [True]

    def test_overlap_opposite_strands_same_strand(self):
        """same_strand=True: + vs - does NOT match."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200], "strand": ["+"]})
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [250], "strand": ["-"]})
        assert list(overlaps(a, b, same_strand=True)) == [False]

    def test_merge_ignores_strand(self):
        """+ and - regions that overlap coordinately merge."""
        rdf = _rdf({
            "contig": ["chr1", "chr1"],
            "start": [100, 150],
            "stop": [200, 300],
            "strand": ["+", "-"],
        })
        merged = merge(rdf)
        assert len(merged) == 1

    def test_anti_ignores_strand(self):
        """+ regions are removed by - blacklist if coordinates overlap."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200], "strand": ["+"]})
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [250], "strand": ["-"]})
        result = overlap_indices(a, b, how="anti")
        assert len(result) == 0


# ═══════════════════════════════════════════════════════════════════════
# cluster
# ═══════════════════════════════════════════════════════════════════════

class TestClusterCornerCases:
    def test_single_frame(self):
        rdf = _rdf({"contig": ["chr1","chr1","chr1"], "start": [100,150,400], "stop": [200,300,500]})
        labels = cluster(rdf)
        assert labels.iloc[0] == labels.iloc[1]
        assert labels.iloc[0] != labels.iloc[2]

    def test_two_frame_transitive(self):
        """A and C don't overlap directly, but both overlap B."""
        a = _rdf({"contig": ["chr1","chr1"], "start": [100,400], "stop": [200,500]})
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [450]})
        labels = cluster(a, b)
        assert labels.iloc[0] == labels.iloc[1]

    def test_book_ended_not_clustered(self):
        rdf = _rdf({"contig": ["chr1","chr1"], "start": [0,10], "stop": [10,20]})
        labels = cluster(rdf)
        assert labels.iloc[0] != labels.iloc[1]


# ═══════════════════════════════════════════════════════════════════════
# nearest
# ═══════════════════════════════════════════════════════════════════════

class TestNearestCornerCases:
    def test_basic_nearest(self):
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})
        b = _rdf({"contig": ["chr1"], "start": [300], "stop": [400]})
        result = nearest(a, b)
        assert len(result) == 1
        assert int(result["distance"].iloc[0]) == 100

    def test_ref_mismatch_raises(self):
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]}, ref="hg38")
        b = _rdf({"contig": ["chr1"], "start": [300], "stop": [400]}, ref="hg19")
        with pytest.raises(ValueError, match="same reference"):
            nearest(a, b)


# ═══════════════════════════════════════════════════════════════════════
# Fraction thresholds
# ═══════════════════════════════════════════════════════════════════════

class TestFractionThresholds:
    def test_min_frac_a(self):
        """Only keep matches where >= 50% of A is covered."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})  # 100bp
        b = _rdf({"contig": ["chr1"], "start": [160], "stop": [300]})  # 40bp overlap
        result = overlap_indices(a, b, min_frac_a=0.5)
        assert len(result) == 0  # 40/100 = 0.4 < 0.5

        result2 = overlap_indices(a, b, min_frac_a=0.3)
        assert len(result2) == 1  # 40/100 = 0.4 >= 0.3

    def test_min_frac_b(self):
        """Only keep matches where >= 50% of B is covered."""
        a = _rdf({"contig": ["chr1"], "start": [100], "stop": [200]})  # 100bp
        b = _rdf({"contig": ["chr1"], "start": [180], "stop": [300]})  # 120bp, 20bp overlap
        result = overlap_indices(a, b, min_frac_b=0.5)
        assert len(result) == 0  # 20/120 < 0.5

        result2 = overlap_indices(a, b, min_frac_b=0.1)
        assert len(result2) == 1  # 20/120 >= 0.1


class TestNonDefaultIndex:
    """Position-based results must be independent of the input index.

    ``overlap_indices`` and ``nearest`` return 0-based positions, so any
    frame — default-indexed, filtered (non-contiguous), or with duplicate
    labels — produces the same ``a_pos``/``b_pos`` values.  These tests
    verify that non-contiguous indices do not raise or corrupt results.
    """

    def _pair(self, a_idx):
        a = RegionDataFrame(
            pd.DataFrame(
                {
                    "contig": ["chr1", "chr1"],
                    "start": [100, 5000],
                    "stop": [200, 5100],
                },
                index=a_idx,
            ),
            ref="hg38",
        )
        b = _rdf({"contig": ["chr1", "chr1"], "start": [100, 5000], "stop": [200, 5100]})
        return a, b

    def test_fraction_filter_survives_a_non_default_index(self):
        a, b = self._pair([10, 20])
        result = overlap_indices(a, b, how="inner", min_frac_a=0.5)
        assert len(result) == 2
        # Positions are always 0-based, regardless of the input index.
        assert sorted(int(i) for i in result["a_pos"]) == [0, 1]

    def test_wiggle_survives_a_non_default_index(self):
        a = RegionDataFrame(
            pd.DataFrame(
                {"contig": ["chr1"], "start": [100], "stop": [200]}, index=[77]
            ),
            ref="hg38",
        )
        b = RegionDataFrame(
            pd.DataFrame(
                {"contig": ["chr1"], "start": [210], "stop": [300]}, index=[88]
            ),
            ref="hg38",
        )
        result = overlap_indices(a, b, how="inner", wiggle=10)
        assert len(result) == 1
        assert int(result["a_pos"].iloc[0]) == 0  # position, not label
        # Gap-bridged pairs share no bases.
        assert int(result["overlap_bases"].iloc[0]) == 0

    def test_anti_returns_positions(self):
        """Callers filter with ``a.iloc[...]``, using 0-based positions."""
        a = RegionDataFrame(
            pd.DataFrame(
                {
                    "contig": ["chr1", "chr1", "chr1"],
                    "start": [100, 5000, 9000],
                    "stop": [200, 5100, 9100],
                },
                index=[10, 20, 30],
            ),
            ref="hg38",
        )
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [300]})
        result = overlap_indices(a, b, how="anti")
        # Rows at positions 1 and 2 do not overlap b.
        assert sorted(int(i) for i in result["a_pos"]) == [1, 2]


class TestReciprocalFraction:
    """bedtools `-r`: `min_frac_a` must hold against BOTH A and B.

    An earlier version computed `frac_a_ok & frac_b_ok` under `reciprocal`,
    which is exactly what the non-reciprocal branch already did whenever both
    thresholds were set, and a no-op otherwise — the flag could not change any
    result. `test_reciprocal_rejects_when_b_is_barely_covered` is the one that
    discriminates: it passes without `reciprocal` and fails with it.
    """

    def _a_inside_b(self):
        # A is 100bp, fully inside a 1000bp B: frac(A)=1.0, frac(B)=0.1
        a = _rdf({"contig": ["chr1"], "start": [1000], "stop": [1100]})
        b = _rdf({"contig": ["chr1"], "start": [500], "stop": [1500]})
        return a, b

    def test_min_frac_a_alone_accepts(self):
        a, b = self._a_inside_b()
        assert len(overlap_indices(a, b, min_frac_a=0.5)) == 1

    def test_reciprocal_rejects_when_b_is_barely_covered(self):
        a, b = self._a_inside_b()
        assert len(overlap_indices(a, b, min_frac_a=0.5, reciprocal=True)) == 0

    def test_reciprocal_accepts_when_both_sides_are_covered(self):
        a = _rdf({"contig": ["chr1"], "start": [1000], "stop": [1100]})
        b = _rdf({"contig": ["chr1"], "start": [1000], "stop": [1100]})
        assert len(overlap_indices(a, b, min_frac_a=0.5, reciprocal=True)) == 1

    def test_reciprocal_without_min_frac_a_raises(self):
        a, b = self._a_inside_b()
        with pytest.raises(ValueError, match="min_frac_a"):
            overlap_indices(a, b, reciprocal=True)


# ── label/position safety ────────────────────────────────────────────

def _labelled(rows, idx):
    return RegionDataFrame(pd.DataFrame(rows, index=idx), ref="hg38")


_A_ROWS = {
    "contig": ["chr1", "chr1", "chr1"],
    "start": [100, 5000, 9000],
    "stop": [200, 5100, 9100],
    "strand": ["+", "+", "-"],
}
_B_ROWS = {
    "contig": ["chr1", "chr1"],
    "start": [150, 9050],
    "stop": [300, 9200],
    "strand": ["+", "+"],
}

# Every public entry point, including the parameter paths that have their own
# internal lookups (wiggle, same_strand and the fraction filter each index
# back into the input frames separately).
_ENTRY_POINTS = {
    "overlap_indices_inner": lambda a, b: overlap_indices(a, b),
    "overlap_indices_anti": lambda a, b: overlap_indices(a, b, how="anti"),
    "overlap_indices_left": lambda a, b: overlap_indices(a, b, how="left"),
    "overlap_indices_wiggle": lambda a, b: overlap_indices(a, b, wiggle=50),
    "overlap_indices_same_strand": lambda a, b: overlap_indices(a, b, same_strand=True),
    "overlap_indices_min_frac": lambda a, b: overlap_indices(a, b, min_frac_a=0.3),
    "overlaps": lambda a, b: overlaps(a, b),
    "overlaps_wiggle": lambda a, b: overlaps(a, b, wiggle=50),
    "overlaps_same_strand": lambda a, b: overlaps(a, b, same_strand=True),
    "nearest": lambda a, b: nearest(a, b),
    "cluster_one_frame": lambda a, b: cluster(a),
    "cluster_two_frames": lambda a, b: cluster(a, b),
    "merge": lambda a, b: merge(a),
}


def _values_only(result):
    """Compare by values, ignoring the index labels themselves."""
    if isinstance(result, pd.Series):
        return list(result.values)
    if isinstance(result, pd.DataFrame):
        cols = [c for c in result.columns if c not in ("a_pos", "b_pos")]
        # a_pos/b_pos are positions — they should be identical regardless
        # of input index labels, so we still compare only the data columns.
        return [len(result)] + [list(result[c].values) for c in cols]
    return result


@pytest.mark.parametrize("name", sorted(_ENTRY_POINTS))
def test_entry_point_is_indifferent_to_index_labels(name):
    """Results must not depend on whether the index is 0..n-1 or arbitrary.

    `bioframe` returns index LABELS in its `index`/`index_` columns. Any
    internal lookup that treats those as POSITIONS breaks on frames produced
    by a filter or a slice — which is most real frames. This has been found
    four separate times in this module: the wiggle path, the fraction filter,
    `overlaps`, and `_strand_mask`.

    Two of those four raised IndexError, which is survivable. The other two
    returned a plausible WRONG ANSWER — `overlaps` handed back an all-False
    mask — which is not. Default-indexed fixtures cannot catch either, and
    `from_bed` produces a default index, so even the real-data manifest run
    exercises only the safe path.
    """
    fn = _ENTRY_POINTS[name]
    default = fn(_labelled(_A_ROWS, [0, 1, 2]), _labelled(_B_ROWS, [0, 1]))
    labelled = fn(_labelled(_A_ROWS, [10, 20, 30]), _labelled(_B_ROWS, [77, 88]))
    assert _values_only(default) == _values_only(labelled)


def test_overlaps_does_not_return_all_false_on_a_labelled_index():
    """The specific silent failure, pinned on its own.

    The parametrized test above would also catch this, but only by comparing
    against the default-index run. This asserts the true answer directly, so
    a future change that broke BOTH paths identically could not slip through.
    """
    a = _labelled(_A_ROWS, [10, 20, 30])
    b = _labelled({"contig": ["chr1"], "start": [150], "stop": [300]}, [0])
    mask = overlaps(a, b)
    assert list(mask.values) == [True, False, False]
    assert list(mask.index) == [10, 20, 30]


class TestDeterministicOrder:
    """Pair results must come back in a stable, sorted order.

    Row order is part of the contract by design: two implementations can agree
    on the set of pairs and still differ on ordering, which silently breaks any
    caller that zips or positionally indexes the result.

    Before this was enforced, order was not stable ACROSS PROCESSES. Three runs
    of identical code on identical input (964,593 CTCF regions vs the hg38
    blacklist) produced three different digests; pinning PYTHONHASHSEED made
    them identical, so something upstream iterates a set or dict ordered by
    Python's hash randomisation. It was stable *within* a process, which is
    what made it invisible to every in-process test.

    Cross-process instability cannot be asserted from inside one process, so
    these pin the sortedness that produces it instead.
    """

    def _many(self):
        # Enough rows, and deliberately not pre-sorted, so an unsorted
        # implementation is very unlikely to come back ordered by luck.
        starts = [900, 100, 500, 300, 700, 200, 800, 400, 600, 1000]
        a = _rdf({
            "contig": ["chr1"] * len(starts),
            "start": starts,
            "stop": [s + 50 for s in starts],
        })
        b = _rdf({
            "contig": ["chr1"] * 6,
            "start": [120, 320, 520, 720, 920, 1020],
            "stop": [140, 340, 540, 740, 940, 1040],
        })
        return a, b

    def test_overlap_indices_is_sorted(self):
        a, b = self._many()
        r = overlap_indices(a, b)
        assert len(r) > 1, "fixture must produce several pairs to be meaningful"
        keys = list(zip(r["a_pos"], r["b_pos"]))
        assert keys == sorted(keys)

    def test_nearest_is_sorted(self):
        a, b = self._many()
        r = nearest(a, b)
        assert len(r) > 1
        keys = list(zip(r["a_pos"], r["b_pos"]))
        assert keys == sorted(keys)

    def test_repeated_calls_agree(self):
        """Within a process this always held; it is the floor, not the ceiling."""
        a, b = self._many()
        first = overlap_indices(a, b)
        for _ in range(3):
            assert overlap_indices(a, b).equals(first)


class TestDuplicateIndexLabels:
    """Duplicate index labels must not leak between rows.

    `pd.concat` without `ignore_index=True` produces a frame with repeated
    labels, which is ordinary rather than exotic. `overlaps` previously built
    its mask with `a.index.isin(pairs.a_index)` -- label membership -- so when
    two rows shared a label and only one overlapped, BOTH came back True.
    """

    def _dup(self):
        # Rows 0 and 1 share the label 7. Only the first overlaps b.
        a = RegionDataFrame(
            pd.DataFrame(
                {
                    "contig": ["chr1", "chr1"],
                    "start": [100, 90000],
                    "stop": [200, 90100],
                },
                index=[7, 7],
            ),
            ref="hg38",
        )
        b = _rdf({"contig": ["chr1"], "start": [150], "stop": [160]})
        return a, b

    def test_overlaps_is_per_row_not_per_label(self):
        a, b = self._dup()
        m = overlaps(a, b)
        assert list(m.values) == [True, False]

    def test_overlaps_length_matches_rows(self):
        """A per-label result would collapse to one entry; a per-row one keeps two."""
        a, b = self._dup()
        assert len(overlaps(a, b)) == len(a) == 2

    def test_overlap_indices_identifies_the_matching_row(self):
        """With positions, duplicate labels no longer cause ambiguity.

        The old label-based API returned label 7 for the matching row, but
        ``a.loc[7]`` selected BOTH rows (since both carry label 7) — the
        caller could not tell which row actually overlapped.

        With ``a_pos`` returning 0-based positions, ``a.iloc[0]`` selects
        exactly the overlapping row. The known limitation is gone.
        """
        a, b = self._dup()
        pairs = overlap_indices(a, b)
        assert len(pairs) == 1
        assert list(pairs["a_pos"]) == [0]  # position 0 = first row
        # iloc with the position selects exactly the matching row.
        assert len(a.iloc[pairs["a_pos"].tolist()]) == 1


# ═══════════════════════════════════════════════════════════════════════
# Windowed frame — the case that motivated the label-to-position change
# ═══════════════════════════════════════════════════════════════════════

class TestWindowedFrameOverlap:
    """bin_regions_into_windows produces N windows per region, all sharing
    the parent region's index label.  The old label-based API could not say
    WHICH window matched — ``a.iloc[label]`` selected the wrong row.

    ``test_overlap_indices_identifies_correct_window`` is **discriminating**:
    under the old label-based API, ``a_index`` was the shared label ``0``,
    so ``windowed.iloc[0]`` pointed to the first window [0,250) rather than
    the matching second window [250,500).  With ``a_pos=1``,
    ``windowed.iloc[1]`` correctly identifies the overlapping window.
    """

    def _windowed(self):
        """Two 1000bp regions → 8 × 250bp windows, index [0,0,0,0,1,1,1,1]."""
        rdf = _rdf({
            "contig": ["chr1", "chr1"],
            "start": [0, 10000],
            "stop": [1000, 11000],
        })
        return rdf.bin_regions_into_windows(250, "valid", stride=250)

    def test_overlap_indices_identifies_correct_window(self):
        windowed = self._windowed()
        assert len(windowed) == 8

        # Blacklist hits only the 2nd window of region 0 ([250,500)).
        bl = _rdf({"contig": ["chr1"], "start": [300], "stop": [400]})

        result = overlap_indices(windowed, bl)
        assert len(result) == 1

        # Position uniquely identifies which window matched.
        pos = int(result["a_pos"].iloc[0])
        row = windowed.iloc[pos]
        assert int(row["start"]) == 250
        assert int(row["stop"]) == 500

    def test_overlaps_mask_is_per_window(self):
        """overlaps() gives a per-window boolean, not per-parent-region."""
        windowed = self._windowed()
        bl = _rdf({"contig": ["chr1"], "start": [300], "stop": [400]})
        mask = overlaps(windowed, bl)

        # Exactly 1 of 8 windows overlaps.
        assert mask.sum() == 1
        assert list(mask.values) == [
            False, True, False, False,
            False, False, False, False,
        ]

    def test_anti_join_drops_only_matching_window(self):
        """Anti-join on a windowed frame drops the one overlapping window."""
        windowed = self._windowed()
        bl = _rdf({"contig": ["chr1"], "start": [300], "stop": [400]})
        result = overlap_indices(windowed, bl, how="anti")

        # 7 of 8 windows survive.
        assert len(result) == 7
        kept_starts = sorted(
            int(windowed.iloc[int(p)]["start"]) for p in result["a_pos"]
        )
        # The dropped window was [250,500); all others present.
        assert 250 not in kept_starts
        assert len(kept_starts) == 7


class TestAttachBlacklistOnWindowedFrame:
    """attach_blacklist_regions must annotate per ROW, not per index label.

    `bin_regions_into_windows` emits N windows per region, all carrying the
    parent region's label. The previous implementation mapped positions back
    to labels and `join`ed on them, which smeared the annotation across every
    window sharing a parent: 8 windows with one real overlap produced FOUR
    annotated rows, three of which do not touch the blacklist at all.

    Wrong in the permissive direction -- it marks clean regions as
    blacklisted -- which is the worst way for a blacklist to be wrong.
    """

    def _windowed_and_bed(self, tmp_path):
        rdf = _rdf({
            "contig": ["chr1", "chr1"],
            "start": [0, 10000],
            "stop": [1000, 11000],
        })
        windowed = rdf.bin_regions_into_windows(250, "valid", stride=250)
        bed = tmp_path / "bl.bed"
        bed.write_text("chr1\t300\t400\n")
        return windowed, str(bed)

    def test_only_the_overlapping_window_is_annotated(self, tmp_path):
        windowed, bed = self._windowed_and_bed(tmp_path)
        assert list(windowed.index) == [0, 0, 0, 0, 1, 1, 1, 1], (
            "fixture must have duplicate labels or this proves nothing"
        )

        out = windowed.attach_blacklist_regions(bed)

        annotated = [bool(v) and v != "" for v in out["blacklist_regions"]]
        assert annotated == [
            False, True, False, False,
            False, False, False, False,
        ]

    def test_row_count_is_preserved(self, tmp_path):
        """A label join on duplicate labels can also change the row count."""
        windowed, bed = self._windowed_and_bed(tmp_path)
        out = windowed.attach_blacklist_regions(bed)
        assert len(out) == len(windowed) == 8

    def test_the_annotation_names_the_right_interval(self, tmp_path):
        windowed, bed = self._windowed_and_bed(tmp_path)
        out = windowed.attach_blacklist_regions(bed)
        hit = out.iloc[1]["blacklist_regions"]
        assert len(hit) == 1
        assert int(hit[0].start) == 300
        assert int(hit[0].stop) == 400
