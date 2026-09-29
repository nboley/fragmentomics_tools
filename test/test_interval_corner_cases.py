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
        assert int(result["a_index"].iloc[0]) == 1


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
        assert int(result["a_index"].iloc[0]) == 1

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
