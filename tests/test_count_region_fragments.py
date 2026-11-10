"""Tests for scripts/band_model/count_region_fragments.py.

Covers shard partitioning (including failure modes), the MAPQ / dedup filter
pipeline, and the midpoint-in-region counting rule.  These tests use only
synthetic data and require no external files.
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "band_model"))
from count_region_fragments import (
    partition_samples,
    filter_and_dedup,
    count_midpoints_in_regions,
)


# ═══════════════════════════════════════════════════════════════════════════
# Shard partitioning
# ═══════════════════════════════════════════════════════════════════════════

class TestPartitionSamples:
    """partition_samples must split deterministically and reject bad input."""

    def test_single_shard(self):
        samples = list(range(5))
        assert partition_samples(samples, 0, 1) == samples

    def test_two_shards_cover_all(self):
        samples = list(range(7))
        s0 = partition_samples(samples, 0, 2)
        s1 = partition_samples(samples, 1, 2)
        assert sorted(s0 + s1) == samples
        assert len(set(s0) & set(s1)) == 0  # disjoint

    def test_n_shards_equal_n_samples(self):
        """Each shard gets exactly one sample."""
        samples = ["a", "b", "c"]
        for i in range(3):
            assert partition_samples(samples, i, 3) == [samples[i]]

    def test_negative_shard_index(self):
        with pytest.raises(ValueError, match="out of range"):
            partition_samples([1, 2, 3], -1, 2)

    def test_shard_index_equals_count(self):
        with pytest.raises(ValueError, match="out of range"):
            partition_samples([1, 2, 3], 2, 2)

    def test_shard_index_exceeds_count(self):
        with pytest.raises(ValueError, match="out of range"):
            partition_samples([1, 2, 3], 5, 3)

    def test_shard_count_zero(self):
        with pytest.raises(ValueError, match="shard_count must be >= 1"):
            partition_samples([1, 2], 0, 0)

    def test_shard_count_exceeds_samples(self):
        with pytest.raises(ValueError, match="exceeds number of samples"):
            partition_samples([1, 2], 0, 5)

    def test_deterministic(self):
        """Same input always yields the same partition."""
        samples = list(range(10))
        a = partition_samples(samples, 2, 4)
        b = partition_samples(samples, 2, 4)
        assert a == b


# ═══════════════════════════════════════════════════════════════════════════
# MAPQ filter + dedup
# ═══════════════════════════════════════════════════════════════════════════

class TestFilterAndDedup:
    """Tests for the combined MAPQ filter + coordinate dedup pipeline.

    The MAPQ comparison is inclusive (``>=``), matching the background-model
    store's ``from_fragments_h5`` (``mapq_vals >= min_mapq``).
    """

    def _make_data(self, starts, stops, mapq_pairs):
        """Helper: build arrays from Python lists."""
        return (
            np.array(starts, dtype=np.int32),
            np.array(stops, dtype=np.int32),
            np.array(mapq_pairs, dtype=np.int32),
        )

    def test_mapq10_boundary_inclusive(self):
        """--min-mapq 10 ADMITS MAPQ exactly 10 and REJECTS 9 (>= semantics)."""
        starts, stops, mapqs = self._make_data(
            [100, 200, 300],
            [150, 250, 350],
            [[10, 10], [9, 9], [9, 60]],  # ==10 admit, ==9 reject, min=9 reject
        )
        s, e = filter_and_dedup(starts, stops, mapqs, min_mapq=10)
        # Only the exactly-10 fragment survives.
        assert len(s) == 1
        assert s[0] == 100
        assert e[0] == 150

    def test_mapq_is_ge_not_gt(self):
        """min_mapq=30 admits 30 (a `> 30` implementation would drop it)."""
        starts, stops, mapqs = self._make_data(
            [100, 200],
            [150, 250],
            [[30, 30], [29, 60]],  # ==30 admit, min=29 reject
        )
        s, e = filter_and_dedup(starts, stops, mapqs, min_mapq=30)
        assert len(s) == 1
        assert s[0] == 100

    def test_mapq_min_of_pair(self):
        """Filter uses min of the two paired-end MAPQs."""
        starts, stops, mapqs = self._make_data(
            [100, 200],
            [150, 250],
            [[60, 9], [10, 40]],  # min=9 (fail at 10), min=10 (pass at 10)
        )
        s, e = filter_and_dedup(starts, stops, mapqs, min_mapq=10)
        assert len(s) == 1
        assert s[0] == 200

    def test_dedup_by_start_stop(self):
        """Fragments with identical (start, stop) collapse to one."""
        starts, stops, mapqs = self._make_data(
            [100, 100, 200],
            [150, 150, 250],
            [[40, 40], [40, 40], [40, 40]],
        )
        s, e = filter_and_dedup(starts, stops, mapqs, min_mapq=10)
        assert len(s) == 2  # two unique (start, stop) pairs

    def test_dedup_preserves_distinct(self):
        """Fragments with same start but different stop are not deduped."""
        starts, stops, mapqs = self._make_data(
            [100, 100],
            [150, 160],
            [[40, 40], [40, 40]],
        )
        s, e = filter_and_dedup(starts, stops, mapqs, min_mapq=10)
        assert len(s) == 2

    def test_all_filtered(self):
        """When every fragment fails MAPQ, return empty arrays."""
        starts, stops, mapqs = self._make_data(
            [100, 200], [150, 250], [[5, 5], [9, 9]],
        )
        s, e = filter_and_dedup(starts, stops, mapqs, min_mapq=10)
        assert len(s) == 0
        assert len(e) == 0

    def test_empty_input(self):
        """Empty input produces empty output without error."""
        starts = np.array([], dtype=np.int32)
        stops = np.array([], dtype=np.int32)
        mapqs = np.zeros((0, 2), dtype=np.int32)
        s, e = filter_and_dedup(starts, stops, mapqs, min_mapq=10)
        assert len(s) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Midpoint-in-region counting
# ═══════════════════════════════════════════════════════════════════════════

class TestCountMidpointsInRegions:
    """Tests for the searchsorted-based midpoint counting."""

    def test_basic_counting(self):
        """Midpoints at known positions counted into known regions."""
        midpoints = np.array([5, 15, 25, 35, 45], dtype=np.int64)
        starts = np.array([0, 10, 20, 30, 40], dtype=np.int64)
        stops = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        counts = count_midpoints_in_regions(midpoints, starts, stops)
        np.testing.assert_array_equal(counts, [1, 1, 1, 1, 1])

    def test_half_open_right_boundary(self):
        """A midpoint exactly at region_stop is NOT counted (half-open)."""
        midpoints = np.array([10, 20], dtype=np.int64)
        starts = np.array([0, 10], dtype=np.int64)
        stops = np.array([10, 20], dtype=np.int64)
        counts = count_midpoints_in_regions(midpoints, starts, stops)
        # midpoint 10 is NOT in [0,10) but IS in [10,20)
        # midpoint 20 is NOT in [10,20)
        np.testing.assert_array_equal(counts, [0, 1])

    def test_half_open_left_boundary(self):
        """A midpoint exactly at region_start IS counted (half-open)."""
        midpoints = np.array([0, 10], dtype=np.int64)
        starts = np.array([0, 10], dtype=np.int64)
        stops = np.array([10, 20], dtype=np.int64)
        counts = count_midpoints_in_regions(midpoints, starts, stops)
        np.testing.assert_array_equal(counts, [1, 1])

    def test_no_fragments(self):
        """Empty midpoints => zero counts everywhere."""
        midpoints = np.array([], dtype=np.int64)
        starts = np.array([0, 100], dtype=np.int64)
        stops = np.array([50, 200], dtype=np.int64)
        counts = count_midpoints_in_regions(midpoints, starts, stops)
        np.testing.assert_array_equal(counts, [0, 0])

    def test_gap_between_regions(self):
        """Midpoints in gaps between regions are not counted."""
        midpoints = np.array([5, 15, 25], dtype=np.int64)
        starts = np.array([0, 20], dtype=np.int64)
        stops = np.array([10, 30], dtype=np.int64)
        counts = count_midpoints_in_regions(midpoints, starts, stops)
        # midpoint 15 falls in the gap [10, 20)
        np.testing.assert_array_equal(counts, [1, 1])

    def test_midpoint_definition(self):
        """Verify midpoint = (start+stop)//2 with integer division."""
        frag_starts = np.array([100, 100], dtype=np.int64)
        frag_stops = np.array([201, 200], dtype=np.int64)
        midpoints = (frag_starts + frag_stops) // 2
        assert midpoints[0] == 150  # (100+201)//2 = 150
        assert midpoints[1] == 150  # (100+200)//2 = 150

        region_starts = np.array([140, 150], dtype=np.int64)
        region_stops = np.array([150, 160], dtype=np.int64)
        counts = count_midpoints_in_regions(
            np.sort(midpoints), region_starts, region_stops,
        )
        np.testing.assert_array_equal(counts, [0, 2])

    def test_large_scale_consistency(self):
        """Brute-force vs searchsorted match on random data."""
        rng = np.random.default_rng(42)
        midpoints = np.sort(rng.integers(0, 1_000_000, size=50_000))
        starts = np.arange(0, 1_000_000, 1000, dtype=np.int64)
        stops = starts + 1000

        counts = count_midpoints_in_regions(midpoints, starts, stops)

        expected = np.array([
            np.sum((midpoints >= s) & (midpoints < e))
            for s, e in zip(starts, stops)
        ])
        np.testing.assert_array_equal(counts, expected)
