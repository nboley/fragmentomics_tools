"""Regression tests for gc field alignment through FragmentArray operations.

The gc field (per-fragment GC fraction) must be carried correctly through all
operations that change fragment count or order. These tests verify that:
1. gc values stay aligned with their corresponding fragments
2. gc=None continues to work (means "never fetched")
3. The validation catches misaligned gc arrays
"""
import numpy as np
import pytest

from fragmentomics_tools import FragmentArray, RegionFragmentArray
from fragmentomics_tools.region import Region


def make_fa_with_gc(starts, stops, gc_values, length=100, max_frag_len=511):
    """Helper to create a FragmentArray with specific gc values."""
    return FragmentArray(
        starts_0=np.array(starts, dtype=np.int32),
        stops_0=np.array(stops, dtype=np.int32),
        length=length,
        max_frag_len=max_frag_len,
        gc=np.array(gc_values, dtype=np.float32) if gc_values is not None else None,
    )


def make_rfa_with_gc(starts, stops, gc_values, region=None, max_frag_len=511):
    """Helper to create a RegionFragmentArray with specific gc values."""
    if region is None:
        region = Region("chr1", 0, 100)
    return RegionFragmentArray(
        starts_0=np.array(starts, dtype=np.int32),
        stops_0=np.array(stops, dtype=np.int32),
        region=region,
        max_frag_len=max_frag_len,
        gc=np.array(gc_values, dtype=np.float32) if gc_values is not None else None,
    )


class TestGcValidation:
    """Test that validate_data catches misaligned gc arrays."""

    def test_gc_length_mismatch_raises(self):
        """gc array with wrong length should raise ValueError."""
        with pytest.raises(ValueError, match="gc length"):
            FragmentArray(
                starts_0=np.array([0, 10, 20], dtype=np.int32),
                stops_0=np.array([5, 15, 25], dtype=np.int32),
                length=100,
                max_frag_len=511,
                gc=np.array([0.4, 0.5], dtype=np.float32),  # Wrong length: 2 != 3
            )

    def test_gc_none_is_valid(self):
        """gc=None should be valid (means "never fetched")."""
        fa = FragmentArray(
            starts_0=np.array([0, 10, 20], dtype=np.int32),
            stops_0=np.array([5, 15, 25], dtype=np.int32),
            length=100,
            max_frag_len=511,
            gc=None,
        )
        assert fa.gc is None

    def test_gc_correct_length_is_valid(self):
        """gc array with correct length should be valid."""
        gc_vals = np.array([0.3, 0.4, 0.5], dtype=np.float32)
        fa = FragmentArray(
            starts_0=np.array([0, 10, 20], dtype=np.int32),
            stops_0=np.array([5, 15, 25], dtype=np.int32),
            length=100,
            max_frag_len=511,
            gc=gc_vals,
        )
        np.testing.assert_array_equal(fa.gc, gc_vals)


class TestGcMask:
    """Test gc alignment through mask() operation."""

    def test_mask_with_gc_values_aligned(self):
        """After masking, gc values should correspond to surviving fragments."""
        # Fragments with unique gc values to track alignment
        fa = make_fa_with_gc(
            starts=[0, 10, 20, 30],
            stops=[5, 15, 25, 35],
            gc_values=[0.1, 0.2, 0.3, 0.4],
        )
        # Keep fragments 0 and 2
        mask = np.array([True, False, True, False])
        result = fa.mask(mask)

        assert result.n_frags == 2
        np.testing.assert_array_equal(result.starts_0, [0, 20])
        np.testing.assert_array_equal(result.stops_0, [5, 25])
        # gc values must match the surviving fragments
        np.testing.assert_array_almost_equal(result.gc, [0.1, 0.3])

    def test_mask_with_gc_none(self):
        """Masking should preserve gc=None."""
        fa = make_fa_with_gc(
            starts=[0, 10, 20, 30],
            stops=[5, 15, 25, 35],
            gc_values=None,
        )
        mask = np.array([True, False, True, False])
        result = fa.mask(mask)

        assert result.gc is None
        assert result.n_frags == 2

    def test_mask_with_indices_gc_aligned(self):
        """Mask with index array should align gc correctly."""
        fa = make_fa_with_gc(
            starts=[0, 10, 20, 30],
            stops=[5, 15, 25, 35],
            gc_values=[0.1, 0.2, 0.3, 0.4],
        )
        # Select fragments in different order
        indices = np.array([3, 1])
        result = fa.mask(indices)

        assert result.n_frags == 2
        np.testing.assert_array_equal(result.starts_0, [30, 10])
        np.testing.assert_array_equal(result.stops_0, [35, 15])
        np.testing.assert_array_almost_equal(result.gc, [0.4, 0.2])


class TestGcSubset:
    """Test gc alignment through subset() operation."""

    def test_subset_with_gc_values_aligned(self):
        """After subsetting, gc values should correspond to selected fragments."""
        fa = make_fa_with_gc(
            starts=[0, 10, 20, 30, 40],
            stops=[5, 15, 25, 35, 45],
            gc_values=[0.1, 0.2, 0.3, 0.4, 0.5],
        )
        indices = np.array([4, 2, 0])
        result = fa.subset(indices)

        assert result.n_frags == 3
        np.testing.assert_array_equal(result.starts_0, [40, 20, 0])
        np.testing.assert_array_almost_equal(result.gc, [0.5, 0.3, 0.1])

    def test_subset_with_gc_none(self):
        """Subsetting should preserve gc=None."""
        fa = make_fa_with_gc(
            starts=[0, 10, 20],
            stops=[5, 15, 25],
            gc_values=None,
        )
        result = fa.subset(np.array([2, 0]))

        assert result.gc is None


class TestGcDropDuplicateFragments:
    """Test gc alignment through drop_duplicate_fragments() operation."""

    def test_drop_duplicates_gc_aligned(self):
        """After dropping duplicates, gc should match surviving fragments."""
        # Fragments 0 and 2 are duplicates (same start/stop)
        fa = make_fa_with_gc(
            starts=[0, 10, 0, 30],
            stops=[5, 15, 5, 35],
            gc_values=[0.1, 0.2, 0.3, 0.4],
        )
        result = fa.drop_duplicate_fragments()

        # Should keep 3 unique fragments
        assert result.n_frags == 3
        # gc values should match the kept fragments
        assert len(result.gc) == 3
        # The unique indices returned by np.unique keep the first occurrence
        # So we expect fragments at positions 0, 1, 3 to be kept
        # Note: np.unique sorts, so order may differ
        # Just verify length matches and values are from original set
        for gc_val in result.gc:
            assert gc_val in [0.1, 0.2, 0.3, 0.4]

    def test_drop_duplicates_gc_none(self):
        """Dropping duplicates should preserve gc=None."""
        fa = make_fa_with_gc(
            starts=[0, 10, 0, 30],
            stops=[5, 15, 5, 35],
            gc_values=None,
        )
        result = fa.drop_duplicate_fragments()

        assert result.gc is None


class TestGcSubsetFragmentLengths:
    """Test gc alignment through subset_fragment_lengths() operation."""

    def test_subset_fragment_lengths_gc_aligned(self):
        """After filtering by fragment length, gc should match surviving fragments."""
        # Fragments with different lengths
        fa = make_fa_with_gc(
            starts=[0, 10, 20, 30],
            stops=[5, 30, 25, 80],  # lengths: 5, 20, 5, 50
            gc_values=[0.1, 0.2, 0.3, 0.4],
        )
        # Keep only fragments with length >= 10
        result = fa.subset_fragment_lengths(min_frag_len=10)

        assert result.n_frags == 2
        np.testing.assert_array_equal(result.starts_0, [10, 30])
        np.testing.assert_array_almost_equal(result.gc, [0.2, 0.4])

    def test_subset_fragment_lengths_gc_none(self):
        """Fragment length filtering should preserve gc=None."""
        fa = make_fa_with_gc(
            starts=[0, 10, 20, 30],
            stops=[5, 30, 25, 80],
            gc_values=None,
        )
        result = fa.subset_fragment_lengths(min_frag_len=10)

        assert result.gc is None


class TestGcReverseStrand:
    """Test gc alignment through reverse_strand() operation."""

    def test_reverse_strand_gc_reversed(self):
        """After reversing strand, gc order should be reversed to match fragments."""
        fa = make_fa_with_gc(
            starts=[0, 10, 20, 30],
            stops=[5, 15, 25, 35],
            gc_values=[0.1, 0.2, 0.3, 0.4],
        )
        result = fa.reverse_strand()

        # Fragment order is reversed
        assert result.n_frags == 4
        # Original starts [0,10,20,30] become new stops when flipped
        # After reversal, fragment order is reversed
        # GC values should be reversed to match the new fragment order
        np.testing.assert_array_almost_equal(result.gc, [0.4, 0.3, 0.2, 0.1])

    def test_reverse_strand_gc_none(self):
        """Reversing strand should preserve gc=None."""
        fa = make_fa_with_gc(
            starts=[0, 10, 20, 30],
            stops=[5, 15, 25, 35],
            gc_values=None,
        )
        result = fa.reverse_strand()

        assert result.gc is None

    def test_reverse_strand_gc_values_unchanged(self):
        """GC fraction values themselves shouldn't change (G↔C, A↔T are symmetric)."""
        fa = make_fa_with_gc(
            starts=[0, 10],
            stops=[5, 15],
            gc_values=[0.35, 0.65],
        )
        result = fa.reverse_strand()

        # Values are the same, just reversed order
        np.testing.assert_array_almost_equal(result.gc, [0.65, 0.35])


class TestGcConcatenation:
    """Test gc alignment through __add__ (concatenation) operation."""

    def test_add_both_have_gc(self):
        """When both arrays have gc, result should concatenate gc values."""
        fa1 = make_fa_with_gc(
            starts=[0, 10],
            stops=[5, 15],
            gc_values=[0.1, 0.2],
        )
        fa2 = make_fa_with_gc(
            starts=[20, 30],
            stops=[25, 35],
            gc_values=[0.3, 0.4],
        )
        result = fa1 + fa2

        assert result.n_frags == 4
        np.testing.assert_array_equal(result.starts_0, [0, 10, 20, 30])
        np.testing.assert_array_almost_equal(result.gc, [0.1, 0.2, 0.3, 0.4])

    def test_add_both_gc_none(self):
        """When both arrays have gc=None, result should have gc=None."""
        fa1 = make_fa_with_gc(starts=[0, 10], stops=[5, 15], gc_values=None)
        fa2 = make_fa_with_gc(starts=[20, 30], stops=[25, 35], gc_values=None)
        result = fa1 + fa2

        assert result.gc is None

    def test_add_mixed_gc_produces_none(self):
        """When one array has gc and other has None, result should be None.

        This is the documented behavior: we can't provide gc for all fragments
        in the merged result, so gc becomes unavailable rather than misaligned.
        """
        fa1 = make_fa_with_gc(
            starts=[0, 10],
            stops=[5, 15],
            gc_values=[0.1, 0.2],
        )
        fa2 = make_fa_with_gc(starts=[20, 30], stops=[25, 35], gc_values=None)
        result = fa1 + fa2

        # Mixed gc/None produces None
        assert result.gc is None

    def test_add_mixed_gc_reverse_order(self):
        """Same as above but with operands reversed."""
        fa1 = make_fa_with_gc(starts=[0, 10], stops=[5, 15], gc_values=None)
        fa2 = make_fa_with_gc(
            starts=[20, 30],
            stops=[25, 35],
            gc_values=[0.3, 0.4],
        )
        result = fa1 + fa2

        assert result.gc is None


class TestGcRegionFragmentArray:
    """Test gc alignment for RegionFragmentArray operations."""

    def test_rfa_mask_gc_aligned(self):
        """RegionFragmentArray mask should align gc correctly."""
        rfa = make_rfa_with_gc(
            starts=[0, 10, 20, 30],
            stops=[5, 15, 25, 35],
            gc_values=[0.1, 0.2, 0.3, 0.4],
        )
        result = rfa.mask(np.array([True, False, True, False]))

        assert result.n_frags == 2
        np.testing.assert_array_almost_equal(result.gc, [0.1, 0.3])

    def test_rfa_reverse_strand_gc_aligned(self):
        """RegionFragmentArray reverse_strand should align gc correctly."""
        rfa = make_rfa_with_gc(
            starts=[0, 10, 20],
            stops=[5, 15, 25],
            gc_values=[0.1, 0.2, 0.3],
            region=Region("chr1", 0, 100, "+"),
        )
        result = rfa.reverse_strand()

        np.testing.assert_array_almost_equal(result.gc, [0.3, 0.2, 0.1])

    def test_rfa_add_gc_aligned(self):
        """RegionFragmentArray concatenation should align gc correctly."""
        rfa1 = make_rfa_with_gc(
            starts=[0, 10],
            stops=[5, 15],
            gc_values=[0.1, 0.2],
        )
        rfa2 = make_rfa_with_gc(
            starts=[20, 30],
            stops=[25, 35],
            gc_values=[0.3, 0.4],
        )
        result = rfa1 + rfa2

        np.testing.assert_array_almost_equal(result.gc, [0.1, 0.2, 0.3, 0.4])


class TestGcChainedOperations:
    """Test gc alignment through chained operations."""

    def test_mask_then_reverse(self):
        """gc should stay aligned through mask followed by reverse_strand."""
        fa = make_fa_with_gc(
            starts=[0, 10, 20, 30],
            stops=[5, 15, 25, 35],
            gc_values=[0.1, 0.2, 0.3, 0.4],
        )
        # Keep fragments 1 and 3
        masked = fa.mask(np.array([False, True, False, True]))
        assert masked.n_frags == 2
        np.testing.assert_array_almost_equal(masked.gc, [0.2, 0.4])

        # Then reverse
        reversed_fa = masked.reverse_strand()
        np.testing.assert_array_almost_equal(reversed_fa.gc, [0.4, 0.2])

    def test_add_then_mask(self):
        """gc should stay aligned through add followed by mask."""
        fa1 = make_fa_with_gc(starts=[0, 10], stops=[5, 15], gc_values=[0.1, 0.2])
        fa2 = make_fa_with_gc(starts=[20, 30], stops=[25, 35], gc_values=[0.3, 0.4])

        combined = fa1 + fa2
        np.testing.assert_array_almost_equal(combined.gc, [0.1, 0.2, 0.3, 0.4])

        # Keep fragments 0 and 3
        masked = combined.mask(np.array([True, False, False, True]))
        np.testing.assert_array_almost_equal(masked.gc, [0.1, 0.4])

    def test_subset_fragment_lengths_then_reverse(self):
        """gc should stay aligned through length filtering then reverse."""
        fa = make_fa_with_gc(
            starts=[0, 10, 20, 30],
            stops=[5, 30, 25, 80],  # lengths: 5, 20, 5, 50
            gc_values=[0.1, 0.2, 0.3, 0.4],
        )
        # Keep fragments with length >= 10 (indices 1 and 3)
        filtered = fa.subset_fragment_lengths(min_frag_len=10)
        np.testing.assert_array_almost_equal(filtered.gc, [0.2, 0.4])

        # Then reverse
        reversed_fa = filtered.reverse_strand()
        np.testing.assert_array_almost_equal(reversed_fa.gc, [0.4, 0.2])
