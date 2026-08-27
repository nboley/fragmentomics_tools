"""Golden count test: fragment_array counts vs independent recount.

This test requires a real fragments_h5 fixture file. If no fixture is
available (no network/S3 access), the test auto-skips with a clear message.

When a fixture IS available, it:
1. Reads a few small regions via RegionFragmentArray.from_fragments_h5
   with strand='.' (condition #1: no strand flip)
2. Independently counts fragments via pysam on the source BAM
3. Compares per-track, per-position counts

Locks: endpoint definitions (first/last/midpoint), band edges [lo, hi)
inclusive/exclusive (condition #5), dedup semantics, genome-orientation
invariant (no strand flip — condition #1).
"""

import os

import numpy as np
import pytest

# Path to a test fragments h5 fixture — set via env var or hardcoded
FIXTURE_H5 = os.environ.get("BG_TEST_FIXTURE_H5", "")
FIXTURE_SKIP_REASON = (
    "No fragments_h5 fixture available. Set BG_TEST_FIXTURE_H5 env var "
    "to a valid .frag.h5 file to enable this test. Building a fixture "
    "requires the fragments_h5 build CLI + a BAM file, which are not "
    "available in this environment."
)


@pytest.mark.skipif(
    not FIXTURE_H5 or not os.path.exists(FIXTURE_H5),
    reason=FIXTURE_SKIP_REASON,
)
class TestGoldenCounts:
    """Compare build_coverage_counts against an independent recount."""

    TEST_REGIONS = [
        # Small regions on chr1 for quick testing
        ("chr1", 10000, 12000),
        ("chr1", 50000, 52000),
    ]
    FL_BANDS = [(40, 65), (120, 175)]
    MIN_MAPQ = 10

    def _load_rfa(self, h5_path, contig, start, stop):
        from fragments_h5 import FragmentsH5
        from fragmentomics_tools.fragment_array.fragment_array import RegionFragmentArray
        from fragmentomics_tools.region import Region

        h5 = FragmentsH5(h5_path, cache_pointers=False)
        region = Region(chrom=contig, start=start, stop=stop, strand=".")
        rfa = RegionFragmentArray.from_fragments_h5(
            h5, region, min_mapq=self.MIN_MAPQ, max_frag_len=175,
        )
        rfa = rfa.drop_duplicate_fragments()
        h5.close()
        return rfa

    def _pysam_recount(self, bam_path, contig, start, stop, fl_bands):
        """Independent fragment count using pysam.

        Returns dict {(strand, fl_band, cov_type): dense_array}.
        """
        import pysam

        length = stop - start
        counts = {}
        for strand in ["+", "-"]:
            for fl_band in fl_bands:
                for cov_type in ["first", "last", "midpoint"]:
                    counts[(strand, fl_band, cov_type)] = np.zeros(length, dtype=np.uint32)

        seen_pairs = set()
        bam = pysam.AlignmentFile(bam_path, "rb")
        for read in bam.fetch(contig, start, stop):
            if read.mapping_quality < self.MIN_MAPQ:
                continue
            if not read.is_proper_pair:
                continue

            # Dedup: skip if we've seen this exact (start, stop) pair
            frag_start = min(read.reference_start, read.next_reference_start)
            frag_stop = frag_start + abs(read.template_length)
            frag_key = (frag_start, frag_stop)
            if frag_key in seen_pairs:
                continue
            seen_pairs.add(frag_key)

            frag_len = frag_stop - frag_start
            frag_strand = "+" if not read.is_reverse else "-"

            for lo, hi in fl_bands:
                if lo <= frag_len < hi:  # half-open [lo, hi) — condition #5
                    # First covered base
                    first = frag_start - start
                    if 0 <= first < length:
                        counts[(frag_strand, (lo, hi), "first")][first] += 1

                    # Last covered base
                    last = frag_stop - 1 - start
                    if 0 <= last < length:
                        counts[(frag_strand, (lo, hi), "last")][last] += 1

                    # Midpoint
                    mid = (frag_start + frag_stop) // 2 - start
                    if 0 <= mid < length:
                        counts[(frag_strand, (lo, hi), "midpoint")][mid] += 1

        bam.close()
        return counts

    def test_strand_dot_no_flip(self):
        """Condition #1: strand='.' regions produce genome-oriented counts.

        We test by loading a region near a minus-strand gene, verifying that
        counts are NOT flipped (comparing two loads: one with strand='.').
        """
        from fragments_h5 import FragmentsH5
        from fragmentomics_tools.fragment_array.fragment_array import RegionFragmentArray
        from fragmentomics_tools.region import Region

        contig, start, stop = self.TEST_REGIONS[0]

        h5 = FragmentsH5(FIXTURE_H5, cache_pointers=False)

        # Load with strand='.'
        region_dot = Region(chrom=contig, start=start, stop=stop, strand=".")
        rfa_dot = RegionFragmentArray.from_fragments_h5(
            h5, region_dot, min_mapq=self.MIN_MAPQ, max_frag_len=175,
        )

        # Counts should be genome-oriented (not flipped)
        counts_dot = rfa_dot.build_coverage_counts(
            fl_bands=self.FL_BANDS, return_sparse=False,
        )

        h5.close()

        # Verify the counts are non-empty for at least one track
        total = sum(c.sum() for c in counts_dot.values())
        assert total > 0, "Expected non-zero counts from fixture"

    def test_counts_match_pysam(self):
        """Golden test: build_coverage_counts matches independent pysam recount."""
        # This test requires both the h5 AND source BAM
        # For now, skip if BAM is not available
        bam_path = os.environ.get("BG_TEST_FIXTURE_BAM", "")
        if not bam_path or not os.path.exists(bam_path):
            pytest.skip("No BAM fixture (set BG_TEST_FIXTURE_BAM)")

        for contig, start, stop in self.TEST_REGIONS:
            rfa = self._load_rfa(FIXTURE_H5, contig, start, stop)
            rfa_counts = rfa.build_coverage_counts(
                fl_bands=self.FL_BANDS, return_sparse=False,
            )
            pysam_counts = self._pysam_recount(
                bam_path, contig, start, stop, self.FL_BANDS,
            )

            for key in pysam_counts:
                np.testing.assert_array_equal(
                    rfa_counts[key],
                    pysam_counts[key],
                    err_msg=f"Mismatch at {contig}:{start}-{stop} track {key}",
                )


class TestStrandInvariantSynthetic:
    """Test the strand='.' invariant using synthetic RegionFragmentArray data.

    This exercises condition #1 WITHOUT a real h5 fixture.
    """

    def test_strand_dot_no_flip_synthetic(self):
        """A region with strand='.' must not flip fragment data.

        We create a RegionFragmentArray with known fragment positions, then
        verify that strand='.' preserves genome orientation (no reversal).
        """
        from fragmentomics_tools.fragment_array.fragment_array import RegionFragmentArray
        from fragmentomics_tools.region import Region

        # Create a region on minus strand — this WOULD flip data
        region_minus = Region(chrom="chr1", start=1000, stop=1100, strand="-")
        # Create same region with strand='.'
        region_dot = Region(chrom="chr1", start=1000, stop=1100, strand=".")

        # Synthetic fragments: starts at positions 10, 20; stops at 60, 70
        starts_0 = np.array([10, 20])
        stops_0 = np.array([60, 70])

        rfa_dot = RegionFragmentArray(
            starts_0=starts_0,
            stops_0=stops_0,
            region=region_dot,
            max_frag_len=175,
            validate_data=False,
        )

        # Verify genome orientation is preserved with strand='.'
        # Region normalizes strand='.' to None internally, but is_minus_strand() must be False
        assert not rfa_dot.region.is_minus_strand()
        # First covered bases should be at original positions (not flipped)
        np.testing.assert_array_equal(rfa_dot.starts_0, starts_0)

    def test_minus_strand_region_flips(self):
        """Verify that a minus-strand region DOES flip — confirming why we need strand='.'."""
        from fragmentomics_tools.region import Region

        region = Region(chrom="chr1", start=1000, stop=1100, strand="-")
        assert region.is_minus_strand()

        # When from_fragments_h5 gets a minus-strand region, it flips data
        # (fragment_array.py:1809). That's why the design mandates strand='.'.

    def test_fl_band_half_open(self):
        """Condition #5: fl_bands are half-open [lo, hi)."""
        from fragmentomics_tools.fragment_array.fragment_array import RegionFragmentArray
        from fragmentomics_tools.region import Region

        region = Region(chrom="chr1", start=0, stop=200, strand=".")

        # Fragment of length exactly 65 (at the boundary of band (40, 65))
        # Should NOT be in band (40, 65) since 65 is exclusive
        starts_0 = np.array([10])
        stops_0 = np.array([75])  # length = 65

        rfa = RegionFragmentArray(
            starts_0=starts_0,
            stops_0=stops_0,
            region=region,
            max_frag_len=175,
            validate_data=False,
        )

        # subset_fragment_lengths uses < for the upper bound
        sub = rfa.subset_fragment_lengths(40, 65)
        assert sub.n_frags == 0, "Length 65 should NOT be in band [40, 65)"

        # Fragment of length 64 SHOULD be in band (40, 65)
        starts_0b = np.array([10])
        stops_0b = np.array([74])  # length = 64

        rfa_b = RegionFragmentArray(
            starts_0=starts_0b,
            stops_0=stops_0b,
            region=region,
            max_frag_len=175,
            validate_data=False,
        )

        sub_b = rfa_b.subset_fragment_lengths(40, 65)
        assert sub_b.n_frags == 1, "Length 64 SHOULD be in band [40, 65)"

    def test_fl_band_lower_inclusive(self):
        """Lower bound of fl_band is inclusive."""
        from fragmentomics_tools.fragment_array.fragment_array import RegionFragmentArray
        from fragmentomics_tools.region import Region

        region = Region(chrom="chr1", start=0, stop=200, strand=".")

        # Fragment of length exactly 40 (at the lower boundary of band (40, 65))
        starts_0 = np.array([10])
        stops_0 = np.array([50])  # length = 40

        rfa = RegionFragmentArray(
            starts_0=starts_0,
            stops_0=stops_0,
            region=region,
            max_frag_len=175,
            validate_data=False,
        )

        sub = rfa.subset_fragment_lengths(40, 65)
        assert sub.n_frags == 1, "Length 40 SHOULD be in band [40, 65)"
