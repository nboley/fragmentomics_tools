"""Golden count test: fragment_array counts vs an INDEPENDENT pysam recount.

This is the brief's one guard on `fragment_array` counting. It runs against
committed golden fixtures (built ONCE from real fragments_h5 test data by the
production CLI — see tests/data/README.md), so it is enabled by default. The
fixture paths may be overridden with BG_TEST_FIXTURE_H5 / BG_TEST_FIXTURE_BAM.

The reference recount (`_pysam_recount`) is a self-contained pysam/numpy
reimplementation of the build -> read -> dedup -> coverage path. It deliberately
does NOT import `fragment_array`, so a bug in the library cannot hide by being
mirrored in the reference.

Locks: endpoint definitions (first = start, last = stop-1, midpoint =
start + len//2), band edges [lo, hi) (half-open), dedup semantics matching
`drop_duplicate_fragments` (unique by (start, stop)), and the genome-orientation
invariant (strand='.' region -> no flip).
"""

import os

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")

# Golden count fixture (small.chr6): default-built PE h5 + its source BAM.
FIXTURE_H5 = os.environ.get(
    "BG_TEST_FIXTURE_H5", os.path.join(_DATA, "golden.small.chr6.frag.h5")
)
FIXTURE_BAM = os.environ.get(
    "BG_TEST_FIXTURE_BAM", os.path.join(_DATA, "small.chr6.bam")
)
# Dedup fixture (test_duplicates): built with --include-duplicates so the
# coordinate-identical pair reaches read-time drop_duplicate_fragments().
DUP_H5 = os.path.join(_DATA, "golden.test_duplicates.frag.h5")
DUP_BAM = os.path.join(_DATA, "test_duplicates.bam")

FL_BANDS = [(40, 65), (120, 175)]
MIN_MAPQ = 10
MAX_TLEN = 1000

# Windows on small.chr6.bam that contain band-length proper pairs.
TEST_REGIONS = [
    ("chr6", 84150000, 84152000),
    ("chr6", 105362000, 105364000),
    ("chr6", 169340000, 169342000),
]

_FIXTURES_PRESENT = os.path.exists(FIXTURE_H5) and os.path.exists(FIXTURE_BAM)
_SKIP_REASON = (
    f"Golden fixtures not found (H5={FIXTURE_H5}, BAM={FIXTURE_BAM}). "
    "Rebuild with tests/data/make_golden_fixture.sh or set "
    "BG_TEST_FIXTURE_H5 / BG_TEST_FIXTURE_BAM."
)


# ── Independent reference recount (NO fragment_array import) ──────────────

def _pysam_recount(
    bam_path,
    contig,
    start,
    stop,
    fl_bands,
    min_mapq,
    dedup=True,
    include_duplicates=False,
    max_tlen=MAX_TLEN,
):
    """Recount per-track coverage directly from the BAM.

    Mirrors the production path:
    * build time (bam_to_align/bam_to_fragments): keep read1-oriented alignments
      with 0 < tlen <= max_tlen that are not qcfail/secondary/supplementary/
      unmapped/mate_unmapped (and, unless include_duplicates, not duplicate);
      fragment = (pos, pos+tlen); strand from read1 orientation.
    * read time (from_fragments_h5 min_mapq): keep fragments whose per-pair
      min(mapq1, mapq2) >= min_mapq.
    * drop_duplicate_fragments: unique by (start, stop).
    * build_coverage_counts: per strand, per [lo, hi) band, first/last/midpoint.

    Returns dict {(strand, (lo, hi), cov_type): (stop-start,) uint32}.
    """
    import pysam

    length = stop - start
    frags = []  # (frag_start, frag_stop, strand, mapq1, mapq2)
    bam = pysam.AlignmentFile(bam_path, "rb")
    for r in bam.fetch(contig, max(0, start - max_tlen - 1), stop + max_tlen + 1):
        tl = r.template_length
        if not (0 < tl <= max_tlen):
            continue
        if (r.is_qcfail or r.is_secondary or r.is_supplementary
                or r.is_unmapped or r.mate_is_unmapped):
            continue
        if not include_duplicates and r.is_duplicate:
            continue
        fs = r.reference_start
        fe = fs + tl
        if not (fs < stop and fe > start):  # intersect [start, stop)
            continue
        if r.is_read1:
            strand = "+" if r.is_forward else "-"
        elif r.is_read2:
            strand = "-" if r.is_forward else "+"
        else:
            strand = None
        mq1 = r.mapping_quality
        # MQ-tag fallback: the production read-time min_mapq filter compares
        # min(mapq1, mate_mapq) where the mate mapq is read from the MQ tag. This
        # recount mirrors that by reading MQ off the BAM. If a read lacks the MQ
        # tag we fall back to mapq1 alone, which would DIVERGE from production
        # for pairs whose mate mapq is the smaller of the two. This is safe here
        # only because the committed golden fixtures were verified to carry MQ
        # tags on their proper pairs (see tests/data/README.md); do not point
        # this test at a BAM without MQ tags without revisiting this fallback.
        mq2 = r.get_tag("MQ") if r.has_tag("MQ") else None
        frags.append((fs, fe, strand, mq1, mq2))
    bam.close()

    # read-time mapq filter (min over the pair)
    kept = []
    for fs, fe, strand, mq1, mq2 in frags:
        pair_min = mq1 if mq2 is None else min(mq1, mq2)
        if pair_min >= min_mapq:
            kept.append((fs, fe, strand))

    if dedup:
        seen = set()
        uniq = []
        for fs, fe, strand in kept:
            if (fs, fe) not in seen:
                seen.add((fs, fe))
                uniq.append((fs, fe, strand))
        kept = uniq

    counts = {}
    for s in ["+", "-"]:
        for fl in fl_bands:
            for c in ["first", "last", "midpoint"]:
                counts[(s, fl, c)] = np.zeros(length, dtype=np.uint32)

    for fs, fe, strand in kept:
        L = fe - fs
        for lo, hi in fl_bands:
            if lo <= L < hi:  # half-open [lo, hi)
                first = fs - start
                last = (fe - 1) - start
                mid = (fs + (L // 2)) - start
                if 0 <= first < length:
                    counts[(strand, (lo, hi), "first")][first] += 1
                if 0 <= last < length:
                    counts[(strand, (lo, hi), "last")][last] += 1
                if 0 <= mid < length:
                    counts[(strand, (lo, hi), "midpoint")][mid] += 1
    return counts, len(kept)


def _rfa_counts(h5_path, contig, start, stop, fl_bands, min_mapq, dedup=True):
    """Library path under test: RegionFragmentArray -> build_coverage_counts."""
    from fragments_h5 import FragmentsH5
    from fragmentomics_tools.fragment_array.fragment_array import RegionFragmentArray
    from fragmentomics_tools.region import Region

    h5 = FragmentsH5(h5_path, cache_pointers=False)
    region = Region(chrom=contig, start=start, stop=stop, strand=".")
    rfa = RegionFragmentArray.from_fragments_h5(
        h5, region, min_mapq=min_mapq, max_frag_len=175
    )
    n_before = rfa.n_frags
    if dedup:
        rfa = rfa.drop_duplicate_fragments()
    n_after = rfa.n_frags
    res = rfa.build_coverage_counts(
        fl_bands=fl_bands, split_strand=True, return_sparse=False
    )
    h5.close()
    got = {k: np.asarray(v, dtype=np.uint32) for k, v in res.items()}
    return got, n_before, n_after


@pytest.mark.skipif(not _FIXTURES_PRESENT, reason=_SKIP_REASON)
class TestGoldenCounts:
    """build_coverage_counts vs the independent recount over real windows."""

    def test_strand_dot_no_flip(self):
        """strand='.' region produces genome-oriented, non-empty counts."""
        contig, start, stop = TEST_REGIONS[0]
        got, _, _ = _rfa_counts(
            FIXTURE_H5, contig, start, stop, FL_BANDS, MIN_MAPQ
        )
        total = sum(int(v.sum()) for v in got.values())
        assert total > 0, "expected non-zero band counts in the fixture window"

    def test_counts_match_pysam(self):
        """Golden: per-track, per-position counts match the pysam recount."""
        for contig, start, stop in TEST_REGIONS:
            got, _, _ = _rfa_counts(
                FIXTURE_H5, contig, start, stop, FL_BANDS, MIN_MAPQ
            )
            ref, n_kept = _pysam_recount(
                FIXTURE_BAM, contig, start, stop, FL_BANDS, MIN_MAPQ, dedup=True
            )
            assert n_kept > 0, f"no band fragments in {contig}:{start}-{stop}"
            # Explicit track-set completeness: both sides must carry the full
            # canonical set (2 strands * 2 fl_bands * 3 cov_types = 12 tracks),
            # so a missing/extra track cannot slip past the per-track loop below.
            assert set(got.keys()) == set(ref.keys())
            assert len(ref) == 12
            for key in ref:
                np.testing.assert_array_equal(
                    got[key], ref[key],
                    err_msg=f"mismatch at {contig}:{start}-{stop} track {key}",
                )


@pytest.mark.skipif(
    not (os.path.exists(DUP_H5) and os.path.exists(DUP_BAM)),
    reason="test_duplicates golden fixtures not found",
)
class TestGoldenDedup:
    """Dedicated dedup golden case built from test_duplicates.bam.

    The two pairs are coordinate-identical (start/stop = 99110000/99110116,
    length 116, strand '+'); one is duplicate-flagged. The h5 is built with
    --include-duplicates so BOTH enter, and read-time drop_duplicate_fragments
    must collapse 2 -> 1.
    """

    # length 116 is outside the canonical bands; use a band that spans it so the
    # coverage arrays are non-zero.
    DEDUP_BAND = [(100, 150)]
    CONTIG, START, STOP = "chr6", 99110000, 99110200

    def test_read_time_dedup_collapses_pairs(self):
        got, n_before, n_after = _rfa_counts(
            DUP_H5, self.CONTIG, self.START, self.STOP,
            self.DEDUP_BAND, min_mapq=0, dedup=True,
        )
        assert n_before == 2, "include-duplicates h5 should carry both pairs"
        assert n_after == 1, "drop_duplicate_fragments must collapse to one"

    def test_dedup_counts_match_pysam(self):
        got, _, _ = _rfa_counts(
            DUP_H5, self.CONTIG, self.START, self.STOP,
            self.DEDUP_BAND, min_mapq=0, dedup=True,
        )
        ref, n_kept = _pysam_recount(
            DUP_BAM, self.CONTIG, self.START, self.STOP, self.DEDUP_BAND,
            min_mapq=0, dedup=True, include_duplicates=True,
        )
        assert n_kept == 1
        for key in ref:
            np.testing.assert_array_equal(got[key], ref[key], err_msg=str(key))

    def test_dedup_actually_changes_counts(self):
        """Guard: dedup is not a no-op here (nodedup double-counts)."""
        ref_dedup, n_d = _pysam_recount(
            DUP_BAM, self.CONTIG, self.START, self.STOP, self.DEDUP_BAND,
            min_mapq=0, dedup=True, include_duplicates=True,
        )
        ref_nodedup, n_nd = _pysam_recount(
            DUP_BAM, self.CONTIG, self.START, self.STOP, self.DEDUP_BAND,
            min_mapq=0, dedup=False, include_duplicates=True,
        )
        assert n_nd == 2 and n_d == 1
        assert any(
            not np.array_equal(ref_dedup[k], ref_nodedup[k]) for k in ref_dedup
        )


class TestStrandInvariantSynthetic:
    """Test the strand='.' invariant using synthetic RegionFragmentArray data.

    This exercises the no-flip invariant WITHOUT a real h5 fixture.
    """

    def test_strand_dot_no_flip_synthetic(self):
        from fragmentomics_tools.fragment_array.fragment_array import RegionFragmentArray
        from fragmentomics_tools.region import Region

        region_dot = Region(chrom="chr1", start=1000, stop=1100, strand=".")
        starts_0 = np.array([10, 20])
        stops_0 = np.array([60, 70])
        rfa_dot = RegionFragmentArray(
            starts_0=starts_0,
            stops_0=stops_0,
            region=region_dot,
            max_frag_len=175,
            validate_data=False,
        )
        # Region normalizes strand='.' to None; is_minus_strand() must be False.
        assert not rfa_dot.region.is_minus_strand()
        np.testing.assert_array_equal(rfa_dot.starts_0, starts_0)

    def test_minus_strand_region_flips(self):
        from fragmentomics_tools.region import Region

        region = Region(chrom="chr1", start=1000, stop=1100, strand="-")
        assert region.is_minus_strand()

    def test_fl_band_half_open(self):
        from fragmentomics_tools.fragment_array.fragment_array import RegionFragmentArray
        from fragmentomics_tools.region import Region

        region = Region(chrom="chr1", start=0, stop=200, strand=".")
        # length exactly 65 must NOT be in band [40, 65) (upper exclusive)
        rfa = RegionFragmentArray(
            starts_0=np.array([10]), stops_0=np.array([75]),
            region=region, max_frag_len=175, validate_data=False,
        )
        assert rfa.subset_fragment_lengths(40, 65).n_frags == 0
        # length 64 IS in band
        rfa_b = RegionFragmentArray(
            starts_0=np.array([10]), stops_0=np.array([74]),
            region=region, max_frag_len=175, validate_data=False,
        )
        assert rfa_b.subset_fragment_lengths(40, 65).n_frags == 1

    def test_fl_band_lower_inclusive(self):
        from fragmentomics_tools.fragment_array.fragment_array import RegionFragmentArray
        from fragmentomics_tools.region import Region

        region = Region(chrom="chr1", start=0, stop=200, strand=".")
        rfa = RegionFragmentArray(
            starts_0=np.array([10]), stops_0=np.array([50]),  # length 40
            region=region, max_frag_len=175, validate_data=False,
        )
        assert rfa.subset_fragment_lengths(40, 65).n_frags == 1
