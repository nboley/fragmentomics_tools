import numpy
import pytest
import pysam

from typing import List, Optional

from fragmentomics_tools.util.liftover import RegionLiftOver
from fragmentomics_tools.region import Region
from fragmentomics_tools.contig import get_reference_path

@pytest.mark.parametrize(
    "start, stop, strand, resizes, flips",
    [
        # Odd length, even resize go smaller first
        (1, 6, "+", [None, 2, 5, None], [True, False, False, True]),  # flip resize resize flip
        # (1, 6, "+", [None, 2, None, 5], [True, False, True, False]),  # flip resize flip resize
        # Even length, even resize go smaller first
        (1, 7, "+", [None, 2, 6, None], [True, False, False, True]),  # flip resize resize flip
        (1, 7, "+", [None, 2, None, 6], [True, False, True, False]),  # flip resize flip resize
        # Even length, odd resize go smaller first
        (1, 7, "+", [None, 3, 6, None], [True, False, False, True]),  # flip resize resize flip
        # (1, 7, "+", [None, 3, None, 6], [True, False, True, False]),  # flip resize flip resize
        # Odd length, odd resize go smaller first
        (1, 6, "+", [None, 3, 5, None], [True, False, False, True]),  # flip resize resize flip
        (1, 6, "+", [None, 3, None, 5], [True, False, True, False]),  # flip resize flip resize
        # Odd length, even resize go bigger first
        (5, 10, "+", [None, 8, 5, None], [True, False, False, True]),  # flip resize resize flip
        # (5, 10, "+", [None, 8, None, 5], [True, False, True, False]),  # flip resize flip resize
        # Even length, even resize go bigger first
        (5, 11, "+", [None, 8, 6, None], [True, False, False, True]),  # flip resize resize flip
        (5, 11, "+", [None, 8, None, 6], [True, False, True, False]),  # flip resize flip resize
        # Even length, odd resize go bigger first
        (5, 11, "+", [None, 9, 6, None], [True, False, False, True]),  # flip resize resize flip
        # (5, 11, "+", [None, 9, None, 6], [True, False, True, False]),  # flip resize flip resize
        # Odd length, odd resize go bigger first
        (5, 10, "+", [None, 9, 5, None], [True, False, False, True]),  # flip resize resize flip
        (5, 10, "+", [None, 9, None, 5], [True, False, True, False]),  # flip resize flip resize
        # Odd length, even resize go smaller first
        (2, 7, "+", [None, 2, 5, None], [True, False, False, True]),  # flip resize resize flip
        # (2, 7, "+", [None, 2, None, 5], [True, False, True, False]),  # flip resize flip resize
        # Even length, even resize go smaller first
        (2, 8, "+", [None, 2, 6, None], [True, False, False, True]),  # flip resize resize flip
        (2, 8, "+", [None, 2, None, 6], [True, False, True, False]),  # flip resize flip resize
        # Even length, odd resize go smaller first
        (2, 8, "+", [None, 3, 6, None], [True, False, False, True]),  # flip resize resize flip
        # (2, 8, "+", [None, 3, None, 6], [True, False, True, False]),  # flip resize flip resize
        # Odd length, odd resize go smaller first
        (2, 7, "+", [None, 3, 5, None], [True, False, False, True]),  # flip resize resize flip
        (2, 7, "+", [None, 3, None, 5], [True, False, True, False]),  # flip resize flip resize
        # Odd length, even resize go bigger first
        (6, 11, "+", [None, 8, 5, None], [True, False, False, True]),  # flip resize resize flip
        # (6, 11, "+", [None, 8, None, 5], [True, False, True, False]),  # flip resize flip resize
        # Even length, even resize go bigger first
        (6, 12, "+", [None, 8, 6, None], [True, False, False, True]),  # flip resize resize flip
        (6, 12, "+", [None, 8, None, 6], [True, False, True, False]),  # flip resize flip resize
        # Even length, odd resize go bigger first
        (6, 12, "+", [None, 9, 6, None], [True, False, False, True]),  # flip resize resize flip
        # (6, 12, "+", [None, 9, None, 6], [True, False, True, False]),  # flip resize flip resize
        # Odd length, odd resize go bigger first
        (6, 11, "+", [None, 9, 5, None], [True, False, False, True]),  # flip resize resize flip
        (6, 11, "+", [None, 9, None, 5], [True, False, True, False]),  # flip resize flip resize
    ],
)
def test_region_resize_identity(
    start: int, stop: int, strand: str, resizes: List[Optional[int]], flips: List[bool]
):
    assert len(resizes) == len(flips)
    region = Region("NA", start, stop, strand)
    r_mu = region
    for i, (r, fl) in enumerate(zip(resizes, flips)):
        assert r is None or not fl, "One of r needs to be None or fl needs to be False so we have ordering"
        if r is not None:
            r_mu = r_mu.resize(r)
        elif fl:
            r_mu = r_mu.flip_strand()
        else:
            raise ValueError(
                f"ERROR: index {i} has None in both resizes and flips, or nothing: {resizes}, {flips}"
            )
        if i == 0:
            assert r_mu != region  # at least one change needs to happen which breaks equality
    assert r_mu == region  # make sure we have indeed perturbed the region


def test_region_resize():
    r = Region("chr1", 1000, 1100)
    assert r.length == 100

    # test that resizing works for even re-sizes with no strand
    r2 = r.resize(500)
    assert r2.length == 500
    assert r2.midpoint == r.midpoint, f"{r2.midpoint} VS {r.midpoint}"

    # test that a resize that is too big raises and error
    try:
        r.resize(5000)
    except ValueError:
        pass
    else:
        assert False, "Resize should be impossible"

    # test that resizing works for re-sizes with strands
    for strand in "-+":
        for resize_size in (500, 501):
            r = Region("chr1", 1000, 1100, strand)
            assert r.length == 100
            r2 = r.resize(resize_size)
            assert r2.length == resize_size
            # assert r2.midpoint == r.midpoint


def test_convert_region_returns_none():
    lo = RegionLiftOver("hg19", "hg38")
    r = Region("chr1", 1000, 1100, ref="hg19")
    assert r.convert_region(lo) is None


def test_convert_region_succeeds():
    lo = RegionLiftOver("hg19", "hg38")
    r = Region("chr1", 100000, 100100, ref="hg19")
    assert str(r.convert_region(lo)) == str(r)


def test_convert_region_fails_ref_mismatch():
    lo = RegionLiftOver("hg38", "hg19")
    r = Region("chr1", 1000, 1100, ref="hg19")
    with pytest.raises(ValueError):
        assert r.convert_region(lo)


def test_convert_region_fails_chrom_length():
    lo = RegionLiftOver("hg19", "hg38")
    orig_r = Region("chr4", 44974182, 54978782, ref="hg19")
    r = orig_r.liftover(lo)
    assert r is None


def test_get_sequence():
    region = Region("chr11", 106022048, 106023072, ref="hg38")
    with pysam.FastaFile(get_reference_path(region.ref)) as fasta:
        sequence = region.get_sequence(fasta)

        assert sequence.shape[0] == region.length, "sequence and region must have the same length"
        assert set(numpy.unique(sequence)) == {0, 1}, "output must be a one-hot encoding"
        assert (sequence.sum(axis=1) == 1).all(), "Only one base must be on at each locus."
        assert (
            numpy.array(
                ["acgt".index(c) for c in "cctagagatccgcttgctgcgctgttccaactgattggggcactggccgc".replace(" ", "")]
            )
            == region.resize(50).get_sequence(fasta).argmax(axis=1)
        ).all(), (
            "computed sequence does not match the fetched sequence from UCSC Genome Browser"
            "http://genome.ucsc.edu/cgi-bin/das/hg38/dna?segment=chr11:106022536,106022585"
        )


def test_get_subregion_from_jitter_and_strand():
    contig = "chr1"
    start = 100000
    stop = 120000
    center = (start + stop) // 2
    region = Region(contig, start, stop, ref="hg38")

    for output_size in [1000, 2000, 5000, 10000]:
        for jitter in [1000, 100, 50, 0, -50, -100, -1000]:
            for strand in [None, "+", "-"]:
                assert region.get_subregion_from_jitter_and_strand(output_size, jitter, strand) == Region(
                    contig, center + jitter - (output_size // 2), center + jitter + (output_size // 2), strand
                )
