import os
import numpy
import pytest

# from fbio.formats import FragmentBigBedReader, FragmentBedReader

from fragmentomics_tools.fragment_matrix import (
    RegionFragmentMatrix,
    Region,
    merge_fragment_matrices,
    # iter_fragments_from_10x_frag_tsv,
)
from fragmentomics_tools.fragment_array import RegionFragmentArray

DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "./data/")


def test_start_counts():
    fa = RegionFragmentArray([-1, 2, 2], [2, 3, 4], Region("chr1", 0, 5), 511)
    # fm = fa.fragment_matrix
    numpy.testing.assert_equal(fa.first_covered_base_counts, numpy.array([0, 0, 2, 0, 0], dtype=numpy.uint32))
    # numpy.testing.assert_equal(fa.first_covered_base_counts, fm.first_covered_base_counts)


def test_end_counts():
    fa = RegionFragmentArray(
        starts_0=[-1, 2, 2], stops_0=[2, 4, 4], region=Region("chr1", 0, 5), max_frag_len=511
    )
    # fm = fa.fragment_matrix
    numpy.testing.assert_equal(fa.last_covered_base_counts, numpy.array([0, 1, 0, 2, 0], dtype=numpy.uint32))
    # numpy.testing.assert_equal(fa.last_covered_base_counts, fm.last_covered_base_counts)


"""
def test_fragment_matrix_from_frag_bed(chrom="chrX", strand="+", pos=6241588):
    start = pos - 256
    stop = pos + 256

    fm = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, start, stop, strand), min_mapq=0
    )

    # these are the frags that should get counted
    with FragmentBedReader(os.path.join(DIR, "frag.bed.gz")) as fbr:
        frags = list(fbr.fetch(chrom))
    frags = [f for f in frags if start < f.midpoint <= stop]
    assert len(list(frags)) == fm.arr.sum()


def test_mapq_filtering(chrom="chrX", strand="+", pos=6241588):
    start = pos - 1024
    stop = pos + 1024

    fm = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, start, stop, strand), min_mapq=60,
    )

    # these are the frags that should get counted
    with FragmentBedReader(os.path.join(DIR, "frag.bed.gz")) as fbr:
        frags = list(fbr.fetch(chrom))
    frags = [f for f in frags if start < f.midpoint <= stop and f.mapq12_min >= 60]
    assert len(list(frags)) == fm.arr.sum()
"""


def test_merged_fragment_matrices_bed(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    ).fragment_matrix

    merged_fm = merge_fragment_matrices([fm, fm])
    assert fm.arr.sum() * 2 == merged_fm.arr.sum()


def test_merged_incompatible_fragment_matrices(chrom="chrX", strand="+", pos=6241588):
    fm1 = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    ).fragment_matrix
    fm2 = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 100, pos + 100, strand), min_mapq=0,
    ).fragment_matrix

    try:
        merge_fragment_matrices([fm1, fm2])
    except ValueError:
        pass
    else:
        assert False, "It should not be possible to merge fm1 and fm2 because they have different shapes"


def test_add_region_fragment_matrices(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    ).fragment_matrix

    merged_fm = fm + fm
    assert fm.arr.sum() * 2 == merged_fm.arr.sum()
    assert merged_fm.region == fm.region


def test_add_fragment_matrices(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    ).fragment_matrix

    merged_fm = fm + fm
    assert fm.arr.sum() * 2 == merged_fm.arr.sum()


def test_add_fragment_matrix_to_region_fragment_matrix(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    ).fragment_matrix

    # make sure that we get a type error if we try to add a region fragment matrix to
    # a fragment matrix
    try:
        fm + fm.fragment_matrix
    except TypeError:
        pass
    else:
        assert False


def test_add_incompatible_fragment_matrices(chrom="chrX", strand="+", pos=6241588):
    fm1 = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 100, pos + 100, strand="+"), min_mapq=0,
    ).fragment_matrix
    fm2 = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 100, pos + 100, strand="-"), min_mapq=0,
    ).fragment_matrix

    try:
        fm1 + fm2
    except ValueError:
        pass
    else:
        assert False, "It should not be possible to add fm1 and fm2 because they have different strands"


def test_downsample(chrom="chrX", strand="+", pos=6241588):
    fa = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    )
    assert fa.downsampled(3).arr.sum() == 3


def test_reverse_strand(chrom="chrX", strand="+", pos=6241588):
    fa = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    )
    flipped_fa = fa.reverse_strand()
    assert flipped_fa.region.strand == "-"
    assert (fa.fragment_strands == numpy.array([b'+', b'+', b'-', b'+'], dtype='<U1')).all()
    assert (fa.fragment_strands == numpy.array(['+', '+', '-', '+'], dtype='<U1')).all()
    assert (flipped_fa.fragment_strands == numpy.array(['-', '+', '-', '-'], dtype='<U1')).all()
    assert sorted(fa.fragment_lengths) == sorted(flipped_fa.fragment_lengths)


def test_downsampled_frag_lens(chrom="chrX", strand="+", pos=6241588):
    fa = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    )
    # downsample so that only fragments with lengths < 100 are allowed
    downsampled_fa = fa.downsampled_frag_lens([1] * 185 + [0] * (fa.max_frag_len-185))
    assert sorted(downsampled_fa.fragment_lengths) == [165, 178, 185]

    downsampled_fa = fa.downsampled_frag_lens([0] * 185 + [1] * (fa.max_frag_len-185))
    assert sorted(downsampled_fa.fragment_lengths) == [200]


def test_shift_and_zero_pad(chrom="chrX", strand="+", pos=6241588):
    fa = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    )
    shifted_fa = fa.shift_and_zero_pad(100)
    assert (fa.fragment_matrix.dense_array[:, :-100] == fa.shift_and_zero_pad(100).fragment_matrix.dense_array[:, 100:]).all()
    assert fa.region == shifted_fa.region.shift(-100)


def test_sample_with_replacement(chrom="chrX", strand="+", pos=6241588):
    fa = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    )
    res = fa.sample_with_replacement(3)
    assert res.n_fragments == 3


def test_get_slice(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 512, pos + 512, strand), min_mapq=0,
    ).fragment_matrix
    # Check that slicing works with defined bounds
    assert fm.get_slice(slice(20, 400)).region.length == 380
    # Check that None bounds make them extend to either end
    assert fm.get_slice(slice(None, 600)).region.length == 600
    assert fm.get_slice(slice(1000, None)).region.length == 24
    # Check that None bounds on both sides does nothing
    assert fm.get_slice(slice(None, None)).region.length == 1024
    # Check that slice bounds must be within the length of the region
    with pytest.raises(AssertionError):
        fm.get_slice(slice(500, 1200))
        fm.get_slice(slice(-20, 50))


def test_split_into_k_nonoverlapping_fms(chrom="chrX", strand="+", pos=6241588):
    fa = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    )
    fa1, fa2 = fa.split_into_k_nonoverlapping_fas(k=2, sample_size=2)
    assert fa1.n_fragments == 2
    assert fa2.n_fragments == 2

    # test that None works
    fa1, fa2 = fa.split_into_k_nonoverlapping_fas(k=2)
    assert fa.n_fragments%2 == 0
    assert fa1.n_fragments == fa.n_fragments // 2
    assert fa2.n_fragments == fa.n_fragments // 2


def test_jitter(chrom="chrX", strand="+", pos=6241588):
    fa = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 512, pos + 512, strand), min_mapq=0,
    )

    fa_j = fa.jitter(jitter_value=100, output_length=500)
    assert fa_j.shape == (fa.shape[0], 500)
    assert (fa_j.fragment_matrix.todense().sum() == fa.fragment_matrix.todense()[:, (1024 // 2 + 100 - 250) : (1024 // 2 + 100 + 250)].sum())


### Plot
@pytest.mark.skip(reason="haven't added plotting code")
def test_region_fragment_array_plot(chrom="chrX", strand="+", pos=6241588):
    fa = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    )
    fa.plot()


@pytest.mark.skip(reason="haven't added plotting code")
def test_fragment_array_plot(chrom="chrX", strand="+", pos=6241588):
    fa = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 512, pos + 512, strand), min_mapq=0,
    )
    fa.plot()
