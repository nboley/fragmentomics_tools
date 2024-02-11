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


def test_merged_fragment_matrices_flip_minus_strand(chrom="chrX", pos=6241588):
    fm_fwd = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, "+"), min_mapq=0,
    ).fragment_matrix
    fm_rev = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, "-"), min_mapq=0,
    ).fragment_matrix

    merged_fm = merge_fragment_matrices([fm_fwd, fm_rev], flip_minus_strand=True)
    assert numpy.all(merged_fm.dense_array == fm_fwd.dense_array + fm_rev.dense_array[:, ::-1])

    fm = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256), min_mapq=0,
    ).fragment_matrix
    try:
        merge_fragment_matrices([fm, fm], flip_minus_strand=True)
    except AssertionError:
        pass
    else:
        assert False, "Cannot flip minus strand with unstranded fragment matrices"


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
    )

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


def test_region_fragment_matrix_plot(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    ).fragment_matrix
    fm.plot()


def test_region_fragment_matrix_plot_with_smoothed(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    ).fragment_matrix
    fm.plot(fragment_matrix_density=fm.to_kernel_smoothed())


def test_downsample(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    ).fragment_matrix
    assert fm.downsampled(3).arr.sum() == 3


def test_reverse_strand(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    ).fragment_matrix
    flipped_fm = fm.reverse_strand()
    fm = fm
    flipped_fm = flipped_fm
    assert flipped_fm.region.strand == "-"
    assert (flipped_fm.arr.todense() == fm.arr.todense()[:, ::-1]).all()


def test_downsampled_frag_lens(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    ).fragment_matrix
    # downsample so that only fragments with lengths < 100 are allowed
    downsampled_fm = fm.downsampled_frag_lens([1] * 100 + [0] * 1000)
    assert all(frag.len() < 100 for frag in downsampled_fm.to_fragments())


def test_shift_and_zero_pad(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    ).fragment_matrix
    unshifted_arr = fm.arr.tocsc()
    shifted_fm = fm.shift_and_zero_pad(100)
    assert (unshifted_arr[:, 100:].todense() == shifted_fm.arr.tocsc()[:, :-100].todense()).all()
    assert fm.region == shifted_fm.region.shift(-100)


def test_sample_with_replacement(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    ).fragment_matrix
    res = fm.sample_with_replacement(3)
    assert res.arr.sum() == 3


def test_fragment_matrix_plot(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 512, pos + 512, strand), min_mapq=0,
    ).fragment_matrix
    fm.plot()


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
    fm = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    ).fragment_matrix
    fm1, fm2 = fm.split_into_k_nonoverlapping_fms(k=2, sample_size=2)
    assert fm1.n_fragments == 2
    assert fm2.n_fragments == 2

    # test that None works
    fm1, fm2 = fm.split_into_k_nonoverlapping_fms(k=2)
    assert fm1.n_fragments == fm.n_fragments // 2
    assert fm2.n_fragments == fm.n_fragments // 2


def test_jitter(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentArray.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 512, pos + 512, strand), min_mapq=0,
    ).fragment_matrix

    fm_j = fm.jitter(jitter_value=100, output_length=500)
    assert fm_j.shape == (fm.shape[0], 500)
    assert (fm_j.todense() == fm.todense()[:, (1024 // 2 + 100 - 250) : (1024 // 2 + 100 + 250)]).all()
