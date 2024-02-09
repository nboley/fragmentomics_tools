import os
import numpy
import pytest
import torch

from fbio.formats import FragmentBigBedReader, FragmentBedReader
from fbio.fragment import Fragment, fragments_to_fragment_coverage_array
from fbio.util.context_managers import environment_variables

from ravel.bio.frag.fragment_matrix import (
    RegionFragmentMatrix,
    Region,
    merge_fragment_matrices,
    iter_fragments_from_10x_frag_tsv,
    FragmentMatrixDensity,
    UNETSmoothedFragmentMatrix,
    batched_log_operator_exp_trick,
)
from ravel.constants import DEFAULT_DATA_MANIFEST_PATH, RAVEL_TEST_DIR
from ravel.data_manifest import load_data_manifest
from ravel.fragment_array import RegionFragmentArray
from ravel.assays import ASSAYS
from ravel.learn.transforms import ProfileSmoothing2D
from ravel.samples import SAMPLE_DF

DIR = os.path.join(RAVEL_TEST_DIR, "bio/frag/data")


def test_start_counts():
    fa = RegionFragmentArray([-1, 2, 2], [2, 3, 4], Region("chr1", 0, 5), 511)
    fm = fa.fragment_matrix
    numpy.testing.assert_equal(fa.first_covered_base_counts, numpy.array([0, 0, 2, 0, 0], dtype=numpy.uint32))
    numpy.testing.assert_equal(fa.first_covered_base_counts, fm.first_covered_base_counts)


def test_end_counts():
    fa = RegionFragmentArray(
        starts_0=[-1, 2, 2], stops_0=[2, 4, 4], region=Region("chr1", 0, 5), max_frag_len=511
    )
    fm = fa.fragment_matrix
    numpy.testing.assert_equal(fa.last_covered_base_counts, numpy.array([0, 1, 0, 2, 0], dtype=numpy.uint32))
    numpy.testing.assert_equal(fa.last_covered_base_counts, fm.last_covered_base_counts)


def test_fragment_coverage_vector():
    frags = [
        Fragment("chr1", 8, 12),
        Fragment("chr1", 10, 15),
        Fragment("chr1", 13, 17),
        Fragment("chr1", 17, 20),
        Fragment("chr1", 17, 22),
    ]

    region = Region("chr1", 10, 20)

    arr1 = fragments_to_fragment_coverage_array(frags, region)
    arr2 = RegionFragmentMatrix.from_frags(frags, region).get_fragment_coverage_array()
    assert list(arr1) == list(arr2)


def test_fragment_matrix_from_frags(chrom="chr1", strand="+", pos=6241588, window_size=8048):
    start = pos - window_size
    stop = pos + window_size

    with load_data_manifest("big_beds") as dm:
        LARGE_TEST_BED = dm.sync_and_get("big_beds/wgs_hg19/IC05.bam.frags.bed.bb").path

    with FragmentBigBedReader(LARGE_TEST_BED) as fbr:
        frags = list(fbr.fetch(chrom, start, stop))

    fm = RegionFragmentMatrix.from_frags(frags, Region(chrom, start, stop, strand), dedup=False)

    # these are the frags that should get counted
    frags = [f for f in frags if start < f.midpoint <= stop]
    assert len(frags) == 150
    assert fm.dense_array.sum() == 150

    # check dedup
    has_dup = fm.dense_array > 1
    num_dups = fm.dense_array[has_dup].sum() - has_dup.sum()
    assert num_dups == 13

    fm2 = RegionFragmentMatrix.from_frags(frags, Region(chrom, start, stop, strand), dedup=True)

    assert fm2.arr.sum() == 137
    assert fm.arr.sum() - num_dups == fm2.arr.sum()


def test_fragment_matrix_from_frag_bed(chrom="chrX", strand="+", pos=6241588):
    start = pos - 256
    stop = pos + 256

    fm = RegionFragmentMatrix.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, start, stop, strand), min_mapq=0
    )

    # these are the frags that should get counted
    with FragmentBedReader(os.path.join(DIR, "frag.bed.gz")) as fbr:
        frags = list(fbr.fetch(chrom))
    frags = [f for f in frags if start < f.midpoint <= stop]
    assert len(list(frags)) == fm.arr.sum()


def test_fragment_matrix_from_fragments_h5(chrom="chr6", pos=99119623):
    start = pos - 512
    stop = pos + 512

    with load_data_manifest("test") as dm:
        frag_h5_path = dm.sync_and_get("test/BO_4_1.capture.chr6_99118615_99121634.fragments.h5").path
        fm = RegionFragmentMatrix.from_fragments_h5(frag_h5_path, Region(chrom, start, stop), min_mapq=0)
        assert fm.n_fragments == 8030

        # These should give 0 counts, as the fh5 is not stranded
        with pytest.raises(ValueError, match=r".*The referenced h5 file does not contain strand info..*"):
            fm_pos_strand = RegionFragmentMatrix.from_fragments_h5(
                frag_h5_path, Region(chrom, start, stop), min_mapq=0, strand="+"
            )
        with pytest.raises(ValueError, match=r".*The referenced h5 file does not contain strand info..*"):
            fm_neg_strand = RegionFragmentMatrix.from_fragments_h5(
                frag_h5_path, Region(chrom, start, stop), min_mapq=0, strand="-"
            )


def test_stranded_fragment_matrix_from_fragments_h5(chrom="chr6", pos=99118615):
    start = pos
    stop = pos + 1000

    with load_data_manifest("test") as dm:
        frag_h5_path = dm.sync_and_get("test.stranded.h5").path
        fm = RegionFragmentMatrix.from_fragments_h5(frag_h5_path, Region(chrom, start, stop), min_mapq=0)
        assert fm.n_fragments == 70

        # These should give 0 counts, as the fh5 is not stranded
        fm_pos_strand = RegionFragmentMatrix.from_fragments_h5(
            frag_h5_path, Region(chrom, start, stop), min_mapq=0, strand="+"
        )
        fm_neg_strand = RegionFragmentMatrix.from_fragments_h5(
            frag_h5_path, Region(chrom, start, stop), min_mapq=0, strand="-"
        )

        assert fm_pos_strand.n_fragments + fm_neg_strand.n_fragments == fm.n_fragments


def test_fragment_matrix_from_10x_frag_tsv(chrom="chrX", start=6240000, stop=6243000):
    fm = RegionFragmentMatrix.from_10x_frag_tsv(
        os.path.join(DIR, "frag.10x.tsv.gz"), Region(chrom, start, stop),
    )

    # these are the frags that should get counted
    frags = list(
        iter_fragments_from_10x_frag_tsv(os.path.join(DIR, "frag.10x.tsv.gz"), Region(chrom, start, stop))
    )
    frags = [f for f in frags if start < f.midpoint <= stop]
    assert len(list(frags)) == fm.arr.sum()


def test_mapq_filtering(chrom="chrX", strand="+", pos=6241588):
    start = pos - 1024
    stop = pos + 1024

    fm = RegionFragmentMatrix.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, start, stop, strand), min_mapq=60,
    )

    # these are the frags that should get counted
    with FragmentBedReader(os.path.join(DIR, "frag.bed.gz")) as fbr:
        frags = list(fbr.fetch(chrom))
    frags = [f for f in frags if start < f.midpoint <= stop and f.mapq12_min >= 60]
    assert len(list(frags)) == fm.arr.sum()


def test_merged_fragment_matrices_bed(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentMatrix.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    )

    merged_fm = merge_fragment_matrices([fm, fm])
    assert fm.arr.sum() * 2 == merged_fm.arr.sum()


def test_merged_fragment_matrices_flip_minus_strand(chrom="chrX", pos=6241588):
    fm_fwd = RegionFragmentMatrix.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, "+"), min_mapq=0,
    )
    fm_rev = RegionFragmentMatrix.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, "-"), min_mapq=0,
    )

    merged_fm = merge_fragment_matrices([fm_fwd, fm_rev], flip_minus_strand=True)
    assert numpy.all(merged_fm.dense_array == fm_fwd.dense_array + fm_rev.dense_array[:, ::-1])

    fm = RegionFragmentMatrix.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256), min_mapq=0,
    )
    try:
        merge_fragment_matrices([fm, fm], flip_minus_strand=True)
    except AssertionError:
        pass
    else:
        assert False, "Cannot flip minus strand with unstranded fragment matrices"


def test_merged_incompatible_fragment_matrices(chrom="chrX", strand="+", pos=6241588):
    fm1 = RegionFragmentMatrix.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    )
    fm2 = RegionFragmentMatrix.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 100, pos + 100, strand), min_mapq=0,
    )

    try:
        merge_fragment_matrices([fm1, fm2])
    except ValueError:
        pass
    else:
        assert False, "It should not be possible to merge fm1 and fm2 because they have different shapes"


def test_add_region_fragment_matrices(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentMatrix.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    )

    merged_fm = fm + fm
    assert fm.arr.sum() * 2 == merged_fm.arr.sum()
    assert merged_fm.region == fm.region


def test_add_fragment_matrices(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentMatrix.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    ).fragment_matrix

    merged_fm = fm + fm
    assert fm.arr.sum() * 2 == merged_fm.arr.sum()


def test_add_fragment_matrix_to_region_fragment_matrix(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentMatrix.from_frag_bed(
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
    fm1 = RegionFragmentMatrix.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 100, pos + 100, strand="+"), min_mapq=0,
    )
    fm2 = RegionFragmentMatrix.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 100, pos + 100, strand="-"), min_mapq=0,
    )

    try:
        fm1 + fm2
    except ValueError:
        pass
    else:
        assert False, "It should not be possible to add fm1 and fm2 because they have different strands"


def test_region_fragment_matrix_plot(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentMatrix.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    )
    fm.plot()


def test_region_fragment_matrix_plot_with_smoothed(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentMatrix.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    )
    fm.plot(fragment_matrix_density=fm.to_kernel_smoothed())


def test_downsample(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentMatrix.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    )
    assert fm.downsampled(3).arr.sum() == 3


def test_reverse_strand(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentMatrix.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    )
    flipped_fm = fm.reverse_strand()
    fm = fm
    flipped_fm = flipped_fm
    assert flipped_fm.region.strand == "-"
    assert (flipped_fm.arr.todense() == fm.arr.todense()[:, ::-1]).all()


def test_downsampled_frag_lens(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentMatrix.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    )
    # downsample so that only fragments with lengths < 100 are allowed
    downsampled_fm = fm.downsampled_frag_lens([1] * 100 + [0] * 1000)
    assert all(frag.len() < 100 for frag in downsampled_fm.to_fragments())


def test_shift_and_zero_pad(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentMatrix.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    )
    unshifted_arr = fm.arr.tocsc()
    shifted_fm = fm.shift_and_zero_pad(100)
    assert (unshifted_arr[:, 100:].todense() == shifted_fm.arr.tocsc()[:, :-100].todense()).all()
    assert fm.region == shifted_fm.region.shift(-100)


def test_sample_with_replacement(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentMatrix.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    )
    res = fm.sample_with_replacement(3)
    assert res.arr.sum() == 3


def test_fragment_matrix_plot(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentMatrix.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 512, pos + 512, strand), min_mapq=0,
    ).fragment_matrix
    fm.plot()


def test_fragment_matrix_to_fragments():
    # make sure that fragments -> RegionFragmentMatrix -> fragments is not lossy
    frags = [Fragment("chr1", start, stop) for start in range(0, 10) for stop in range(start + 1, 10)]
    rfm = RegionFragmentMatrix.from_frags(frags, Region("chr1", 0, 10))
    assert list(rfm.to_fragments()) == frags


def test_fragment_matrix_density(chrom="chrX", strand="+", pos=6241588):
    # test that we can load a density
    density = numpy.ones((10, 20)) / 200
    fm_density = FragmentMatrixDensity(density)
    assert numpy.isclose(fm_density.density.sum(), 1.0)

    # test that the pos and fl checks work
    fm_density = FragmentMatrixDensity(density, fl=numpy.arange(0, 20, 2), pos=numpy.arange(0, 40, 2))


def test_fragment_matrix_to_kernel_density_smoothed(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentMatrix.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 512, pos + 512, strand), min_mapq=0,
    )
    smoothed_fm = fm.to_kernel_smoothed()
    assert fm.arr.shape == smoothed_fm.density.shape
    assert abs(smoothed_fm.density.sum() - 1.0) < 1e-6


def test_sample_from_kernel_density_smoothed(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentMatrix.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 512, pos + 512, strand), min_mapq=0,
    )
    smoothed_fm = fm.to_kernel_smoothed()
    fm_sample = smoothed_fm.sample_with_replacement(100)
    assert fm_sample.shape == fm.shape


def test_UNET_smoothed_fm(chrom="chrX", strand="+", pos=6241588):
    from ravel.assays import ASSAYS

    region_name = ASSAYS.get("breast_cancer_v1").get_one_target_name(
        Region.from_region_str("chr10:103551262-103552286")
    )

    fm = UNETSmoothedFragmentMatrix.from_sample_id_and_region("SEQRUN9_LIBRARY35", region_name)
    fm.plot()

    region = ASSAYS.get("breast_cancer_v2").get_random_region()

    fm = UNETSmoothedFragmentMatrix.from_sample_id_and_region("SEQRUN14_LIBRARY65", region)
    fm.plot()

    # test that we can load a unet density straight from s3 without downloading the whole file
    fm_from_s3 = UNETSmoothedFragmentMatrix.from_sample_id_and_region(
        "SEQRUN14_LIBRARY65", region, use_s3_fs=True
    )

    assert (fm_from_s3.density == fm.density).all()

    fm = UNETSmoothedFragmentMatrix.from_sample_id_and_region("SEQRUN14_LIBRARY65", region.resize(500))
    assert numpy.shape(fm)[1] == 500


def test_get_slice(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentMatrix.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 512, pos + 512, strand), min_mapq=0,
    )
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
    fm = RegionFragmentMatrix.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 256, pos + 256, strand), min_mapq=0,
    )
    fm1, fm2 = fm.split_into_k_nonoverlapping_fms(k=2, sample_size=2)
    assert fm1.n_fragments == 2
    assert fm2.n_fragments == 2

    # test that None works
    fm1, fm2 = fm.split_into_k_nonoverlapping_fms(k=2)
    assert fm1.n_fragments == fm.n_fragments // 2
    assert fm2.n_fragments == fm.n_fragments // 2


def test_split_into_test_and_train_fms():
    from ravel.samples import SAMPLE_DF
    from ravel.assays import ASSAYS

    sample = SAMPLE_DF.get_sample_id("SEQRUN10_LIBRARY91")
    region = next(ASSAYS.get("breast_cancer_v1").iter_regions())
    region = region.resize(1024)
    fm = sample.get_region_fragment_matrix(region=region)
    _, test_fm = fm.split_into_train_and_test(500)

    expected_starts = [
        2584956,
        2584990,
        2584991,
        2584995,
        2584998,
        2585007,
        2585014,
        2585034,
        2585034,
        2585034,
        2585037,
        2585040,
        2585041,
        2585042,
        2585046,
        2585046,
        2585047,
        2585052,
        2585053,
        2585055,
        2585055,
        2585055,
        2585055,
        2585055,
        2585056,
        2585059,
        2585060,
        2585061,
        2585064,
        2585068,
        2585071,
        2585071,
        2585072,
        2585072,
        2585073,
        2585073,
        2585074,
        2585074,
        2585075,
        2585075,
        2585075,
        2585076,
        2585077,
        2585080,
        2585084,
        2585084,
        2585085,
        2585085,
        2585085,
        2585085,
        2585086,
        2585096,
        2585096,
        2585098,
        2585123,
        2585125,
        2585135,
        2585140,
        2585142,
        2585143,
        2585143,
        2585144,
        2585157,
        2585158,
        2585159,
        2585163,
        2585165,
        2585165,
        2585165,
        2585166,
        2585167,
        2585167,
        2585167,
        2585168,
        2585169,
        2585170,
        2585172,
        2585172,
        2585176,
        2585176,
        2585177,
        2585178,
        2585178,
        2585181,
        2585183,
        2585184,
        2585185,
        2585185,
        2585186,
        2585188,
        2585191,
        2585196,
        2585197,
        2585203,
        2585203,
        2585203,
        2585203,
        2585203,
        2585203,
        2585203,
        2585203,
        2585203,
        2585203,
        2585203,
        2585203,
        2585204,
        2585204,
        2585204,
        2585204,
        2585205,
        2585205,
        2585205,
        2585207,
        2585207,
        2585212,
        2585212,
        2585215,
        2585217,
        2585218,
        2585221,
        2585221,
        2585221,
        2585222,
        2585224,
        2585225,
        2585225,
        2585227,
        2585232,
        2585232,
        2585236,
        2585238,
        2585239,
        2585239,
        2585239,
        2585239,
        2585240,
        2585240,
        2585241,
        2585241,
        2585246,
        2585251,
        2585256,
        2585256,
        2585257,
        2585262,
        2585263,
        2585263,
        2585265,
        2585266,
        2585266,
        2585266,
        2585266,
        2585269,
        2585270,
        2585271,
        2585271,
        2585271,
        2585272,
        2585274,
        2585274,
        2585278,
        2585279,
        2585284,
        2585285,
        2585291,
        2585291,
        2585293,
        2585293,
        2585305,
        2585305,
        2585305,
        2585306,
        2585307,
        2585307,
        2585307,
        2585307,
        2585310,
        2585320,
        2585320,
        2585324,
        2585326,
        2585329,
        2585330,
        2585330,
        2585331,
        2585331,
        2585333,
        2585333,
        2585336,
        2585338,
        2585340,
        2585341,
        2585341,
        2585342,
        2585345,
        2585347,
        2585350,
        2585351,
        2585353,
        2585354,
        2585355,
        2585356,
        2585358,
        2585358,
        2585362,
        2585363,
        2585365,
        2585365,
        2585365,
        2585366,
        2585368,
        2585370,
        2585370,
        2585370,
        2585370,
        2585370,
        2585370,
        2585370,
        2585370,
        2585371,
        2585371,
        2585371,
        2585371,
        2585371,
        2585371,
        2585373,
        2585374,
        2585375,
        2585379,
        2585379,
        2585383,
        2585387,
        2585389,
        2585389,
        2585390,
        2585391,
        2585399,
        2585400,
        2585402,
        2585404,
        2585407,
        2585412,
        2585415,
        2585417,
        2585417,
        2585419,
        2585424,
        2585426,
        2585426,
        2585428,
        2585429,
        2585430,
        2585432,
        2585432,
        2585433,
        2585436,
        2585437,
        2585440,
        2585440,
        2585442,
        2585452,
        2585456,
        2585456,
        2585456,
        2585460,
        2585460,
        2585463,
        2585489,
        2585489,
        2585489,
        2585491,
        2585491,
        2585492,
        2585494,
        2585500,
        2585502,
        2585504,
        2585505,
        2585505,
        2585510,
        2585513,
        2585519,
        2585522,
        2585523,
        2585529,
        2585530,
        2585531,
        2585533,
        2585538,
        2585541,
        2585541,
        2585554,
        2585557,
        2585557,
        2585560,
        2585560,
        2585561,
        2585565,
        2585569,
        2585574,
        2585574,
        2585574,
        2585574,
        2585579,
        2585579,
        2585579,
        2585580,
        2585580,
        2585583,
        2585585,
        2585586,
        2585594,
        2585595,
        2585596,
        2585596,
        2585600,
        2585602,
        2585602,
        2585603,
        2585604,
        2585610,
        2585612,
        2585614,
        2585615,
        2585615,
        2585616,
        2585617,
        2585621,
        2585621,
        2585622,
        2585623,
        2585625,
        2585625,
        2585625,
        2585626,
        2585627,
        2585628,
        2585628,
        2585628,
        2585628,
        2585630,
        2585630,
        2585638,
        2585638,
        2585639,
        2585641,
        2585641,
        2585642,
        2585642,
        2585643,
        2585644,
        2585644,
        2585651,
        2585653,
        2585655,
        2585661,
        2585662,
        2585664,
        2585665,
        2585666,
        2585672,
        2585675,
        2585675,
        2585676,
        2585677,
        2585680,
        2585680,
        2585684,
        2585684,
        2585684,
        2585685,
        2585689,
        2585691,
        2585692,
        2585693,
        2585696,
        2585704,
        2585710,
        2585711,
        2585711,
        2585714,
        2585720,
        2585720,
        2585723,
        2585731,
        2585731,
        2585732,
        2585739,
        2585742,
        2585745,
        2585752,
        2585752,
        2585753,
        2585754,
        2585766,
        2585766,
        2585770,
        2585772,
        2585775,
        2585789,
        2585789,
        2585790,
        2585790,
        2585790,
        2585790,
        2585791,
        2585791,
        2585793,
        2585793,
        2585797,
        2585806,
        2585808,
        2585811,
        2585817,
        2585822,
        2585823,
        2585823,
        2585825,
        2585825,
        2585825,
        2585825,
        2585829,
        2585831,
        2585832,
        2585833,
        2585833,
        2585833,
        2585834,
        2585835,
        2585836,
        2585836,
        2585838,
        2585838,
        2585838,
        2585840,
        2585841,
        2585841,
        2585843,
        2585845,
        2585849,
        2585850,
        2585850,
        2585851,
        2585863,
        2585863,
        2585868,
        2585870,
        2585872,
        2585873,
        2585873,
        2585875,
        2585875,
        2585880,
        2585882,
        2585883,
        2585893,
        2585893,
        2585903,
        2585904,
        2585906,
        2585911,
        2585911,
        2585916,
        2585916,
        2585917,
        2585932,
        2585939,
        2585942,
        2585964,
        2585969,
        2585973,
        2585973,
        2585974,
        2585983,
        2585987,
        2585993,
        2586002,
        2586005,
        2586009,
        2586009,
        2586010,
        2586013,
        2586013,
        2586015,
        2586016,
        2586021,
        2586027,
        2586030,
        2586030,
        2586032,
        2586037,
        2586037,
        2586038,
        2586046,
        2586050,
        2586053,
        2586062,
        2586070,
        2586070,
        2586071,
    ]
    expected_stops = [
        2585207,
        2585214,
        2585225,
        2585227,
        2585230,
        2585231,
        2585232,
        2585236,
        2585240,
        2585240,
        2585242,
        2585248,
        2585249,
        2585256,
        2585258,
        2585260,
        2585260,
        2585263,
        2585270,
        2585301,
        2585307,
        2585307,
        2585309,
        2585309,
        2585310,
        2585311,
        2585312,
        2585313,
        2585320,
        2585320,
        2585321,
        2585321,
        2585322,
        2585323,
        2585323,
        2585324,
        2585332,
        2585337,
        2585341,
        2585341,
        2585342,
        2585345,
        2585345,
        2585348,
        2585350,
        2585351,
        2585354,
        2585354,
        2585355,
        2585355,
        2585355,
        2585355,
        2585357,
        2585361,
        2585365,
        2585366,
        2585368,
        2585368,
        2585368,
        2585368,
        2585368,
        2585368,
        2585369,
        2585370,
        2585370,
        2585374,
        2585375,
        2585375,
        2585382,
        2585385,
        2585388,
        2585389,
        2585398,
        2585398,
        2585403,
        2585404,
        2585404,
        2585407,
        2585408,
        2585409,
        2585412,
        2585412,
        2585415,
        2585416,
        2585422,
        2585422,
        2585431,
        2585431,
        2585431,
        2585433,
        2585434,
        2585434,
        2585437,
        2585440,
        2585442,
        2585442,
        2585442,
        2585448,
        2585449,
        2585451,
        2585453,
        2585453,
        2585453,
        2585454,
        2585454,
        2585454,
        2585456,
        2585456,
        2585458,
        2585464,
        2585471,
        2585476,
        2585479,
        2585481,
        2585482,
        2585483,
        2585485,
        2585485,
        2585487,
        2585487,
        2585490,
        2585493,
        2585493,
        2585493,
        2585496,
        2585496,
        2585503,
        2585503,
        2585503,
        2585503,
        2585503,
        2585503,
        2585503,
        2585503,
        2585503,
        2585503,
        2585503,
        2585503,
        2585503,
        2585503,
        2585504,
        2585504,
        2585504,
        2585504,
        2585504,
        2585504,
        2585504,
        2585504,
        2585505,
        2585505,
        2585505,
        2585505,
        2585505,
        2585505,
        2585506,
        2585508,
        2585509,
        2585510,
        2585515,
        2585517,
        2585517,
        2585517,
        2585517,
        2585517,
        2585518,
        2585518,
        2585524,
        2585524,
        2585524,
        2585525,
        2585525,
        2585525,
        2585525,
        2585525,
        2585525,
        2585525,
        2585525,
        2585532,
        2585532,
        2585533,
        2585533,
        2585533,
        2585533,
        2585533,
        2585534,
        2585537,
        2585537,
        2585542,
        2585543,
        2585543,
        2585543,
        2585544,
        2585544,
        2585545,
        2585547,
        2585547,
        2585547,
        2585547,
        2585548,
        2585549,
        2585550,
        2585550,
        2585550,
        2585550,
        2585550,
        2585552,
        2585553,
        2585556,
        2585556,
        2585557,
        2585557,
        2585557,
        2585557,
        2585559,
        2585560,
        2585561,
        2585563,
        2585564,
        2585566,
        2585568,
        2585568,
        2585569,
        2585569,
        2585572,
        2585572,
        2585574,
        2585576,
        2585576,
        2585578,
        2585579,
        2585579,
        2585580,
        2585581,
        2585584,
        2585585,
        2585587,
        2585587,
        2585589,
        2585590,
        2585590,
        2585590,
        2585590,
        2585595,
        2585596,
        2585596,
        2585598,
        2585601,
        2585605,
        2585608,
        2585610,
        2585611,
        2585615,
        2585617,
        2585617,
        2585618,
        2585619,
        2585627,
        2585631,
        2585637,
        2585641,
        2585641,
        2585644,
        2585646,
        2585649,
        2585649,
        2585651,
        2585658,
        2585661,
        2585663,
        2585670,
        2585671,
        2585672,
        2585676,
        2585680,
        2585681,
        2585684,
        2585686,
        2585691,
        2585696,
        2585699,
        2585700,
        2585719,
        2585732,
        2585738,
        2585750,
        2585752,
        2585753,
        2585758,
        2585762,
        2585762,
        2585762,
        2585765,
        2585767,
        2585770,
        2585773,
        2585776,
        2585777,
        2585778,
        2585778,
        2585779,
        2585780,
        2585781,
        2585781,
        2585781,
        2585786,
        2585788,
        2585791,
        2585791,
        2585792,
        2585793,
        2585797,
        2585800,
        2585806,
        2585806,
        2585807,
        2585809,
        2585810,
        2585810,
        2585810,
        2585810,
        2585813,
        2585814,
        2585821,
        2585823,
        2585823,
        2585828,
        2585829,
        2585833,
        2585835,
        2585835,
        2585840,
        2585840,
        2585846,
        2585847,
        2585849,
        2585852,
        2585853,
        2585853,
        2585859,
        2585860,
        2585861,
        2585861,
        2585862,
        2585865,
        2585868,
        2585869,
        2585869,
        2585874,
        2585877,
        2585879,
        2585879,
        2585880,
        2585881,
        2585882,
        2585882,
        2585883,
        2585890,
        2585896,
        2585906,
        2585909,
        2585915,
        2585916,
        2585916,
        2585916,
        2585924,
        2585927,
        2585928,
        2585929,
        2585930,
        2585931,
        2585932,
        2585940,
        2585941,
        2585941,
        2585945,
        2585964,
        2585970,
        2585971,
        2585973,
        2585974,
        2585977,
        2585980,
        2585981,
        2585985,
        2585986,
        2585986,
        2585986,
        2585987,
        2585988,
        2585988,
        2585989,
        2585989,
        2585989,
        2585989,
        2585990,
        2585990,
        2585992,
        2585992,
        2585993,
        2585996,
        2585998,
        2585998,
        2585998,
        2585998,
        2585999,
        2585999,
        2585999,
        2586000,
        2586000,
        2586002,
        2586002,
        2586002,
        2586002,
        2586002,
        2586002,
        2586008,
        2586009,
        2586012,
        2586013,
        2586013,
        2586013,
        2586013,
        2586013,
        2586015,
        2586015,
        2586019,
        2586021,
        2586024,
        2586026,
        2586026,
        2586028,
        2586032,
        2586034,
        2586035,
        2586035,
        2586037,
        2586041,
        2586041,
        2586047,
        2586052,
        2586061,
        2586063,
        2586063,
        2586065,
        2586069,
        2586091,
        2586094,
        2586094,
        2586105,
        2586125,
        2586139,
        2586144,
        2586145,
        2586145,
        2586151,
        2586163,
        2586164,
        2586170,
        2586170,
        2586171,
        2586174,
        2586175,
        2586176,
        2586184,
        2586186,
        2586187,
        2586187,
        2586189,
        2586193,
        2586193,
        2586195,
        2586197,
        2586197,
        2586198,
        2586200,
        2586201,
        2586204,
        2586204,
        2586206,
        2586212,
        2586212,
        2586213,
        2586214,
        2586216,
        2586217,
        2586225,
        2586232,
        2586233,
        2586242,
        2586243,
        2586254,
        2586306,
        2586307,
        2586320,
        2586331,
        2586333,
        2586341,
        2586351,
        2586359,
        2586363,
    ]
    assert test_fm.n_fragments == 500
    assert sorted(test_fm.starts.tolist()) == sorted(expected_starts)
    assert sorted(test_fm.stops.tolist()) == sorted(expected_stops)


def test_jitter(chrom="chrX", strand="+", pos=6241588):
    fm = RegionFragmentMatrix.from_frag_bed(
        os.path.join(DIR, "frag.bed.gz"), Region(chrom, pos - 512, pos + 512, strand), min_mapq=0,
    )

    fm_j = fm.jitter(jitter_value=100, output_length=500)
    assert fm_j.shape == (fm.shape[0], 500)
    assert (fm_j.todense() == fm.todense()[:, (1024 // 2 + 100 - 250) : (1024 // 2 + 100 + 250)]).all()


def test_batched_log_operator_exp_trick():
    seed = 312
    eps = 3e-8
    rng = numpy.random.RandomState(seed)
    data = torch.tensor(rng.randn(3, 1, 10, 30), dtype=torch.float32)  # assume in log space
    smoother = ProfileSmoothing2D(smoothing_width=9, kind="gaussian", gaussian_sigma=1.0, channels=1)
    lse_smoothed = batched_log_operator_exp_trick(data, smoother, eps=eps)
    lse_smoothed2 = (smoother(data.exp()) + eps).log()
    torch.testing.assert_allclose(lse_smoothed, lse_smoothed2)
