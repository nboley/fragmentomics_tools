import os
import tempfile

import numpy
import pandas as pd
import pytest

from fragments_h5.fragments_h5 import build_fragments_h5, FragmentsH5


from fragmentomics_tools.fragment import Fragment
from fragmentomics_tools.region import Region

from fragmentomics_tools.fragment_array.fragment_array import (
    FragmentDoesNotIntersect,
    InvalidCoordinates,
)

from fragmentomics_tools import (
    RegionFragmentArray,
    FragmentArray,
    merge_fragment_arrays,
)


DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "./data/")


def test_no_fragments():
    fa = RegionFragmentArray([], [], Region("chr1", 0, 100), 100)
    assert fa.n_frags == 0


@pytest.fixture(scope="module")
def bam_path():
    return os.path.join(DATA_DIR, "./small.chr6.bam")


@pytest.fixture(scope="module")
def fasta_file_path():
    return os.path.join(DATA_DIR, "./GRCh38.p12.genome.chr6_99110000_99130000.fa.gz")


@pytest.fixture(scope="module")
def small_h5_path(bam_path, fasta_file_path):
    with tempfile.TemporaryDirectory() as dirname:
        ofname = os.path.join(dirname, os.path.basename(bam_path) + ".frag.h5")
        build_fragments_h5(
            bam_path, ofname, "test_sample", "hg38", fasta_file=fasta_file_path
        )
        yield ofname


regions = [
    Region("chr6", 99119615, 99119634),  # CTCF Constitutive
    # ("chr10", 127907312, 127907911),
    # ("chr10", 25933676, 25934876),
]


@pytest.mark.parametrize(
    "min_mapq, max_frag_len, region",
    [
        (0, 127, regions[0]),
    ],  # , (20, 511, regions[1]), (60, 1023, regions[2])],
)
def test_fragment_array_against_fragment_matrix(
    small_h5_path, min_mapq, max_frag_len, region
):
    # make sure creating a fragment matrix from a fragment array produces is equivalent to
    # creating the fragment matrix from scratch
    assert isinstance(max_frag_len, int)
    assert isinstance(region, Region)

    frag_array = RegionFragmentArray.from_fragments_h5(
        FragmentsH5(small_h5_path, "r"),
        region=region,
        min_mapq=min_mapq,
        max_frag_len=max_frag_len,
    )
    frag_mat = frag_array.fragment_matrix

    numpy.testing.assert_equal(
        frag_array.fragment_matrix.dense_array, frag_mat.dense_array
    )


def test_fragment_array_assertions():
    with pytest.raises(FragmentDoesNotIntersect):
        # 6,7 is out of bounds
        RegionFragmentArray([-1, 6], [3, 7], Region("chr1", 0, 5), 511)

    with pytest.raises(FragmentDoesNotIntersect):
        # -1, 0 is out of bounds
        RegionFragmentArray([-1, 3], [0, 6], Region("chr1", 0, 5), 511)

    with pytest.raises(TypeError):
        # no floating point numbers
        RegionFragmentArray(
            numpy.array([0, 3.0]), numpy.array([2, 4]), Region("chr1", 0, 5), 511
        )

    with pytest.raises(InvalidCoordinates):
        RegionFragmentArray([2], [-1], Region("chr1", 0, 5), 511)


@pytest.mark.parametrize(
    "region, expected_starts, expected_stops",
    [
        (  # The full region
            Region("chr6", 99119500, 99129900),
            numpy.array([-40, 120, 173, 197, 229, 337, 368, 368]),
            numpy.array([137, 363, 317, 511, 404, 668, 447, 531]),
        ),
        (  # Slice with no fragments is empty
            Region("chr1", 0, 100),
            numpy.array([], dtype=int),
            numpy.array([], dtype=int),
        ),
    ],
)
def test_boundary_conditions(small_h5_path, region, expected_starts, expected_stops):
    fa = RegionFragmentArray.from_fragments_h5(
        FragmentsH5(small_h5_path, "r"), region=region
    )
    fm = fa.fragment_matrix
    numpy.testing.assert_equal(fa.starts_0, expected_starts)
    numpy.testing.assert_equal(fa.stops_0, expected_stops)
    numpy.testing.assert_equal(fa.dense_array, fm.dense_array)


@pytest.mark.parametrize(
    "frag_arr",
    [
        RegionFragmentArray(
            starts_0=[-3, -2, 0, 1, 2, 3, 4],
            stops_0=[1, 2, 2, 3, 4, 5, 6],
            weights=[1.0, 0.9, 1.0, 0.1, 1.0, 1.5, 1.0],
            first_covered_base_weights=[1.0, 0.8, 1.0, 0.1, 1.0, 1.5, 1.0],
            last_covered_base_weights=[1.0, 0.9, 1.0, 0.1, 2.0, 1.5, 1.0],
            region=Region("chr1", 0, 5, strand="+"),
            max_frag_len=100,
        ),
        RegionFragmentArray(
            starts_0=[-3, -2, 0, 1, 2, 3, 4],
            stops_0=[1, 2, 2, 3, 4, 5, 6],
            region=Region("chr1", 0, 5, strand="+"),
            max_frag_len=100,
        ),
        RegionFragmentArray(
            starts_0=[],
            stops_0=[],
            region=Region("chr1", 0, 5, strand="+"),
            max_frag_len=100,
        ),
        FragmentArray(
            starts_0=[-3, -2, 0, 1, 2, 3, 4],
            stops_0=[1, 2, 2, 3, 4, 5, 6],
            length=5,
            max_frag_len=100,
        ),
        FragmentArray(
            starts_0=[],
            stops_0=[],
            length=5,
            max_frag_len=100,
        ),
    ],
)
def test_sum(frag_arr):
    additive_identity_downcast = FragmentArray(
        starts_0=[],
        stops_0=[],
        length=frag_arr.length,
        max_frag_len=frag_arr.max_frag_len,
    )
    summed = sum([frag_arr, frag_arr]).sort_in_place()
    merged = merge_fragment_arrays([frag_arr, frag_arr]).sort_in_place()
    summed_downcast = sum(
        [frag_arr, additive_identity_downcast, frag_arr]
    ).sort_in_place()
    merged_downcast = merge_fragment_arrays(
        [frag_arr, additive_identity_downcast, frag_arr]
    ).sort_in_place()
    added = (frag_arr + frag_arr).sort_in_place()
    assert summed == added
    assert merged == summed
    assert type(added) == type(summed)
    assert type(added) == type(frag_arr)
    assert type(summed_downcast) == type(additive_identity_downcast)
    assert type(merged_downcast) == type(additive_identity_downcast)
    numpy.testing.assert_equal(summed_downcast.starts_0, summed.starts_0)
    numpy.testing.assert_equal(merged_downcast.starts_0, summed.starts_0)
    numpy.testing.assert_equal(summed_downcast.stops_0, summed.stops_0)
    numpy.testing.assert_equal(merged_downcast.stops_0, summed.stops_0)
    numpy.testing.assert_equal(summed_downcast.weights, summed.weights)
    numpy.testing.assert_equal(merged_downcast.weights, summed.weights)
    numpy.testing.assert_equal(
        summed_downcast.first_covered_base_weights, summed.first_covered_base_weights
    )
    numpy.testing.assert_equal(
        merged_downcast.first_covered_base_weights, summed.first_covered_base_weights
    )
    numpy.testing.assert_equal(
        summed_downcast.last_covered_base_weights, summed.last_covered_base_weights
    )
    numpy.testing.assert_equal(
        merged_downcast.last_covered_base_weights, summed.last_covered_base_weights
    )


@pytest.mark.parametrize(
    "input_fa, expected_fa",
    [
        (  # The full region
            RegionFragmentArray(
                starts_0=[-3, -2, 0, 1, 2, 3, 4],
                stops_0=[1, 2, 2, 3, 4, 5, 6],
                region=Region("chr1", 0, 5, strand="+"),
                max_frag_len=100,
            ),
            RegionFragmentArray(
                starts_0=[-1, 0, 1, 2, 3, 3, 4],
                stops_0=[1, 2, 3, 4, 5, 7, 8],
                region=Region("chr1", 0, 5, strand="-"),
                max_frag_len=100,
            ),
        ),
        (  # Slice with two reads at begin/end edges
            FragmentArray(
                starts_0=[-3, -2, 0, 1, 2, 3, 4],
                stops_0=[1, 2, 2, 3, 4, 5, 6],
                length=5,
                max_frag_len=100,
            ),
            FragmentArray(
                starts_0=[-1, 0, 1, 2, 3, 3, 4],
                stops_0=[1, 2, 3, 4, 5, 7, 8],
                length=5,
                max_frag_len=100,
            ),
        ),
        (  # Slice with no fragments is empty
            RegionFragmentArray(
                starts_0=[-3, -2, 0, 1, 2, 3, 4],
                stops_0=[1, 2, 2, 3, 4, 5, 6],
                weights=[1.0, 0.9, 1.0, 0.1, 1.0, 1.5, 1.0],
                first_covered_base_weights=[1.0, 0.8, 1.0, 0.1, 1.0, 1.5, 1.0],
                last_covered_base_weights=[1.0, 0.9, 1.0, 0.1, 2.0, 1.5, 1.0],
                region=Region("chr1", 0, 5),
                max_frag_len=100,
            ),
            RegionFragmentArray(
                starts_0=[-1, 0, 1, 2, 3, 3, 4],
                stops_0=[1, 2, 3, 4, 5, 7, 8],
                weights=[1.0, 1.5, 1.0, 0.1, 1.0, 0.9, 1.0],
                first_covered_base_weights=[1.0, 1.5, 2.0, 0.1, 1.0, 0.9, 1.0],
                last_covered_base_weights=[1.0, 1.5, 1.0, 0.1, 1.0, 0.8, 1.0],
                region=Region("chr1", 0, 5),
                max_frag_len=100,
            ),
        ),
    ],
)
def test_reverse_complement(input_fa: FragmentArray, expected_fa: FragmentArray):
    assert type(input_fa) == type(expected_fa)
    rc_input_fa = input_fa.reverse_strand()
    rc_rc_input_fa = rc_input_fa.reverse_strand()
    assert rc_rc_input_fa == input_fa
    assert input_fa != expected_fa
    assert type(rc_input_fa) == type(expected_fa)
    assert rc_input_fa == expected_fa


@pytest.mark.parametrize(
    "input_fa, expected_covered, expected_n_frags",
    [
        (
            RegionFragmentArray(
                starts_0=[],
                stops_0=[],
                region=Region("chr1", 1, 6, strand="+"),
                max_frag_len=100,
            ),
            0,
            0,
        ),
        (
            FragmentArray(
                starts_0=[],
                stops_0=[],
                length=6,
                max_frag_len=100,
            ),
            0,
            0,
        ),
        (
            RegionFragmentArray(
                starts_0=[-5, -2],
                stops_0=[7, 8],
                region=Region("chr1", 1, 6, strand="+"),
                max_frag_len=100,
            ),
            0,
            2,
        ),
        (
            FragmentArray(
                starts_0=[-5, -2],
                stops_0=[7, 8],
                length=5,
                max_frag_len=100,
            ),
            0,
            2,
        ),
    ],
)
def test_empty_array_covered_base_counts(
    input_fa: FragmentArray, expected_covered: int, expected_n_frags: int
):
    first_covered_arr = input_fa.first_covered_base_counts
    last_covered_arr = input_fa.last_covered_base_counts
    assert len(first_covered_arr) == input_fa.length
    assert len(last_covered_arr) == input_fa.length
    assert input_fa.n_frags == expected_n_frags
    assert numpy.sum(first_covered_arr) == expected_covered
    assert numpy.sum(last_covered_arr) == expected_covered


@pytest.mark.parametrize(
    "input_fa, shift, expected_fa",
    [
        (  # The full region
            RegionFragmentArray(
                starts_0=[-2, 0, 1, 2, 3],
                stops_0=[2, 2, 3, 4, 5],
                region=Region("chr1", 1, 6, strand="+"),
                max_frag_len=100,
            ),
            -1,
            RegionFragmentArray(
                starts_0=[-3, -1, 0, 1, 2],
                stops_0=[1, 1, 2, 3, 4],
                region=Region("chr1", 0, 5, strand="+"),
                max_frag_len=100,
            ),
        ),
        (
            RegionFragmentArray(
                starts_0=[-2, 0, 1, 2, 3],
                stops_0=[2, 2, 3, 4, 5],
                region=Region("chr1", 1, 6, strand="+"),
                max_frag_len=100,
            ),
            1,
            RegionFragmentArray(
                starts_0=[-1, 1, 2, 3, 4],
                stops_0=[3, 3, 4, 5, 6],
                region=Region("chr1", 2, 7, strand="+"),
                max_frag_len=100,
            ),
        ),
        (
            RegionFragmentArray(
                starts_0=[-2, 0, 1, 2, 3],
                stops_0=[2, 2, 3, 4, 5],
                region=Region("chr1", 1, 6, strand="+"),
                max_frag_len=100,
            ),
            2,
            RegionFragmentArray(
                starts_0=[0, 2, 3, 4],
                stops_0=[4, 4, 5, 6],
                region=Region("chr1", 3, 8, strand="+"),
                max_frag_len=100,
            ),
        ),
        (
            RegionFragmentArray(
                starts_0=[-2, 0, 1, 2, 3],
                stops_0=[2, 2, 3, 4, 5],
                region=Region("chr1", 2, 7, strand="+"),
                max_frag_len=100,
            ),
            -2,
            RegionFragmentArray(
                starts_0=[-1, 0, 1],
                stops_0=[1, 2, 3],
                region=Region("chr1", 0, 5, strand="+"),
                max_frag_len=100,
            ),
        ),
        (
            FragmentArray(
                starts_0=[-2, 0, 1, 2, 3],
                stops_0=[2, 2, 3, 4, 5],
                length=5,
                max_frag_len=100,
            ),
            -2,
            FragmentArray(
                starts_0=[-1, 0, 1],
                stops_0=[1, 2, 3],
                length=5,
                max_frag_len=100,
            ),
        ),
        (
            FragmentArray(
                starts_0=[-2, 0, 1, 2, 3],
                stops_0=[2, 2, 3, 4, 5],
                length=5,
                max_frag_len=100,
            ),
            2,
            FragmentArray(
                starts_0=[0, 2, 3, 4],
                stops_0=[4, 4, 5, 6],
                length=5,
                max_frag_len=100,
            ),
        ),
    ],
)
def test_shift(input_fa: FragmentArray, shift: int, expected_fa: FragmentArray):
    assert type(input_fa) == type(expected_fa)
    shift_input_fa = input_fa.shift_and_zero_pad(shift)
    if abs(shift) < 2:
        # For the small shift also make sure that the identity holds.
        # This means we save boundary condition tests involving expected drops
        #  for larger shifts in our tests.
        re_shift_input_fa = shift_input_fa.shift_and_zero_pad(-shift)
        assert input_fa == re_shift_input_fa
    assert input_fa != expected_fa
    assert type(shift_input_fa) == type(expected_fa)
    assert shift_input_fa == expected_fa


@pytest.mark.parametrize(
    "input_fa, new_size, expected_fa",
    [
        (  # The full region
            RegionFragmentArray(
                starts_0=[-2, 0, 1, 2, 3],
                stops_0=[2, 2, 3, 4, 5],
                region=Region("chr1", 1, 6, strand="+"),
                max_frag_len=100,
            ),
            3,
            RegionFragmentArray(
                starts_0=[-3, -1, 0, 1, 2],
                stops_0=[1, 1, 2, 3, 4],
                region=Region("chr1", 2, 5, strand="+"),
                max_frag_len=100,
            ),
        ),
        (  # The full region
            FragmentArray(
                starts_0=[-2, 0, 1, 2, 3],
                stops_0=[2, 2, 3, 4, 5],
                length=5,
                max_frag_len=100,
            ),
            3,
            FragmentArray(
                starts_0=[-3, -1, 0, 1, 2],
                stops_0=[1, 1, 2, 3, 4],
                length=3,
                max_frag_len=100,
            ),
        ),
        (
            RegionFragmentArray(
                starts_0=[-2, 0, 1, 2, 3],
                stops_0=[2, 2, 3, 4, 5],
                region=Region("chr1", 1, 6, strand="+"),
                max_frag_len=100,
            ),
            2,
            RegionFragmentArray(
                starts_0=[-3, -1, 0, 1],
                stops_0=[1, 1, 2, 3],
                region=Region("chr1", 2, 4, strand="+"),
                max_frag_len=100,
            ),
        ),
        (
            RegionFragmentArray(
                starts_0=[-2, 0, 1, 2, 3],
                stops_0=[2, 2, 3, 4, 5],
                region=Region("chr1", 1, 6, strand="-"),
                max_frag_len=100,
            ),
            2,
            RegionFragmentArray(
                starts_0=[-1, 0, 1],
                stops_0=[1, 2, 3],
                region=Region("chr1", 3, 5, strand="-"),
                max_frag_len=100,
            ),
        ),
        (
            RegionFragmentArray(
                starts_0=[-2, 0, 1, 2, 3],
                stops_0=[2, 2, 3, 4, 5],
                region=Region("chr1", 1, 7, strand="+"),
                max_frag_len=100,
            ),
            3,
            RegionFragmentArray(
                starts_0=[-1, 0, 1],
                stops_0=[1, 2, 3],
                region=Region("chr1", 3, 6, strand="+"),
                max_frag_len=100,
            ),
        ),
        (  # The full region
            FragmentArray(
                starts_0=[-2, 0, 1, 2, 3],
                stops_0=[2, 2, 3, 4, 5],
                length=6,
                max_frag_len=100,
            ),
            3,
            FragmentArray(
                starts_0=[-1, 0, 1],
                stops_0=[1, 2, 3],
                length=3,
                max_frag_len=100,
            ),
        ),
        (
            RegionFragmentArray(
                starts_0=[-2, 0, 1, 2, 3],
                stops_0=[2, 2, 3, 4, 5],
                region=Region("chr1", 1, 7, strand="+"),
                max_frag_len=100,
            ),
            2,
            RegionFragmentArray(
                starts_0=[-1, 0, 1],
                stops_0=[1, 2, 3],
                region=Region("chr1", 3, 5, strand="+"),
                max_frag_len=100,
            ),
        ),
        (
            RegionFragmentArray(
                starts_0=[-2, 0, 1, 2, 3],
                stops_0=[2, 2, 3, 4, 5],
                region=Region("chr1", 1, 7, strand="-"),
                max_frag_len=100,
            ),
            2,
            RegionFragmentArray(
                starts_0=[-1, 0, 1],
                stops_0=[1, 2, 3],
                region=Region("chr1", 3, 5, strand="-"),
                max_frag_len=100,
            ),
        ),
        (  # The full region
            FragmentArray(
                starts_0=[-2, 0, 1, 2, 3],
                stops_0=[2, 2, 3, 4, 5],
                length=6,
                max_frag_len=100,
            ),
            2,
            FragmentArray(
                starts_0=[-1, 0, 1],
                stops_0=[1, 2, 3],
                length=2,
                max_frag_len=100,
            ),
        ),
    ],
)
def test_resize(input_fa: FragmentArray, new_size: int, expected_fa: FragmentArray):
    resized = input_fa.resize(new_size)
    assert type(resized) == type(input_fa)
    assert resized == expected_fa


def test_data_validation():
    with pytest.raises(
        ValueError, match=".*The length of starts_0 must be the same as stops_0.*"
    ):
        _ = RegionFragmentArray(
            starts_0=[-3, -2, 0, 1, 2, 3, 4, 33],
            stops_0=[1, 2, 2, 3, 4, 5, 6],
            weights=[1.0, 0.9, 1.0, 0.1, 1.0, 1.5, 1.0],
            first_covered_base_weights=[1.0, 0.8, 1.0, 0.1, 1.0, 1.5, 1.0],
            last_covered_base_weights=[1.0, 0.9, 1.0, 0.1, 2.0, 1.5, 1.0],
            region=Region("chr1", 0, 5),
            max_frag_len=100,
        )
    with pytest.raises(ValueError, match=".*Weights length should match data length.*"):
        _ = RegionFragmentArray(
            starts_0=[-3, -2, 0, 1, 2, 3, 4],
            stops_0=[1, 2, 2, 3, 4, 5, 6],
            weights=[1.0, 0.9, 1.0, 0.1, 1.0, 1.5, 1.0, 33.0],
            first_covered_base_weights=[1.0, 0.8, 1.0, 0.1, 1.0, 1.5, 1.0],
            last_covered_base_weights=[1.0, 0.9, 1.0, 0.1, 2.0, 1.5, 1.0],
            region=Region("chr1", 0, 5),
            max_frag_len=100,
        )
    with pytest.raises(
        ValueError,
        match=".*First covered base weights length should match data length.*",
    ):
        _ = RegionFragmentArray(
            starts_0=[-3, -2, 0, 1, 2, 3, 4],
            stops_0=[1, 2, 2, 3, 4, 5, 6],
            weights=[1.0, 0.9, 1.0, 0.1, 1.0, 1.5, 1.0],
            first_covered_base_weights=[1.0, 0.8, 1.0, 0.1, 1.0, 1.5, 1.0, 33.0],
            last_covered_base_weights=[1.0, 0.9, 1.0, 0.1, 2.0, 1.5, 1.0],
            region=Region("chr1", 0, 5),
            max_frag_len=100,
        )
    with pytest.raises(
        ValueError,
        match=".*Last covered base weights length should match data length.*",
    ):
        _ = RegionFragmentArray(
            starts_0=[-3, -2, 0, 1, 2, 3, 4],
            stops_0=[1, 2, 2, 3, 4, 5, 6],
            weights=[1.0, 0.9, 1.0, 0.1, 1.0, 1.5, 1.0],
            first_covered_base_weights=[1.0, 0.8, 1.0, 0.1, 1.0, 1.5, 1.0],
            last_covered_base_weights=[1.0, 0.9, 1.0, 0.1, 2.0, 1.5, 1.0, 33.0],
            region=Region("chr1", 0, 5),
            max_frag_len=100,
        )


@pytest.mark.parametrize(
    "rfa",
    [
        RegionFragmentArray(
            starts_0=[-3, -2, 0, 1, 2, 3, 4],
            stops_0=[1, 2, 2, 3, 4, 5, 6],
            region=Region("chr1", 0, 5, strand="+"),
            max_frag_len=100,
        ),
        # Slice with no fragments is empty
        RegionFragmentArray(
            starts_0=[-2, -3, 0, 1, 2, 3, 4],
            stops_0=[2, 1, 2, 3, 4, 5, 6],
            weights=[1.0, 0.9, 1.0, 0.1, 1.0, 1.5, 1.0],
            first_covered_base_weights=[1.0, 0.8, 1.0, 0.1, 1.0, 1.5, 1.0],
            last_covered_base_weights=[1.0, 0.9, 1.0, 0.1, 2.0, 1.5, 1.0],
            region=Region("chr1", 0, 5),
            max_frag_len=100,
        ),
    ],
)
def test_save_load_region_fragment_array(rfa: RegionFragmentArray, tmpdir):
    pth = tmpdir.mkdir("test_save_load_region_fragment_array").join("test.rfa.h5")
    rfa.save(pth)
    lrfa = rfa.load(pth)
    assert rfa == lrfa


def test_from_fragments_h5(small_h5_path):
    region = Region("chr6", 99118615, 99121634).resize(2048)
    rfa = RegionFragmentArray.from_fragments_h5(
        small_h5_path, region, include_fragment_strand=True
    )
    assert rfa.n_frags == 9
