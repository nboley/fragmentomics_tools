import os
import tempfile
from collections import Counter
from typing import Sequence
import numpy
import pandas as pd
import pytest
from pathlib import Path

from fragmentomics_tools.dataframe import RegionDataFrame, intersect_region_dataframes, SampleAndRegionDataFrame
from fragmentomics_tools.region import Region
from fragmentomics_tools.fragment_array import RegionFragmentArray

# from datamanifest import DataManifest


class TSSs(RegionDataFrame):
    pass


TEST_DATA_DIR = Path(__file__).parent.joinpath("./data")

TEST_GENE_ENSEMBLE_IDS = [
    "ENSG00000000419",
    "ENSG00000001167",
    "ENSG00000002586",
    "ENSG00000002834",
    "ENSG00000003056",
    "ENSG00000003756",
]

TSS_ANNOTATION_FILE = os.path.join(TEST_DATA_DIR, "tss.all_rampage.hg19.bed.gz")
TSS_ANNOTATION_FILE_HG38 = os.path.join(TEST_DATA_DIR, "tss.all_rampage.hg38.bed.gz")
CTCF_BED_FILE = os.path.join(TEST_DATA_DIR, "CTCF.matches.known.hg38.bed.gz")
H3K4ME3_SIGNAL = os.path.join(TEST_DATA_DIR, "E001-H3K4me3.imputed.pval.signal.hg38.bigwig")
BLACK_LIST_FILE_HG38 = os.path.join(TEST_DATA_DIR, "hg38-blacklist.v2.sorted.bed.gz")

def load_bed_from_path(fpath, ref):
    pass


def test_load_region_data_frame():
    rdf = RegionDataFrame(pd.read_table(TSS_ANNOTATION_FILE), ref="hg19")
    assert len(rdf) == 18263


def test_load_region_data_frame_from_fname():
    rdf = RegionDataFrame(pd.read_table(TSS_ANNOTATION_FILE), ref="hg19")
    assert len(rdf) == 18263


def test_load_tss_s():
    rdf = RegionDataFrame(pd.read_table(TSS_ANNOTATION_FILE), ref="hg19")
    assert len(rdf) == 18263


def test_resize_vs_vector():
    rdf0 = RegionDataFrame(pd.read_table(TSS_ANNOTATION_FILE), ref="hg19").resize_regions(
        2000, vectorized=True
    )
    rdf1 = RegionDataFrame(pd.read_table(TSS_ANNOTATION_FILE), ref="hg19").resize_regions(
        2000, vectorized=False
    )
    pd.testing.assert_frame_equal(rdf0, rdf1)


def test_get_pfm_consistent_with_resize():
    ctcf_rdf = (
        RegionDataFrame.from_bed(CTCF_BED_FILE, ref="hg38")
        .query("name == 'CTCF_known1' and contig == 'chr1'")
        .sample(100)
    )

    pfms = {}
    freqs = {}
    for region_size in [99, 100]:
        for motif_size in range(16, 21):
            pfms[(region_size, motif_size)] = ctcf_rdf.resize_regions(100).get_pfm(motif_size)
            freqs[(region_size, motif_size)] = pfms[(region_size, motif_size)].freqs.sum(axis=0)

    assert (freqs[(99, 16)] == freqs[(100, 16)]).all()
    expected_freq = freqs[(100, 16)]

    for region_size in [99, 100]:
        for motif_size in range(17, 21):
            if motif_size % 2 == 1:
                total_cut_off = motif_size - 16
                left_cut_off = total_cut_off // 2
                right_cut_off = total_cut_off - left_cut_off
                assert (freqs[(region_size, motif_size)][left_cut_off:-right_cut_off] == expected_freq).all()
            else:
                cut_off = (motif_size - 16) // 2
                assert (freqs[(region_size, motif_size)][cut_off:-cut_off] == expected_freq).all()


def test_load_tss_s_from_fname():
    rdf = RegionDataFrame.from_fname_s3_or_local(TSS_ANNOTATION_FILE, ref="hg19")
    assert len(rdf) == 18263


def test_filter_tss_s(TSS_ANNOTATION_FILE=TSSs(pd.read_table(TSS_ANNOTATION_FILE), ref="hg19")):
    tss_s = TSS_ANNOTATION_FILE
    res = tss_s.query("peak_tpm > 10")
    assert len(res) == 8489


def test_lift_over():
    rdf = RegionDataFrame(pd.read_table(TSS_ANNOTATION_FILE), ref="hg19").resize_regions(2000).iloc[:5]
    lifted_rdf = rdf.lift_over("hg38", transfer_columns=True)
    assert list(lifted_rdf["start"]) == [719150, 777818, 777974, 783373, 826656]
    assert list(lifted_rdf["stop"]) == [721150, 779818, 779974, 785373, 828656]
    assert list(lifted_rdf["id"]) == [
        "ENSG00000230021.8_pk1",
        "ENSG00000237491.8_pk1",
        "ENSG00000237491.8_pk2",
        "ENSG00000237491.8_pk3",
        "ENSG00000228794.8_pk1",
    ]


def test_lift_over_no_id_column():
    """The reason behind this test is that in a previous pr, 'id' column was removed from RegionDataFrame
    constructor, however it wasn't tested on liftover with inbput that doesn't have an 'id' column. In a fix,
    the line which checked for id column in lif_over function was removed"""

    rdf = RegionDataFrame.from_regions(
        [Region("chr1", 1000 + i * 500, 1500 + i * 500) for i in range(20, 100)], ref="hg19"
    )
    rdf_lift_over = rdf.lift_over("hg38", transfer_columns=True)

    assert (rdf.values == rdf_lift_over.values).all()


def test_get_fragment_coverage_sum():
    # tests that the sum bigwig signal over the first 10 TSSs is the values found below
    rdf = RegionDataFrame(pd.read_table(TSS_ANNOTATION_FILE_HG38), ref="hg38").resize_regions(2000)
    values = rdf.iloc[:10].get_fragment_coverage_sum(H3K4ME3_SIGNAL)
    assert numpy.isclose(
        values,
        numpy.array(
            [
                226617.27285767,
                125626.76041555,
                185629.98674583,
                953.75800273,
                97040.2532236,
                148298.5844841,
                63485.53418653,
                76950.97574207,
                97327.95037329,
                72133.48155385,
            ]
        ),
    ).all


def test_get_fragment_coverage_track():
    # tests that the bigwig signal tracks over the first 10 TSSs is the values found below
    rdf = RegionDataFrame(pd.read_table(TSS_ANNOTATION_FILE_HG38), ref="hg38")
    values = rdf.iloc[:10].get_fragment_coverage_track(H3K4ME3_SIGNAL)
    assert numpy.isclose(
        values,
        numpy.array(
            [
                [
                    [108.98120117, 94.61039734, 94.61039734, 94.61039734, 94.61039734, 94.61039734],
                    [97.06439972, 97.06439972, 97.06439972, 97.06439972, 97.06439972, 97.06439972],
                    [56.41839981, 56.41839981, 56.41839981, 56.41839981, 56.41839981, 56.41839981],
                    [0.99199998, 0.99199998, 1.09200001, 1.09200001, 1.09200001, 1.09200001],
                    [25.83200073, 25.83200073, 25.83200073, 25.83200073, 25.83200073, 25.83200073],
                    [11.46199989, 11.46199989, 11.46199989, 11.46199989, 11.46199989, 12.21640015],
                    [48.31000137, 48.31000137, 48.31000137, 48.31000137, 48.31000137, 48.31000137],
                    [46.19800186, 46.19800186, 46.19800186, 46.19800186, 65.94999695, 65.94999695],
                    [55.59400177, 55.59400177, 55.59400177, 37.52799988, 37.52799988, 37.52799988],
                    [34.10879898, 34.10879898, 34.10879898, 34.10879898, 34.10879898, 34.10879898],
                ]
            ]
        ),
    ).all


def test_tf_annotation():
    rdf_1 = RegionDataFrame.from_regions([Region("chr1", 1000, 1500), Region("chr1", 2000, 2100)], ref="hg19")
    motifs = rdf_1.center_regions_on_tf_motif(
        target_tfs=["CTCF"],
        target_len=11,
        tf_search_width=50,
        num_workers=1,
        batch_size=2,
        cuda=False,
        verify_motif_scores=True,
        inplace=False,
        unique_regions=True,
        shuffle_tf_motif=False,
        shuffle_tf_seed=314,
    )
    assert motifs.shape[0] == 2
    assert motifs.shape[-1] >= 12
    missing_cols = {
        "strand",
        "tf_on_rev_strand",
        "tf_top_offset",
        "tf_top_score",
        "target_tfs",
        "original_start",
        "original_stop",
        "query_start",
        "query_stop",
    } - set(motifs.columns)
    assert len(missing_cols) == 0, missing_cols


def test_intersect_region_dataframe():
    rdf_1 = RegionDataFrame.from_regions([Region("chr1", 1000, 1500), Region("chr1", 2000, 2100)], ref="hg19")
    rdf_2 = RegionDataFrame.from_regions([Region("chr1", 1300, 2100), Region("chr3", 4000, 4100)], ref="hg19")
    intersection_rdf = intersect_region_dataframes([rdf_1, rdf_2])
    assert isinstance(intersection_rdf, RegionDataFrame)
    intersection_regions = list(intersection_rdf.iter_regions())
    assert intersection_regions[0] == Region("chr1", 1300, 1500, ref="hg19")
    assert intersection_regions[1] == Region("chr1", 2000, 2100, ref="hg19")
    # Also check that the operator on an RDF gives the same results
    for region_1, region_2 in zip(
        rdf_1.intersect_with_rdf(rdf_2).iter_regions(),
        intersect_region_dataframes([rdf_1, rdf_2]).iter_regions(),
    ):
        assert region_1 == region_2, f"{region_1}, {region_2}"


def test_overlaps_with_bed():
    num_regions_checked = 20
    rdf = RegionDataFrame(
        pd.read_table(TSS_ANNOTATION_FILE_HG38), ref="hg38"
    )[:num_regions_checked]

    expected_values = []
    blacklist_df = pd.read_csv(BLACK_LIST_FILE_HG38, sep="\t", names=["contig", "start", "stop"])
    for region in rdf.iter_regions():
        q = blacklist_df.query("contig == @region.chrom and start < @region.stop and stop > @region.start")
        expected_values.append(len(q) > 0)

    values = rdf.overlaps_with_bed(blacklist_path)

    assert numpy.all(values == numpy.array(expected_values))


@pytest.mark.parametrize("inplace", [True, False])
def test_drop_unlabeled_records(inplace):
    tss_s = TSSs(pd.read_table(TSS_ANNOTATION_FILE), ref="hg19")
    tss_s = tss_s.set_binary_label_by_thresholds(
        "peak_tpm", on_threshold=25, off_threshold=10, drop_unlabeled_records=False
    )
    assert Counter(tss_s.label.tolist()) == Counter({0: 9773, 1: 5566, -1: 2924})
    tss_s = tss_s.drop_unlabeled_records(inplace=inplace)
    assert Counter(tss_s.label.tolist()) == Counter({0: 9773, 1: 5566})


def test_set_binary_label_by_threshold():
    tss_s = TSSs(pd.read_table(TSS_ANNOTATION_FILE), ref="hg19")

    # test that we can set labels using a string columns
    tss_s = tss_s.set_binary_label_by_thresholds(
        "peak_tpm", on_threshold=25, off_threshold=10, drop_unlabeled_records=False
    )
    assert Counter(tss_s.label.tolist()) == Counter({0: 9773, 1: 5566, -1: 2924})

    # test that we can set labels using a string columns, and that drop_unlabeled_records works
    tss_s = tss_s.set_binary_label_by_thresholds(
        ["peak_tpm",], on_threshold=25, off_threshold=10, drop_unlabeled_records=True
    )
    assert Counter(tss_s.label.tolist()) == Counter({0: 9773, 1: 5566})


def test_set_binary_label():
    tss_s = TSSs(pd.read_table(TSS_ANNOTATION_FILE), ref="hg19")

    # test that just an on query works
    tss_s = tss_s.set_binary_label("peak_tpm > 25")
    assert Counter(tss_s.label.tolist()) == Counter({0: 12697, 1: 5566})

    # test that an on and off work
    tss_s = tss_s.set_binary_label("peak_tpm > 25", "peak_tpm < 10", drop_unlabeled_records=False)
    assert Counter(tss_s.label.tolist()) == Counter({0: 9773, 1: 5566, -1: 2924})

    # test that drop_unlabeled_records works
    tss_s = tss_s.set_binary_label("peak_tpm > 25", "peak_tpm < 10", drop_unlabeled_records=True)
    assert Counter(tss_s.label.tolist()) == Counter({0: 9773, 1: 5566})


def test_to_tsv():
    # FIXME: load_bed_from_path is not standard for reading region dataframes
    #  Also, it uses some bw_df bigwig. Is it supposed to be for bigwigs?
    rdf = RegionDataFrame.from_bed(TSS_ANNOTATION_FILE_HG38, ref="hg38").iloc[:10, :]

    with tempfile.TemporaryDirectory() as tmp_dir:
        rdf.to_tsv(f"{tmp_dir}/tmp.tsv")
        # FIXME: This is not a region dataframe, it's a pandas dataframe
        rdf_r = pd.read_csv(f"{tmp_dir}/tmp.tsv", sep="\t")

    assert (rdf.columns == rdf_r.columns).all()
    assert rdf.shape == rdf_r.shape


@pytest.fixture
def rdf():
    return RegionDataFrame.from_bed(TSS_ANNOTATION_FILE_HG38, ref="hg38").iloc[:5, :]


@pytest.fixture
def sdf():
    assert False
    return SAMPLE_DF.select_sample_set("ml_capture_batch3").get_sample_names(["W_1", "W_2"])


@pytest.fixture
def sample_id_to_h5_path():
    return {
        "SEQRUN8_LIBRARY8": "./fragment_h5s/v1/SEQRUN8_LIBRARY8.fragments.h5",
        "SEQRUN9_LIBRARY7": "./fragment_h5s/v1/SEQRUN9_LIBRARY7.fragments.h5",
    }


def test_build_sample_region_dataframe(rdf, sdf):
    srdf = rdf.build_sample_region_dataframe(sdf)
    assert srdf.shape[0] == 2 * rdf.shape[0]
    assert srdf.shape[1] == rdf.shape[1] + 1


def test_get_sample_df(rdf, sdf):
    srdf = rdf.build_sample_region_dataframe(sdf)
    sdf_2 = srdf.get_sample_df()
    assert sdf.sort_index().equals(sdf_2.sort_index())


def test_get_sample_df_keep_sample_metadata(rdf, sdf):
    srdf = rdf.build_sample_region_dataframe(sdf, keep_sample_metadata=True)
    assert all(x in srdf.columns for x in sdf)


def test_get_sample_df_keep_sample_metadata_error_on_wrong_type(rdf, sample_id_to_h5_path):
    with pytest.raises(ValueError):
        srdf = rdf.build_sample_region_dataframe(sdf, keep_sample_metadata=True)


def test_split_on_contig():
    rdf = RegionDataFrame(pd.read_table(TSS_ANNOTATION_FILE), ref="hg19")

    d1, d2, d3 = rdf.split_on_contig(["chr1", "chr2", ["chr3", "chr4"]])
    assert d1.shape[0] == 1659
    assert d2.shape[0] == 1327
    assert d3.shape[0] == 1878

    with pytest.raises(ValueError):
        d1, d2 = rdf.split_on_contig([["chr1", "chr2"], ["chr2", "chr3"]])


def test_rdfs_equal():
    rdf = RegionDataFrame(pd.read_table(TSS_ANNOTATION_FILE_HG38), ref="hg38").iloc[:5, :]
    rdf_same = RegionDataFrame(pd.read_table(TSS_ANNOTATION_FILE_HG38), ref="hg38").iloc[:5, :]
    rdf_different_1 = RegionDataFrame(pd.read_table(TSS_ANNOTATION_FILE_HG38), ref="hg38").iloc[
        :10, :
    ]
    rdf_different_2 = RegionDataFrame(TSS_ANNOTATION_FILE_HG38, ref="hg38").iloc[5:10, :]

    assert rdf.equals(rdf_same)
    assert not rdf.equals(rdf_different_1)
    assert not rdf.equals(rdf_different_2)


def test__get_fragment_h5_paths_from_sdf(rdf, sdf):
    srdf = rdf.build_sample_region_dataframe(sdf)
    fragment_h5_paths = srdf._get_fragment_h5_paths()
    assert sorted(fragment_h5_paths.keys()) == ["SEQRUN8_LIBRARY8", "SEQRUN9_LIBRARY7"]
    assert fragment_h5_paths["SEQRUN8_LIBRARY8"].endswith("fragment_h5s/v1/SEQRUN8_LIBRARY8.fragments.h5")
    assert fragment_h5_paths["SEQRUN9_LIBRARY7"].endswith("fragment_h5s/v1/SEQRUN9_LIBRARY7.fragments.h5")


def test__get_fragment_h5_paths_from_mapping(rdf, sample_id_to_h5_path):
    srdf = rdf.build_sample_region_dataframe(sample_id_to_h5_path)
    fragment_h5_paths = srdf._get_fragment_h5_paths()
    assert sorted(sample_id_to_h5_path.values()) == sorted(fragment_h5_paths.values())


def test__get_fragment_h5_paths_from_collection(rdf, sample_id_to_h5_path):
    h5_paths = sorted(sample_id_to_h5_path.values())

    # make sure that we get an error when the paths aren't reachable
    with pytest.raises(OSError):
        srdf = rdf.build_sample_region_dataframe(h5_paths)

    h5_paths = [os.path.join(REPO_DATA_DIR, path) for path in list(h5_paths)]
    srdf = rdf.build_sample_region_dataframe(h5_paths)
    fragment_h5_paths = srdf._get_fragment_h5_paths()
    assert sorted(fragment_h5_paths.values()) == sorted(h5_paths)
