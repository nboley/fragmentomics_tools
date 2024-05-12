import glob

import pandas as pd
import numpy as np

from fragments_h5 import FragmentsH5
from fragmentomics_tools.plot.tracks import (
    Tracks,
    VLine,
    VplotTrack,
    CoverageDifferenceTrack,
    StrandSplitCoverageTrack,
    VectorTrack,
)
from fragmentomics_tools.dataframe import SampleDataFrame
from fragmentomics_tools.fragment_array import merge_fragment_arrays

NUM_CORES = 16
DEFAULT_MIN_MAPQ = 10
DEFAULT_MAX_FRAG_LEN = 511
# the amount we expand regions beyond the tss and polya
BOUNDARY_EXPANSION_SIZE = 1024 * 4

##### Samples ############################################################################################################


def load_ibd_sample_df():
    label_mapping = {
        "Asymptomatic": 0,
        "Remission": 1,
        "Mild": 2,
        "Moderate": 3,
        "Severe": 4,
    }

    sample_ids_to_samples = {}
    for fname in glob.glob("/ssd/karius/frag_h5s/*.fragments.h5"):
        sample_id = fname.split("/")[4].split(".")[1].split("_")[0][:-5]
        sample_ids_to_samples[sample_id] = fname

    assert (
        1
        == pd.read_csv("/scratch/karius/Merged_meta_data_CD_UC_foundation.csv")[
            "SampleID"
        ]
        .value_counts()
        .max()
    )
    metadata = pd.read_csv("/scratch/karius/Merged_meta_data_CD_UC_foundation.csv")
    metadata = metadata.query("SampleID in @sample_ids_to_samples.keys()")[
        ["SampleID", "SeqRun", "DIAGNOSIS", "ENDO_CATEGORY", "Cluster", "SITE_NUMBER"]
    ].rename(columns={"SampleID": "sample_id"})
    metadata = metadata.assign(
        label=metadata.apply(lambda x: label_mapping[x.ENDO_CATEGORY], 1)
    )
    metadata = metadata.assign(
        frag_h5=metadata.apply(
            lambda x: FragmentsH5(sample_ids_to_samples[x.sample_id]), 1
        )
    )[
        [
            "sample_id",
            "SeqRun",
            "label",
            "frag_h5",
            "DIAGNOSIS",
            "ENDO_CATEGORY",
            "Cluster",
            "SITE_NUMBER",
        ]
    ]
    return SampleDataFrame(metadata).sort_values("label")


def build_marker_gene_rdf(genes_rdf):
    marker_genes = pd.read_table(
        "/scratch/karius/reference/markers.immune_vs_epithelial.tsv", sep=" "
    )
    up_marker_genes = marker_genes.query("p_val_adj < 0.01 and avg_log2FC < -3").assign(
        direction="up_in_colon"
    )
    down_marker_genes = marker_genes.query(
        "p_val_adj < 0.01 and avg_log2FC > 7"
    ).assign(direction="down_in_colon")
    marker_genes = pd.concat([up_marker_genes, down_marker_genes])
    return (
        genes_rdf.set_index("gene_name")
        .join(marker_genes, how="inner")
        .reset_index()
        .reset_index(drop=True)
    )  # .query("direction == 'up_in_colon' and expression < 0.2")


def make_single_strand_bndry(
    fa,
    vline_pos=None,
    include_vplot=True,
    coverage_type="summed_endpoints",
    smooth_window=None,
):
    FRAGMENT_BANDS = [(40, 60), (60, 100), (110, 500)]

    if vline_pos is None:
        vline_pos = fa.length / 2
    region_size = fa.shape[1]

    region_size // 2 + 145
    first_nuc = 132
    offset = 175
    tracks = Tracks()
    if include_vplot:
        tracks.append(VplotTrack(fa, fa.plot_region, sum_pool_by=2))

    for fraction in FRAGMENT_BANDS:
        for TrackCls, ymin in zip(
            (StrandSplitCoverageTrack, CoverageDifferenceTrack), (0, None)
        ):
            tracks.append(
                TrackCls(
                    fa,
                    fa.plot_region,
                    coverage_type=coverage_type,
                    n_boot_coverage=0,
                    fraction=fraction,
                    smooth_window=smooth_window,
                    ylim=(ymin, None),
                    vlines=[VLine(x=vline_pos)],
                    name=f"{coverage_type} coverage ({fraction[0]} - {fraction[1]} bp Fraction)",
                )
            )

    return tracks


def make_gene_cov(rdf, tss_or_polya):
    lengths = pd.DataFrame(dict(start=0, stop=rdf.region_lengths))
    cov = np.zeros(lengths.stop.max())
    if tss_or_polya == "tss":
        for r in lengths.itertuples():
            cov[r.start : r.stop] += 1
    elif tss_or_polya == "polya":
        for r in lengths.itertuples():
            cov[-(r.stop - r.start) :] += 1
    else:
        raise ValueError(
            f"tss_or_polya must be either 'tss' or 'polya' (saw {tss_or_polya})"
        )
    cov = cov / lengths.shape[0]
    return cov


def make_on_off_plots(_rdf_w_fas, tss_or_polya, max_region_length=1024 * 24):
    fl_lb, fl_ub = _rdf_w_fas.n_fragments.quantile([0.05, 0.95])

    on_rdf = _rdf_w_fas.query(
        f"expression > 25 and n_fragments > {fl_lb} and n_fragments < {fl_ub}"
    )
    off_rdf = _rdf_w_fas.query(
        f"expression < 0.1 and n_fragments > {fl_lb} and n_fragments < {fl_ub}"
    )
    n_regions = min(on_rdf.shape[0], off_rdf.shape[0])

    if tss_or_polya == "tss":
        resize_method_name = "three_prime_resize"
        vline_pos = BOUNDARY_EXPANSION_SIZE
    elif tss_or_polya == "polya":
        resize_method_name = "five_prime_resize"
        vline_pos = max_region_length - BOUNDARY_EXPANSION_SIZE
    else:
        assert False, "UNREACHABLE"

    all_on_fa = merge_fragment_arrays(
        [
            getattr(_, resize_method_name)(max_region_length)
            for _ in on_rdf.sample(n_regions).fragment_array.tolist()
        ]
    )  # cell_type == 'all' and
    tracks = make_single_strand_bndry(
        all_on_fa,
        vline_pos=vline_pos,
        include_vplot=True,
        coverage_type="summed_endpoints",
        smooth_window=100,
    )
    tracks.append(
        VectorTrack(
            make_gene_cov(on_rdf, tss_or_polya)[
                (
                    slice(None, all_on_fa.length)
                    if tss_or_polya == "tss"
                    else slice(-all_on_fa.length, None)
                )
            ],
            all_on_fa.plot_region,
            name="Gene Coverage",
            ylim=(0, 1),
        )
    )
    tracks.plot()

    all_off_fa = merge_fragment_arrays(
        [
            getattr(_, resize_method_name)(max_region_length)
            for _ in off_rdf.sample(n_regions).fragment_array.tolist()
        ]
    )  # cell_type == 'all' and
    tracks = make_single_strand_bndry(
        all_off_fa,
        vline_pos=vline_pos,
        include_vplot=True,
        coverage_type="summed_endpoints",
        smooth_window=100,
    )
    tracks.append(
        VectorTrack(
            make_gene_cov(off_rdf, tss_or_polya=tss_or_polya)[
                (
                    slice(None, all_on_fa.length)
                    if tss_or_polya == "tss"
                    else slice(-all_on_fa.length, None)
                )
            ],
            all_off_fa.plot_region,
            name="Gene Coverage",
            ylim=(0, 1),
        )
    )
    tracks.plot()

    return
