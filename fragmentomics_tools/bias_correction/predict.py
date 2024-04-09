import tqdm

import numpy as np
from scipy.stats import multinomial, nbinom
import pandas as pd
import torch
import matplotlib

from fragmentomics_tools.region import Region
from fragmentomics_tools.dataframe import SampleAndRegionDataFrame, DataFrameBase
from fragmentomics_tools.fragment_array import merge_fragment_arrays
from fragmentomics_tools.plot.tracks import Tracks, GeneTrack, VectorTrack, BedTrack, smooth1d

from .data import FragmentEndpointsDataset

FASTA_PATH = "/home/nboley/src/Ravel/data/repo_data_manifest/reference/GRCh38/GRCh38.p12.genome.fa.gz"

def make_qc_plots(stats_df):
    obs = stats_df.fwd_start_counts_count/stats_df.bkwd_stop_counts_count
    pred = stats_df.fwd_start_counts_pred_count/stats_df.bkwd_stop_counts_pred_count

    fig, axarr = matplotlib.pyplot.subplots(ncols=2, nrows=2, figsize=(12, 10))
    df = pd.DataFrame(dict(obs=obs, pred=pred), index=stats_df.index)
    df.plot.scatter(x='obs', y='pred', ax=axarr[0, 0])
    axarr[0, 0].axline((0,0), slope=1)
    axarr[0, 0].set_title(f"Fwd/Rev Strand Ratios (corr = {round(np.corrcoef(df.T)[0, 1], 3)})")


    stats_df.set_index("gene_name").plot.scatter(x='fwd_start_counts_count', y='fwd_start_counts_pred_count', ax=axarr[0, 1]) # , xlim=(0, 20000), ylim=(0, 20000))
    stats_df.set_index("gene_name").plot.scatter(x='bkwd_stop_counts_count', y='bkwd_stop_counts_pred_count', color='orange', ax=axarr[0, 1]) # , xlim=(0, 20000), ylim=(0, 20000))
    axarr[0, 1].axline((0,0), slope=1)
    axarr[0, 1].set_title(f"Counts")

    stats_df[['fwd_start_counts_frac_over_95_conf']].hist(bins=50, ax=axarr[1, 0])
    stats_df[['bkwd_stop_counts_frac_over_95_conf']].hist(bins=50, ax=axarr[1, 1])

    fig.suptitle("Predicted vs Observed Region Counts")

def calc_stats(record):
    base_columns = [c for c in record.index.tolist() if c.endswith("_counts")]

    all_counts = [record[c] for c in base_columns]
    dists = [record[c + "_dist"] for c in base_columns]

    rv = {}
    for c, counts, dist in zip(base_columns, all_counts, dists):
        rv['gene_id'] = record.gene_id
        rv['gene_name'] = record.gene_name
        rv[c + "_count"] = counts.sum()
        rv[c + "_pred_count"] = dist.mean().sum()
        rv[c + "_l1_loss"] = np.abs(dist.mean() - counts).mean()
        if hasattr(dist, 'ppf'):
            rv[c + "_frac_over_95_conf"] = (counts > dist.ppf(0.95)).mean()
        rv[c + "_frac_over_99_conf"] = (counts > dist.ppf(0.99)).mean()
    return pd.Series(rv)

def make_tracks(record, sharey=False, smooth=False, max_scale=False):
    region = Region(record.contig, record.start, record.stop, strand=record.strand)
    base_columns = [c for c in record.index.tolist() if c.endswith("_counts")]
    tracks = Tracks()
    tracks.append(GeneTrack("/scratch/karius/annotation/gencode/gencode.v45.basic.annotation.bed.gz", region))

    all_counts = [record[c] for c in base_columns]
    dists = [record[c + "_dist"] for c in base_columns]

    if sharey:
        ylim = (0, int(self.max().max()*1.1))
    else:
        ylim = None

    for base_column, counts, dist in zip(base_columns, all_counts, dists):
        if max_scale:
            scale = (counts.max()/dist.mean().max())
        else:
            scale = 1
        track = VectorTrack(counts, region, name=f"Observed - {base_column}", color='black', alpha=0.7, ylim=ylim, smooth=smooth) + \
                VectorTrack(scale*dist.mean(), region, name=f"Predicted - {base_column}", color='purple', alpha=0.5, ylim=ylim, smooth=smooth) \
                + VectorTrack(scale*dist.ppf(0.95), region, name=f"Predicted - {base_column}", color='orange', alpha=0.5, ylim=ylim, smooth=smooth)
            

        track.name = base_column
        tracks.append(track)
        tracks.append(VectorTrack(smooth1d(counts, smooth)/smooth1d(dist.mean(), smooth), region, name="smoothed_ratio", color='black', alpha=1, ylim=ylim))
        tracks.append(VectorTrack(np.log10(counts + 1), region, name="log10(1+count)", color='black', alpha=1, ylim=ylim))

    tracks.append(BedTrack("/home/nboley/src/Ravel/data/repo_data_manifest/annotations/GRCh38_extras/hg38-repeats.sorted.bed.gz", region))
    tracks.append(BedTrack("/scratch/karius/annotation/mappability.simple_repeats.sorted.bed.gz", region))

    return tracks

