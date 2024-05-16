import os
import math
import copy
import logging
import warnings
from collections import defaultdict, Counter
from functools import reduce
from itertools import chain
from typing import Dict, List, Union, Optional, Sequence, Iterable, Tuple
from zlib import crc32
import pandas.core.internals
import numpy
import numpy as np
import pandas
import pandas as pd
import pybedtools
from intervaltree import IntervalTree
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import shuffle as sk_shuffle
from smart_open import open
from scipy.stats.mstats import trimmed_std

from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

tqdm.pandas()
from joblib import delayed, Parallel

import multiprocessing
import pickle
import traceback

# from tqdm.contrib.concurrent import process_map
# from p_tqdm import p_map

import seaborn as sns

import pysam
import logging

from fragmentomics_tools.region import Region, OutOfBoundsError
from fragmentomics_tools.formats import BedReader, BigWigReader
from fragmentomics_tools.contig import CONTIG_LENGTHS
from fragmentomics_tools.motif import Pfm


# from fbio.formats import BedReader, BigWigReader
# from fbio.fragments_h5 import FragmentsH5
# from fbio.liftover import RegionLiftOver
# from fbio.region import Region, OutOfBoundsError
# from fbio.util import aws_utils
# from fbio.util.iter_utils import windowed_range
# from fbio.util.misc_utils import progress_bar
# from ravel.data_manifest import load_data_manifest, DataManifest
# from ravel.util.ml_utils import get_indices_of_balanced_labels
# from ravel.util.pandas_utils import dataframe_region_mask


logger = logging.getLogger(__name__)


from fragmentomics_tools import (
    RegionFragmentArray,
    FragmentArray,
    merge_fragment_arrays,
)


NUM_CORES = -1
DEFAULT_MIN_MAPQ = 10
DEFAULT_MAX_FRAG_LEN = 511


def _apply_fn(df, fn, counter, send_conn, send_conn_lock):
    rv = []
    while True:
        with counter.get_lock():
            idx = counter.value
            counter.value += 1
        if idx >= df.shape[0]:
            return

        try:
            res = fn(df.iloc[idx])
        except Exception as inst:
            # traceback.print_exception(inst)
            msg = pickle.dumps(inst)
        else:
            msg = pickle.dumps((idx, res))
        with send_conn_lock:
            send_conn.send_bytes(msg)


def get_indices_of_balanced_labels(labels, random_state=None):
    """
    >>> labels = np.array([1, 0, 0, 0, 1])
    >>> idxs = get_indices_of_balanced_labels(labels, random_state=1)
    >>> idxs
    array([0, 1, 3, 4])
    >>> Counter(labels[idxs])
    Counter({1: 2, 0: 2})
    >>> get_indices_of_balanced_labels([])
    array([], dtype=int64)
    >>> get_indices_of_balanced_labels([1])
    array([0])
    >>> get_indices_of_balanced_labels([0,1,2,2], random_state=1)
    array([0, 1, 2])
    """
    if len(labels) == 0:
        return np.array([], int)

    labels = np.array(labels)
    counts = Counter(labels)
    min_count = min(counts.values())

    keep_idxs = []
    for label in counts.keys():
        label_idxs = np.where(labels == label)[0]
        keep_label_idxs = sk_shuffle(label_idxs, random_state=random_state)[:min_count]
        keep_idxs += keep_label_idxs.tolist()

    # sanity check
    assert len(set(Counter(labels[keep_idxs]).values())) == 1

    return np.array(sorted(keep_idxs))


def windowed_range(start, stop, window_size):
    """
    >>> list(windowed_range(0, 5, 2))
    [(0, 2), (2, 4), (4, 5)]
    >>> list(windowed_range(0, 1, 2))
    [(0, 1)]
    >>> list(windowed_range(0, 11, 3))
    [(0, 3), (3, 6), (6, 9), (9, 11)]
    >>> list(windowed_range(-3, 3, 3))
    [(-3, 0), (0, 3)]
    """
    if window_size <= 0:
        raise ValueError("invalid window size")
    if stop <= start:
        raise ValueError("invalid start/stop")

    for start in range(start, stop, window_size):
        yield start, min(stop, start + window_size)


def _bytes_to_float(b):
    return float(crc32(b) & 0xFFFFFFFF) / 2**32


class DataFrameBase(pandas.DataFrame):
    _metadata = []  # Metadata is optional, you can pass it in
    _required_metadata = (
        []
    )  # This must be a subset of metadata, but it is required for init
    _required_columns = []  # These columns will be checked for existence during init.
    _potentially_confused_columns = {}

    @property
    def _constructor(self):
        def f(*args, **kwargs):
            for attr_name in self._metadata:
                kwargs[attr_name] = getattr(self, attr_name)
            return type(self)(*args, **kwargs)

        return f

    @classmethod
    def from_fname_s3_or_local(cls, fname, *args, **kwargs):
        warnings.warn("deprecated, use pd.read_table()", DeprecationWarning)
        # allow this to be initialized from a path
        if not isinstance(fname, (str, bytes)):
            raise TypeError("Expecting a file path, got {}".format(repr(fname)))

        data = pandas.read_table(fname)
        return cls(data, *args, **kwargs)

    def __init__(self, data, *args, **kwargs):
        # ensure that required_metadata is a subset of metadata
        assert all(x in self._metadata for x in self._required_metadata)

        # hack around pandas not correctly using _constructor internally
        if not isinstance(data, pandas.core.internals.BlockManager):
            for attr_name in self._required_metadata:
                if attr_name not in kwargs:
                    raise ValueError(f"arg '{attr_name}' required")

        # set the metadata args
        for attr_name in self._metadata:
            # we pop off the attr_name because it shouldn't be passed into
            # the parent initializer. This must default to None b/c of how
            # pandas passes around data internally, but we check that the
            # argument is required (if it is) above
            setattr(self, attr_name, kwargs.pop(attr_name, None))

        super().__init__(data, *args, **kwargs)
        if isinstance(data, pandas.core.internals.BlockManager):
            return

        # After super and we know this is not a BlockManager,
        # we can check if there's any columns that may have gotten confused
        # TODO: account for capitalization
        for needed_column in self._potentially_confused_columns.keys():
            if needed_column not in self.columns:
                for potential_confused_column in self._potentially_confused_columns[
                    needed_column
                ]:
                    if potential_confused_column in self.columns:
                        self.rename(
                            columns={potential_confused_column: needed_column},
                            inplace=True,
                        )

        # Sanity check on whether self.columns has everything needed
        missing_key_cols = set(self._required_columns) - set(self.columns)
        assert len(missing_key_cols) == 0, (
            f"Missing these columns in dataframe: {missing_key_cols}, "
            f"found {self.columns}."
        )


class RegionDataFrame(DataFrameBase):
    _metadata = ["ref"]
    _required_metadata = ["ref"]
    _critical_bed_columns = ["contig", "start", "stop"]
    _potentially_confused_columns = {
        "contig": ["seqname", "chrom", "chromosome", "chr"],
        "start": ["begin"],
        "stop": ["end"],
    }
    _optional_bed_columns = ["id", "score", "strand"]
    _standard_bed_columns = _critical_bed_columns + _optional_bed_columns
    _additional_required_columns = []
    _required_columns = _critical_bed_columns + _additional_required_columns

    def get_fasta_path(self):
        if self.ref == 'hg38':
            return "/scratch/karius/annotation/GRCh38.p12.genome.fa.gz"
        else:
            raise ValueError(f"No fasta associated with reference '{self.ref}'")

    @property
    def df(self):
        """Cast to a normal pandas dataframe"""
        return pd.DataFrame(self)

    @classmethod
    def from_fname_s3_or_local(cls, fname, *args, **kwargs):
        if fname.endswith(".bed") or fname.endswith(".bed.gz"):
            return cls.from_bed(fname, *args, **kwargs)

        return super().from_fname_s3_or_local(fname, *args, **kwargs)

    def __init__(self, data, *args, **kwargs):
        # hack around pandas not correctly using _constructor internally
        # This line should always be first.  Pandas incorrectly passes this BlockManager to the constructor sometimes.
        if (
            len(data) == 0
            and hasattr(data, "columns")
            and len(set(self._critical_bed_columns) - set(data.columns)) > 0
        ):
            # Handle empty dataframes by imposing null columns
            data = pd.DataFrame(
                columns=self._standard_bed_columns + self._critical_bed_columns
            )

        super().__init__(data, *args, **kwargs)

        if isinstance(data, pandas.core.internals.BlockManager):
            return

        if "strand" not in self.columns:
            self["strand"] = "."

    def __and__(self, other):
        for metadata_key in self._metadata:
            if not self.__dict__[metadata_key] == other.__dict__[metadata_key]:
                raise ValueError(
                    f"{self} and {other} have different metadata for {metadata_key}: "
                    f"{self.__dict__[metadata_key]} and {other.__dict__[metadata_key]}"
                )
        return RegionDataFrame(pandas.concat([self, other]), ref=self.ref)

    @staticmethod
    def concat(rdfs):
        # copy the references so that I can pop
        rdfs = list(rdfs)
        merged_rdf = rdfs.pop().copy()
        for rdf in rdfs:
            merged_rdf = merged_rdf & rdf
        return merged_rdf

    def __eq__(self, other):
        """
        Checks if two region dataframes have identical regions, in the same order
        """
        if len(self) != len(other):
            return False
        for r1, r2 in zip(self.iter_regions(), other.iter_regions()):
            if r1 != r2:
                return False
        return True

    @property
    def nrow(self):
        return len(self.index)

    @classmethod
    def from_bed(cls, in_bed_file, ref):
        """Convenience function to load from a bed file."""
        with open(in_bed_file) as infile:
            first_line_fields = infile.readline().split()
            if first_line_fields[0] in ["chrom", "chr", "contig"]:
                has_header = True
            else:
                has_header = False

        df = BedReader.load_dataframe(in_bed_file)
        df = df.rename(columns=dict(chrom="contig"))

        return cls(df, ref=ref)

    @classmethod
    def from_beds_merged(cls, in_bed_files, ref, chroms=None, bed_filter_callback=None):
        """
        Reads in multiple beds, concatenates and merges them, safely truncating to get rid of strand
        :param in_bed_files: a BED file or list of BED files
        :param ref: reference genome
        :param chroms: A list of chromosomes to keep, STANDARD_CHROMS by default
        :param bed_filter_callback: A pybedtools filter function (https://daler.github.io/pybedtools/filtering.html)
         This is a function that operates on each feature of the bedtool object and returns a True/False
         For example, to get scores above some threshold:
            def bed_filter(region):
                return int(region.score) >= 100
            RegionDataFrame.from_beds_merged(in_bed_files, bed_filter=bed_filter)
        :return: a RegionDataframe
        """
        if bed_filter_callback is None:

            def filter_func(_):
                return True

        else:
            filter_func = bed_filter_callback

        if isinstance(in_bed_files, str):
            in_bed_files = [in_bed_files]

        assert isinstance(in_bed_files, list) and all(
            isinstance(in_bed_file, str) for in_bed_file in in_bed_files
        ), "Must pass a peak BED file or a list of BED files"
        assert numpy.all(
            [
                in_bed_file.lower().endswith(
                    (".bed", ".bed.gz", ".bed.npk.gz", ".narrowpeak")
                )
                for in_bed_file in in_bed_files
            ]
        ), "All files must be BED files"
        assert len(in_bed_files) != 0, "Empty list of BED files"
        if len(in_bed_files) == 1:
            df = cls.from_bed(in_bed_files[0], ref)
        else:
            bedtool_files = [
                pybedtools.BedTool(in_bed_file).filter(filter_func)
                for in_bed_file in in_bed_files
            ]
            merged_bedtool = bedtool_files[0].cat(
                *bedtool_files[1:], postmerge=True, force_truncate=True
            )
            df = cls(
                merged_bedtool.to_dataframe(names=["contig", "start", "stop"]), ref=ref
            )

        if chroms is not None:
            df = df.query("contig == @chroms")

        return df

    @property
    def region_lengths(self):
        return self.stop - self.start

    def get_interval_dict(
        self,
        data_cols: Optional[List[str]] = ["id"],
        expand_upstream: int = 0,
        expand_downstream: int = 0,
    ) -> Dict[str, IntervalTree]:
        """
        :param data_cols: if None, use dataframe's row index as the return
            value, otherwise use the columns defined in data_cols.
        :param expand_upstream: basepairs upstream to expand each interval
        :param expand_downstream: basepairs downstream to expand each interval
        :return: per chromosome (dict by chromosome name) IntervalTree
        """
        interval_dict = defaultdict(IntervalTree)
        for idx, row in self.iterrows():
            if data_cols is None:
                data = idx
            else:
                data = tuple(row.loc[data_cols].tolist())
            if expand_upstream != 0 or expand_downstream != 0:
                assert row["strand"] in {"-", "+"}
                if row["strand"] == "+":
                    start = row["start"] - expand_upstream
                    stop = row["stop"] + expand_downstream
                elif row["strand"] == "-":
                    start = row["start"] - expand_downstream
                    stop = row["stop"] + expand_upstream
            else:
                start = row["start"]
                stop = row["stop"]
            interval_dict[row["contig"]][start:stop] = data
        return dict(interval_dict)

    def overlaps_rdf(
        self, query: "RegionDataFrame", max_distance: int = 0
    ) -> pd.Series:
        """Returns a boolean series of which regions overlap the other dataframe

        :param query: the query dataframe
        :param max_distance: maximum distance (>=0) (edge to edge) to an item in the
            query dataframe to consider a row an overlap
        :return: pd.Series[bool] of whether each row overlaps.
        """
        assert max_distance >= 0

        query_intervals = query.get_interval_dict(
            data_cols=None,
            expand_upstream=max_distance,
            expand_downstream=max_distance,
        )

        def is_olap(row: pd.Series) -> bool:
            contig = row["contig"]
            start = row["start"]
            stop = row["stop"]
            return len(query_intervals[contig][start:stop]) > 0

        return self.apply(is_olap, axis=1)

    def attach_num_tss_overlaps(
        self,
        tss_intervals=None,
        from_midpoint: bool = True,
        expand_upstream: int = 4000,
        expand_downstream: int = 0,
        conservative: bool = False,
    ) -> "RegionDataFrame":
        """
        :param tss_intervals: If provided, will skip re-querying the TSS intervals
        :param from_midpoint: If true, regions are resized to 1bp so the overalps are calculated relative to the center.
        # The following options are only used if tss_intervals is None for re-querying it.
        :param expand_upstream: How far upstream of the TSS to look for an overlap
        :param expand_downstream: How far downstream of the TSS to look for an overlap
        :param conservative: If the conservative set should be used
        :return:
        """
        if tss_intervals is None:
            tss_intervals = self.get_tss_intervals(
                expand_upstream=expand_upstream,
                expand_downstream=expand_downstream,
                conservative=conservative,
            )
        n_tss = np.zeros(len(self), dtype=int)

        if from_midpoint:
            rdf = self.resize_regions(1)
        else:
            rdf = self

        for i, (_, row) in enumerate(rdf.iterrows()):
            n_tss[i] = len(tss_intervals[row["contig"]][row["start"] : row["stop"]])
        self.loc[:, "num_tss_overlaps"] = n_tss
        return self

    @property
    def ref_path(self) -> str:
        assert False
        with DataManifest() as dm:
            ref_path = dm.sync_and_get(REFERENCE_FASTA_DATA_MANIFEST_KEY[self.ref]).path
            _ = dm.sync_and_get(
                REFERENCE_FASTA_DATA_MANIFEST_KEY[self.ref] + ".fai"
            ).path
            _ = dm.sync_and_get(
                REFERENCE_FASTA_DATA_MANIFEST_KEY[self.ref] + ".gzi"
            ).path
        assert os.path.exists(ref_path)
        assert os.path.exists(ref_path + ".fai")
        assert os.path.exists(ref_path + ".gzi")
        # check that we can actually open the fasta file
        with pysam.FastaFile(
            filename=ref_path, filepath_index_compressed=ref_path + ".fai"
        ) as _:
            pass

        return ref_path

    def center_on_summit(self):
        """Center regions on summit, resize the regions, and then drop the summit column."""
        if "summit" not in self.columns:
            raise TypeError("Must contain a 'summit' column to center on the summit.")

        region_lengths = (self.stop - self.start).copy()

        # check if the summit is within start
        if ((self.summit >= self.start) & (self.summit <= self.stop)).all():
            self["start"] = self.summit - region_lengths // 2
        elif (self.summit <= self.region_lengths).all():
            self["start"] = self.start + self.summit - region_lengths // 2
        else:
            raise ValueError("summits must either be within the region interval or less than the length of the region.")

        self["stop"] = self.start + region_lengths

        return self.drop(columns=["summit"])

    def center_regions_on_tf_motif(
        self,
        target_tfs: List[str],
        target_len: int = 15,
        tf_search_width: Optional[int] = None,
        num_workers: int = 16,
        batch_size: int = 30,
        cuda: bool = False,
        verify_motif_scores: bool = True,
        inplace: bool = False,
        unique_regions: bool = False,
        shuffle_tf_motif: bool = False,
        default_jaspar: bool = False,
        shuffle_tf_seed: int = 314,
        quiet: bool = False,
    ) -> "RegionDataFrame":
        """Add TF related columns and set start/stop to be the start/stop of the tf. Each region in the output will be
         `target_len` long.

        :param shuffle_tf_seed: the seed to use, _if_ shuffle_tf_motif is requested. Ignored otherwise.
        :param shuffle_tf_motif: usually do not change this. If you want to use a shuffled version of the tf motif as
         background, set this to True.
        :param unique_regions: Do you want to drop non-unique regions after the top tfs are found in each
         `tf_search_width` bp window?
        :param inplace: Should these operations be performed in place?
        :param verify_motif_scores: Do you want a final verification that the coordinates match the tf top scores?
        :param cuda: Run this on cuda. Pretty fast on CPU for regions of size ~500.
        :param tf_search_width: width to search for TFs, regions are temporarily resized to this width.
        :param batch_size: size of region batch
        :param num_workers: number of workers
        :param target_len: length of the tf motif to use. If shorter than the motif of interest the best subset is
         found, if longer the edges are zero padded.
        :param target_tfs: List of tfs to search for.
        :return:
        """
        # Keep the following two import lines here to avoid a big circular import refactor
        import torch
        from fragmentomics_tools.motif import TFConv1D, SeqDataSet

        if min(self.region_lengths) < 500 and tf_search_width is None:
            warnings.warn(
                "WARNING! Scanning for TFs with a window size of less than 500bp is not advised",
                UserWarning,
            )
        assert (
            len(self.region_lengths.unique()) == 1
        ), f"All regions must be of the same size. Hint: {self.region_lengths.unique()}"
        if isinstance(target_tfs, str):
            target_tfs = [target_tfs]

        if not inplace:
            self = self.copy()

        tf_conv = TFConv1D(
            tf_width=target_len,
            only_logo=True,
            output_both_strands=True,
            channels=4,
            default_jaspar=default_jaspar,
            tfs=set(target_tfs),
            shuffle_motifs=shuffle_tf_motif,
            shuffle_seed=shuffle_tf_seed,
        ).eval()
        if cuda:
            tf_conv = tf_conv.cuda()

        with torch.no_grad():
            sds = SeqDataSet(
                region_dataframe=self
                if tf_search_width is None
                else self.resize_regions(tf_search_width),
                ref_path="/home/nboley/src/Ravel/data/repo_data_manifest/reference/GRCh38/GRCh38.p12.genome.fa.gz",
            )
            sdl = torch.utils.data.DataLoader(
                sds,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
                batch_size=batch_size,
            )
            max_scores = []
            max_idxs = []
            max_strands = []

            for sequences in tqdm(sdl, disable=quiet):
                with torch.no_grad():
                    sequences_gpu = sequences.to(tf_conv.kernel_f.device)
                    res = tf_conv(sequences_gpu)
                    scores_v, _ = res.max(dim=1)
                    (max_v, max_i) = (scores_v * 1).max(dim=-1)  # type: torch.Tensor
                    pos_strand: torch.Tensor = max_v[:, 0] > max_v[:, 1]
                    top_scores: torch.Tensor = max_v[:, 1].clone()
                    top_idx: torch.Tensor = max_i[:, 1].clone()
                    top_scores[pos_strand] = max_v[pos_strand, 0]
                    top_idx[pos_strand] = max_i[pos_strand, 0]
                    max_scores.append(top_scores.cpu())
                    max_idxs.append(top_idx.cpu())
                    max_strands.append(pos_strand.cpu())
            max_idxs = torch.cat(max_idxs, dim=0)
            max_strands = torch.cat(max_strands, dim=0)
            max_scores = torch.cat(max_scores, dim=0)
            resized_rdf = sds.region_dataframe
            self["tf_on_rev_strand"] = ~max_strands
            self["tf_top_offset"] = max_idxs
            self["tf_top_score"] = max_scores
            self["target_tfs"] = ",".join(target_tfs)
            self["original_start"] = self["start"]
            self["original_stop"] = self["stop"]
            self["query_start"] = resized_rdf.start
            self["query_stop"] = resized_rdf.stop
            self["start"] = self["query_start"] + self["tf_top_offset"]
            self["stop"] = self["start"] + target_len
            self["strand"] = "+"
            self.loc[self.tf_on_rev_strand, "strand"] = "-"
        if verify_motif_scores:
            self.verify_motif_scores(
                tf_names=target_tfs,
                num_workers=num_workers,
                batch_size=batch_size,
                default_jaspar=default_jaspar,
                cuda=cuda,
                shuffle_tf_motif=shuffle_tf_motif,
                shuffle_tf_seed=shuffle_tf_seed,
                quiet=quiet,
            )
        if unique_regions:
            return self.unique_regions()
        return self

    def annotate_regions_with_max_tf_scores(
        self,
        target_tfs: List[str],
        target_len: int = 15,
        num_workers: int = 10,
        batch_size: int = 30,
        cuda: bool = False,
        inplace: bool = False,
        default_jaspar: bool = False,
    ) -> "RegionDataFrame":
        """Add TF related columns and set start/stop to be the start/stop of the tf. Each region in the output will be `target_len` long. The max tf score involves
            first fitting a smoothed spline to the data with knots in key regions near the center of the region. The relative smooth coverage at these key knots
            is used to generate a score. The score is normalized for both coverage and variance.

        :param shuffle_tf_seed: the seed to use, _if_ shuffle_tf_motif is requested. Ignored otherwise.
        :param shuffle_tf_motif: usually do not change this. If you want to use a shuffled version of the tf motif as background, set this to True.
        :param unique_regions: Do you want to drop non-unique regions after the top tfs are found in each `tf_search_width`bp window?
        :param inplace: Should these operations be performed in place?
        :param verify_motif_scores: Do you want a final verification that the coordinates match the tf top scores?
        :param cuda: Run this on cuda. Pretty fast on CPU for regions of size ~500.
        :param tf_search_width: width to search for TFs, regions are temporarily resized to this width.
        :param batch_size: size of region batch
        :param num_workers: number of workers
        :param target_len: length of the tf motif to use. If shorter than the motif of interest the best subset is found, if longer the edges are zero padded.
        :param target_tfs: List of tfs to search for.
        :return:
        """
        # Keep the following two import lines here to avoid a big circular import refactor
        from ravel.learn.functional_genomics.data import SeqDataSet
        from ravel.learn.functional_genomics.model import TFConv1D

        if not inplace:
            self = self.copy()

        tf_conv = TFConv1D(
            tf_width=target_len,
            only_logo=True,
            output_both_strands=True,
            channels=4,
            default_jaspar=default_jaspar,
            tfs=set(target_tfs),
            shuffle_motifs=False,
            shuffle_seed=0,
        ).eval()
        if cuda:
            tf_conv = tf_conv.cuda()

        with torch.no_grad():
            region_lengths = set(self.region_lengths)
            assert (
                len(region_lengths) == 1
            ), "Found more than 1 region length, please first resize this RDF to be one size."
            sds = SeqDataSet(region_dataframe=self)
            sdl = torch.utils.data.DataLoader(
                sds,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
                batch_size=batch_size,
            )
            max_scores = []
            max_strands = []
            max_idxs = []

            for sequences in tqdm(sdl):
                sequences_device = sequences.to(tf_conv.kernel_f.device)
                res = tf_conv(sequences_device)
                # res is batch x n_tfs x region_length
                (max_v, max_i) = res.max(dim=-1)  # type: torch.Tensor
                _, strand_idx = max_v.max(dim=-1)
                max_idx = max_i.gather(-1, strand_idx[..., None]).squeeze(-1)
                max_val = max_v.gather(-1, strand_idx[..., None]).squeeze(-1)
                max_scores.append(max_val.cpu())
                max_strands.append(strand_idx.cpu())
                max_idxs.append(max_idx.cpu())
            max_idxs = torch.cat(max_idxs, dim=0).numpy()
            assert max_idxs.shape == (len(self), len(tf_conv.tf_names)), max_idxs.shape
            max_strands = torch.cat(max_strands, dim=0).numpy()
            assert max_strands.shape == (
                len(self),
                len(tf_conv.tf_names),
            ), max_strands.shape
            all_strands = max_strands.flatten().astype(int)
            assert np.all((all_strands == 0) | (all_strands == 1)), all_strands[
                (all_strands != 0) & (all_strands != 1)
            ][:10]
            max_scores = torch.cat(max_scores, dim=0).numpy()
            assert max_scores.shape == (
                len(self),
                len(tf_conv.tf_names),
            ), max_scores.shape
            for i, name in tqdm(
                enumerate(tf_conv.tf_names), total=len(tf_conv.tf_names)
            ):
                self[f"{name}_max_motif_score"] = max_scores[:, i]
                self[f"{name}_max_motif_strand"] = "+"
                self.loc[max_strands[:, i] == 1, f"{name}_max_motif_strand"] = "-"
                self[f"{name}_max_motif_region_offset"] = max_idxs[:, i]
        return self

    def unique_regions(
        self,
        by: List[str] = ["contig", "start", "stop"],
        best_by: Optional[str] = "score",
        ascending_best: bool = False,
    ):
        """Remove
        :param by: columns you want to use to define what a region is. Default is contig,start,stop
        :param best_by: which column you want to use to select ties. If None or the column doesn't exist, just do a random one
        :param ascending_best: Should the best column be ascending or descending, the first match is taken. Score should typically be False for example.
        :return:
        """
        sort_cols = by
        ascending = [False, False, False]
        if best_by is not None:
            if best_by in self.columns:
                sort_cols = sort_cols + [best_by]
                ascending = ascending + [ascending_best]
            else:
                logger.warning(
                    f"Column name {best_by} not found in {self.columns}. Will pick first row by current order when dropping duplicates."
                )
        return (
            self.sort_values(by=sort_cols, ascending=ascending)
            .reset_index(drop=True)
            .groupby(by=by, as_index=False)
            .head(1)
            .reset_index(drop=True)
        )

    @classmethod
    def rdf_from_bed3(cls, fname, ref, nrows, label):
        df = BedReader.load_dataframe(fname, nrows=nrows).rename(
            columns=dict(chrom="contig")
        )
        df["id"] = [f"{r.contig}:{r.start}-{r.stop}" for _, r in df.iterrows()]
        df["strand"] = None
        df["label"] = label
        return cls(df, ref=ref)

    @classmethod
    def from_regions(cls, regions: List[Region], ref: str) -> "RegionDataFrame":
        """
        Create a RegionDataFrame from a set of regions
        :param regions: an iterable of regions
        :param ref: 'hg38' or 'hg19'
        """
        # refs = set([reg.ref for reg in regions])
        # assert len(refs) == 1
        # ref_from_regions = refs.pop()
        # if ref is None:
        #     ref = ref_from_regions
        # else:
        #     assert ref == ref_from_regions, f"{ref} is not the same as {ref_from_regions}"
        chroms, starts, stops, strands = zip(
            *[
                (region.chrom, region.start, region.stop, region.strand)
                for region in regions
            ]
        )
        df = pandas.DataFrame(
            dict(contig=chroms, start=starts, stop=stops, strand=strands)
        )
        return cls(df, ref=ref)

    @classmethod
    def from_random_regions(
        cls, size, region_lengths, ref, chroms=None
    ) -> "RegionDataFrame":
        """
        Create a RegionDataFrame from a random set of regions
        :param size: number of regions
        :param region_lengths: length of each region
        :param ref: 'hg38' or 'hg19'
        :param chroms: chromosomes to use, defaults to STANDARD_CHROMS
        """
        return cls.from_regions(
            [
                Region.random(region_lengths, assembly=ref, chroms=chroms)
                for _ in range(size)
            ],
            ref=ref,
        )

    def merge_regions(self, **kwargs):
        bedtool = pybedtools.BedTool.from_dataframe(self).sort()
        return type(self)(
            bedtool.merge(**kwargs).to_dataframe(names=list(self.columns)),
            ref=self.ref,
        )

    def intersect_with_rdf(self, other, sorted=False, rsuff="other"):
        """
        Creates the intersection of RegionDataFrames
        :param other: other RegionDataFrame
        :param sorted: RegionDataFrames are both sorted -- use Bedtools chromsweep algorithm
        :param rsuff: Suffix to append to other dataframe
        :param intersect_kwargs: Bedtools intersect kwargs, such as wa, wo, etc. For more info, consult
            https://daler.github.io/pybedtools/autodocs/pybedtools.bedtool.BedTool.intersect.html
        :return: RegionDataFrame that is the intersection of two RegionDataFrames
        """

        def reordered_columns(rdf):
            bed_columns = ["contig", "start", "stop", "strand"]
            return bed_columns + [c for c in rdf.columns if c not in bed_columns]

        # assert isinstance(other, RegionDataFrame)
        assert (
            self.ref == other.ref
        ), f"RegionDataFrames must have the same reference: {self.ref, other.ref}"

        self_index = self.index
        self = self.reset_index()
        this_bed_df = self.bed_df
        this_bed_df["index"] = self_index
        this_bedtool = pybedtools.BedTool.from_dataframe(this_bed_df)
        other_bedtool = pybedtools.BedTool.from_dataframe(
            other[reordered_columns(other)]
        )
        other_column_names = [f"{c}_{rsuff}" for c in reordered_columns(other)]
        intersection_rdf = pd.DataFrame(
            this_bedtool.intersect(
                other_bedtool, sorted=sorted, wa=True, wb=True
            ).to_dataframe(names=this_bed_df.columns.tolist() + other_column_names),
        )
        if len(intersection_rdf) == 0:
            return pd.DataFrame(
                columns=[c[: -(1 + len(rsuff))] for c in other_column_names]
            )

        intersection_rdf = intersection_rdf.set_index("index")
        intersection_rdf = intersection_rdf[other_column_names]
        intersection_rdf.columns = [c[: -(1 + len(rsuff))] for c in other_column_names]
        return RegionDataFrame(intersection_rdf, ref=self.ref)

    def intersect_with_bed(
        self, bed_file_path, sorted=False, rsuff="other", **intersect_kwargs
    ):
        """
        Finds intersection between RegionDataFrame and some bed file
        :param bed_file_path: A BED file path, for example a blacklist or repeat annotation
        :param rsuff: Suffix to append to fields of bed file
        :param sorted: Whether BED is sorted
        :return: a RegionDataFrame with all intersections, adding addition columns demarcated "_bed"
        """
        other_rdf = RegionDataFrame.from_bed(bed_file_path, ref=self.ref)
        overlap_rdf = self.intersect_with_rdf(
            other_rdf, sorted=sorted, rsuff=rsuff, **intersect_kwargs
        )

        return overlap_rdf

    def sort(self, inplace=False):
        """
        Sorts dataframe by contig, start, and stop (same as pybedtools)
        :param inplace: sort in place
        :return: None or sorted RegionDataFrame
        """
        rdf = self.sort_values(["contig", "start", "stop"], inplace=inplace)
        if not inplace:
            return rdf

    def lift_over(
        self, new_ref, transfer_columns=False, remove_non_liftoverable_regions=True
    ):
        """
        Lifts over RegionDataFrame to a new reference
        :param new_ref: str of new reference name ("hg18", "hg19", "hg38")
        :param transfer_columns: transfer additional columns found in dataframe, such as id, etc
        :parm remove_non_liftoverable_regions: If true, don't return regions that don't have unique mappings.
                If False, report un-liftoverer regions as (None, -1, -1, None)
        :return: lifted over RegionDataFrame of the same subclass as self
        """
        liftoverer = RegionLiftOver(self.ref, new_ref)

        def _liftover(record):
            lifted_coord = liftoverer.uniquely_convert_region(
                record.contig, record.start, record.stop, record.strand
            )
            if lifted_coord is not None:
                return lifted_coord
            else:
                return (None, -1, -1, None)

        res = [_liftover(record) for record in tqdm(self.itertuples(), total=self.nrow)]
        contigs, starts, stops, strands = zip(*res)

        rv = self.copy()
        rv.ref = new_ref
        rv["contig"] = contigs
        rv["start"] = starts
        rv["stop"] = stops
        rv["strand"] = strands

        if not transfer_columns:
            rv = rv[["contig", "start", "stop", "strand"]]
        if remove_non_liftoverable_regions:
            rv = rv.query("not contig.isnull()")
        return rv

    def get_overlapping_base_counts(self, bed_file, rsuff="bed", sorted=False):
        """
        Returning the number of overlapping bases in an intersection between a RDF and bed file
        :param bed_file: A BED file, for example a blacklist or repeat annotation
        :param rsuff: suffix to use in bedtools region intersection
        :param sorted: whether BED file is sorted
        :return: a dict with "counts" corresponding to the total overlap between the rdf and annotation,
         "max_counts" corresponding to the longest interval that overlaps rdf
        """
        counts = numpy.zeros(len(self), dtype=int)
        max_counts = numpy.zeros(len(self), dtype=int)

        region2idx = {
            (x.contig, x.start, x.stop): pp for pp, x in enumerate(self.itertuples())
        }
        counts_column_key = "overlap"
        for region, overlaps in self.intersect_with_bed(
            bed_file,
            rsuff=rsuff,
            sorted=sorted,
            wao=True,
        ).groupby(["contig", "start", "stop"]):
            inner_counts = overlaps[counts_column_key]
            max_counts[region2idx[region]] = inner_counts.max()
            counts[region2idx[region]] = inner_counts.sum()

        return {"counts": counts, "max_counts": max_counts}

    def overlaps_with_bed(self, bed_file, invert=False, min_size=0):
        """
        Finds intersection between RegionDataFrame and some bed file, returning a True/False array
        :param bed_file: A BED file, for example a blacklist or repeat annotation
        :param invert: Whether to invert intersection.
         If invert==False, regions that overlap bed return True
         If invert==True, regions that overlap bed return False
        :param min_size: Minimum size of overlap to mark as overlapping
        :return: Returns whether or not the rdf overlaps with a bed file with no more than a min_size region
        """
        max_counts = self.get_overlapping_base_counts(bed_file)["max_counts"]
        mask = max_counts > min_size
        if invert:
            mask = numpy.logical_not(mask)
        return mask

    def overlaps_with_beds(self, bed_files, num_cores=1, *args, **kwargs):
        """
        Finds intersection between RegionDataFrame and a list of bed files, returning a list of True/False arrays
        :param bed_files: A list of BED files, for example a blacklist or repeat annotation
        :param num_cores: Number of threads
        :return: Returns whether or not the rdf overlaps with a bed file with no more than a min_size region
        """
        if isinstance(bed_files, str):
            bed_files = [bed_files]
        overlap_per_bed = Parallel(num_cores)(
            delayed(self.overlaps_with_bed)(bed_file, *args, **kwargs)
            for bed_file in bed_files
        )
        return overlap_per_bed

    def bases_overlap_with_bed(self, bed_file):
        """
        Finds intersection between RegionDataFrame and some bed file, returning the number of overlapping bases
        :param bed_file: A BED file, for example a blacklist or repeat annotation
        :return: Gets the total number of bases that rdf overlaps with a bed file
        """
        return self.get_overlapping_base_counts(bed_file)["counts"]

    def bases_overlap_with_beds(self, bed_files, num_cores=1):
        """
        Finds intersection between RDF and a list of bed files, returning list of number of overlapping bases
        :param bed_files: A list of BED files, for example a blacklist or repeat annotation
        :return: Gets the total number of bases that rdf overlaps with a bed file
        """
        if isinstance(bed_files, str):
            bed_files = [bed_files]
        bases_overlap_per_bed = Parallel(num_cores)(
            delayed(self.bases_overlap_with_bed)(bed_file) for bed_file in bed_files
        )
        return bases_overlap_per_bed

    def drop_overlapping_regions(self, other_rdf):
        """
        Masks blacklist regions, returning a new dataframe with regions that don't overlap blacklist regions.
        OPTIONALLY, also masks repeat regions. To do this, must provide a max repeat size allowed
        """
        assert (
            self.ref == "hg38"
        ), "Must use hg38 reference if excluding blacklist regions"
        tmp = self.intersect_with_rdf(other_rdf)
        return self.loc[self.index.difference(tmp.index), :]

    def attach_blacklist_regions(self, bed_fname):
        tmp = self.intersect_with_rdf(RegionDataFrame.from_bed(bed_fname, ref=self.ref))
        if len(tmp) == 0:
            self["blacklist_regions"] = ""
            return self

        blacklist_regions = (
            tmp.groupby(tmp.index.names)
            .apply(lambda x: list(x.iter_regions()))
            .rename("blacklist_regions")
        )
        return self.join(blacklist_regions).fillna("")

    def region_mask(self, region):
        """
        Returns a mask of this dataframe of all regions which intersect region
        :param region: the region to intersect
        """
        return dataframe_region_mask(self, region)

    @property
    def bed_df(self):
        columns = self._required_columns
        assert all(c in self.columns for c in columns)

        bed_df = self.copy(deep=True)

        return bed_df[columns]

    def save_as_bed(self, path):
        self.bed_df.to_csv(path, index=False, sep="\t", header=False)

    def iter_regions(self):
        for r in self.itertuples():
            strand = (
                None
                if r.strand in (".", "None", None) or pandas.isnull(r.strand)
                else r.strand
            )
            # if "name" in r.columns:
            #    data = dict(name=r["name"])
            # else:
            #    data = None
            data = None
            yield Region(r.contig, r.start, r.stop, strand, ref=self.ref, data=data)

    def _get_fragment_coverage_track(self, in_fname: str):
        """
        :param in_fname: a bigwig or fragments h5 file
        :return: coverage profiles over regions in self
        """
        if in_fname.lower().endswith((".bw", ".bigwig")):
            with BigWigReader(in_fname) as reader:
                numpy.warnings.filterwarnings(
                    "ignore", category=numpy.VisibleDeprecationWarning
                )
                return numpy.array(
                    [
                        reader.values(region.chrom, region.start, region.stop)
                        for region in self.iter_regions()
                    ]
                )
        # For fragment H5s, get the fragment matrix and make a coverage track
        elif in_fname.lower().endswith(".h5"):
            frag_h5 = FragmentsH5(in_fname)
            fms = self.apply(
                lambda row: RegionFragmentMatrix.from_fragments_h5(
                    frag_h5,
                    Region(row["contig"], row["start"], row["stop"], ref=self.ref),
                ),
                axis=1,
            ).values
            return numpy.array([fm.get_fragment_coverage_array() for fm in fms])
        else:
            raise NotImplementedError(f"{in_fname} is not supported")

    def get_fragment_coverage_track(
        self, in_fnames: Union[List, str], num_cores=NUM_CORES, verbose=0
    ):
        """
        :param in_fnames: file or list of bigwig or fragment h5 files
        :return: coverage profiles over regions in self
        """
        if isinstance(in_fnames, str):
            in_fnames = [in_fnames]

        return numpy.array(
            Parallel(num_cores, verbose=verbose)(
                delayed(self._get_fragment_coverage_track)(in_fname)
                for in_fname in in_fnames
            )
        )

    def _get_fragment_coverage_sum(self, in_fname: str, sorted=False):
        """
        For each region, get the fragment coverage for in_fname, which is a fragment coverage bigwig or fragment bed
        """

        def get_column_names(bed_file):
            default_cols = [
                "chrom_1",
                "start_1",
                "end_1",
                "name_1",
                "score_1",
                "strand_1",
            ]
            with open(bed_file) as infile:
                num_fields = len(infile.readline().split())
            return default_cols[:num_fields]

        if in_fname.lower().endswith((".bw", ".bigwig")):
            return numpy.array(
                [track.sum() for track in self._get_fragment_coverage_track(in_fname)]
            )
        elif in_fname.lower().endswith((".bed", ".bed.gz")):
            # Use pybedtools to intersect the read / fragment beds with the regions
            bed_columns = get_column_names(in_fname)
            rdf_columns = [col + "_2" for col in self.columns]
            intersect_df = (
                pybedtools.BedTool(in_fname)
                .intersect(
                    pybedtools.BedTool.from_dataframe(self),
                    wa=True,
                    wb=True,
                    sorted=sorted,
                )
                .to_dataframe(names=bed_columns + rdf_columns)
            )
            # This is just to map back to an array
            peak2idx = {peak: pp for pp, peak in enumerate(self.id)}

            counts_vect = numpy.zeros(len(self))
            if intersect_df.shape[0] == 0:
                return numpy.array([])
            for peak, group in intersect_df.groupby("id_2"):
                counts_vect[peak2idx[peak]] = len(group)
            return counts_vect
        elif in_fname.lower().endswith(".h5"):
            frag_h5 = FragmentsH5(in_fname)
            return self.apply(
                lambda row: frag_h5.fetch_counts(
                    row["contig"], row["start"], row["stop"]
                ),
                axis=1,
            ).values
        else:
            raise NotImplementedError(f"{in_fname} is not supported")

    def get_fragment_coverage_sum(
        self, in_fnames: Union[List, str], num_cores=NUM_CORES, sorted=False, verbose=0
    ):
        """
        For each region, return the sum of the number of reads in in_fnames
        If in_fnames is a list, returns the sum over all files for each region
        """
        if isinstance(in_fnames, str):
            in_fnames = [in_fnames]

        return numpy.array(
            Parallel(num_cores, verbose=verbose)(
                delayed(self._get_fragment_coverage_sum)(in_fname, sorted=sorted)
                for in_fname in in_fnames
            )
        )

    @staticmethod
    def _error_on_invalid_new_starts(new_start):
        if (new_start < 0).any():
            raise OutOfBoundsError(
                f"There is not enough flanking sequence to modify this region"
                f"(would result in a start coordinate of '{new_start.min()} at idx {new_start.argmin()}')"
            )

    @staticmethod
    def _error_on_invalid_new_stops(rdf, new_stop):
        if rdf.ref == 'NA':
            return

        valid_contig_set = set(CONTIG_LENGTHS[rdf.ref].keys())
        for contig in sorted(set(rdf.contig)):
            new_stops_for_contig = new_stop[rdf.contig == contig]
            if (
                contig in valid_contig_set
                and new_stops_for_contig.max() > CONTIG_LENGTHS[rdf.ref][contig]
            ):
                raise OutOfBoundsError(
                    f"There is not enough flanking sequence to modify a region "
                    f"(would result in a stop coordinate of '{new_stops_for_contig.max()}' at "
                    f"idx {new_stops_for_contig.argmax()} but the "
                    f"chrom length is '{CONTIG_LENGTHS[rdf.ref][contig]}')"
                )

    def _valid_regions_mask(self, new_start, new_stop, discard_buffer_bp=0):
        # assert self.shape[0] == new_start.shape
        # assert self.shape[0] == new_stop.shape
        ok = (new_start - discard_buffer_bp) >= 0
        for contig in sorted(set(self.contig)):
            max_len = CONTIG_LENGTHS[self.ref][contig]
            contig_good = (new_stop + discard_buffer_bp) <= max_len
            contig_good |= self.contig != contig
            ok &= contig_good

        return ok

    def _resize_region_boundaries(
        self,
        left: int = 0,
        right: int = 0,
        inplace: bool = False,
        strand_aware: bool = False,
        discard_invalid_resizes: bool = False,
    ):
        if not inplace:
            rdf = self.copy()
        else:
            rdf = self

        if strand_aware:
            neg_mask = self.strand == "-"

        new_starts = rdf.start + left
        if strand_aware:
            new_starts[neg_mask] = rdf.loc[neg_mask, "start"] - right
        # assert (new_starts > 0).all(), pd.DataFrame(dict(right=right, length=self.region_lengths, total=self.region_lengths+right)).total.value_counts()

        new_stops = rdf.stop + right
        if strand_aware:
            new_stops[neg_mask] = rdf.loc[neg_mask, "stop"] - left

        valid_regions_mask = self._valid_regions_mask(
            new_starts, new_stops, discard_buffer_bp=0
        )

        rdf["start"] = new_starts
        rdf["stop"] = new_stops
        if discard_invalid_resizes:
            rdf = rdf.loc[valid_regions_mask, :]
        else:
            self._error_on_invalid_new_starts(new_starts)
            self._error_on_invalid_new_stops(self, new_stops)

        return rdf

    def expand_regions(
        self,
        /,
        left_amt: int = 0,
        right_amt: int = 0,
        inplace: bool = False,
        strand_aware: bool = False,
        discard_invalid_resizes: bool = False,
    ):
        assert (np.array(left_amt) >= 0).all()
        assert (np.array(right_amt) >= 0).all()
        return self._resize_region_boundaries(
            -left_amt, right_amt, inplace, strand_aware, discard_invalid_resizes
        )

    def truncate(
        self,
        /,
        left_amt: int = 0,
        right_amt: int = 0,
        inplace: bool = False,
        strand_aware: bool = False,
        discard_invalid_resizes: bool = False,
    ):
        assert (np.array(left_amt) >= 0).all()
        assert (np.array(right_amt) >= 0).all()
        return self._resize_region_boundaries(
            left_amt, -right_amt, inplace, strand_aware, discard_invalid_resizes
        )

    def resize_regions(
        self,
        new_size: Union[int, Sequence[int]],
        inplace: bool = False,
        discard_invalid_resizes: bool = False,
        discard_buffer_bp: int = 0,
    ):
        if not inplace:
            rdf = self.copy()
        else:
            rdf = self

        sizes = rdf.stop - rdf.start
        # midpoints = rdf.start + sizes // 2
        # new_start = midpoints - new_size // 2
        new_start = Region.get_resize_starts(rdf.start, sizes, new_size, rdf.strand)
        new_stop = new_start + new_size

        if discard_invalid_resizes:
            ok = (new_start - discard_buffer_bp) >= 0
            filter = None
            for contig in sorted(set(rdf.contig)):
                max_len = CONTIG_LENGTHS[rdf.ref][contig]
                contig_bad = ((new_stop + discard_buffer_bp) > max_len) & (
                    rdf.contig == contig
                )
                if filter is None:
                    filter = contig_bad
                else:
                    filter = filter | contig_bad
            ok = ok & ~filter

            rdf = rdf.loc[ok, :]
            new_start = new_start[ok]
            new_stop = new_stop[ok]
            logger.warning(
                f"Discarded {np.sum(~ok)} of {len(ok)} regions due to invalid resize."
            )

        self._error_on_invalid_new_starts(new_start)
        self._error_on_invalid_new_stops(rdf, new_stop)
        rdf["start"] = new_start
        rdf["stop"] = new_stop
        return rdf

    def bin_regions_into_windows(self, window_size, mode, stride=None):
        """Multiply all regions by tiling windows across each region in self.

        :param window_size: the size of the window
        :param mode: 'full' or valid'
                      full: expand the region boundaries to produce ceil(region_length/window_size) windows
                      valid: shrink the region boundaries to produce floor(region_length/window_size) windows
                      exact: raise an error if any region_length isn't even divisble by stride and window_size
        :param stride: stride length. window_size%stride must equal 0. Default: window_size
        """
        assert mode in ["full", "valid", "exact"]
        if stride is None:
            stride = window_size
        else:
            if window_size % stride != 0:
                raise ValueError(
                    f"window size ({window_size}) must be evenly divisble by stride ({stride})"
                )

        def _resize(region):
            if mode == "full":
                return region.resize(int(stride * math.ceil(region.length / stride)))
            elif mode == "valid":
                return region.resize(int(stride * math.floor(region.length / stride)))
            elif mode == "exact":
                assert window_size % stride == 0
                if region.length % stride != 0:
                    raise ValueError(
                        "region length ({region.length}) must be evenly divisible by stride ({stride}) in 'exact' mode."
                    )
                return region
            else:
                assert False, "UNREACHABLE"

        # first build all the windows
        index_name = self.index.name
        self_copy = self.reset_index()
        all_windows = []
        for region, record in tqdm(
            self_copy.iter_region_row(), total=self_copy.shape[0], disable=False
        ):
            region = _resize(region)
            all_windows.extend(
                (record.name, x[0], x[1])
                for x in windowed_range(region.start, region.stop, stride)
            )

        window_df = pd.DataFrame(
            all_windows, columns=["index", "new_start", "new_stop"]
        ).set_index("index")
        window_df["new_stop"] = window_df["new_stop"] + window_size - stride

        rv = (
            self_copy.join(window_df)
            .rename(
                columns=dict(
                    new_start="start",
                    new_stop="stop",
                    start="old_start",
                    stop="old_stop",
                )
            )
            .drop(columns=["old_start", "old_stop"])
            .set_index("index")
        )
        rv.index.rename(index_name, inplace=True)
        return rv

    def split_on_query(self, query):
        """Split self into two dataframes.

        Returns:
        df1: subset of self selected by 'query'
        df2: subset of self *not* selected by 'query'
        """
        df1 = self.query(query)
        df2 = self.query("not ({})".format(query))
        return df1, df2

    def split_on_column(
        self, column_name: Union[bytes, str], value_groups: Iterable[str]
    ):
        # fix any string/bytes contigs

        if column_name not in self.columns:
            raise KeyError(f"Column {column_name} does not exist")

        value_groups = [
            [x] if (isinstance(x, str) or isinstance(x, bytes)) else x
            for x in value_groups
        ]

        # make sure that no value is in multiple groups
        all_values = set(chain.from_iterable(value_groups))
        if sum(len(x) for x in value_groups) != len(all_values):
            raise ValueError(
                f"contigs in groups must not overlap (saw '{value_groups}')"
            )

        dfs = []
        for values in value_groups:
            dfs.append(self.query("column_name in @values"))

        return dfs

    def split_on_contig(self, contig_groups):
        # fix any string/bytes contigs
        contig_groups = [
            [x] if (isinstance(x, str) or isinstance(x, bytes)) else x
            for x in contig_groups
        ]

        # make sure that no chromosomes are in multiple groups
        all_chroms = set(chain.from_iterable(contig_groups))
        if sum(len(x) for x in contig_groups) != len(all_chroms):
            raise ValueError(
                f"contigs in groups must not overlap (saw '{contig_groups}')"
            )

        dfs = []
        for contigs in contig_groups:
            dfs.append(self.query("contig in @contigs"))

        return dfs

    def iter_region_row(self):
        """
        iterates over dataframe rows with the region, each item yielded will be: (region, row)
        """
        for _, row in self.iterrows():
            yield Region(row.contig, row.start, row.stop), row

    def to_tsv(
        self,
        path_or_buf=None,
        columns=None,
    ):
        """
        Using this function to write regions dataframes to disk facilitates writing and also
        prevents discrepancies and corrupt dataframes by enforcing certain keywords
        :param path_or_buf: a file path or buffer
        :param columns: list of column names
        :return:
        """

        self.to_csv(
            path_or_buf=path_or_buf,
            sep="\t",
            columns=columns,
            header=True,
            index=False,
            index_label=None,
        )

    def _get_seq(
        self,
        fasta_path,
        seq_type,
        reverse_complement_sequence_if_minus_strand,
        verbose=False,
    ):
        if fasta_path is None:
            fasta_path = self.get_fasta_path()

        assert seq_type in ["one_hot_encoded", "bytearray"]
        if seq_type == "one_hot_encoded":
            method = "get_one_hot_encoded_sequence"
            name = "one_hot_encoded_sequence"
        elif seq_type == "bytearray":
            method = "get_sequence"
            name = "sequence"
        else:
            assert False, "UNREACHABLE"

        seqs = []
        with pysam.FastaFile(fasta_path) as fasta:
            for region in tqdm(
                self.iter_regions(), total=len(self), disable=(not verbose), desc="get sequences"
            ):
                seqs.append(
                    getattr(region, method)(
                        fasta,
                        reverse_complement_sequence_if_minus_strand=reverse_complement_sequence_if_minus_strand,
                    )
                )
        return pd.Series(seqs, index=self.index, name=name)

    def get_sequence(
        self,
        fasta_path=None,
        reverse_complement_sequence_if_minus_strand=False,
        verbose=False,
    ):
        return self._get_seq(
            fasta_path,
            "bytearray",
            reverse_complement_sequence_if_minus_strand=reverse_complement_sequence_if_minus_strand,
            verbose=verbose,
        )

    def attach_sequence(self, *args, rebuild=False, **kwargs):
        if "sequence" in self.columns and not rebuild:
            return self
        return self.join(self.get_sequence(*args, **kwargs))

    def get_one_hot_encoded_sequence(
        self,
        fasta_path=None,
        reverse_complement_sequence_if_minus_strand=False,
        verbose=False,
    ):
        return self._get_seq(
            fasta_path,
            "one_hot_encoded",
            reverse_complement_sequence_if_minus_strand=reverse_complement_sequence_if_minus_strand,
            verbose=verbose,
        )

    def attach_one_hot_encoded_sequence(self, *args, rebuild=False, **kwargs):
        if "one_hot_encoded_sequence" in self.columns and not rebuild:
            return self
        return self.join(self.get_one_hot_encoded_sequence(*args, **kwargs))

    def get_pfm(
        self,
        reverse_complement_sequence_if_minus_strand=True,
        verbose=True,
    ):
        """Get the pfm by stacking up the sequence over all regions.

        :param reverse_complement_sequence_if_minus_strand: If the region is on the minus strand,
            and this is true, sequences will be reverse complemented
        :param verbose: ignored
        :param return_series: if True the result is returned as a
        :return:
        """

        region_lengths = self.region_lengths.unique().tolist()
        if len(region_lengths) > 1:
            raise ValueError(
                "window_size must be set if all regions don't have the same length"
            )

        if not set(self.strand) == {"-", "+"}:
            warnings.warn(
                "RegionDataFrame does not seem to have strand information. Finding PFM may be problematic"
            )
        # (window_size, 4)
        sequences = numpy.array(
            self.get_one_hot_encoded_sequence(
                reverse_complement_sequence_if_minus_strand=reverse_complement_sequence_if_minus_strand,
                verbose=verbose,
            )
        )
        sequences = np.stack(sequences, axis=0)
        sequences = numpy.swapaxes(sequences, 1, 2)
        pfm = Pfm(freqs=sequences)
        return pfm

    def get_pwm(self, *args, **kwargs):
        return self.get_pfm(*args, **kwargs).pwm

    def split(self, num_sections):
        num_each_section, extras = divmod(self.nrow, num_sections)
        section_sizes = (
            [0]
            + extras * [num_each_section + 1]
            + (num_sections - extras) * [num_each_section]
        )
        div_points = np.array(section_sizes, dtype=np.intp).cumsum()

        sub_rdfs = []
        for i in range(num_sections):
            st = div_points[i]
            end = div_points[i + 1]
            sub_rdfs.append(self.iloc[st:end, :])

        return sub_rdfs

    def _parallel_apply(self, fn, n_workers, verbose):
        # do this in a fork context so that we don't need to serialize the data into the new process
        ctx = multiprocessing.get_context("fork")

        # set a shared counter to track the row index to process
        counter = ctx.Value("i", 0)
        recv_conn, send_conn = ctx.Pipe(duplex=False)
        pipe_lock = multiprocessing.Lock()
        ps = [
            ctx.Process(
                target=_apply_fn, args=(self, fn, counter, send_conn, pipe_lock)
            )
            for _ in range(n_workers)
        ]
        for p in ps:
            p.start()

        # clear any cache -- works around a jupyter display bug
        tqdm._instances.clear()
        if verbose:
            pbar = tqdm(total=self.shape[0])

        def _cleanup():
            # close the pipe
            send_conn.close()
            recv_conn.close()

            # join all of the worker processes
            for p in ps:
                p.join()

            if verbose:
                pbar.close()

        # store the returned row indices and records
        # we track the indices so that we can re-sort into the original order and then
        # join with the input dataframe index
        indices = []
        records = []
        # keep processing new data until the returned records equals the inputs
        while len(indices) < self.shape[0]:
            if recv_conn.poll(0.1):
                res = pickle.loads(recv_conn.recv_bytes())

                # if a worker raised an exception kill any active process, cleanup, and then re-raise the exception
                if isinstance(res, Exception):
                    for p in ps:
                        p.kill()
                    _cleanup()
                    raise res

                i, rec = res
                indices.append(i)
                records.append(rec)
                if verbose:
                    pbar.update(1)

        _cleanup()
        return indices, records

    def parallel_apply(self, fn, n_workers=None, verbose=True):
        # special case n_workers == 1 so that it runs in the main thread -- mostly used for debugging purposes
        if n_workers == 1:
            indices = []
            records = []
            for idx in tqdm(range(self.shape[0]), disable=(not verbose)):
                indices.append(idx)
                records.append(fn(self.iloc[idx]))
        else:
            # if the number of workers isn't set then use all available cpus
            if n_workers is None:
                n_workers = multiprocessing.cpu_count()
            indices, records = self._parallel_apply(fn, n_workers, verbose)

        # concatanate all records into a dataframe
        rv = pandas.DataFrame(records)
        # add the numeric indices and sort to the original order
        rv.index = indices
        rv = rv.sort_index()
        # re-attach the original index
        rv.index = self.index

        return rv

    ###############################################################################################
    # #  These are methods that require a label column
    # #  this should probably be split into a subclass

    def label_balanced(self, column_name, random_state=None):
        """Return a copy of self with balanced labels."""
        if not hasattr(self, column_name):
            raise ValueError(f"The data frame must have column '{column_name}'")

        keep_idxs = get_indices_of_balanced_labels(
            self[column_name], random_state=random_state
        )

        return self.iloc[keep_idxs].copy()

    def set_binary_label(
        self,
        on_query,
        off_query=None,
        drop_unlabeled_records=True,
        inplace=False,
        label_column="label",
    ):
        """Add a label column.

        Set the label to 1 for records that satisfy on_query, and 0 for records that
        satisdy off query. If off_query is not provided, use the complement of the off
        query.
        """
        # if no off_query is provided, use the complement of the on query
        if off_query is None:
            off_query = "not ({})".format(on_query)

        # get the indices for records that should have an 'on' label
        on_records_index = self.query(on_query).index
        off_records_index = self.query(off_query).index
        # make sure that the on and off queries were mutually exclusive
        if len(on_records_index.intersection(off_records_index)) > 0:
            raise ValueError(
                "'on_query' and 'off_query' returned some of the same records.\n"
                "Hint: the on and off queries must be mutually exclusive"
            )

        # create the label column, and set the labels
        self["label"] = -1
        # self.loc[:, 'label'] = -1
        self.loc[on_records_index, "label"] = 1
        self.loc[off_records_index, "label"] = 0

        if drop_unlabeled_records:
            return self.drop_unlabeled_records(inplace=inplace)
        else:
            return self

    # REQUIRES LABEL
    def set_binary_label_by_thresholds(
        self,
        columns,
        on_threshold,
        off_threshold,
        drop_unlabeled_records=True,
        inplace=False,
        label_column="label",
    ):
        """Applies on/off thresholds to columns and sets the label column.

        0 if all columns are below off_threshold
        1 if all columns are above on_threshold
        -1 if they are neither
        """
        assert on_threshold >= off_threshold

        # if columns is a string, then assume that we want to select a single column
        if isinstance(columns, (str, bytes)):
            columns = [
                columns,
            ]

        on_query = " and ".join(
            "{} > {}".format(column, on_threshold) for column in columns
        )
        off_query = " and ".join(
            "{} < {}".format(column, off_threshold) for column in columns
        )
        return self.set_binary_label(
            on_query,
            off_query,
            drop_unlabeled_records=drop_unlabeled_records,
            inplace=inplace,
            label_column=label_column,
        )

    # REQUIRES LABEL
    def downsample_stratified_by_label(self, max_features_per_label):
        """
        Downsample so that there is at most `max_number_of_samples_per_label` for each label.  Useful for quick
        testing.
        """
        # FIXME should we break this class in two
        #  (one which has sample_ids/fragment_matrices and another which doesnt?)
        assert "sample_id" not in self.columns
        keep_idxs = []
        for label in self["label"].unique():
            rdf_label = self.query(f"label == {label}")
            n = min(len(rdf_label), max_features_per_label)
            idxs = rdf_label.sample(n=n, replace=False).index
            keep_idxs += list(idxs)
        return self.loc[keep_idxs]

    # #  END -- These are methods that require a label column
    ###############################################################################################


def _set_fragment_array_weights_from_pred_record(fragment_array, record, left_expansion):
    assert False
    # avoid circular import
    from fragmentomics_tools.bias_correction.data import track_name_to_index_key, index_key_to_track_name

    cov_type_to_weights_attr = {'first': 'first_covered_base_weights', 'last': 'last_covered_base_weights', 'midpoint': 'weights'}
    cov_type_to_fragment_coord = {'first': 'starts_0', 'last': 'stops_0', 'midpoint': 'midpoints_0'}

    pred_column_to_key = {c[10:]: track_name_to_index_key(c[10:]) for c in record.index if c.startswith("pred_dist.")}
    strands = list(set(x[0] for x in pred_column_to_key.values()))
    fl_bands = list(set(x[1] for x in pred_column_to_key.values()))
    cov_types = list(set(x[2] for x in pred_column_to_key.values()))

    # zero out all of the weights
    for attr in cov_type_to_weights_attr.values():
        getattr(fragment_array, attr)[:] = 0

    # set all of the valid weights
    for strand in strands:
        assert strand in ".+-"
        for fl_lb, fl_ub in [(40, 65), (120, 175)]:
            for cov_type in ['first', 'last', 'midpoint']:
                mask = numpy.zeros(fragment_array.n_fragments).astype(bool)
                mask = (mask | (fragment_array.fragment_lengths >= fl_lb) & (fragment_array.fragment_lengths <= fl_ub))
                if strand in '-+':
                    mask = (mask | (fragment_array.fragment_strands == strand))
                means = record["pred_dist." + index_key_to_track_name((strand, (fl_lb, fl_ub), cov_type))]
                weights = means.mean()/means
                count_indices = (getattr(fragment_array, cov_type_to_fragment_coord[cov_type]) + left_expansion)
                assert (count_indices >= 0).all()
                getattr(fragment_array, cov_type_to_weights_attr[cov_type])[mask] = weights[count_indices[mask]]

    # drop all fragments with 0 weights (this probably means that they were out of the fl bands)
    return fragment_array.mask((fragment_array.weights > 1e-6) | (fragment_array.first_covered_base_weights > 1e-6) | (fragment_array.last_covered_base_weights > 1e-6))


def _set_fragment_array_weights_from_weights_record(fragment_array, record, left_expansion):
    # avoid circular import
    from fragmentomics_tools.bias_correction.data import track_name_to_index_key, index_key_to_track_name

    cov_type_to_weights_attr = {'first': 'first_covered_base_weights', 'last': 'last_covered_base_weights', 'midpoint': 'weights'}
    cov_type_to_fragment_coord = {'first': 'starts_0', 'last': 'stops_0', 'midpoint': 'midpoints_0'}

    pred_column_to_key = {c[10:]: track_name_to_index_key(c[10:]) for c in record.index if c.startswith("pred_dist.")}
    strands = list(set(x[0] for x in pred_column_to_key.values()))
    fl_bands = list(set(x[1] for x in pred_column_to_key.values()))
    cov_types = list(set(x[2] for x in pred_column_to_key.values()))

    # zero out all of the weights
    for attr in cov_type_to_weights_attr.values():
        getattr(fragment_array, attr)[:] = 0

    # set all of the valid weights
    for strand in strands:
        assert strand in ".+-"
        for fl_lb, fl_ub in fl_bands:
            for cov_type in cov_types:
                mask = numpy.zeros(fragment_array.n_fragments).astype(bool)
                mask = (mask | (fragment_array.fragment_lengths >= fl_lb) & (fragment_array.fragment_lengths <= fl_ub))
                if strand in '-+':
                    mask = (mask | (fragment_array.fragment_strands == strand))
                weights = record["pred_dist." + index_key_to_track_name((strand, (fl_lb, fl_ub), cov_type))]
                count_indices = (getattr(fragment_array, cov_type_to_fragment_coord[cov_type]) + left_expansion)
                assert (count_indices >= 0).all()
                assert not np.isnan(weights[count_indices[mask]]).any()
                getattr(fragment_array, cov_type_to_weights_attr[cov_type])[mask] = 1./weights[count_indices[mask]]

    # drop all fragments with 0 weights (this probably means that they were out of the fl bands)
    return fragment_array.mask((fragment_array.weights > 1e-6) | (fragment_array.first_covered_base_weights > 1e-6) | (fragment_array.last_covered_base_weights > 1e-6))


class SampleAndRegionDataFrame(RegionDataFrame):
    _additional_required_columns = ["sample_id", "frag_h5"]

    @classmethod
    def init_from_rdf_and_sdf(cls, rdf, sdf):
        return cls(rdf.merge(sdf, how="cross"), ref=rdf.ref)

    @property
    def has_fragment_array(self):
        return bool("fragment_array" in self.columns)

    def _check_has_fragment_array(self):
        assert self.has_fragment_array, (
            "Please first call `srdf = srdf.attach_fragment_array(...)` to generate "
            "the required/missing fragment_array column"
        )

    def load_fragment_arrays(
        self,
        n_workers=None,
        verbose=1,
        min_mapq: int = 0,
        max_frag_len: int = DEFAULT_MAX_FRAG_LEN,
        fragment_array_callback = None,
        **kwargs,
    ):
        """ """
        assert self.index.is_unique
        # reset the progress bar
        tqdm._instances.clear()

        def get_fa(record):
            region = Region(record.contig, record.start, record.stop, record.strand, ref=self.ref)
            _kwargs = dict(
                region=region,
                min_mapq=min_mapq,
                max_frag_len=max_frag_len,
                include_fragment_strand=True,
                **kwargs,
            )
            fa = RegionFragmentArray.from_fragments_h5(record.frag_h5, **_kwargs)
            if fragment_array_callback is not None:
                fa = fragment_array_callback(fa)
            return fa

        if n_workers == 1:
            res = [
                get_fa(x)
                for x in tqdm(
                    self.itertuples(), total=len(self), disable=(verbose <= 0)
                )
            ]
            return pandas.Series(res, index=self.Index, name="fragment_array")
        else:
            if n_workers == None:
                n_workers = multiprocessing.cpu_count()
            field_subset = ["contig", "start", "stop", "strand", "sample_id", "frag_h5"]
            rv = self[field_subset].parallel_apply(get_fa, n_workers=n_workers, verbose=verbose)
            rv.columns = ['fragment_array']
            return rv


    def attach_fragment_arrays(self, *args, rebuild_fragment_arrays=False, **kwargs):
        # If we've already attached
        if not rebuild_fragment_arrays and "fragment_array" in self.columns:
            return self

        fas = self.load_fragment_arrays(*args, **kwargs)
        self["fragment_array"] = fas
        return self

    def bin_regions_into_windows(self, *args, mode, **kwargs):
        # we can only shrink fragment arrays without going back to the fragmnet h5s, so
        # the binning mode needs to be set accordingly
        if self.has_fragment_array and mode not in ("exact", "valid"):
            raise ValueError(
                "bin_regions_into_windows mode must be 'exact' or 'valid' if the srdf has fragment arrays."
                "Hint: If you need to grow regions with fragment arrays you'll need to drop the fragment arrays, resize the regions, and then re-attach the fragment arrays"
            )
        self = super().bin_regions_into_windows(*args, mode=mode, **kwargs)

        # if we have fragmnet arrays then resize them
        if self.has_fragment_array:
            fragment_arrays = [
                record.fragment_array.subset_by_region(region)
                for region, record in tqdm(self.iter_region_row(), total=self.nrow)
            ]
            self["fragment_array"] = fragment_arrays

        return self

    def _resize_region_boundaries(
        self,
        left: int = 0,
        right: int = 0,
        inplace: bool = False,
        strand_aware: bool = False,
        discard_invalid_resizes: bool = False,
    ):
        self = super()._resize_region_boundaries(
            left=left,
            right=right,
            inplace=inplace,
            strand_aware=strand_aware,
            discard_invalid_resizes=discard_invalid_resizes,
        )

        # if we have fragmnet arrays then resize them
        if self.has_fragment_array:
            fragment_arrays = [
                record.fragment_array.subset_by_region(region)
                for region, record in tqdm(self.iter_region_row(), total=self.nrow)
            ]
            self["fragment_array"] = fragment_arrays
            assert False

        return self

    def expand_regions(self, *args, **kwargs):
        if self.has_fragment_array:
            raise ValueError(
                "can not expand regions if the srdf has fragment arrays."
                "Hint: If you need to grow regions with fragment arrays you'll need to drop the fragment arrays, resize the regions, and then re-attach the fragment arrays"
            )
        self = super().expand_regions(*args, **kwargs)

    def resize_regions(self, new_size, *args, **kwargs):
        if self.has_fragment_array:
            if (self.region_lengths < numpy.array(new_size)).any():
                raise ValueError(
                    "can not expand regions if the srdf has fragment arrays."
                    "Hint: If you need to grow regions with fragment arrays you'll need to drop the fragment arrays, resize the regions, and then re-attach the fragment arrays"
                )
        self = super().resize_regions(new_size, *args, **kwargs)
        if self.has_fragment_array:
            fragment_arrays = [
                record.fragment_array.subset_by_region(region)
                for region, record in tqdm(self.iter_region_row(), total=self.nrow)
            ]
            self["fragment_array"] = fragment_arrays

        return self

    def get_sample_count_bounds(self, num_sd):
        res = []
        for sample_id, sub_df in self.groupby("sample_id"):
            counts = pd.DataFrame(
                [x.n_fragments for x in sub_df.fragment_array], columns=[sample_id]
            )
            means = counts.median().rename("mean_fragment_counts")
            stds = counts.apply(lambda x: trimmed_std(x, (0.05, 0.05))).rename(
                "std_fragment_counts"
            )
            mins = (means - num_sd * stds).rename("min_fragments")
            maxs = (means + num_sd * stds).rename("max_fragments")
            res.append(pd.DataFrame([mins, maxs]))
        return pd.concat(res, axis=1).T

    def filter_outlier_counts(self, min_frags, num_sd=3, return_stat_columns=False):
        n_fragments = self.df.progress_apply(
            lambda x: pd.Series(
                dict(sample_id=x.sample_id, n_fragments=x.fragment_array.n_fragments)
            ),
            axis=1,
        )
        n_fragments.index = self.index
        tmp = n_fragments.merge(
            self.get_sample_count_bounds(num_sd=num_sd),
            left_on="sample_id",
            right_index=True,
            how="inner",
        ).drop(columns="sample_id")
        self = self.join(tmp)
        mask = (
            (self.n_fragments >= min_frags)
            & (self.n_fragments >= self.min_fragments)
            & (self.n_fragments <= self.max_fragments)
        )
        rv = self.loc[mask.values, :]
        if not return_stat_columns:
            rv = rv.drop(columns=["n_fragments", "min_fragments", "max_fragments"])
        return rv

    def set_fragment_array_weights(self, model, n_workers=None):
        # find the maximum fragment length so that we can ensure that we have enough context to set
        # start and end wweights
        expansion = 511 # self.fragment_array.apply(lambda x: x.max_frag_len)
        # these are the columns that we use to produce fixed regions
        columns = ['contig', 'strand', 'start', 'stop']
        # predict on all of the unique regions
        tmp_rdf = RegionDataFrame(self[columns].drop_duplicates(), ref=self.ref)
        # join the model back
        pred = tmp_rdf.join(model.predict_weights_from_rdf(tmp_rdf.expand_regions(left_amt=expansion, right_amt=expansion)))
        pred = SampleAndRegionDataFrame(self.df.set_index(columns).join(pred.df.set_index(columns), how='inner').reset_index(), ref=self.ref)
        assert pred.shape[0] == self.shape[0]
        fas = pred.parallel_apply(
            lambda record: _set_fragment_array_weights_from_weights_record(record.fragment_array, record, expansion),
            n_workers=n_workers
        ).iloc[:, 0].rename('fragment_array')
        fas.index = self.index
        self['fragment_array'] = fas
        return

    def reset_fragment_array_weights(self):
        """Set the fragment array weights to zero.

        """
        self.progress_apply(lambda record: record.fragment_array.reset_cutsite_bias_weights(), axis=1)
        return None


class FlDist:
    @classmethod
    def init_from_sdf(cls, sdf):
        max_frag_len = 512
        columns = {}
        for frag_h5 in sdf.frag_h5:
            cnts = frag_h5.fragment_length_counts[:max_frag_len]
            # normalize to library depth
            cnts = cnts / cnts.sum()
            columns["RD-" + frag_h5.sample_id.split("-")[1]] = cnts

        fl_df = pd.DataFrame(columns)
        fl_df = fl_df.set_index(fl_df.index + 1)

        return cls(fl_df)

    def subset_by_sample_ids(self, sample_ids):
        fl_df = self.fl_df.T.loc[sample_ids].T
        return type(self)(fl_df)

    def __init__(self, fl_df):
        self.fl_df = fl_df

    def plot(self, figsize=(20, 8)):
        sns.set(rc={"figure.figsize": figsize})

        # add the reference
        fl_df = self.fl_df.copy()
        ref_fl_dist = fl_df.mean(axis=1)
        ref_fl_dist = ref_fl_dist / ref_fl_dist.sum()
        fl_df.loc[:, "Reference"] = ref_fl_dist

        fl_df.plot(legend=False)


class SampleDataFrame(DataFrameBase):
    _metadata = ["_fl_dist"]
    _required_columns = ["sample_id", "frag_h5"]

    @property
    def df(self):
        return pd.DataFrame(self)

    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)

        # hack around pandas not correctly using _constructor internally
        # This line should always be first.  Pandas incorrectly passes this BlockManager to the constructor sometimes.
        if isinstance(data, pd.core.internals.BlockManager):
            return

        self._fl_dist = FlDist.init_from_sdf(self)

        return

    @property
    def fl_dist(self):
        return self._fl_dist.subset_by_sample_ids(self.sample_id)

    def label_balanced(self, column_name, random_state=None):
        """Return a copy of self with balanced labels."""
        if not hasattr(self, column_name):
            raise ValueError(f"The data frame must have column '{column_name}'")

        keep_idxs = get_indices_of_balanced_labels(
            self[column_name], random_state=random_state
        )

        return self.iloc[keep_idxs].copy()


def intersect_region_dataframes(region_dataframes, sort=False):
    """
    Finds the intersection of a list of RegionDataFrames
    :param region_dataframes: a list of region_dataframes
    :param sort: pre-sort RegionDataFrames before intersecting
    :return: A RegionDataFrame that is the intersection of all passed region_dataframes
    """
    if isinstance(region_dataframes, RegionDataFrame):
        return region_dataframes
    assert isinstance(region_dataframes, (list, tuple)) and all(
        [isinstance(rdf, RegionDataFrame) for rdf in region_dataframes]
    ), "Must pass list or tuple of RegionDataFrames"
    if sort:
        region_dataframes = [rdf.sort() for rdf in region_dataframes]
    intersected_rdf = region_dataframes[0]
    for rdf in region_dataframes[1:]:
        intersected_rdf = intersected_rdf.intersect_with_rdf(rdf, sorted=sort)
    return intersected_rdf
