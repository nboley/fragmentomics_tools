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
from sklearn.utils import shuffle as sk_shuffle
from smart_open import open
from scipy.stats.mstats import trimmed_std

from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

tqdm.pandas()
from joblib import delayed, Parallel

import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor
import traceback

# from tqdm.contrib.concurrent import process_map
# from p_tqdm import p_map

import seaborn as sns

import pysam
import logging

from fragmentomics_tools.region import Region, OutOfBoundsError
from fragmentomics_tools.formats import BedReader, BigWigReader
from fragmentomics_tools.contig import CONTIG_LENGTHS
# Pfm import deferred to avoid top-level torch dependency (see motif.py)
# from fragmentomics_tools.motif import Pfm
from fragmentomics_tools.util.liftover import RegionLiftOver

# from fbio.formats import BedReader, BigWigReader
# from fbio.fragments_h5 import FragmentsH5
# from fbio.liftover import RegionLiftOver
# from fbio.region import Region, OutOfBoundsError
# from fbio.util import aws_utils
# from fbio.util.iter_utils import windowed_range
# from fbio.util.misc_utils import progress_bar
# from ravel.util.ml_utils import get_indices_of_balanced_labels


logger = logging.getLogger(__name__)


from fragmentomics_tools import (
    RegionFragmentArray,
    FragmentArray,
    merge_fragment_arrays,
)


NUM_CORES = -1
DEFAULT_MIN_MAPQ = 10
DEFAULT_MAX_FRAG_LEN = 511


# Per-worker state. This global is only ever written inside a worker process,
# by _init_parallel_apply_worker, which the executor runs once per worker at
# startup. It is deliberately NOT set in the parent: doing so shares one
# mutable slot across every concurrent caller, and two threads calling
# parallel_apply at once then silently compute each other's data. Each
# executor forks its own workers, so per-worker state keeps concurrent calls
# isolated.
#
# The frame and callable reach the workers through the executor's `initargs`
# rather than through the task queue. Under a "fork" context initargs are
# inherited rather than pickled, so the DataFrame is never serialized and `fn`
# may be a lambda. Sending them per task would pickle both on every row.
_PARALLEL_APPLY_STATE = {}


def _init_parallel_apply_worker(df, fn):
    """Runs once per worker process, at fork time."""
    _PARALLEL_APPLY_STATE["df"] = df
    _PARALLEL_APPLY_STATE["fn"] = fn


def _apply_fn(idx):
    """Apply the pending callable to one row. Runs in a forked worker."""
    df = _PARALLEL_APPLY_STATE["df"]
    fn = _PARALLEL_APPLY_STATE["fn"]
    return idx, fn(df.iloc[idx])


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

    def _repr_html_(self, *args, **kwargs):
        return self.df._repr_html_(*args, **kwargs)

    def to_string(self, *args, **kwargs):
        """When formatting for pandas dfs columns are sometimes dropped and then the
           constructor can raise an error if those were required columns. This just
           calls to_string on the base dataframe since it doesn't matter for this type 
           of printing"""
        return self.df.to_string(*args, **kwargs)

    def reorder_columns(self):
        return self[self._required_columns + list(filter(lambda x: x not in set(self._required_columns), self.columns))]

    @classmethod
    def from_fname_s3_or_local(cls, fname, *args, **kwargs):
        warnings.warn("deprecated, use pd.read_table()", DeprecationWarning)
        # allow this to be initialized from a path
        if not isinstance(fname, (str, bytes)):
            raise TypeError("Expecting a file path, got {}".format(repr(fname)))

        data = pandas.read_table(fname)
        return cls(data, *args, **kwargs)

    @property
    def df(self):
        """Cast to a normal pandas dataframe"""
        return pd.DataFrame(self)

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

    def _parallel_apply(self, fn, n_workers, verbose):
        # Use a fork context so the frame is inherited copy-on-write rather
        # than serialized into each worker. Only the row index is sent through
        # the task queue.
        #
        # NOTE: "fork" is load-bearing here -- it is what avoids serializing
        # the frame and what allows `fn` to be unpicklable (e.g. a lambda).
        # Python 3.12+ warns that forking a multi-threaded process may deadlock
        # (tqdm's monitor thread alone is enough to trigger that warning), and
        # the default start method changes in later versions. Moving to "spawn"
        # would require pickling both the frame and `fn` on every call, so it is
        # not a drop-in substitution; it would need a different design.
        #
        # This previously hand-rolled a shared counter, a pipe, a lock and a
        # polling loop. That loop waited for exactly shape[0] results with no
        # liveness check, so a worker dying (OOM kill, segfault, an
        # unpicklable result or exception) hung the parent forever -- verified
        # by execution before this change. ProcessPoolExecutor supervises its
        # workers and raises BrokenProcessPool instead.
        #
        # `self` and `fn` are handed over via initargs, which the executor
        # applies once per worker. Under fork those are inherited, not pickled,
        # so the frame is not serialized and `fn` may be a lambda. Keeping the
        # state per-worker (rather than in a module global in the parent) is
        # what makes concurrent calls from different threads independent.
        ctx = multiprocessing.get_context("fork")

        # clear any cache -- works around a jupyter display bug
        tqdm._instances.clear()

        indices = []
        records = []
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_parallel_apply_worker,
            initargs=(self, fn),
        ) as executor:
            results = executor.map(_apply_fn, range(self.shape[0]))
            for idx, record in tqdm(
                results, total=self.shape[0], disable=(not verbose)
            ):
                indices.append(idx)
                records.append(record)

        return indices, records

    def parallel_apply(self, fn, n_workers=None, verbose=True):
        """Apply `fn` to each row across worker processes.

        :param fn: called with one row (a Series). It may be a lambda or other
            unpicklable callable -- workers inherit it via fork rather than
            receiving it pickled.
        :param n_workers: worker count; None uses every CPU. 1 runs in-process
            without forking, which is useful when debugging `fn`. 0 or negative
            raises ``ValueError``.
        :param verbose: show a progress bar.
        :return: a plain ``pandas.DataFrame``, NOT this subclass, because `fn`
            decides the output columns and they need not satisfy this class's
            required-column contract.

            If `fn` returns DataFrames they are concatenated and an
            ``original_index`` column is added mapping each output row back to
            the input row it came from; `fn` must not itself return that
            column. Otherwise records are treated as rows and the result
            carries this frame's index.

            **An empty input returns an empty DataFrame with no columns** --
            `fn` is never called, so the column set is unknowable.

        `fn` must return the same kind of value for every row, and must not
        return None; both raise ``ValueError`` rather than producing a quietly
        malformed frame.

        Raises whatever `fn` raises. If a worker dies outright (an OOM kill,
        say) this raises ``BrokenProcessPool`` rather than hanging. One
        limitation inherited from multiprocessing: an exception that cannot be
        pickled (one holding a lambda or a lock, say) reaches the caller as a
        pickling error instead of itself, losing the original message.

        Concurrent calls from different threads are independent. Calls may also
        nest: `fn` may itself call ``parallel_apply``.
        """
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

        # An empty input yields no records, and `all(...)` is vacuously true on
        # an empty list, which sent this down the concat branch and died in
        # pd.concat([]) with "No objects to concatenate".
        if len(records) == 0:
            return pandas.DataFrame(index=self.index[:0])

        # Reject a partially-DataFrame result rather than falling through to the
        # row branch, where each DataFrame would be stuffed into a cell as a
        # Series and the output would be quietly garbage.
        # fn returning None is a common mistake (a fn that mutates and forgets
        # to return). Left alone it surfaces as "'NoneType' object is not
        # iterable" from the DataFrame constructor, which names neither fn nor
        # the row.
        #
        # The kind tally distinguishes Series from plain rows as well as from
        # DataFrames. An earlier version only counted DataFrames, so a fn
        # returning a Series for some rows and a dict for others slipped past
        # and died in pandas with "'dict' object has no attribute 'dtype'" --
        # which names neither fn nor the contract it broke.
        n_none = 0
        kinds = set()
        for x in records:
            if x is None:
                n_none += 1
            elif isinstance(x, pd.DataFrame):
                kinds.add("DataFrame")
            elif isinstance(x, pd.Series):
                kinds.add("Series")
            else:
                kinds.add("row")

        if n_none:
            raise ValueError(
                f"fn returned None for {n_none} of {len(records)} rows; it must "
                f"return a value for every row"
            )
        if len(kinds) > 1:
            raise ValueError(
                f"fn must return the same kind of value for every row, but "
                f"returned a mix of {', '.join(sorted(kinds))} across "
                f"{len(records)} rows"
            )

        # if everything is a data frame
        if all(isinstance(x, pd.DataFrame) for x in records):
            # concatanate all records into a dataframe
            index_col = "original_index"
            if any(index_col in x.columns for x in records):
                # previously a bare `assert`, which both crashed the caller and
                # vanishes under `python -O`
                raise ValueError(
                    f"records returned by fn must not contain a "
                    f"'{index_col}' column; it is added here to restore the "
                    f"original row order"
                )
            # assign() rather than mutating: these frames belong to the caller's
            # fn, and adding a column to them in place was visible to anyone
            # holding a reference.
            records = [
                x.assign(**{index_col: self.index[o]})
                for o, x in zip(indices, records)
            ]
            rv = pd.concat([records[x] for x in np.argsort(indices)])
        else:
            rv = pandas.DataFrame(records)
            # add the numeric indices and sort to the original order
            rv.index = indices
            rv = rv.sort_index()
            # re-attach the original index
            rv.index = self.index

        return rv


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

    @property
    def _required_columns(self):
        return self._critical_bed_columns + self._additional_required_columns

    def get_fasta_path(self):
        if self.ref == 'hg38':
            return "/scratch/karius/annotation/GRCh38.p12.genome.fa.gz"
        else:
            raise ValueError(f"No fasta associated with reference '{self.ref}'")

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

    def equals_rdf(self, other):
        """Check if two region dataframes have identical regions, in the same order.

        This was previously ``__eq__``, which broke the pandas contract
        (element-wise comparison) and made instances unhashable.
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

    def center_on_summit(self):
        """Center regions on summit, resize the regions, and then drop the summit column."""
        if "summit" not in self.columns:
            raise TypeError("Must contain a 'summit' column to center on the summit.")

        region_lengths = (self.stop - self.start).copy()

        # check if the summit is within start
        if ((self.summit >= self.start) & (self.summit < self.stop)).all():
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
        verify_motif_scores: bool = False,
        inplace: bool = False,
        unique_regions: bool = False,
        shuffle_tf_motif: bool = False,
        default_jaspar: bool = False,
        shuffle_tf_seed: int = 314,
        quiet: bool = False,
        ref_path: Optional[str] = None,
        pfms: Optional[List] = None,
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

        if pfms is not None:
            tf_conv = TFConv1D(
                tf_width=target_len,
                only_logo=True,
                output_both_strands=True,
                channels=4,
                pfms=pfms,
                shuffle_motifs=shuffle_tf_motif,
                shuffle_seed=shuffle_tf_seed,
            ).eval()
        else:
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

        if ref_path is None:
            ref_path = self.get_fasta_path()

        with torch.no_grad():
            sds = SeqDataSet(
                region_dataframe=self
                if tf_search_width is None
                else self.resize_regions(tf_search_width),
                ref_path=ref_path,
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
            # `self.verify_motif_scores` has never existed -- it is an orphaned
            # reference to the removed `ravel` package (see the commented-out
            # imports at the top of this module), so this branch always raised
            # AttributeError. Every caller in biomarker-projects passes
            # verify_motif_scores=False, which is how a method with 24
            # consumers survived a broken default. The default is now False so
            # the working path is the default one; asking for verification
            # fails loudly and accurately instead of with an AttributeError
            # about a missing attribute.
            raise NotImplementedError(
                "verify_motif_scores=True is not implemented: the underlying "
                "verify_motif_scores() method does not exist (it was lost in "
                "the migration away from `ravel`). Pass verify_motif_scores="
                "False, which is what every existing caller does."
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
        # bedtools merge emits BED3 (chrom/start/end) unless column aggregation
        # is requested via -c/-o, so the output column set is decided by
        # bedtools, not by self.columns. Passing `names=list(self.columns)`
        # here silently produced an all-NaN column for every field beyond the
        # first three -- merging a frame with strand/name/score returned those
        # three columns entirely NaN rather than dropping them.
        merged_df = bedtool.merge(**kwargs).to_dataframe()
        merged_df.columns = ["contig", "start", "stop"] + list(merged_df.columns[3:])
        return type(self)(merged_df, ref=self.ref)

    def intersect_with_rdf(self, other, sorted=False, rsuff="other", **intersect_kwargs):
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
        this_bed_df = self[reordered_columns(self)]
        this_bed_df["index"] = self_index
        this_bedtool = pybedtools.BedTool.from_dataframe(this_bed_df)
        other_bedtool = pybedtools.BedTool.from_dataframe(
            other[reordered_columns(other)]
        )
        other_column_names = [(f"{c}_{rsuff}" if c in this_bed_df.columns else c) for c in reordered_columns(other)]

        # `intersect_kwargs` was documented in the docstring but not accepted,
        # so any caller using it died with TypeError -- which is why
        # get_overlapping_base_counts (which passes wao=True) could never run.
        # -wo/-wao already imply writing both A and B, so only default to
        # wa/wb when the caller has not chosen its own output mode.
        bedtools_kwargs = dict(intersect_kwargs)
        if not any(k in bedtools_kwargs for k in ("wa", "wb", "wo", "wao", "u", "c")):
            bedtools_kwargs.update(wa=True, wb=True)

        names = this_bed_df.columns.tolist() + other_column_names
        # -wo/-wao append a trailing column holding the overlap length.
        if bedtools_kwargs.get("wo") or bedtools_kwargs.get("wao"):
            names = names + ["overlap"]

        intersection_rdf = pd.DataFrame(
            this_bedtool.intersect(
                other_bedtool, sorted=sorted, **bedtools_kwargs
            ).to_dataframe(names=names),
        )
        if len(intersection_rdf) == 0:
            return type(self)(
                pd.DataFrame(columns=names).drop(columns=["index"]),
                ref=self.ref,
            )

        intersection_rdf = intersection_rdf.set_index("index")
        return type(self)(intersection_rdf, ref=self.ref)

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
        self, new_ref, transfer_columns=True, remove_non_liftoverable_regions=True
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

    def attach_blacklist_regions(self, bed_fname, rsuff="other"):
        tmp = self.intersect_with_rdf(
            RegionDataFrame.from_bed(bed_fname, ref=self.ref), rsuff=rsuff
        )
        if len(tmp) == 0:
            self["blacklist_regions"] = ""
            return self

        # NOTE: iter_regions() reads the contig/start/stop columns, which on
        # the intersection result belong to *self*, not to the blacklist. Using
        # it here attached each query region to itself instead of the
        # overlapping blacklist interval -- silently, with no error. The
        # blacklist coordinates live in the `_{rsuff}`-suffixed columns that
        # intersect_with_rdf produces.
        def _blacklist_regions_for(group):
            return [
                Region(
                    row[f"contig_{rsuff}"],
                    row[f"start_{rsuff}"],
                    row[f"stop_{rsuff}"],
                    ref=self.ref,
                )
                for _, row in group.iterrows()
            ]

        blacklist_regions = (
            tmp.groupby(tmp.index.names)
            .apply(_blacklist_regions_for)
            .rename("blacklist_regions")
        )
        return self.join(blacklist_regions).fillna("")

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
                return counts_vect
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

    def truncate_regions(
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
            # NOTE: this was `self.query("column_name in @values")` -- a literal
            # string, so it looked for a column actually named "column_name"
            # and the parameter was never used. The method could never have
            # worked for any input.
            dfs.append(self.query(f"{column_name} in @values"))

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
        from fragmentomics_tools.motif import Pfm
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

    # #  END -- These are methods that require a label column
    ###############################################################################################




def _detach_h5_inplace(df):
    """Replace live FragmentsH5 handles with their file paths in-place."""
    if "frag_h5" in df.columns:
        df["frag_h5"] = df["frag_h5"].apply(
            lambda h5: h5._f_fname if hasattr(h5, '_f_fname') else h5
        )


class SampleAndRegionDataFrame(RegionDataFrame):
    _additional_required_columns = ["sample_id", "frag_h5"]

    def detach_h5(self):
        """Replace live FragmentsH5 handles with their file paths.

        After detaching, the object can be pickled portably. If the stored
        paths still resolve on the current system, load_fragment_arrays will
        transparently re-open them.
        """
        _detach_h5_inplace(self)
        return self

    def reorder_columns(self):
        # hacky way to make sure that fragmnet array is displayed at the start if it exists
        req = self._required_columns
        not_req = list(filter(lambda x: x not in set(self._required_columns), self.columns))
        if 'fragment_array' in not_req:
            not_req.remove('fragment_array')
            req.append('fragment_array')
        return self[req + not_req]

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
        max_frag_len: int = DEFAULT_MAX_FRAG_LEN,
        generate_weights_callback = None,
        fragment_array_callback = None,
        fetch_array_kwargs: dict = None,
        min_mapq: int = None,
    ):
        """ """
        assert self.index.is_unique
        # reset the progress bar
        tqdm._instances.clear()

        def get_fa(record):
            region = Region(record.contig, record.start, record.stop, record.strand, ref=self.ref)
            _kwargs = dict(
                region=region,
                max_frag_len=max_frag_len,
                generate_weights_callback=generate_weights_callback,
                fetch_array_kwargs=fetch_array_kwargs,
                min_mapq=min_mapq,
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
            return pandas.Series(res, index=self.index, name="fragment_array")
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
            # NOTE: a bare `assert False` sat here, immediately after the
            # fragment arrays were resized, so any resize of an SRDF carrying
            # fragment arrays crashed unconditionally. It is debug debris --
            # there is no condition it was guarding, and the assignment above
            # is the intended end of this branch.

        return self

    def expand_regions(self, *args, **kwargs):
        if self.has_fragment_array:
            raise ValueError(
                "can not expand regions if the srdf has fragment arrays."
                "Hint: If you need to grow regions with fragment arrays you'll need to drop the fragment arrays, resize the regions, and then re-attach the fragment arrays"
            )
        # NOTE: this previously ended with `self = super().expand_regions(...)`
        # and no return. Rebinding the local name has no effect on the caller,
        # so the method returned None and the expansion was silently discarded.
        return super().expand_regions(*args, **kwargs)

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

    def set_fragment_array_weights(self, weight_fn, n_workers=None, verbose=True):
        """Apply a weight callback to all fragment arrays in place.

        The callback protocol is::

            weight_fn(fa) -> numpy.ndarray   # shape (n_fragments,)

        The callback takes a FragmentArray and returns a 1-D array of
        per-fragment weights. These weights are applied uniformly to all
        coverage types (first, last, midpoint).

        Args:
            weight_fn: A callable that takes a FragmentArray and returns
                a 1-D array of weights. See ``fragmentomics_tools.fragment_array.weights``
                for available implementations.
            n_workers: Number of parallel workers. None uses all CPUs,
                1 runs single-threaded (useful for debugging).

        Returns:
            self: The same SampleAndRegionDataFrame, for method chaining.

        Raises:
            NotImplementedError: If called with an old v1 bias_correction model
                (detected via ``hasattr(weight_fn, "predict_weights_from_rdf")``).

        Example::

            from fragmentomics_tools.fragment_array.weights import UniformWeights, GCFlWeights

            # Reset to uniform weights
            srdf.set_fragment_array_weights(UniformWeights())

            # Apply GC/FL correction
            from flgc.model import GCFlDistModel
            normalizer = GCFlDistModel.load(...)
            srdf.set_fragment_array_weights(GCFlWeights(normalizer))
        """
        # Guard against old v1 usage
        if hasattr(weight_fn, "predict_weights_from_rdf") or not callable(weight_fn):
            raise NotImplementedError(
                "set_fragment_array_weights now takes a callback returning one weight "
                "per fragment, not a v1 bias_correction model.\n"
                "  was:  srdf.set_fragment_array_weights(model)\n"
                "  now:  srdf.set_fragment_array_weights(GCFlWeights(normalizer))\n"
                "See fragmentomics_tools/fragment_array/weights.py for available "
                "weight functions."
            )

        self._check_has_fragment_array()

        # The vector is wrapped in a dict so parallel_apply keeps each array in
        # ONE cell. Returning a bare ndarray makes pandas treat it as a row of
        # per-fragment columns, so the caller silently gets scalars instead of
        # vectors -- and unequal fragment counts would NaN-pad on top of that.
        def _compute(record):
            return {"weights": weight_fn(record.fragment_array)}

        # Workers return only the weight vectors, which are cheap to ship. They
        # must not assign: a forked worker mutating its own copy is discarded,
        # so the parent does the assignment below.
        computed = self.parallel_apply(_compute, n_workers=n_workers, verbose=verbose)

        # Assign positionally -- parallel_apply preserves input order.
        for fa, weights in zip(self["fragment_array"], computed["weights"]):
            fa.assign_weights(weights)

        return self


class FlDist:
    @classmethod
    def init_from_sdf(cls, sdf):
        max_frag_len = 512
        columns = {}
        for record in sdf.itertuples():
            cnts = record.frag_h5.fragment_length_counts
            if len(cnts) < max_frag_len:
                cnts = np.pad(cnts, (0, max_frag_len - len(cnts)))
            else:
                cnts = cnts[:max_frag_len]
            # normalize to library depth
            total = cnts.sum()
            if total > 0:
                cnts = cnts / total
            columns[record.sample_id] = cnts

        fl_df = pd.DataFrame(columns)
        fl_df = fl_df.set_index(fl_df.index + 1)

        return cls(fl_df)

    def subset_by_sample_ids(self, sample_ids):
        fl_df = self.fl_df.T.loc[sample_ids].T
        return type(self)(fl_df)

    def __init__(self, fl_df):
        self.fl_df = fl_df


    def plot(self, figsize=(20, 8), legend=False, max_frag_len=None, include_reference=True):
        sns.set(rc={"figure.figsize": figsize})

        # add the reference
        fl_df = self.fl_df.copy()
        if include_reference:
            ref_fl_dist = fl_df.mean(axis=1)
            ref_fl_dist = ref_fl_dist / ref_fl_dist.sum()
            fl_df.loc[:, "Reference"] = ref_fl_dist

        return fl_df.loc[0:max_frag_len, :].plot(legend=legend)


class SampleDataFrame(DataFrameBase):
    _metadata = ["_fl_dist"]
    _required_columns = ["sample_id", "frag_h5"]

    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)

        # hack around pandas not correctly using _constructor internally
        # This line should always be first.  Pandas incorrectly passes this BlockManager to the constructor sometimes.
        if isinstance(data, pd.core.internals.BlockManager):
            return

        if (
            "frag_h5" in self.columns
            and len(self) > 0
            and hasattr(self["frag_h5"].iloc[0], "fragment_length_counts")
        ):
            self._fl_dist = FlDist.init_from_sdf(self)
        else:
            self._fl_dist = None

        return

    def detach_h5(self):
        """Replace live FragmentsH5 handles with their file paths.

        After detaching, the object can be pickled portably. The fl_dist
        property remains available if it was built before detach (it is
        preserved in pandas _metadata through pickle).
        """
        _detach_h5_inplace(self)
        return self

    def dropna(self, *args, **kwargs):
        return type(self)(self.df.dropna(*args, **kwargs))

    @property
    def fl_dist(self):
        if self._fl_dist is None:
            raise RuntimeError(
                "fl_dist unavailable: frag_h5 handles were not provided or have "
                "been detached. Build the SampleDataFrame with live FragmentsH5 "
                "handles to compute fl_dist."
            )
        return self._fl_dist.subset_by_sample_ids(self.sample_id)

    def label_balanced(self, column_name, random_state=None):
        """Return a copy of self with balanced labels."""
        if not hasattr(self, column_name):
            raise ValueError(f"The data frame must have column '{column_name}'")

        keep_idxs = get_indices_of_balanced_labels(
            self[column_name], random_state=random_state
        )

        return self.iloc[keep_idxs].copy()


def str_concat_columns(input_df, agg_column_names):
    """Group a DataFrame and concatenate string columns within each group.

    Groups ``input_df`` by all columns *except* those in ``agg_column_names``,
    then joins the values of each aggregation column with commas. Also adds an
    'n' column with the group size.

    Args:
        input_df: The input DataFrame.
        agg_column_names: Column names whose values should be comma-concatenated
            within each group.

    Returns:
        A DataFrame with the groupby columns, the concatenated aggregation
        columns, and an 'n' column indicating group size.
    """
    def apply_fn(sub_df):
        res = {key: ",".join(sub_df[key]) for key in agg_column_names}
        assert 'n' not in input_df.columns
        res['n'] = sub_df.shape[0]
        return pd.Series(res)

    groupby_columns = list(set(input_df.columns) - set(agg_column_names))
    return input_df.groupby(groupby_columns).apply(apply_fn, include_groups=False).reset_index()


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
