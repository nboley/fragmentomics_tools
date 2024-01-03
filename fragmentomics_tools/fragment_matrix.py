import copy
import itertools as it
import warnings
from dataclasses import dataclass, replace
from typing import Iterable, Dict, Tuple, List, Union, Optional

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches, cm
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from scipy.interpolate import RectBivariateSpline

from scipy.sparse import coo_matrix, csr_matrix

from fbio.annotations import ANNOTATIONS
import fbio.logging as logging
from fbio import pysam
from fbio.fragment import (
    Fragment,
    bam_to_fragments,
)
from fbio.fragments_h5 import FragmentsH5
from fbio.region import Region
from ravel.bio.frag.fragment_matrix_math import (
    make_ones_coo_arr,
    reverse_sum_pool,
)
from ravel.bio.frag.fragment_matrix_math import sum_pool
from ravel.constants import (
    DEFAULT_MAX_FRAG_LEN,
    DEFAULT_MIN_MAPQ,
    DEFAULT_DEDUP,
    DEFAULT_VPLOT_SUMPOOL_BY,
)
from fbio.util.numpy_utils import jitter_matrix

logger = logging.getLogger(__name__)


def iter_fragments_from_10x_frag_tsv(in_frag_tsv, region, max_frags=100000000):
    with pysam.TabixFile(in_frag_tsv) as frag_tsv:
        for line in it.islice(frag_tsv.fetch(region.chrom, region.start, region.stop), max_frags):
            contig, start, stop = line.split("\t")[:3]
            start, stop = int(start), int(stop)
            yield Fragment(contig, start, stop)


def _rescale_shape(shape, frag_step_size, pos_step_size):
    return (shape[0] * frag_step_size, shape[1] * pos_step_size)


def dense_array_into_coo_arr(arr, frag_step_size=1, pos_step_size=1):
    """Build a smoothed fragment matrix from a dense array.
    """
    # convert the dense array into a coo_matrix where every data entry is 1.
    # This makes conversion into the data frame of positions straightforward.
    coo_arr = make_ones_coo_arr(coo_matrix(arr))
    assert coo_arr.shape == arr.shape
    # if the frag_step_size or pos_step_size is > 1, then we need to rescale the
    # shape so that the coo_arr has the correct dimensions. This will happen if
    # we want to load a sum_pooled array.
    if frag_step_size > 1 or pos_step_size > 1:
        coo_arr = coo_matrix(
            # multiplying each row entry by the step sizes puts them onto the
            # correct scale
            (coo_arr.data, (coo_arr.row * frag_step_size, coo_arr.col * pos_step_size),),
            # it is critical that we set the shape. Otherwise, the downstream code
            # will yield a smoothed array that is the wrong size.
            shape=_rescale_shape(arr.shape, frag_step_size, pos_step_size),
        )
    return coo_arr


@dataclass
class FragmentMatrix:
    # 2 dimensions coo matrix
    # the first dimension (shape[0]) is the fragment length
    # the second dimension (shape[1]) is the position
    arr: coo_matrix

    def __post_init__(self,):
        if isinstance(self.arr, (np.ndarray, csr_matrix)):
            self.arr = make_ones_coo_arr(coo_matrix(self.arr))
        # assert np.issubdtype(self.arr.dtype, np.integer)  # FIXME re-enable this and test get tests working

    @staticmethod
    def from_fragment_matrices(fragment_matrices):
        return merge_fragment_matrices(fragment_matrices)

    @property
    def dense_array(self):
        return self.arr.toarray()

    def todense(self):
        """Return a fragment matrix array.

        Mimics the coo_matrix interface.
        """
        return self.dense_array

    def get_density(self, pseudo_count=0):
        arr = self.dense_array + pseudo_count
        return arr / arr.sum()

    def get_fragment_length_density(self, pseudo_count=0):
        arr = self.dense_array.sum(1) + pseudo_count
        return (arr / arr.sum()).flatten()

    def get_coverage_density(self, pseudo_count=0):
        arr = self.dense_array.sum(0) + pseudo_count
        return arr / arr.sum()

    def sum_pooled(self, sum_pool_by):
        # deepcopy is to copy all attributes (for example the region in RegionFragmentMatrix)
        return replace(self, arr=sum_pool(self.arr, sum_pool_by))

    def reverse_sum_pooled(self, sum_pool_by, preserve_sum: bool = False):
        """Upscale the array.

        e.g.
        if we have an array [[2]] and reverse_sum_pool with sum_pool_by=2 then we should return
        [[2, 2], [2, 2]]

        This is used in smooth_by_sum_pool for 2D smoothing.
        """
        if sum_pool_by is None:
            return replace(self)
        else:
            return replace(
                self,
                arr=coo_matrix(reverse_sum_pool(self.dense_array, sum_pool_by, preserve_sum=preserve_sum)),
            )

    def smooth_by_sum_pool(self, sum_pool_by, preserve_sum: bool = False):
        return self.sum_pooled(sum_pool_by).reverse_sum_pooled(sum_pool_by, preserve_sum=preserve_sum)

    def to_kernel_smoothed(self, *args, **kwargs) -> "KernelSmoothedFragmentMatrix":
        """Return a KernelSmoothedFragmentMatrix constructed from self.

        *args and **kwargs are passed through to the KernelSmoothedFragmentMatrix constructor.
        """
        # if the weights aren't all integers, then it's not possible to make a ones_coo_arr.
        arr = make_ones_coo_arr(self.arr)
        assert (arr.data == 1).all()
        return KernelSmoothedFragmentMatrix(self.arr, *args, **kwargs)

    def make_tracks(
        self,
        sum_pool_by: int = DEFAULT_VPLOT_SUMPOOL_BY,
        title: str = None,
        figsize: Tuple[int] = None,
        fragment_matrix_density: Dict = dict(),
        endpoints: bool = False,
        coverage_window_len: int = 20,
        n_boot_coverage: int = 25,
        contour: bool = False,
        display_tfs: List[str] = [],
        gene_track: bool = False,
        epilogos_track: bool = False,
        midpoints: bool = True,
        coverage_fraction=None,
        binding_site_track_rdf: "RegionDataFrame" = None,
    ):
        from fbio.plot.tracks import (
            VplotTrack,
            VplotDensityTrack,
            CoverageTrack,
            GeneTrack,
            EpilogosTrack,
            TfConvTrack,
            Tracks,
            VLine,
            TfRegionTrack,
            MultiRegonTfRegionTrack,
        )

        if not isinstance(fragment_matrix_density, dict):
            fragment_matrix_density = {"density": fragment_matrix_density}

        tracks = []
        if gene_track:
            ann = ANNOTATIONS.get("gencode_basic", self.region.ref)
            tracks.append(GeneTrack(ann.local_path, self.region, name="Gencode", height=2,))

        if epilogos_track:
            ann = ANNOTATIONS.get("epilogos", self.region.ref)
            tracks.append(EpilogosTrack(ann.local_path, self.region, legend=True,))
        tracks.append(
            VplotTrack(
                self,
                self.plot_region,
                sum_pool_by=sum_pool_by,
                vlines=[VLine(x=self.plot_region.midpoint)],
                name=title,
            )
        )
        for key, density in fragment_matrix_density.items():
            tracks.append(
                VplotDensityTrack(
                    density,
                    self.plot_region,
                    sum_pool_by=sum_pool_by,
                    vlines=[VLine(x=self.plot_region.midpoint)],
                )
            )
        coverage_types = []
        if midpoints:
            coverage_types.append("midpoint")
        if endpoints:
            coverage_types.extend(("left_endpoint", "right_endpoint"))

        colors = cm.rainbow(np.linspace(0, 1, len(fragment_matrix_density)))
        coverage_track = CoverageTrack(
            self,
            self.plot_region,
            coverage_type=coverage_types,
            fraction=coverage_fraction,
            smooth_window=coverage_window_len,
            vlines=[VLine(x=self.plot_region.midpoint)],
            name="coverage" if coverage_fraction is None else f"{coverage_fraction} coverage",
        )
        scaling_term = coverage_track._build_coverage("midpoint").sum()
        for i, (key, density) in enumerate(fragment_matrix_density.items()):
            coverage_track += CoverageTrack(
                density,
                self.plot_region,
                coverage_type=coverage_types,
                fraction=coverage_fraction,
                smooth_window=coverage_window_len,
                name=key,
                scaling_factor=scaling_term,
                color=colors[i],
            )
        tracks.append(coverage_track)

        if binding_site_track_rdf is not None:
            if self.plot_region.chrom == "NA":
                tracks.append(MultiRegonTfRegionTrack(binding_site_track_rdf, self.plot_region, log=False))
            else:
                tracks.append(TfRegionTrack(binding_site_track_rdf, self.plot_region, noisy_tf_threshold=5,))
        for tf in display_tfs:
            tracks.append(TfConvTrack(self, self.plot_region, tf_conv_width=17, factors=[tf],))
        return Tracks(tracks)

    def plot(self, title=None, *args, **kwargs) -> Figure:
        """
        Plots tracks related to this Fragment Matrix

        :param BindingSiteData binding_site_data: if set, adds binding site data tracks
        :param cut_site_log_pval_cutoff: set the pvalue cutoff for determining whether a site is a cut site
        :return: a matplotlib figure
        """
        tracks = self.make_tracks(*args, **kwargs)
        tracks.plot(title=title)

    def to_density(self) -> numpy.ndarray:
        if self.n_fragments:
            return self.dense_array / self.n_fragments
        else:
            return self.dense_array


@dataclass
class RegionFragmentMatrix(FragmentMatrix):
    arr: coo_matrix
    region: Region

    @property
    def plot_region(self):
        """
        The Region to use for plotting.  It's overridden, since the FragmentMatrix plots over a pseudo-region (a
        region with an NA chrom)
        """
        return self.region

    @property
    def starts(self):
        return self.midpoints - self.lengths // 2 + self.region.start

    @property
    def stops(self):
        # add one to the stop if length is odd, since the actual midpoint was midpoint + .5
        return self.midpoints + self.lengths // 2 + (self.lengths % 2) + self.region.start

    @property
    def chrom(self):
        return self.region.chrom

    @property
    def strand(self):
        return self.region.strand

    @property
    def start(self):
        return self.region.start

    @property
    def stop(self):
        return self.region.stop

    @property
    def midpoint(self) -> int:
        return self.region.midpoint

    def __post_init__(self):
        super().__post_init__()
        assert self.region.length > 0, "Invalid region length"

    @property
    def fragment_matrix(self):
        return FragmentMatrix(self.arr)

    @classmethod
    def from_bam(
        cls,
        in_bam: str,
        region: Region,
        min_mapq: int = DEFAULT_MIN_MAPQ,
        max_frag_len: int = DEFAULT_MAX_FRAG_LEN,
        max_frags=None,
        dedup: bool = DEFAULT_DEDUP,
        isize_downsample_factor=None,
    ):
        if isize_downsample_factor is not None:
            raise NotImplementedError("isize_downsample_factor is not supported")

        with pysam.AlignmentFile(in_bam) as alignment_file:
            return cls.from_frags(
                it.islice(
                    bam_to_fragments(
                        alignment_file,
                        fasta_file=None,
                        max_tlen=max_frag_len,
                        min_mapq=min_mapq,
                        chrom=region.chrom,
                        start=region.start,
                        stop=region.stop,
                    ),
                    0,
                    max_frags,
                ),
                region=region,
                dedup=dedup,
                max_allowed_frag_len=max_frag_len,
            )

    @classmethod
    def from_fragment_iterator(
        cls,
        fragment_iterator,
        region: Region,
        min_mapq: int = DEFAULT_MIN_MAPQ,
        max_frag_len: int = DEFAULT_MAX_FRAG_LEN,
        max_frags=None,
        dedup: bool = DEFAULT_DEDUP,
    ) -> "RegionFragmentMatrix":
        def _iter_filtered_fragments(
            fragments: Iterable[Fragment], min_mapq: int = DEFAULT_MIN_MAPQ, max_frag_len: int = DEFAULT_MAX_FRAG_LEN,
        ):
            for frag in fragments:
                if not frag.mapq_gte(min_mapq):
                    continue
                if frag.length > max_frag_len:
                    continue

                yield frag

        fragment_iterator = _iter_filtered_fragments(fragment_iterator, min_mapq, max_frag_len)

        # intialize the class and return
        return cls.from_frags(
            fragments=fragment_iterator, region=region, dedup=dedup, max_allowed_frag_len=max_frag_len,
        )


    @classmethod
    def from_fragments_h5(
        cls,
        fragments_h5: Union[str, bytes, FragmentsH5],
        region: Region,
        min_mapq: int = DEFAULT_MIN_MAPQ,
        max_frag_len: int = DEFAULT_MAX_FRAG_LEN,
        strand: Optional[str] = None,
    ) -> "RegionFragmentMatrix":
        if strand in ("+", "-"):
            strand = strand.encode()
        assert strand in (None, b"+", b"-")
        if isinstance(fragments_h5, (str, bytes)):
            do_close = True
            fragments_h5 = FragmentsH5(fragments_h5, cache_pointers=False)
        else:
            do_close = False

        try:
            starts, stops, supp_data = fragments_h5.fetch_array(
                region.chrom,
                region.start,
                region.stop,
                max_frag_len=max_frag_len,
                return_mapqs=(min_mapq > 0),  # only get mapqs if we need them to filter
                return_strand=(strand is not None),
            )
            lengths = stops - starts
            midpoints = starts - region.start + lengths // 2
            mask = (midpoints >= 0) & (midpoints < region.length)
            if min_mapq > 0:
                mask &= supp_data["mapq"].min(axis=1) >= min_mapq

            if strand is not None:
                mask &= supp_data["strand"] == strand

            # filter fragments that don't meet the filter criteria
            midpoints = midpoints[mask]
            lengths = lengths[mask]

            arr = coo_matrix(
                (np.ones(len(midpoints), dtype=int), (lengths, midpoints)),
                shape=(max_frag_len + 1, region.length),
            )
            return cls(arr, region)
        finally:
            if do_close:
                fragments_h5.close()

    @classmethod
    def from_10x_frag_tsv(
        cls, in_frag_tsv: str, region: Region, max_frags=None, dedup: bool = DEFAULT_DEDUP,
    ) -> "FragmentMatrix":
        # iterate over fragments from that region
        fragment_iterator = iter_fragments_from_10x_frag_tsv(in_frag_tsv, region, max_frags)
        # intialize the class and return
        return cls.from_frags(fragment_iterator, region, dedup)

    @property
    def region_str(self) -> str:
        return f"{self.chrom}:{self.start}-{self.stop}"

    def plot(self, title=None, *args, **kwargs):
        if title is None:
            title = f"{self.region_str}({self.strand}) {int(self.arr.sum())} reads"
        return super().plot(*args, **kwargs, title=title)

    def get_slice(self, sl):
        # Returns a slice of an RFM. Takes dense array and slices as normal, changes coordinates of region
        region_length = self.region.stop - self.region.start
        if sl.start is None:
            sl = slice(0, sl.stop)
        if sl.stop is None:
            # Defaults to go to the end of the region
            sl = slice(sl.start, region_length)
        if sl == slice(0, region_length):
            return self

        assert sl.start >= 0, f"slice start is before the start of {self.region}"
        assert sl.stop <= region_length, f"slice stop is past stop of {self.region}"
        assert sl.step in (None, 1), f"only a step of None or 1 is allowed"

        arr = make_ones_coo_arr(self.arr.tocsc()[:, sl].tocoo())

        shifted_region = replace(
            self.region, start=self.region.start + sl.start, stop=self.region.start + sl.stop
        )
        return replace(self, arr=arr, region=shifted_region)

    def get_subregion_slice(self, subregion):
        # Returns the slice of an RFM that corresponds to the coordinates given by subregion
        assert isinstance(subregion, Region)
        if subregion not in self.region:
            raise ValueError(f"subregion {subregion} is not a subset of {self.region}")
        sl = slice(subregion.start - self.region.start, subregion.stop - self.region.start)
        return self.get_slice(sl)


class FragmentMatrixDensity:
    def __init__(self, density, fl=None, pos=None, sum_pool=None):
        # fl = fragment length
        # pos = genomic position
        # the density
        self.density = density
        assert self.density.min() >= 0, self.density.min()
        assert self.density.max() <= 1, self.density.max()
        assert np.isclose(self.density.sum(), 1.0), self.density.sum()

        self.density = self.density.astype("float128")
        self.density /= self.density.sum() + 1e-12
        self.density = self.density.astype("float64")
        self.density /= self.density.sum()

        self.sum_pool = sum_pool
        if self.sum_pool is not None:
            assert fl is None and pos is None
            fl = np.arange(0, density.shape[0] * self.sum_pool, self.sum_pool)
            pos = np.arange(0, density.shape[1] * self.sum_pool, self.sum_pool)

        # fragment lengths
        # if fl is None, assume that the grid density is 1
        if fl is None:
            fl = np.arange(0, density.shape[0], 1)
        else:
            if len(fl) != self.density.shape[0]:
                raise ValueError(
                    f"'fl' (len {len(fl)}) must be the same length as the first "
                    f"dimension of 'density' (shape {density.shape})"
                )

        self.fl = fl

        # genomic positions
        # if pos is None, assume that the grid density is 1
        if pos is None:
            pos = np.arange(0, density.shape[1], 1)
        else:
            if len(pos) != self.density.shape[1]:
                raise ValueError(
                    f"'pos' (len {len(pos)}) must be the same length as the second "
                    f"dimension of 'density' (shape {density.shape})"
                )
        self.pos = pos

    @property
    def arr(self):
        return self.density

    def subset_fragment_lengths(self, low: int, high: int):
        res = copy.deepcopy(self)
        res.density[:low, :] = 0
        res.density[high:, :] = 0
        res.density = res.density / (res.density.sum() + 1e-12)
        return res

    def todense(self):
        return self.density

    @property
    def dense_array(self):
        return self.density

    def get_midpoint_coverage_array(self) -> numpy.ndarray:
        return self.dense_array.sum(0).squeeze()

    @classmethod
    def from_dense_array(cls, arr, frag_step_size=1, pos_step_size=1):
        """Build a smoothed fragment matrix from a dense array.
        """
        return cls(dense_array_into_coo_arr(arr, frag_step_size=frag_step_size, pos_step_size=pos_step_size))

    def rescale(self, scaling_factor, in_place=False):
        density = self.density * scaling_factor[:, None]
        density /= density.sum()
        if in_place:
            self.density = density
        else:
            return FragmentMatrixDensity(density, fl=self.fl, pos=self.pos)

    def jitter(self, jitter_value, output_length):
        """
        Used for input images, jitters the image forward by jitter_value
        :param jitter_value: int value by which the matrix is shifted forward.
        :param output_length: int the length of the output
        :return:
        """

        assert isinstance(jitter_value, int), f"jitter_value must be an integer, received {jitter_value}."

        arr_j = jitter_matrix(self.density, jitter_value=jitter_value, output_length=output_length)

        return type(self)(arr_j)  # replace(self, arr=arr_j)

    @property
    def max_frag_len(self):
        return self.density.shape[0] - 1

    def sum_pooled(self, sum_pool_by):
        new_density = sum_pool(self.density, sum_pool_by)
        # renormalize to handle issues like values being > 1 due to precision error
        new_density /= new_density.sum()
        new_fl = self.fl[numpy.arange(self.fl.min(), self.fl.max() + 1, sum_pool_by)]
        new_pos = self.pos[numpy.arange(self.pos.min(), self.pos.max() + 1, sum_pool_by)]
        # this is potentially dropping the extra child attributes, but I'm not sure
        # how else to this unless I make a copy. The problem with a copy is that some
        # of the attributes may be wrong after the sumpool, so I'll leave as is
        # until it becomes an issue
        return FragmentMatrixDensity(new_density, fl=new_fl, pos=new_pos)

    def reverse_sum_pooled(self, sum_pool_by, preserve_sum: bool = True):
        """Upscale the array.

        e.g.
        if we have an array [[2]] and reverse_sum_pool with sum_pool_by=2 then we should return
        [[2, 2], [2, 2]]

        This is used in smooth_by_sum_pool for 2D smoothing.
        """
        if preserve_sum == False:
            warnings.warn("Ignoring preserve_sum=False in density, invalid for density.")
        new_density = reverse_sum_pool(self.density, sum_pool_by, preserve_sum=True)
        new_fl = np.linspace(self.fl.min(), self.fl.max() + 1, len(self.fl) * sum_pool_by)
        new_pos = np.linspace(self.pos.min(), self.pos.max() + 1, len(self.pos) * sum_pool_by)
        return FragmentMatrixDensity(new_density, fl=new_fl, pos=new_pos)

    def smooth_by_sum_pool(self, sum_pool_by, preserve_sum: bool = True):
        if preserve_sum == False:
            warnings.warn("Ignoring preserve_sum=False in density, invalid for density.")
        if sum_pool_by == 1:
            return self
        return self.sum_pooled(sum_pool_by).reverse_sum_pooled(sum_pool_by, preserve_sum=True)

    @staticmethod
    def _rescale_shape(shape, frag_step_size, pos_step_size):
        return (shape[0] * frag_step_size, shape[1] * pos_step_size)

    def sample_with_replacement(self, n_fragments):
        rng = np.random.default_rng()
        assert numpy.isclose(self.density.sum(), 1), f"1st Axiom! Sum is {self.density.sum()}"
        arr = rng.multinomial(n_fragments, self.density.ravel()).reshape(self.density.shape)
        return FragmentMatrix(arr)

    @classmethod
    def contour_plot(
        cls,
        ax,
        density_to_plot,
        poss,
        fls,
        percentile_levels=tuple(
            it.chain(np.arange(0, 95, 1), np.arange(95, 99, 0.2), np.arange(99, 100, 0.02))
        ),
        cmap="nipy_spectral_r",
    ):
        """Contour plotting code designed for plotting smoothed v-plots."""
        # find level values from the provided percentile levels.
        levels = np.unique(np.percentile(density_to_plot, percentile_levels))
        # levels = np.arange(0, 20, 0.5)
        # if there aren't enough levels, then let it decide. This is an edge case for when we are, for example,
        # plotting the difference between something and itself.

        if len(levels) < 2:
            levels = None
        res = ax.contourf(list(poss), list(fls), density_to_plot, levels=levels, cmap=cmap)
        ax.set_xticks(ticks=[0, poss[-1] // 2, poss[-1]])
        ax.set_xticklabels([-poss[-1] // 2, 0, poss[-1] // 2])
        return res

    def plot_diff(self, other, figsize=(20, 10), **kwargs):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()
        self.plot_diff_on_axis(ax, other, **kwargs)

    def plot_diff_on_axis(self, ax, other, **kwargs):
        """Plot the difference between the density of self and other on ax"""
        # if not isinstance(other, type(self)):
        #    raise TypeError(f"Expected 'other' to be instance of {type(self)} but found {type(other)}")
        pos_diff = (
            sorted(set(self.pos) - set(other.pos)),
            sorted(set(other.pos) - set(self.pos)),
        )
        if len(pos_diff[0]) > 0:
            raise ValueError(f"'self' ({self}) has additional positions: {pos_diff[0]}")
        if len(pos_diff[1]) > 0:
            raise ValueError(f"'other' ({other}) has additional positions: {pos_diff[1]}")

        fl_diff = (
            sorted(set(self.fl) - set(other.fl)),
            sorted(set(other.fl) - set(self.fl)),
        )
        if len(fl_diff[0]) > 0:
            raise ValueError(f"'self' ({self}) has additional frag lens: {fl_diff[0]}")
        if len(fl_diff[1]) > 0:
            raise ValueError(f"'other' ({other}) has additional frag lens: {fl_diff[1]}")
        return self.contour_plot(ax, self.density - other.density, self.pos, self.fl, **kwargs)

    def plot_vplot(
        self,
        density_to_plot,
        ax=None,
        cbar_ax=None,
        xlab_start: int = None,
        plot_kwargs: Union[Dict, None] = None,
        plot_midpoint: bool = True,
        cmap: Union[numpy.ndarray, None, str] = "nipy_spectral",
        add_cbar: bool = False,
        # other pretty good cmaps include RdYlBu_r, jet, jet_r, nipy_spectral
        # you can get the list of possibilities by putting in something invalid like "asdf" and reading the
        # exception
        vmax: Union[int, float] = None,
        position_mask: Union[numpy.ndarray, None] = None,
    ):
        """
        :param ax: main axis to plot the vplot on
        :param cbar_ax: axis to plot color bar on
        :param xlab_start: start coordinate for x label
        :param sum_pool_by: sum_pool_by
        :param plot_kwargs: extra kwargs for seaborn.heatmap()
        :param plot_midpoint: plot a line at the midpoint
        :param cmap: colormap to use for the heatmap
        :param position_mask: A mask designating which positions where used and which were removed.  Used for example
          in downsampling.
        :param vmax: the vmax parameter to seaborn.heatmap.  Sets the maximum value to use for the colorbar.
        :return:
        """
        if ax is None:
            assert cbar_ax is None, "cbar_ax should not be specified if ax is None"

            fig = plt.figure()
            gs = plt.GridSpec(1, 10)
            ax = fig.add_subplot(gs[0, :9])
            cbar_ax = fig.add_subplot(gs[0, 9:])
        else:
            fig = ax.figure

        if plot_kwargs is None:
            plot_kwargs = dict()

        if xlab_start is None:
            # default is to label at half the length
            xlab_start = -density_to_plot.shape[-1] // 2

        xlab_stop = xlab_start + density_to_plot.shape[-1]

        # reverse sum_pool_by so that coordinate system is still valid
        arr = density_to_plot

        df = pd.DataFrame(arr, columns=range(xlab_start, xlab_stop), index=range(0, self.max_frag_len + 1))

        sns.heatmap(df, ax=ax, vmax=vmax, cmap=cmap, cbar=add_cbar, **plot_kwargs)

        if cbar_ax is not None:
            colorbar = fig.colorbar(
                ax.get_children()[0], cax=cbar_ax, orientation="horizontal", fraction=0.5, pad=0.2,
            )
        else:
            colorbar = None

        # xticks
        mid_point_idx = len(df.columns) // 2 + 1
        ax.set_xticks(ticks=[0, mid_point_idx, len(df.columns)])
        ax.set_xticklabels([xlab_start, 0, xlab_stop])
        plt.xticks(rotation=0)
        if plot_midpoint:
            ax.axvline(mid_point_idx, color="maroon", linestyle="--", alpha=0.5)

        ax.invert_yaxis()
        ax.set_ylabel("fragment length")

        if position_mask is not None:
            # plot which positions were masked
            rectangles = []
            for idx in numpy.where(~position_mask)[0]:
                # add a rectangle of width 1 and height self.max_frag_len at position idx
                rectangles.append(patches.Rectangle((idx, 0), 1, self.max_frag_len, linewidth=1))

            collection = PatchCollection(rectangles, facecolor="grey")
            ax.add_collection(collection)

        # fig.tight_layout()

        return ax, colorbar

    def plot_on_axis(self, ax, num_fragments=None, contour: bool = True, vplot_cbar_ax=None, **kwargs):
        """Plot the smoothed fragment matrix onto an axis."""
        if num_fragments:
            density_to_plot = self.density * num_fragments
        else:
            density_to_plot = self.density
        if contour:
            return type(self).contour_plot(ax, density_to_plot, self.pos, self.fl, **kwargs)
        else:
            return self.plot_vplot(density_to_plot, ax=ax, cbar_ax=vplot_cbar_ax, **kwargs)[0]

    def plot(
        self,
        figsize=(20, 10),
        title="",
        cmap="nipy_spectral_r",
        use_expected_counts=False,
        contour: bool = False,
    ):
        """A high level plotting interface."""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, title=title)
        self.plot_on_axis(ax, cmap=cmap, num_fragments=None, contour=contour)
        return fig

    @staticmethod
    def interpolate_to_single_bp_resolution(density, fl, pos, max_fl, max_pos):
        with numpy.errstate(under="ignore"):
            # calculate the density at bp resolution by interpolating over the grid
            interpolator = RectBivariateSpline(fl, pos, density)
            interpolated_density = interpolator(np.arange(0, max_fl), np.arange(0, max_pos),)
            interpolated_density /= interpolated_density.sum()
            interpolated_density = interpolated_density.clip(1e-12, 1 - 1e-12)
            interpolated_density /= interpolated_density.sum()
            assert np.isclose(interpolated_density.sum(), 1)

        return interpolated_density

    def to_single_bp_resolution(self):
        with numpy.errstate(under="ignore"):
            fl_grid_size = self.fl[1] - self.fl[0]
            pos_grid_size = self.pos[1] - self.pos[0]

            # calculate the density at bp resolution by interpolating over the grid
            interpolator = RectBivariateSpline(
                self.fl + fl_grid_size // 2, self.pos + pos_grid_size // 2, self.density
            )

            fl_max = self.fl.max() + fl_grid_size
            pos_max = self.pos.max() + pos_grid_size

            interpolated_density = interpolator(np.arange(0, fl_max), np.arange(0, pos_max),)
            interpolated_density /= interpolated_density.sum()
            interpolated_density = interpolated_density.clip(1e-12, 1 - 1e-12)
            interpolated_density /= interpolated_density.sum()
            assert np.isclose(interpolated_density.sum(), 1)

        return type(self)(interpolated_density, sum_pool=1)

    @property
    def shape(self):
        return self.density.shape

    def truncate(self, new_size):
        if new_size > self.shape[1]:
            raise ValueError(f"Can not truncate to '{new_size}' because the array shape is '{self.shape[1]}'")
        size_diff = self.shape[1] - new_size
        assert size_diff // 2 + new_size <= self.shape[1]
        sl = slice(size_diff // 2, size_diff // 2 + new_size)
        new_density = self.density[:, sl]
        new_density = new_density / new_density.sum()
        return FragmentMatrixDensity(new_density)
        # return replace(self)(new_density)
        # return type(self)(new_density)

    def resize(self, new_size):
        """
        Resizes density to size `resize`
        """

        return self.truncate(new_size)


class KernelSmoothedFragmentMatrix(FragmentMatrixDensity):
    def __init__(self, arr, alpha=0.15, grid_step_size=DEFAULT_VPLOT_SUMPOOL_BY):
        assert isinstance(arr, coo_matrix)
        self.arr_original = make_ones_coo_arr(arr)
        assert self.arr_original.shape == arr.shape

        self.data = pd.DataFrame(dict(pos=self.arr_original.col, fl=self.arr_original.row))

        # put this in the init so that we can avoid the R startup overhead in cases
        # where this class isn't being used
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri

        pandas2ri.activate()
        ro.r("library('pdfCluster')")

        smoother = ro.r(
            """
        function(pos, fl, data, alpha) {
            pdf = kepdf(data, bwtype='adaptive', alpha=alpha, expand.grid(pos, fl))
            density = matrix(pdf@estimate/sum(pdf@estimate), nrow=length(pos), ncol=length(fl))
            t(density)
        }
        """
        )

        fl = np.arange(0, self.arr_original.shape[0] + 1, grid_step_size)
        pos = np.arange(0, self.arr_original.shape[1] + 1, grid_step_size)
        max_fl = self.arr_original.shape[0]
        max_pos = self.arr_original.shape[1]
        density = smoother(pos, fl, self.data, alpha=alpha)
        assert np.isclose(density.sum(), 1)

        interpolated_density = self.interpolate_to_single_bp_resolution(
            density, fl, pos, max_fl=max_fl, max_pos=max_pos
        )

        super().__init__(interpolated_density)


def merge_fragment_matrices(
    fragment_matrices: Iterable[Union[FragmentMatrix, RegionFragmentMatrix, FragmentMatrixDensity]],
    flip_minus_strand: bool = False,
) -> Union[FragmentMatrix, RegionFragmentMatrix, FragmentMatrixDensity]:
    # FIXME: flip_minus_strand is broken. Fragment matrices are now natively flipped for minus strand
    # raise NotImplementedError("Deprecated. Use FragmentArray and merge those instead")

    regions = set()
    arr = None
    if flip_minus_strand:
        assert all(
            [fm.strand is not None for fm in fragment_matrices]
        ), "If flipping minus strand, all fragment matrices must have a strand"
    for i, fm in enumerate(fragment_matrices):
        if isinstance(fm, RegionFragmentMatrix) or fm.__class__.__name__ == "RegionFragmentMatrix":
            regions.add(fm.region)
        else:
            if not (
                isinstance(fm, FragmentMatrix)
                or isinstance(fm, FragmentMatrixDensity)
                or isinstance(fm, RegionFragmentMatrix)
                or fm.__class__.__name__
                in {"FragmentMatrix", "FragmentMatrixDensity", "RegionFragmentMatrix",}
            ):
                raise TypeError(f"{fm}, name={fm.__class__.__name__} is not a recognized FragmentMatrix")
            regions.add(None)

        if flip_minus_strand and fm.strand == "-":
            _arr = fm.reverse_strand().arr
        else:
            _arr = fm.arr

        if i == 0:
            arr = _arr
        else:
            arr += _arr

    if arr is None:
        raise StopIteration(f"fragment_matrices was an empty iterable")

    if any([isinstance(fm, FragmentMatrixDensity) for fm in fragment_matrices]):
        assert all([isinstance(fm, FragmentMatrixDensity) for fm in fragment_matrices])
        return FragmentMatrixDensity(arr / arr.sum())

    if None in regions or len(regions) != 1:
        return FragmentMatrix(arr)
    elif len(regions) == 1:
        return RegionFragmentMatrix(arr, regions.pop())
    else:
        raise ValueError(f"Unsure how to process {regions}, {arr}")


class TooFewReadsToDownsample(Exception):
    pass
