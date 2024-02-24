import abc
import itertools
import json
import os
import re
import warnings
from collections import UserList, defaultdict
from dataclasses import dataclass, field, replace
from typing import Union, List, Tuple, Optional

import matplotlib.patheffects as path_effects
import numpy
import numpy as np
import seaborn
import copy

import pysam

from fbio.formats import (
    BigWigReader,
    NarrowPeakBigBedReader,
    BedIntervalTreeReader,
    TabixBedReader,
    GenePredReader,
)
from fragmentomics_tools.region import intervals_intersect

from fbio.util import aws_utils
from fbio.util.misc_utils import igv_link, obj_type_name_matches_class_name_str
from fbio.viz_sequence import plot_weights_given_ax

from matplotlib import pyplot, colors
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Polygon
from matplotlib.pyplot import axis
from matplotlib.ticker import FuncFormatter

from ravel.bio.frag.fragment_matrix_math import sum_pool, reverse_sum_pool
from scipy.ndimage import gaussian_filter

from ravel.learn.functional_genomics.model import TFConv1D
from ravel.public_data_resources.public_chromhmm import CHROMHMM_COLORS
from ravel.fragment_array import FragmentArray
from ravel.util.numpy_utils import smooth1d


@dataclass
class VLine:
    """Vertical Line plot object (at an x coordinate)"""

    x: float
    color: str = "black"
    alpha: float = 1
    linestyle: str = "-"
    linewidth: float = 1


@dataclass
class Line:
    """Line plot object (y values over a region)"""

    y: numpy.ndarray
    x: Optional[numpy.ndarray] = None  # defaults to the range of the region
    color: str = None
    alpha: float = 0.5
    linestyle: str = "-"
    label: str = None


@dataclass
class GenomeTrack:
    """
    Base class for a Genome Track
    """

    # NOTE: A class attribute must have a type annotation for the dataclass to add it to the Track.__init__ method.
    # We must have a default of None here for overlaid tracks
    input: object = None
    region: Region = None

    name: str = None
    color: str = None

    height: int = 2
    vlines: List[VLine] = field(default_factory=list)
    lines: List[Line] = field(default_factory=list)

    ylim: Tuple[Union[int, float], Union[int, float]] = None
    legend: bool = False
    title_color: str = None
    title_fontweight: str = "bold"

    def __add__(self, other):
        # This can be used with track1 + track2 + ...
        if hasattr(self, "region") and hasattr(other, "region"):
            assert self.region == other.region
        return OverlaidTracks([self, other])

    def __radd__(self, other):
        # This can be used with sum([track1, track2, ...])
        if other == 0:
            return self
        else:
            return self.__add__(other)

    @property
    def title(self):
        return self.name

    @property
    def label(self):
        warnings.warn("deprecated, use .name")
        return self.name

    def _plot_extras(self, ax):
        for vline in self.vlines:
            ax.axvline(
                vline.x, color=vline.color, alpha=vline.alpha, linestyle=vline.linestyle
            )

    def replace(self, **kwargs):
        return replace(self, **kwargs)

    @property
    def chrom(self):
        return self.region.chrom

    @property
    def start(self):
        return self.region.start

    @property
    def stop(self):
        return self.region.stop

    _data = None  # this is a cache for the data property

    @property
    def data(self):
        """
        The cached result of self._materialize_data()
        """
        if self._data is None:
            data = self._materialize_data()
            self._data = data

        if self._data is None:
            raise ValueError(f"{self}.materialize() failed to materialize {self}._data")

        return self._data

    def __post_init__(self):
        if isinstance(self.input, str) and not aws_utils.path_exists_s3_or_local(
            self.input
        ):
            raise ValueError(f"{self.input} does not exist")
        if self.region is None:
            self.region = self.input.plot_region

    @abc.abstractmethod
    def _plot(self, ax):
        pass

    def _plot_right(self, ax_main, ax_right):
        ax_right.remove()

    def _plot_left(self, ax_main, ax_right):
        ax_right.remove()

    def _materialize_data(self):
        return self.input

    def plot(self, ax_main=None, plot_region=None, ax_right=None):
        """
        Create a plot of all tracks

        :param figsize: figure size
        :param ax_main: axis to plot on
        :param plot_region: the region to plot accross all tracks (the default is to use each Track.region).
        :return: matplotlib.axis
        """
        if ax_main is None:
            return Tracks([self]).plot()

        if plot_region is None:
            plot_region = self.region

        self._plot(ax_main)
        self._plot_extras(ax_main)

        ax_main.set_xlim(plot_region.start, plot_region.stop)

        ax_main.xaxis.set_major_formatter(FuncFormatter(lambda x, y: f"{int(x):,}"))
        ax_main.set_xticks(numpy.linspace(plot_region.start, plot_region.stop, 4))

        if self.ylim is not None:
            ax_main.set_ylim(self.ylim)

        # plot vertical line elements
        for vline in self.vlines:
            ax_main.axvline(
                vline.x, color=vline.color, alpha=vline.alpha, linestyle=vline.linestyle
            )

        # plot Line elements
        for line in self.lines:
            if line.x is not None:
                x = line.x
            else:
                x = numpy.arange(self.region.start, self.region.stop)
            ax_main.plot(
                x,
                line.y,
                linestyle=line.linestyle,
                color=line.color,
                alpha=line.alpha,
                label=line.label,
            )

        if self.legend:
            ax_main.legend(
                bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0
            )

        # Making this dictionary because we can't pass None values for set_title parameters
        title_params = dict()
        if self.title_color:
            title_params["color"] = self.title_color
        if self.title_fontweight:
            title_params["fontweight"] = self.title_fontweight
        ax_main.set_title(self.title, **title_params)

        if ax_right is not None:
            self._plot_right(ax_main, ax_right)

        return ax_main


class Tracks(UserList):
    def get_igv_link(self, local_mount, region, ipython_link=False):
        """
        Load all tracks into IGV.  Prepends 'local_mout' to track.data
        """
        return igv_link(
            [
                f"{local_mount}/{os.path.abspath(track.data)}".replace("//", "/")
                for track in self
            ],
            region=region,
            names=[track.name for track in self],
            ipython_link=ipython_link,
        )

    def plot(
        self,
        plot_region=None,
        width=25,
        height_multiplier=1,
        vlines: List[VLine] = None,
        out_fname=None,
        title=None,
        plot_region_coords=(-1,),
        sharey=False,
    ):
        """

        :param plot_region: Override the regions being plotted
        :param height_multiplier: Controls figure height.  Multiplier to use for individual and thus total track height.
        :param width: figure width
        :param plot_region_coords: List or tuple of indexes where to display region coordinates and chromosome.
         For example, [0] means to display on the top plot only, [0, -1] for top and bottom plots
        :return: figure, axes
        """
        if vlines is None:
            vlines = []

        height_sum = sum(t.height for t in self)
        fig = pyplot.figure(figsize=(width, height_multiplier * height_sum))
        if title is not None:
            pyplot.suptitle(title, fontsize=16, x=0.50, y=1.0)

        gs = pyplot.GridSpec(height_sum, 20, figure=fig, hspace=0.8, wspace=0.75)

        row = 0
        axes = []
        axes_right = []

        all_regions_same = (
            len(
                set(
                    [
                        (track.region.chrom, track.region.start, track.region.stop)
                        for track in self
                    ]
                )
            )
            == 1
        )

        if not isinstance(plot_region_coords, (list, tuple)):
            plot_region_coords = (plot_region_coords,)

        prev_ax_main = None
        for i, track in enumerate(self):
            ax_main = fig.add_subplot(
                gs[row : row + track.height, 1:18],
                sharey=(prev_ax_main if sharey else None),
            )
            ax_left = fig.add_subplot(gs[row : row + track.height, :1])
            ax_right = fig.add_subplot(
                gs[row : row + track.height, 18:], sharey=ax_main
            )

            axes.append(ax_main)
            axes_right.append(ax_right)

            track.plot(ax_main=ax_main, plot_region=plot_region)
            # ax.set_title(track.data)

            row += track.height

            for vline in vlines:
                ax_main.axvline(
                    vline.x,
                    color=vline.color,
                    alpha=vline.alpha,
                    linestyle=vline.linestyle,
                    linewidth=vline.linewidth,
                )

            if all_regions_same:
                # if plot_region_coords = [-1], then we want to plot when i == len(self)-1, hence 2nd statement
                if (i in plot_region_coords) or (i - len(self) in plot_region_coords):
                    ax_main.set_xlabel(track.chrom)
                else:
                    ax_main.set_xticks([])

            track._plot_right(ax_main, ax_right)
            track._plot_left(ax_main, ax_left)
            ax_left.yaxis.set_ticks_position("left")

            prev_ax_main = ax_main

        if out_fname:
            fig.savefig(out_fname)

        return fig, axes


@dataclass
class EmptyTrack(GenomeTrack):
    def _plot(self, ax):
        pass


class OverlaidTracks(GenomeTrack):
    def __init__(self, tracks_to_overlay):
        self.tracks_to_overlay = tracks_to_overlay
        self.region = tracks_to_overlay[0].region
        assert numpy.all(
            [
                not hasattr(tracks_to_overlay, "region")
                or track_to_overlay.region == self.region
                for track_to_overlay in tracks_to_overlay
            ]
        ), "All tracks must have the same region"
        self.vlines = list(itertools.chain(*[tr.vlines for tr in tracks_to_overlay]))
        self.lines = list(itertools.chain(*[tr.lines for tr in tracks_to_overlay]))
        names = [
            tr.name
            for tr in tracks_to_overlay
            if hasattr(tr, "name") and tr.name is not None
        ]
        self.name = " ".join(names) if names else ""

    def _plot(self, ax):
        ylims = []
        for track in self.tracks_to_overlay:
            track.plot(ax)
            ylims.append(track.ylim)

        # if none of the ylims are set, then autoscale
        if all(x is None for x in ylims):
            ax.autoscale(enable=True, axis="y", tight=None)
        # if one of the tracks autoscale is set to true, then still
        # autoscale but print a warning
        elif any(x is None for x in ylims):
            print(
                "Warning: some of the overlaid tracks have specified ylims but others have not. Defaulting to autoscale"
            )
            ax.autoscale(enable=True, axis="y", tight=None)
        # if none of the ylimits are set to None, then split into lower and upper bounds
        else:
            # if all of the ylims are set and the same, then set that. We first enable autoscale in case
            # only one of the limits is set
            ylim_lowers = [x[0] for x in ylims]
            if any(x is None for x in ylim_lowers):
                ylim_lower = None
            else:
                ylim_lower = min(ylim_lowers)

            ylim_uppers = [x[1] for x in ylims]
            if any(x is None for x in ylim_uppers):
                ylim_upper = None
            else:
                ylim_upper = max(ylim_uppers)

            # enable auto-scale to account for any None values
            ax.autoscale(enable=True, axis="y", tight=None)
            ax.set_ylim((ylim_lower, ylim_upper))


@dataclass
class VectorTrack(GenomeTrack):
    # plots a 1d numpy vector
    alpha: float = 1.0
    linewidth: Optional[float] = None
    color: str = "xkcd:blue"
    linestyle: str = "-"
    ymin: Optional[float] = None
    ymax: Optional[float] = None
    scale: float = 1.0
    smooth: bool = False
    smooth_num_std: int = 2
    smooth_window_len: int = 15

    def _plot(self, ax):
        if self.smooth:
            data = smooth1d(
                self.data,
                window_len=self.smooth_window_len,
                filter_shape="gaussian",
                num_std=self.smooth_num_std,
            )
        else:
            data = self.data
        ax.plot(
            numpy.arange(self.region.start, self.region.stop),
            data * self.scale,
            linewidth=self.linewidth,
            alpha=self.alpha,
            color=self.color,
            linestyle=self.linestyle,
        )
        ax.set_ylim(self.ymin, self.ymax)


@dataclass
class VectorTracks(GenomeTrack):
    # plots multiple 1d numpy vectors or a 2d numpy array as overlaid plots
    # Optionally takes in alpha, color, or a list of both for specific tracks
    alpha: float = 1.0
    alphas: list = None
    color: str = "xkcd:blue"
    colors: list = None
    labels: Optional[List[str]] = None
    legend_loc: str = "upper right"
    ymin: Union[float, str] = "infer"
    ymax: Union[float, str] = "infer"
    fill: bool = False
    fills: Optional[List[bool]] = None
    order: Optional[List] = None

    def _plot(self, ax):
        num_tracks = len(self.data)
        if isinstance(self.alpha, float) and self.alphas is None:
            self.alphas = [self.alpha] * num_tracks
        if isinstance(self.color, str) and self.colors is None:
            self.colors = [self.color] * num_tracks
        if self.ymin == "infer" and self.ymax == "infer":
            ymax = np.array(self.data).max()
            ymin = np.array(self.data).min()
            self.ymin = ymin - ((ymax - ymin) * 0.02)
            self.ymax = ymax + ((ymax - ymin) * 0.02)
        elif self.ymin == "infer":
            ymin = np.array(self.data).min()
            self.ymin = ymin - ((self.ymax - ymin) * 0.02)
        elif self.ymax == "infer":
            ymax = np.array(self.data).max()
            self.ymax = ymax + ((ymax - self.ymin) * 0.02)
        if self.ymin == self.ymax:
            self.ymin -= 0.1
            self.ymax += 0.1
        if self.labels is None:
            self.labels = [None] * num_tracks
        if self.fills is None and self.fill is not None:
            self.fills = [self.fill] * num_tracks
        if self.order is None:
            self.order = list(range(len(self.data)))

        for ii in self.order:
            plot_func = ax.fill_between if self.fills[ii] else ax.plot
            plot_func(
                numpy.arange(self.region.start, self.region.stop),
                self.data[ii],
                alpha=self.alphas[ii],
                color=self.colors[ii],
                label=self.labels[ii],
            )

        ax.set_ylim(self.ymin, self.ymax)
        if any([label is not None for label in self.labels]):
            ax.legend(loc=self.legend_loc)


def _make_cutsites(numpy_array):
    # This takes a numpy array corresponding to a V-Plot Density
    # and finds where the cut sites are by looking at the total signal on diagonals
    _, region_length = numpy.shape(numpy_array)
    marginal_density = numpy.zeros((2, region_length))
    for rr, row in enumerate(numpy_array):
        offset = rr // 2
        marginal_density[0, : region_length - offset] += row[offset:]
        marginal_density[1, offset:] += row[: region_length - offset]
    return marginal_density / numpy.sum(marginal_density, axis=1)[:, numpy.newaxis]


@dataclass
class CutsiteTrack(GenomeTrack):
    # plots cutsites for a single fragment matrix or numpy array
    # Optionally takes in alpha, color, or a list of both for specific tracks
    alpha: float = 1.0
    colors: list = None
    plot_left: bool = True
    plot_right: bool = True

    def _materialize_data(self):
        return _make_cutsites(self.input)

    def _plot(self, ax):
        if self.colors is None:
            self.colors = ["xkcd:blue", "xkcd:red"]
        else:
            assert len(self.colors) == 2, "Must provide a list of two colors"

        assert self.data.ndim == 2
        for idx, do_plot in enumerate([self.plot_left, self.plot_right]):
            if do_plot:
                ax.plot(
                    numpy.arange(self.region.start, self.region.stop),
                    self.data[idx],
                    alpha=self.alpha,
                    color=self.colors[idx],
                )


@dataclass
class ChipTrack(GenomeTrack):
    """
    This is different from other tracks in that it draws lines which can be overlaid
    WiggleTrack fills on the bottom. Also, can normalize this and smooth.

    Input is anything that VplotTrack.materialize_numpy_array can take. It will sum along the FL axis
    smooth_by: uses smooth1d to smooth with this window size
    normalize: normalizes by total depth in region
    scale: for scaling al values by a constant. Useful for comparing tracks of different sequencing depths
    """

    alpha: float = 1.0
    color: str = None
    linestyle: str = "-"
    smooth_by: int = None
    filter_shape: str = "uniform"
    num_std: float = 3.0
    mode: str = "mirror"
    normalize: bool = False
    scale: float = 1.0

    def _materialize_data(self):
        if isinstance(self.input, str):
            chrom, start, stop = self.region.chrom, self.region.start, self.region.stop
            values = WiggleTrack.materialize_numpy_track(
                self.input, chrom, start, stop, smooth=False
            )
        else:
            values = numpy.sum(
                VplotTrack.materialize_numpy_array(self.input, self.region), axis=0
            )
        if self.smooth_by is not None:
            values = smooth1d(
                values,
                self.smooth_by,
                filter_shape=self.filter_shape,
                num_std=self.num_std,
                mode=self.mode,
            )
        if self.normalize and numpy.sum(values) != 0:
            values = values / numpy.sum(values)
        values = values * self.scale
        return values

    def _plot(self, ax):
        VectorTrack(
            self.data,
            self.region,
            alpha=self.alpha,
            color=self.color,
            linestyle=self.linestyle,
        ).plot(ax)


@dataclass
class IntervalTrack(GenomeTrack):
    """
    Plots intervals in a given color. Can add more tracks using +
    Input is a 1d numpy array with values indicating positions marked
    A peak from position 105 to 109 should be [105, 106, 107, 108]
    """

    color: str = "red"
    alpha: float = 1.0
    genomic_coords: bool = True
    height: int = 1

    def _materialize_data(self):
        offset = 0 if self.genomic_coords else self.region.start
        return self.input + offset

    def _plot(self, ax):
        ax.vlines(self.data, ymin=0, ymax=1, color=self.color, alpha=self.alpha)
        # ax.set_ylim([-self.height, 0])
        ax.axis("off")


@dataclass
class CpGTrack(GenomeTrack):
    """
    Plots the positions of GpGs in a region. Input is a fasta filepath
    """

    height: int = 1

    @property
    def title(self):
        return "CpG positions" if self.name is None else self.name

    def _materialize_data(self):
        with pysam.FastaFile(self.input) as ff:
            seq = ff.fetch(self.region.chrom, self.region.start, self.region.stop)

        local_start_positions = numpy.array([m.start() for m in re.finditer("CG", seq)])

        return local_start_positions + self.region.start

    def _plot(self, ax):
        ax.vlines(self.data, ymin=0, ymax=1, color="g")


@dataclass
class VplotDiffTrack(GenomeTrack):
    """
    Plots a difference of two Vplots.
    """

    cmap: str = "bwr"
    vmax: float = None
    sum_pool_by: int = 16
    center: bool = True

    def _plot(self, ax):
        arr = self.data

        if self.sum_pool_by:
            arr = reverse_sum_pool(sum_pool(arr, self.sum_pool_by), self.sum_pool_by)

        if self.center and self.vmax is None:
            self.vmax = numpy.max([numpy.max(arr), -numpy.min(arr)])

        ax.imshow(
            arr[::-1, :],
            extent=[self.region.start, self.region.stop, 0, self.data.shape[0]],
            cmap=self.cmap,
            vmin=(None if self.vmax is None else -self.vmax),
            vmax=self.vmax,
        )
        ax.set_aspect("auto")

    def _plot_right(self, ax_main, ax_right):
        arr = self.data.sum(1)

        ax_right.plot(arr, range(len(arr)))

    def _plot_left(self, ax_main, ax_right):
        fig = ax_main.get_figure()
        axes_data = ax_main.get_children()[0]
        fig.colorbar(axes_data, cax=ax_right)


@dataclass
class VplotDensityDiffTrack(GenomeTrack):
    cmap: str = "bwr"
    vmin: float = None
    vmax: float = None
    height: int = 4
    sum_pool_by: int = 1
    equal_vmin_vmax: bool = True

    def _plot(self, ax):
        fm = self.data
        plot_data = sum_pool(fm, self.sum_pool_by)
        if self.equal_vmin_vmax:
            abs_max = max(abs(numpy.min(plot_data)), abs(numpy.min(plot_data)))
            self.vmin, self.vmax = (-abs_max, abs_max)
        norm = colors.TwoSlopeNorm(vcenter=0, vmin=self.vmin, vmax=self.vmax)

        ax.imshow(
            plot_data,
            extent=[self.region.start, self.region.stop, 0, fm.shape[0]],
            cmap=self.cmap,
            origin="lower",
            norm=norm,
        )
        ax.set_aspect("auto")

    def _plot_right(self, ax_main, ax_right):
        arr = self.data.sum(1)

        ax_right.plot(arr, range(len(arr)))

    def _plot_left(self, ax_main, ax_right):
        fig = ax_main.get_figure()
        axes_data = ax_main.get_children()[0]
        fig.colorbar(axes_data, cax=ax_right)


@dataclass
class VplotTrack(GenomeTrack):
    """
    :param input: a str or a list of strs of fnames to stack, or a 2d numpy array, or a FragmentMatrix
    """

    # smooth = None
    height: int = 4
    sum_pool_by: int = 16
    cmap: str = "nipy_spectral_r"
    gaussian_filter: float = None  # if set, gaussian smooth with this sigma
    vmax: int = None

    add_n_frags_to_title: bool = True

    @property
    def title(self):
        if self.name is None:
            title = ""
        else:
            title = self.name

        if self.add_n_frags_to_title:
            title += f" {int(round(self.data.sum()))} frags"

        return title

    def _materialize_data(self):
        return self.materialize_numpy_array(self.input, self.region)

    def _plot_right(self, ax_main, ax_right):
        arr = self.data.sum(1)
        ax_right.plot(arr, range(len(arr)))

    def _plot_left(self, ax_main, ax_right):
        fig = ax_main.get_figure()
        axes_data = ax_main.get_children()[0]
        fig.colorbar(axes_data, cax=ax_right)

    def _plot(self, ax):
        arr = self.data  #: Dense array

        if self.sum_pool_by:
            arr = reverse_sum_pool(sum_pool(arr, self.sum_pool_by), self.sum_pool_by)

        arr = arr[::-1]

        if self.gaussian_filter:
            arr = gaussian_filter(arr.astype(float), self.gaussian_filter)

        ax.imshow(
            arr,
            extent=[self.region.start, self.region.stop, 0, arr.shape[0]],
            cmap=self.cmap,
            vmax=self.vmax,
        )

        # imshow sets aspect to be square, reset to auto
        ax.set_aspect("auto")
        # ax.invert_yaxis()

    @staticmethod
    def materialize_numpy_array(input, region):
        """
        Returns a numpy array of a FM from a variety input types (strings, fragment matrix instances, arrays, ...)

        :param input: a str or a list of strs of fnames to stack, or a 2d numpy array, or a FragmentMatrix
        :return: a numpy array
        """
        from ravel.bio.frag.fragment_matrix import (
            RegionFragmentMatrix,
            FragmentMatrix,
            merge_fragment_matrices,
        )
        from ravel.fragment_array import RegionFragmentArray

        if isinstance(input, (FragmentMatrix)) or input.__class__.__name__ in [
            "FragmentMatrix",
            "RegionFragmentMatrix",
        ]:
            return input.dense_array.astype(float)
        if isinstance(input, (FragmentArray)) or input.__class__.__name__ in [
            "FragmentArray",
            "RegionFragmentArray",
        ]:
            return input.fragment_matrix.dense_array.astype(float)
        elif obj_type_name_matches_class_name_str(input, RegionFragmentArray):
            return input.fragment_matrix.dense_array.astype(float)
        elif isinstance(input, numpy.ndarray):
            # this is a 2d numpy array, just return it
            assert input.ndim == 2, "expected 2 dimensions"
            return input.astype(float)
        else:
            raise ValueError(f"cannot handle {input}")


@dataclass
class MotifTrack(GenomeTrack):
    """
    input is a pwm numpy array
    """

    rel_width: float = 0.25

    def _plot(self, ax):
        assert 0 < self.rel_width <= 1
        logo_width = self.region.length * self.rel_width
        motif_start_pos = (
            self.region.start + (0.5 * self.region.length) - (0.5 * logo_width)
        )
        motif_end_pos = motif_start_pos + logo_width
        plot_weights_given_ax(
            ax,
            self.data,
            subticks_frequency=None,
            x_start=motif_start_pos,
            x_end=motif_end_pos,
            min_height_to_plot=0.001,
            length_padding=0,
            align_to="top",
        )
        # Draw diagonal lines to position in genome
        motif_len = len(self.data)
        seq_start_pos = self.region.start + (self.region.length / 2) - (motif_len / 2)
        seq_end_pos = self.region.start + (self.region.length / 2) + (motif_len / 2)
        ax_ymin = ax.get_ylim()[0]

        motif_max_y = max(0, numpy.max(self.data.sum(axis=1)))
        motif_min_y = min(0, numpy.min(self.data.sum(axis=1)))

        # Trapezoid mapping motif to sequence
        trap = Polygon(
            (
                [seq_start_pos, ax_ymin],
                [motif_start_pos, motif_min_y],
                [motif_end_pos, motif_min_y],
                [seq_end_pos, ax_ymin],
            ),
            facecolor="xkcd:grey",
            edgecolor="black",
            alpha=0.5,
        )
        ax.add_patch(trap)

        # Bounding box for motif
        ax.add_patch(
            Rectangle(
                (motif_start_pos, motif_min_y),
                motif_end_pos - motif_start_pos,
                motif_max_y - motif_min_y,
                facecolor="none",
                edgecolor="k",
                linewidth=2,
            )
        )


@dataclass
class CoverageTrack(GenomeTrack):
    smooth_window: int = 10
    n_boot_coverage: int = 25
    coverage_type: tuple = "midpoint"
    coverage_type_to_color: tuple = (
        ("midpoint", "black"),
        ("summed_endpoints", "black"),
        ("left_endpoint", "red"),
        ("right_endpoint", "blue"),
        ("fragment", "black"),
    )
    bootstrap_alpha: float = 0.1
    color: str = None
    fraction: str = None
    # scale the coverage to allow for putting densities on the
    # same scale as counts
    scaling_factor: int = 1
    linestyle: str = "solid"
    legend: bool = True
    center: bool = False

    def _materialize_data(self):
        return self.input

    def _smooth(self, cov):
        if self.smooth_window:
            return smooth1d(cov, self.smooth_window)
        else:
            return cov

    def _build_coverage(self, coverage_type):
        data = copy.deepcopy(self.data)

        if self.fraction is None:
            pass
        elif isinstance(self.fraction, str):
            if self.fraction in ("small", "sub_nucleosome"):
                data = data.subset_fragment_lengths(0, 125)
            elif self.fraction == "single_nucleosome":
                data = data.subset_fragment_lengths(125, 250)
            elif self.fraction == "dual_nucleosome":
                data = data.subset_fragment_lengths(250, 375)
            elif self.fraction == "triple_nucleosome":
                data = data.subset_fragment_lengths(375, 500)
            else:
                raise ValueError(f"Unrecognized value for fraction: '{self.fraction}'")
        elif isinstance(self.fraction, tuple):
            data = data.subset_fragment_lengths(self.fraction[0], self.fraction[1])
        else:
            raise ValueError(f"Unrecognized value for fraction: '{self.fraction}'")

        if coverage_type == "midpoint":
            return data.get_midpoint_coverage_array()
        elif coverage_type == "fragment":
            return data.get_fragment_coverage_array()
        elif coverage_type == "summed_endpoints":
            return data.first_covered_base_counts + data.last_covered_base_counts
        elif coverage_type == "left_endpoint":
            return data.first_covered_base_counts
        elif coverage_type == "right_endpoint":
            return data.last_covered_base_counts
        else:
            raise ValueError(f"Unrecognized coverage type: '{self.coverage_type}'")

    def _plot(self, ax):
        if isinstance(self.coverage_type, str):
            coverage_types = [self.coverage_type]
        else:
            coverage_types = self.coverage_type

        for coverage_type in coverage_types:
            if self.color is None:
                color = dict(self.coverage_type_to_color)[coverage_type]
            else:
                color = self.color

            cov = (
                self._build_coverage(coverage_type=coverage_type) * self.scaling_factor
            )
            n = cov.sum()
            if self.name is None:
                label = f"\nsm_win {self.smooth_window}"
            else:
                label = self.name
            # label = f"{self.name + ' ' if self.name is not None else ''}{coverage_type}"
            # if self.smooth_window is not None:
            #    label += f"\nsm_win {self.smooth_window}"
            cov = self._smooth(cov)
            ax.plot(
                numpy.arange(self.region.start, self.region.stop),
                # cov - (cov.max() - cov.min())//2 if self.center else cov,
                cov - (cov[0] + cov[-1]) / 2 if self.center else cov,
                color=color,
                label=label,
                linestyle=self.linestyle,
            )
            if (
                self.n_boot_coverage is not None
                and self.n_boot_coverage > 0
                and numpy.array(cov, dtype=float).sum() > 0
            ):
                ps = numpy.array(cov, dtype=float)
                ps = ps / ps.sum()
                rng = numpy.random.default_rng()
                for i in range(self.n_boot_coverage):
                    cov = self._smooth(rng.multinomial(n, ps))
                    ax.plot(
                        numpy.arange(self.region.start, self.region.stop),
                        cov,
                        alpha=self.bootstrap_alpha,
                        color=color,
                    )
        if self.legend:
            ax.legend(loc="right", bbox_to_anchor=(1.12, 0.5), borderaxespad=0.0)


@dataclass
class TfRegionTrack(GenomeTrack):
    noisy_tf_threshold: int = 8

    def validate_binding_df_scores(self):
        assert self.input is not None
        binding_site_df = self.input
        region_contigs = sorted(set(binding_site_df["contig"]))
        region_starts = sorted(set(binding_site_df["original_start"]))
        region_stops = sorted(set(binding_site_df["original_stop"]))
        assert len(region_starts) == 1
        assert len(region_stops) == 1
        assert len(region_contigs) == 1
        r_contig = region_contigs[0]
        r_start = region_starts[0]
        r_stop = region_stops[0]
        assert self.region.chrom == r_contig, (self.region, r_contig, r_start, r_stop)
        assert self.region.start == r_start, (self.region, r_contig, r_start, r_stop)
        assert self.region.stop == r_stop, (self.region, r_contig, r_start, r_stop)
        tf_counts = binding_site_df.tf_name.value_counts()
        noisy_tfs = [t for t, c in tf_counts.items() if c >= self.noisy_tf_threshold]
        return binding_site_df.query("tf_name not in @noisy_tfs")

    def _plot(self, ax):
        binding_sitedf = self.validate_binding_df_scores()
        factors = sorted(set(binding_sitedf.tf_name))
        self.ylim = (-(len(factors) + 1), len(factors) + 1)
        ax.set_ylim(-(len(factors) + 1), len(factors) + 1)
        self.name = f"Factor sites for {', '.join(factors)}"
        import matplotlib.pyplot as plt

        ax.axhline(0, linestyle="--", color="black", label="+/- strand split")
        for y, tf in enumerate(factors):
            tfdf = binding_sitedf.query("tf_name==@tf")
            for strand, mult in zip(["+", "-"], [1, -1]):
                tfsdf = tfdf.query("strand==@strand")
                if len(tfsdf) == 0:
                    continue
                for start, stop in zip(tfsdf.start.values, tfsdf.stop.values):
                    ax.plot(
                        (start, stop),
                        ((y + 1) * mult, (y + 1) * mult),
                        label=f"{tf}",
                        c=plt.cm.tab10(y),
                    )
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="right",
            bbox_to_anchor=(1.12, 0.5),
            borderaxespad=0.0,
        )


@dataclass
class MultiRegonTfRegionTrack(GenomeTrack):
    top_k_to_plot: int = 10
    flip_strand_based_on_ori: bool = True
    log: bool = True

    def get_tf_cov_arrays(self):
        assert self.input is not None
        binding_site_df = self.input
        binding_site_df["start_offset"] = (
            binding_site_df["start"] - binding_site_df["original_start"]
        )
        binding_site_df["stop_offset"] = (
            binding_site_df["stop"] - binding_site_df["original_start"]
        )
        binding_site_df["same_strand"] = (
            binding_site_df["original_strand"] == binding_site_df["strand"]
        )
        tf_arrays = defaultdict(
            lambda: defaultdict(lambda: np.zeros(self.region.length))
        )
        for (tf, same_strand, original_strand), tfdf in binding_site_df.groupby(
            ["tf_name", "same_strand", "original_strand"]
        ):
            this_region = np.zeros(self.region.length)
            for start, stop in zip(tfdf["start_offset"], tfdf["stop_offset"]):
                this_region[start:stop] += 1
            if original_strand == "-" and self.flip_strand_based_on_ori:
                this_region = this_region[::-1]
            tf_arrays[tf][same_strand] += this_region
        for tf in tf_arrays.keys():
            for same_strand in tf_arrays[tf].keys():
                mult = 1 if same_strand else -1
                if self.log:
                    tf_arrays[tf][same_strand] = np.log2(tf_arrays[tf][same_strand] + 1)
                tf_arrays[tf][same_strand] = tf_arrays[tf][same_strand] * mult
        tf_maxes = {}
        for tf in tf_arrays.keys():
            max_peak = max(
                np.abs(np.max(tf_arrays[tf][True])),
                np.abs(np.max(tf_arrays[tf][False])),
            )
            tf_maxes[tf] = max_peak
        tf_mean = {}
        for tf in tf_arrays.keys():
            mean_peak = np.abs(np.max(tf_arrays[tf][True])) + np.abs(
                np.max(tf_arrays[tf][False])
            )
            tf_mean[tf] = mean_peak
        tf_enrichment = {}
        for tf in tf_arrays.keys():
            tf_enrichment[tf] = tf_maxes[tf] / (tf_mean[tf] + (1 / self.region.length))
        sorted_tfs = sorted(tf_enrichment.items(), key=lambda xy: xy[1], reverse=True)
        top_tfs = set([k for k, v in sorted_tfs[: self.top_k_to_plot]])
        return {k: v for k, v in tf_arrays.items() if k in top_tfs}

    def _plot(self, ax):
        tf_arrays = self.get_tf_cov_arrays()
        factors = sorted(list(tf_arrays.keys()))
        import matplotlib.pyplot as plt

        ax.axhline(0, linestyle="--", color="black", label="+/- strand split")
        for i, tf in enumerate(factors):
            for same_strand in [True, False]:
                ax.plot(
                    np.arange(self.region.length),
                    tf_arrays[tf][same_strand],
                    label=f"{tf}",
                    c=plt.cm.tab10(i),
                )
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="right",
            bbox_to_anchor=(1.12, 0.5),
            borderaxespad=0.0,
        )


@dataclass
class MidpointCoverageTrack(CoverageTrack):
    coverage_type: tuple = "midpoint"


@dataclass
class EndpointCoverageTrack(CoverageTrack):
    coverage_type: tuple = ("left_endpoint", "right_endpoint")
    smooth_window: int = 1


@dataclass
class SummedEndpointsCoverageTrack(CoverageTrack):
    coverage_type: tuple = "summed_endpoints"


@dataclass
class ReadCoverageTrack(GenomeTrack):
    """
    Input is a BAM file
    """

    smooth_window: int = 10

    def _materialize_data(self):
        with pysam.AlignmentFile(self.input) as alignment_file:
            return numpy.stack(
                alignment_file.count_coverage(self.chrom, self.start, self.stop), -1
            ).sum(-1)

    def _plot(self, ax):
        ax.plot(numpy.arange(self.start, self.stop), self.data)


@dataclass
class WiggleTrack(GenomeTrack):
    """
    Input is a wiggle file
    """

    smooth: bool = True
    num_bins: int = 700

    @staticmethod
    def materialize_numpy_track(
        input_file, chrom, start, stop, smooth=True, num_bins=700
    ):
        with BigWigReader(input_file) as reader:
            if smooth:
                # dtype = float sets Nones returned by pybigwig to nans
                y = numpy.array(
                    reader.stats(
                        chrom, start=start, stop=stop, nBins=num_bins, type="mean"
                    ),
                    dtype=float,
                )
                y[numpy.isnan(y)] = 0
            else:
                y = reader.values(chrom, start, stop)

        return y

    def _plot(self, ax):
        x = (
            numpy.linspace(self.start, self.stop, 700)
            if self.smooth
            else numpy.arange(self.start, self.stop)
        )
        y = self.materialize_numpy_track(
            self.input,
            self.chrom,
            self.start,
            self.stop,
            smooth=self.smooth,
            num_bins=self.num_bins,
        )
        ax.fill_between(x, [0] * len(y), y, linewidth=0.1, color=self.color)

        # pad default ylim by 20%
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], ylim[1] * 1.2)


@dataclass
class SmallFraction(GenomeTrack):  # pycharm thinks this causes an error but it isn't
    smooth: Union[int, None] = 20

    def _materialize_data(self):
        return VplotTrack.materialize_numpy_array(self.input, self.region)

    def _plot(self, ax):
        smooth = min(self.smooth, self.data.shape[1] // 2)

        data = self.data
        # +1 to avoid divide by zero
        y = data[0:125].sum(0) / (data.sum(0) + 1)
        y = smooth1d(y, smooth)

        ax.plot(numpy.arange(self.region.start, self.region.stop), y)


@dataclass
class NarrowPeakTrack(GenomeTrack):
    """
    Input is a narrowpeak bed file
    """

    def _plot(self, ax):
        chrom, start, stop = self.chrom, self.start, self.stop
        with NarrowPeakBigBedReader(self.input) as reader:
            patches = []
            for rec in reader.fetch(chrom, start, stop):
                patches += [
                    Rectangle((rec.start, 0.1), rec.stop - rec.start, 0.8, linewidth=1)
                ]
                ax.axvline(
                    x=rec.start + rec.peak, ymin=0.1, ymax=0.8, linewidth=3, color="red"
                )

            collection = PatchCollection(
                patches, facecolor=self.color, edgecolor="black"
            )
            ax.add_collection(collection)


@dataclass
class PeakTrack(GenomeTrack):
    """
    Input is a bed file or list of regions. This is different from BedTrack in that it plots all intervals on the same line
    """

    color: str = "black"
    height: int = 1

    def _materialize_data(self):
        if isinstance(self.input, list):
            return self.input
        elif self.input.endswith(".gz"):
            reader = TabixBedReader
        elif self.input.endswith(".bed"):
            reader = BedIntervalTreeReader
        else:
            raise ValueError(f"invalid file extension: {self.input}")

        with reader(self.input) as reader:
            return list(reader.fetch(self.chrom, self.start, self.stop))

    def _plot(self, ax):
        for rec in self.data:
            ax.vlines(
                numpy.arange(rec.start, rec.stop), ymin=0, ymax=1, color=self.color
            )
        ax.axis("off")


@dataclass
class BedTrack(GenomeTrack):
    """
    :param input: a path to a bed file or a list of Regions
    """

    record_height: float = 0.1
    font_size: float = 6
    color: str = None

    def _materialize_data(self):
        if isinstance(self.input, list):
            return self.input
        elif isinstance(self.input, str):
            if self.input.endswith(".gz"):
                reader = TabixBedReader
            elif self.input.endswith(".bed"):
                reader = BedIntervalTreeReader
            else:
                raise ValueError(f"invalid file extension: {self.input}")

            with reader(self.input) as reader:
                return list(reader.fetch(self.chrom, self.start, self.stop))
        else:
            raise ValueError(f"`{input}` is not a valid input")

    def _plot(self, ax):
        if self.color is None:
            cmap_iter = itertools.cycle(seaborn.color_palette(n_colors=10))
        else:
            cmap_iter = itertools.cycle([self.color])

        arr = numpy.arange(0.1, 1, self.record_height)
        arr = arr[arr + self.record_height <= 1]
        y_coord_iter = itertools.cycle(arr)
        for i, rec in enumerate(self.data):
            rect = Rectangle(
                (rec.start, next(y_coord_iter)),
                width=rec.stop - rec.start,
                height=self.record_height,
                linewidth=1,
                facecolor=next(cmap_iter),
                edgecolor="black",
            )
            ax.add_patch(rect)

            if hasattr(rec, "name"):
                # write record name into the rectangle
                rx, ry = rect.get_xy()
                cx = rx + rect.get_width() + 1  # / 2.0
                cy = ry + rect.get_height() / 2.0

                ax.annotate(
                    rec.name,
                    (cx, cy),
                    color="black",
                    weight="bold",
                    fontsize=self.font_size,
                    ha="left",
                    va="center",
                )


@dataclass
class EpilogosTrack(GenomeTrack):
    def _materialize_data(self):
        if self.input.endswith("qcat"):
            reader = BedIntervalTreeReader
        elif self.input.endswith(".gz"):
            reader = TabixBedReader
        else:
            raise ValueError(f"invalid file extension: {self.input}")

        with reader(self.input, check_extension=False) as reader:
            return list(reader.fetch(self.chrom, self.start, self.stop))

    def _plot(self, ax):
        """
        Main plotting function. Plots chromosome region to matplotlib axis.
        :param ax: axis to plot the track
        :param chrom: chromosome name to plot
        :param start: starting coordinate
        :param stop: ending coordinate
        """
        # Load categories file for coloring
        from fbio.constants import FBIO_DATA_DIR

        with open(FBIO_DATA_DIR + "/misc/chrom_hmm_epilogo_colors.json", "r") as file:
            categories = json.load(file)["categories"]
        colors = {key: categories[key][1] for key in categories.keys()}
        labels = {key: categories[key][0] for key in categories.keys()}
        used_labels = []

        ymin = 0
        ymax = 0

        # a qcat file contains epigenetic scores for chromosome region.
        # every record in it consists of: chromosome name, start position, end position, record id and
        # scores list of scores with category id attached.

        for i, record in enumerate(self.data):
            values_list = []
            locations_list = []
            qcat_id_list = []

            values = json.loads(record.name.split("qcat:")[1])
            for value, qcat_id in values:
                values_list.append(value)
                locations_list.append([record.start, record.stop])
                qcat_id_list.append(qcat_id)

            # Check if there are any negative values
            neg_values = [x for x in values_list if x < 0]
            if len(neg_values) > 0:
                min_neg_sum = sum([x for x in values_list if x < 0])
            else:
                min_neg_sum = 0

            # Draw a rectangle for each value
            y_low = min_neg_sum
            for value, location, qcat_id in zip(
                values_list, locations_list, qcat_id_list
            ):
                if value == 0:
                    continue
                # use color and label from categories file
                qcat_color = colors[str(qcat_id)]
                label = labels[str(qcat_id)]

                # make sure we don't add a label twice
                if label in used_labels:
                    label = None
                else:
                    used_labels.append(label)

                height = abs(value)

                # configuring min and max y values for correct y axis scale
                ymax = max(ymax, y_low + height)
                ymin = min(ymin, y_low)

                rectangle_start, rectangle_end = location[0], location[1]

                # Rectangle(xy, width, height, angle=0.0, **kwargs)
                # Draw a rectangle with lower left at xy = (x, y) with specified width, height and rotation angle.
                ax.add_patch(
                    Rectangle(
                        (rectangle_start, y_low),
                        rectangle_end - rectangle_start,
                        height,
                        edgecolor="black",
                        facecolor=qcat_color,
                        linewidth=0.5,
                        label=label,
                    )
                )
                y_low += height
        ax.set_ylim(ymin * 1.03, ymax * 1.03)
        # ax.set_xlabel(self.chrom)
        ax.set_ylabel("Epigenetic score")


@dataclass
class ChromHmmTrack(GenomeTrack):
    """
    Plots a ChromHMM annotation track with colors
    Input is a (num_states, num_positions) matrix, where the values are either one-hot encoded or proportions
    If one-hot, each position is plotted as a single ChromHMM state. If proportions, states are stacked.
    Resolution can be either the same as the region or coarser, in which case the matrix is aligned
    'left', 'right', or 'center'
    """

    color_set: str = "blueprint"
    colors: List[str] = None
    pad: bool = False
    pad_state: int = -1
    resolution: int = 200
    align: str = "center"
    annotate: bool = False

    def _materialize_data(self):
        assert isinstance(self.input, numpy.ndarray)
        num_states, num_pos = self.input.shape
        if num_pos == self.region.length:
            return self.input

        if num_pos * self.resolution > self.region.length:
            raise ValueError(
                f"Number of positions {num_pos} * resolution {self.resolution} is longer than region"
            )

        if not self.pad and num_pos * self.resolution < self.region.length:
            raise ValueError(
                "If passing annotation for smaller region than for plotting, pass pad=True"
            )

        if self.pad_state < 0:
            self.pad_state = num_states + self.pad_state

        out_mat = numpy.zeros((num_states, self.region.length))
        out_mat[self.pad_state] = 1.0
        to_insert = numpy.repeat(self.input, self.resolution, axis=1)
        to_insert_len = to_insert.shape[1]
        if self.align == "left":
            out_mat[:, :to_insert_len] = to_insert
        elif self.align == "right":
            out_mat[:, -to_insert_len:] = to_insert
        elif self.align == "center":
            double_pad_amount = self.region.length - to_insert_len
            assert (
                double_pad_amount % 2 == 0
            ), "Cannot pad evenly on both sides with 'center' alignment"
            pad_amount = double_pad_amount // 2
            out_mat[:, pad_amount:-pad_amount] = to_insert
        else:
            raise NotImplementedError(f"Alignment '{self.align}' not implemented")

        return out_mat

    def _plot(self, ax):
        if not hasattr(self.data, "shape"):
            raise ValueError(f"{type(self.data)} object has no property 'shape'")
        num_states, num_pos = self.data.shape

        if self.color_set is not None:
            if self.color_set not in CHROMHMM_COLORS.keys():
                raise ValueError(
                    f"Invalid color_set. Options are {CHROMHMM_COLORS.keys()}"
                )
            color_dict = CHROMHMM_COLORS[self.color_set]
            try:
                self.colors = [color_dict[str(k)] for k in range(1, num_states + 1)]
            except KeyError:
                raise ValueError(
                    f"Found {num_states} in data, but only {len(color_dict)} colors: {color_dict}"
                )

        if self.colors is None:
            raise ValueError(
                f"Must pass either colors or color_set parameter, options are {CHROMHMM_COLORS.keys()}"
            )

        assert isinstance(self.colors, list)
        if len(self.colors) != num_states:
            raise ValueError(
                f"colors is of length {len(self.colors)} instead of expected {num_states}"
            )

        if self.region.length != num_pos:
            raise ValueError(
                f"Passed matrix has length {num_pos} instead of expected {self.region.length}"
            )

        cumsums = numpy.zeros(self.region.length)
        for cc, color in enumerate(self.colors):
            width = 1
            x_pos = self.region.start
            y_pos = cumsums[0]
            for pp, pos in enumerate(range(self.region.start, self.region.stop)):
                # We typically see blocky resolution here, so we consolidate rectangles to save on rendering
                curr_prop = self.data[cc, pp]
                # If we're at the last position or the next position is different from the current, plot rectangle
                if pp == num_pos - 1 or curr_prop != self.data[cc, pp + 1]:
                    rect = Rectangle(
                        (x_pos, y_pos),
                        width=width,
                        height=curr_prop,
                        fill=True,
                        facecolor=color,
                        linewidth=0,
                    )
                    ax.add_patch(rect)

                    if pp == num_pos - 1 and self.annotate:
                        # TODO: Implement me
                        raise NotImplementedError()

                    # Setting up for the next rectangle
                    width = 1
                    x_pos = pos + 1
                    if pp != num_pos - 1:
                        y_pos = cumsums[pp + 1]

                else:
                    assert curr_prop == self.data[cc, pp + 1]
                    width += 1

                cumsums[pp] += curr_prop

        assert numpy.allclose(cumsums, 1.0)

        ax.set_ylim(0, 1)


@dataclass
class GeneTrack(GenomeTrack):
    """
    Input is a bed or bed.gz file
    """

    show_duplicates: bool = True

    def _plot(self, ax: axis):
        """
        Main plotting function. Plots chromosome region to matplotlib axis.
        :param ax: axis to plot the track
        :param chrom: chromosome name to plot
        :param start: starting coordinate
        :param stop: ending coordinate
        :param show_duplicates: if to plot duplicated genes
        """
        with GenePredReader(self.input) as reader:
            # Init min and max values for plotting
            y_min = float("Inf")
            y_max = 0

            # Rectangles parameters
            height = 2
            space = 4
            gene_rows = [[]]

            # Matplotlib colormap
            cmap = ScalarMappable()

            arrow_small = 0.003 * (self.stop - self.start)
            used_names = []
            for record in reader.fetch(self.chrom, self.start, self.stop):
                if not self.show_duplicates:
                    # Check if a gene with this name has been plotted already
                    if record.name in used_names:
                        continue
                    else:
                        used_names.append(record.name)

                # Get exon data from the record
                exons_length = list(map(int, record.exon_start.split(",")[:-1]))
                exons_offset = list(map(int, record.exon_end.split(",")[:-1]))

                # Get color from record score
                color = cmap.to_rgba(record.score)

                # Get the row where the genes doesn't overlap with the ones previously plotted
                free_row = self.get_free_row(gene_rows, [record.start, record.stop])
                y = free_row * space

                # Draw gene
                vertices = [(record.start, y), (record.stop, y)]
                ax.add_patch(
                    Polygon(
                        vertices,
                        closed=True,
                        fill=True,
                        edgecolor=color,
                        facecolor=color,
                        linewidth=2,
                    )
                )

                # Draw thick part of the gene
                vertices = [(record.thickStart, y), (record.thickEnd, y)]
                ax.add_patch(
                    Polygon(
                        vertices,
                        closed=True,
                        fill=True,
                        edgecolor=color,
                        facecolor=color,
                        linewidth=4,
                    )
                )

                # Record this space is used by the gene; arrow_small * 20 is used to leave space for text
                gene_rows[free_row].append(
                    [record.start - arrow_small * 20, record.stop + 20 * arrow_small]
                )

                # Draw an arrow
                if record.strand == "+":
                    text_offset = arrow_small
                    vertices = [
                        (record.stop, y + height / 2),
                        (record.stop + arrow_small, y),
                        (record.stop, y - height / 2),
                        (record.stop, y),
                    ]
                elif record.strand == "-":
                    text_offset = arrow_small * 2
                    vertices = [
                        (record.start, y + height / 2),
                        (record.start - arrow_small, y),
                        (record.start, y - height / 2),
                        (record.start, y),
                    ]
                else:
                    text_offset = arrow_small
                    vertices = [(record.start, y), (record.stop, y)]

                ax.add_patch(
                    Polygon(
                        vertices,
                        closed=True,
                        fill=True,
                        edgecolor=color,
                        facecolor=color,
                        linewidth=1,
                    )
                )

                # Write gene name
                if record.start < self.start:
                    text_start = self.start
                else:
                    text_start = record.start
                ax.text(
                    text_start - text_offset,
                    y + height / 1.5,
                    s=record.name,
                    size=11,
                    ha="left",
                    path_effects=[path_effects.withStroke(linewidth=2, foreground="w")],
                )
                # Adjust minmax values to scale the plot properly
                y_min = min([y, y + height, y_min])
                y_max = max([y, y + height, y_max])

                # Draw exons
                for exon_length, exon_offset in zip(exons_length, exons_offset):
                    exon_start = record.start + exon_offset
                    exon_end = exon_start + exon_length
                    vertices = [
                        (exon_start, y + height / 2),
                        (exon_end, y + height / 2),
                        (exon_end, y - height / 2),
                        (exon_start, y - height / 2),
                        (exon_start, y + height / 2),
                    ]
                    ax.add_patch(
                        Polygon(
                            vertices,
                            closed=True,
                            fill=True,
                            edgecolor=color,
                            facecolor=color,
                            linewidth=1,
                        )
                    )

            # Set min max values for plotting
            ax.set_ylim(-height * 1.5, y_max * 1.03)

            # # Remove y axis from plot
            # ax.get_yaxis().set_visible(False)

    @staticmethod
    def get_free_row(rows, new_gene):
        """
        This function returns the first row that can be used to plot the gene.
        :param rows: [[[x0, x1], ...], ...] - list of rows. Each row is a list of genes. Each gene is [start, stop].
        :param new_gene: [x0, x1] - gene start and stop.
        :return: index of the first row that has space to plot the gene.
        """
        for i, row in enumerate(rows):
            good_row = True
            for gene in row:
                if intervals_intersect(gene[0], gene[1], new_gene[0], new_gene[1]):
                    good_row = False
                    break
            if good_row:
                return i
        rows.append([])
        return len(rows) - 1
