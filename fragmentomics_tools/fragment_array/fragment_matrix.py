import copy
import itertools as it
import warnings
from dataclasses import dataclass, replace, fields
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

from fragments_h5 import FragmentsH5

import pysam


from ..region import Region

from .fragment_matrix_math import make_ones_coo_arr, reverse_sum_pool, sum_pool


def _rescale_shape(shape, frag_step_size, pos_step_size):
    return (shape[0] * frag_step_size, shape[1] * pos_step_size)


def dense_array_into_coo_arr(arr, frag_step_size=1, pos_step_size=1):
    """Build a smoothed fragment matrix from a dense array."""
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
            (
                coo_arr.data,
                (coo_arr.row * frag_step_size, coo_arr.col * pos_step_size),
            ),
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

    def __post_init__(
        self,
    ):
        if isinstance(self.arr, (np.ndarray, csr_matrix)):
            self.arr = make_ones_coo_arr(coo_matrix(self.arr))
        # assert np.issubdtype(self.arr.dtype, np.integer)  # FIXME re-enable this and test get tests working

    def __add__(self, other):
        # ensure that the types match. eg, we shouldn't be able to add a RegionFragmentMatrix
        # to a normal FragmentMatrix
        if type(other) != type(self):
            raise TypeError(
                f"Can only add FragmentMatrices of the same type. "
                f"(self is '{type(self)}', other is type {type(other)}\n"
                f"Hint: use 'merge_fragment_matrices' if you want everything to be a FragmentMatrix"
            )

        # ensure that the two fragment matrices that we're adding have the same non-array. This,
        # for example, prevents one from adding two RegionFragmentMatrix's from different regions
        # without first casting to a normal FragmentMatrix
        self_attrs = [getattr(self, f.name) for f in fields(self) if f.name != "arr"]
        other_attrs = [getattr(other, f.name) for f in fields(self) if f.name != "arr"]
        if self_attrs != other_attrs:
            raise ValueError(
                f"Can only add FragmentMatrices that have the same metadata. "
                f"(self has attributes '{self_attrs}', other has {other_attrs}\n"
                f"Hint: use 'merge_fragment_matrices' if you want everything to be a FragmentMatrix"
            )

        return replace(self, arr=(self.arr + other.arr))

    def __radd__(self, other):
        return other + self

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
                arr=coo_matrix(
                    reverse_sum_pool(
                        self.dense_array, sum_pool_by, preserve_sum=preserve_sum
                    )
                ),
            )

    def smooth_by_sum_pool(self, sum_pool_by, preserve_sum: bool = False):
        return self.sum_pooled(sum_pool_by).reverse_sum_pooled(
            sum_pool_by, preserve_sum=preserve_sum
        )


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
        return (
            self.midpoints + self.lengths // 2 + (self.lengths % 2) + self.region.start
        )

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

    @property
    def region_str(self) -> str:
        return f"{self.chrom}:{self.start}-{self.stop}"

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
            self.region,
            start=self.region.start + sl.start,
            stop=self.region.start + sl.stop,
        )
        return replace(self, arr=arr, region=shifted_region)

    def get_subregion_slice(self, subregion):
        # Returns the slice of an RFM that corresponds to the coordinates given by subregion
        assert isinstance(subregion, Region)
        if subregion not in self.region:
            raise ValueError(f"subregion {subregion} is not a subset of {self.region}")
        sl = slice(
            subregion.start - self.region.start, subregion.stop - self.region.start
        )
        return self.get_slice(sl)


def merge_fragment_matrices(
    fragment_matrices: Iterable[Union[FragmentMatrix, RegionFragmentMatrix]]
) -> Union[FragmentMatrix, RegionFragmentMatrix]:
    # FIXME: flip_minus_strand is broken. Fragment matrices are now natively flipped for minus strand
    # raise NotImplementedError("Deprecated. Use FragmentArray and merge those instead")

    regions = set()
    arr = None
    for i, fm in enumerate(fragment_matrices):
        if (
            isinstance(fm, RegionFragmentMatrix)
            or fm.__class__.__name__ == "RegionFragmentMatrix"
        ):
            regions.add(fm.region)
        else:
            if not (
                isinstance(fm, FragmentMatrix)
                or isinstance(fm, RegionFragmentMatrix)
                or fm.__class__.__name__
                in {
                    "FragmentMatrix",
                    "RegionFragmentMatrix",
                }
            ):
                raise TypeError(
                    f"{fm}, name={fm.__class__.__name__} is not a recognized FragmentMatrix"
                )
            regions.add(None)

        if i == 0:
            arr = fm.arr
        else:
            arr += fm.arr

    if arr is None:
        raise StopIteration(f"fragment_matrices was an empty iterable")

    if None in regions or len(regions) != 1:
        return FragmentMatrix(arr)
    elif len(regions) == 1:
        return RegionFragmentMatrix(arr, regions.pop())
    else:
        assert False, "Unreachable"


class TooFewReadsToDownsample(Exception):
    pass
