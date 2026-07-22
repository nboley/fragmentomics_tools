import functools
import logging
import warnings
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import List, Union, Optional, Tuple

import h5py
import numpy
import numpy as np
import pandas
import sparse
import scipy
from scipy.sparse import coo_matrix

import numba

from fragments_h5 import FragmentsH5


from ..constants import DEFAULT_VPLOT_SUMPOOL_BY, DEFAULT_MIN_MAPQ, DEFAULT_MAX_FRAG_LEN
from ..region import Region
from .fragment_matrix import RegionFragmentMatrix, FragmentMatrix
from .fragment_matrix_math import reverse_sum_pool

DEFAULT_MIN_SCALING_FACTOR: float = 1e-6
DEFAULT_MAX_SCALING_FACTOR: float = 10.0


import numpy

@numba.njit
def _todense(coords, values, length):
    rv = numpy.zeros(length, dtype=values.dtype)
    for x, v in zip(coords, values):
        rv[x] += v
    return rv

class SparseIntVector():
    def __init__(self, coords, data, length, check=True):
        self.coords = numpy.array(coords).astype(int)
        if check and len(self.coords) > 0 and self.coords.min() < 0:
            raise IndexError("All coordinates must be >= 0")
        if check and len(self.coords) > 0 and self.coords.max() >= length:
            raise IndexError(f"All coordinates must be < the array length ({length})")

        self.data = numpy.array(data)
        self.length = length

    def __add__(self, other):
        if other.length != self.length:
            raise ValueError("Can not add sparse arrays of different lengths")

        return type(self)(coords=numpy.concatenate([self.coords, other.coords]), data=numpy.concatenate([self.data, other.data]), length=self.length, check=False)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def todense(self):
        return _todense(self.coords, self.data, self.length)

    def __repr__(self):
        return self.coords.__repr__().replace('array', type(self).__name__)

    def __str__(self):
        return self.coords.__str__().replace('array', type(self).__name__)

    @staticmethod
    def sum(ars):
        coords = numpy.concatenate([x.coords for x in ars])
        data = numpy.concatenate([x.data for x in ars])
        lengths = numpy.unique([x.length for x in ars])
        if len(lengths) != 1:
            raise ValueError("All arrays must be the same size to sum over them")
        else:
            length = lengths[0]
        return SparseIntVector(coords, data, length)

@numba.njit
def add_at_intervals_inplace(arr, starts, stops, amount):
    """
    Optimized method to increment arr by amount over intervals specified by starts and stops

    .. warning:: out of bounds starts/stops will result in unexpected behavior

    >>> arr = numpy.zeros(10)
    >>> add_at_intervals_inplace(arr, [0, 3], [1, 4], 1)
    >>> arr
    array([1., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
    """
    for start, stop in zip(starts, stops):
        for i in range(start, stop):
            arr[i] += amount


def _switch_plus_with_minus_and_minus_with_plus(fragment_strands):
    # plus_mask = ((fragment_strands == '+') | (fragment_strands == b'+'))
    # minus_mask = ((fragment_strands == '-') | (fragment_strands == b'-'))
    plus_mask = fragment_strands == "+"
    minus_mask = fragment_strands == "-"
    fragment_strands = fragment_strands.copy()
    fragment_strands[plus_mask] = "-"
    fragment_strands[minus_mask] = "+"
    return fragment_strands


def _concat_fragment_strands(fs1, fs2):
    if fs1 is None and fs2 is None:
        return None
    elif (len(fs1) == 0) and (fs2 is None):
        return None
    elif (fs1 is None) and (len(fs2) == 0):
        return None
    elif fs1 is None or fs2 is None:
        raise ValueError(
            "Can not add two fragment arrays where one has fragment strands and the other does not."
        )
    else:
        return numpy.concatenate([fs1, fs2])


def _fragment_strands_are_equal(fs1, fs2):
    if (fs1 is None or len(fs1) == 0) and (fs2 is None or len(fs2) == 0):
        return True
    return numpy.all(fs1 == fs2)


logger = logging.getLogger(__name__)

already_warned_diff_region_add = False


class FragmentDoesNotIntersect(Exception):
    pass


class InvalidCoordinates(Exception):
    pass


class FragmentArray:
    """
    A FragmentArray is a collection of fragments stored in a spare-array-like coordinate format.  The
    fragment coordinates (starts_0 and stops_0) are the coordinates relative to the region start and region end.

    It is distinct from a FragmentMatrix because a FragmentMatrix stores midpoints/lengths, and has the limitation of
    filtering out fragments which intersect a region, but who's midpoints are out of bounds.

    >>> fa = FragmentArray(starts_0=[-1,2,3], stops_0=[3,4,5], length=5, max_frag_len=10)
    >>> fa.starts_0
    array([-1,  2,  3], dtype=int32)
    >>> fa.stops_0
    array([3, 4, 5], dtype=int32)

    Counts of number of fragment starts over this region. Out of bounds starts are not counted.
    >>> fa.first_covered_base_counts
    array([0., 0., 1., 1., 0.])

    Counts of the number of fragment ends over this region (ends are stops-1).  Out of bounds ends are not counted.
    Note that this is generally more useful than fa.stop_counts, since the "end" is the specific position the fragment
    ends.
    >>> fa.last_covered_base_counts
    array([0., 0., 1., 1., 1.])

    You can add two FragmentArrays from identical regions (example, same region over two different BAMs)
    >>> fa2 = FragmentArray(starts_0=[3,4,4], stops_0=[6,7,8], length=5, max_frag_len=10)
    >>> fa + fa2
    FragmentArray(n_frags=6, length=5, starts_0=[-1, 2, 3, 3, 4, 4], stops_0=[3, 4, 5, 6, 7, 8], weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], first_covered_base_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], last_covered_base_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], num_cpgs=[0, 0, 0, 0, 0, 0], num_meth_cpgs=[0, 0, 0, 0, 0, 0], max_frag_len=10)

    You can add two FragmentArrays over different regions as long as the region lengths are the same.  It
    produces a fragment array with a "Pseudo-Region" (a region with "NA" for its chromosome, and a start of 0)
    >>> fa2 = FragmentArray(starts_0=[-1,2,3], stops_0=[3,4,5], length=5, max_frag_len=10)
    >>> fa + fa2
    FragmentArray(n_frags=6, length=5, starts_0=[-1, 2, 3, -1, 2, 3], stops_0=[3, 4, 5, 3, 4, 5], weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], first_covered_base_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], last_covered_base_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], num_cpgs=[0, 0, 0, 0, 0, 0], num_meth_cpgs=[0, 0, 0, 0, 0, 0], max_frag_len=10)
    """

    @property
    def plot_region(self):
        """
        The Region instance to use for plotting
        """
        return Region("NA", 0, self.shape[1])

    def _ones_if_none(self, _x):
        if _x is None:
            return numpy.ones(self.starts_0.shape, dtype=float)
        else:
            return numpy.asarray(_x, dtype=float)

    def _zeros_if_none(self, _x):
        if _x is None:
            return numpy.zeros(self.starts_0.shape, dtype=int)
        else:
            return numpy.asarray(_x, dtype=int)

    def __init__(
        self,
        starts_0: Union[numpy.ndarray, List],
        stops_0: Union[numpy.ndarray, List],
        length: int,
        max_frag_len: int,
        validate_data: bool = True,
        fragment_strands: Union[numpy.ndarray, List, None] = None,
        weights: Union[numpy.ndarray, List, None] = None,
        first_covered_base_weights: Union[numpy.ndarray, List, None] = None,
        last_covered_base_weights: Union[numpy.ndarray, List, None] = None,
        num_cpgs: Union[numpy.ndarray, List, None] = None,
        num_converted_cpgs: Union[numpy.ndarray, List, None] = None,
        num_cytosines: Union[numpy.ndarray, List, None] = None,
        num_converted_cytosines: Union[numpy.ndarray, List, None] = None,
        is_flipped: bool = False,
        gc: Union[numpy.ndarray, None] = None,
    ):
        """
        A FragmentArray is a collection of fragments stored in a spare-array-like coordinate format.  The
        fragment coordinates (starts_0 and stops_0) are all relative to a region over the fragment [0, length)

        :param validate_data: Perform data validation on init.
            Skip validation -- primarily used when the data has already been validated
        :param starts_0: start coordinates of the fragment relative to a region over the fragment [0, length), 0 based inclusive
        :param stops_0: stop coordinates of the fragment  a region over the fragment [0, length), 0 based exclusive
        :param max_frag_len: maximum fragment length (ex 511). We usually use 511 because when converting to a dense
        array, this produces an array who's index 10 represents fragment length 10.
        :param length: length of the region these fragments intersect
        :param weights: weight to be applied to each fragment
        :param first_covered_base_weights: regularization weights for the first base
        :param last_covered_base_weights: regularization weights for the last base
        :param num_cpgs: number of cpgs in the sequenced reads. Overlaps are only counted once.
        :param num_converted_cpgs: number of converted cpgs (e.g. from emseq)
        :param num_cytosines: number of cytosines in the entire fragment (not just the sequenced reads)
        :param num_comverted_cytosines: number of converted cytosines (e.g. from emseq). Overlaps are only counted once.
        :param gc: per-fragment GC content as fraction (0.0-1.0). From FragmentsH5 fetch_array(return_gc=True).
        """
        # do not add validate_data yet.
        self.init_kwargs = dict(
            starts_0=starts_0,
            stops_0=stops_0,
            length=length,
            max_frag_len=max_frag_len,
            weights=weights,
            fragment_strands=fragment_strands,
            first_covered_base_weights=first_covered_base_weights,
            last_covered_base_weights=last_covered_base_weights,
            num_cpgs=num_cpgs,
            num_converted_cpgs=num_converted_cpgs,
            num_cytosines=num_cytosines,
            num_converted_cytosines=num_converted_cytosines,
            is_flipped=is_flipped,
            gc=gc,
        )
        if isinstance(starts_0, numpy.ndarray) and starts_0.dtype not in (numpy.int32, numpy.int64):
            raise TypeError("start array must be int32 or int64")
        if isinstance(stops_0, numpy.ndarray) and stops_0.dtype not in (numpy.int32, numpy.int64):
            raise TypeError("stop array must be int32 or int64")

        # cast starts/stops to array
        self.starts_0 = numpy.asarray(starts_0, dtype=numpy.int32)
        self.stops_0 = numpy.asarray(stops_0, dtype=numpy.int32)

        self.weights = self._ones_if_none(weights)
        self.first_covered_base_weights = self._ones_if_none(first_covered_base_weights)
        self.last_covered_base_weights = self._ones_if_none(last_covered_base_weights)

        self.fragment_strands = fragment_strands
        if self.fragment_strands is not None:
            self.fragment_strands = numpy.asarray(self.fragment_strands, dtype="U1")


        self.num_cpgs = self._zeros_if_none(num_cpgs)
        self.num_converted_cpgs = self._zeros_if_none(num_converted_cpgs)
        self.num_cytosines = self._zeros_if_none(num_cytosines)
        self.num_converted_cytosines = self._zeros_if_none(num_converted_cytosines)

        self.gc = gc

        self.length = length
        self.max_frag_len = max_frag_len
        self.is_flipped = is_flipped

        # Update the things that were just set above
        for k in self.init_kwargs.keys():
            self.init_kwargs[k] = getattr(self, k)

        if validate_data:
            if len(self.starts_0) != len(self.stops_0):
                raise ValueError("The length of starts_0 must be the same as stops_0")
            if len(self.starts_0) != len(self.weights):
                raise ValueError("Weights length should match data length")
            if len(self.starts_0) != len(self.first_covered_base_weights):
                raise ValueError(
                    "First covered base weights length should match data length"
                )
            if len(self.starts_0) != len(self.last_covered_base_weights):
                raise ValueError(
                    "Last covered base weights length should match data length"
                )
            if self.fragment_strands is not None:
                uq_strands = numpy.unique(self.fragment_strands)
                if len(uq_strands) == 1 and uq_strands[0] == "N":
                    self.fragment_strands = None
                else:
                    if not all(x in ["+", "-"] for x in uq_strands):
                        raise ValueError(
                            f"All members of fragment strands must be either '+' or '-', saw {uq_strands}"
                        )
                    if len(self.fragment_strands) != len(self.starts_0):
                        raise ValueError(
                            "The length of fragment_strands must be the same as starts_0"
                        )

            if not numpy.all(self.starts_0 < self.stops_0):
                raise InvalidCoordinates("stops must be greater than starts")

            not_valid = numpy.logical_not(
                self.valid_idxs(self.starts_0, self.stops_0, self.length)
            )
            if numpy.any(not_valid):
                raise FragmentDoesNotIntersect(
                    f"Fragments at indices {self.frag_str(numpy.arange(len(self.starts_0))[not_valid])} "
                    f"do not overlap given length {self.length}: \n"
                    f"invalid_starts: {self.frag_str(self.starts_0[not_valid])}"
                    f"invalid_stops: {self.frag_str(self.stops_0[not_valid])}"
                )

            weird_starts = self.starts_0 >= self.stops_0
            if numpy.any(weird_starts):
                raise FragmentDoesNotIntersect(
                    f"Fragments at indices {self.frag_str(numpy.arange(len(self.starts_0))[weird_starts])} have start >= end: \n"
                    f"invalid_starts: {self.frag_str(self.starts_0[weird_starts])}"
                    f"invalid_stops: {self.frag_str(self.stops_0[weird_starts])}"
                )

            if numpy.any(self.weights < 0):
                raise ValueError(
                    "All fragment weights must be greater than or equal to 0."
                )

    @property
    def pct_meth_cpgs(self):
        assert False
        output = np.zeros_like(self.num_cpgs, dtype=float)
        valid_pos = self.num_cpgs > 0
        output[~valid_pos] = -1
        output[valid_pos] = self.num_meth_cpgs[valid_pos] / self.num_cpgs[valid_pos]
        return output

    @staticmethod
    def frag_str(frags: numpy.ndarray):
        """Dump out a pretty formatted string for a fragment position array."""
        if len(frags) <= 6:
            return f"[{', '.join(map(str, frags))}]"
        return (
            f"[{','.join(map(str, frags[:2]))}, ..., {','.join(map(str, frags[-2:]))}]"
        )

    def reset_cutsite_bias_weights(self):
        self.first_covered_base_weights = numpy.ones(self.starts_0.shape, dtype=float)
        self.last_covered_base_weights = numpy.ones(self.starts_0.shape, dtype=float)
        self.weights = numpy.ones(self.starts_0.shape, dtype=float)

    def _replace(self, validate_data: bool = True, **kwargs):
        """Reinitalize self, potentially replacing input args with entries from **kwargs"""
        assert (
            len(set(kwargs.keys()) - set(self.init_kwargs.keys())) == 0
        ), f"Keys {set(kwargs.keys()) - set(self.init_kwargs.keys())} not found in init_kwargs"
        init_kwargs = {k: v for k, v in self.init_kwargs.items() if k not in kwargs}
        for k, v in kwargs.items():
            init_kwargs[k] = v
        return type(self)(**init_kwargs, validate_data=validate_data)

    def __radd__(self, other):
        # NOTE: __radd__ is only called when __add__ is not defined on both left/right.
        #  so typically this will be if the user is doing something weird. It also happens
        #  when you call sum([a b c]) by default which adds 0 as the base case. 0 is an ok
        #  additive identity though so hack things so that sum does what we want.
        #  Worst case this misses a user error where the user says 0+fragment_array, but the
        #  behavior here is arguably what you would want in that case.
        if other == 0:
            return self._replace(validate_data=False)
        else:
            raise ValueError(
                f"Do not know how to add type {type(other)} to {type(self)}"
            )

    def __add__(
        self, other: Union["RegionFragmentArray", "FragmentArray"]
    ) -> "FragmentArray":
        """
        Adds two FragmentArrays together.  Both FragmentArrays must have the same region length.
        If the regions are identical, the new fragment array with will have the same region.
        If the region are not identical (but have the same length), the new FragmentArray's region will be a
        psuedo-region (chromosome value of "NA", and a start of 0)
        """
        assert (
            self.shape == other.shape
        ), f"shape mismatch! {self.shape} != {other.shape}"
        starts_0 = numpy.concatenate([self.starts_0, other.starts_0])
        stops_0 = numpy.concatenate([self.stops_0, other.stops_0])
        weights = numpy.concatenate([self.weights, other.weights])
        first_covered_base_weights = numpy.concatenate(
            [self.first_covered_base_weights, other.first_covered_base_weights]
        )
        last_covered_base_weights = numpy.concatenate(
            [self.last_covered_base_weights, other.last_covered_base_weights]
        )
        fragment_strands = _concat_fragment_strands(
            self.fragment_strands, other.fragment_strands
        )
        num_cpgs = numpy.concatenate([self.num_cpgs, other.num_cpgs])
        num_converted_cpgs = numpy.concatenate([self.num_converted_cpgs, other.num_converted_cpgs])
        num_cytosines = numpy.concatenate([self.num_cytosines, other.num_cytosines])
        num_converted_cytosines = numpy.concatenate([self.num_converted_cytosines, other.num_converted_cytosines])


        assert (
            self.max_frag_len == other.max_frag_len
        ), "Max fragment length mismatch: {self.max_frag_len} != {other.max_frag_len}"
        # Data has presumably been validated already for left/right, so skip the check for performance.
        return self._replace(
            starts_0=starts_0,
            stops_0=stops_0,
            weights=weights,
            first_covered_base_weights=first_covered_base_weights,
            last_covered_base_weights=last_covered_base_weights,
            fragment_strands=fragment_strands,
            num_cpgs=num_cpgs,
            num_converted_cpgs=num_converted_cpgs,
            num_cytosines=num_cytosines,
            num_converted_cytosines=num_converted_cytosines,
            validate_data=False,
        )

    def sort_in_place(self):
        starts_ends = numpy.array(
            list(
                zip(
                    self.starts_0,
                    self.stops_0,
                    self.weights,
                    self.num_cpgs,
                    self.num_converted_cpgs,
                    self.num_cytosines,
                    self.num_converted_cytosines,
                )
            ),
            dtype=[
                ("s", numpy.int32),
                ("e", numpy.int32),
                ("w", numpy.float32),
                ("m1", numpy.uint32),
                ("m2", numpy.uint32),
                ("m3", numpy.uint32),
                ("m4", numpy.uint32),
            ],
        )
        srt_idx = numpy.argsort(starts_ends, order=("s", "e", "w", "m1", "m2", "m3", "m4"))
        return self.mask(srt_idx)

    def __eq__(self, other: "FragmentArray"):
        """
        >>> fa1 = FragmentArray([-1,2,3], [3,4,500], 100, 511)
        >>> fa2 = FragmentArray([-1,2,3], [3,4,500], 100, 511)
        >>> fa2r = RegionFragmentArray([-1,2,3], [3,4,500], Region('chr1', 0, 100), 511)
        >>> fa3 = FragmentArray([-1,2,4], [3,4,500], 100, 511)
        >>> fa1 == fa2
        True
        >>> fa1 == fa2r # python calls the __eq__ method of the subclass, which is stricter.
        False
        >>> fa1 == fa3
        False
        """
        if (
            self.length != other.length
            or self.max_frag_len != other.max_frag_len
            or self.n_frags != other.n_frags
        ):
            return False
        self.sort_in_place()
        other.sort_in_place()

        return (
            numpy.all(self.starts_0 == other.starts_0)
            and numpy.all(self.stops_0 == other.stops_0)
            and numpy.allclose(self.weights, other.weights)
            and numpy.allclose(
                self.first_covered_base_weights, other.first_covered_base_weights
            )
            and numpy.allclose(
                self.last_covered_base_weights, other.last_covered_base_weights
            )
            and _fragment_strands_are_equal(
                self.fragment_strands, other.fragment_strands
            )
            and numpy.all(self.num_cpgs == other.num_cpgs)
            and numpy.all(self.num_converted_cpgs == other.num_converted_cpgs)
            and numpy.all(self.num_cytosines == other.num_cytosines)
            and numpy.all(self.num_converted_cytosines == other.num_converted_cytosines)
        )

    @staticmethod
    def valid_idxs(
        starts_0: numpy.ndarray, stops_0: numpy.ndarray, length: int
    ) -> numpy.ndarray:
        """Returns a boolean array to filter out fragments that do not overlap a desired length
        :param starts_0: array of 0 based start coordinates relative to this region (not the genome). Can be negative.
        :param stops_0: array of 1 based stop coordiantes (equivalently 0 based exclusive). Can extend beyond length.
        :param length: length of the region

        Observations:
            stop (1 based) must land on position 1 for a covered base to at least be at the 0th position
            the last start (0 based) must begin before the length or else we would have a fragment starting
                at array index [len] which should raise an error.
        """
        # the start needs to be within the range of [0, length) or the stop needs to be in the range of [1, length]
        starts_or_stops_overlap_array = (
            # Stop lands in the range of (0, length]
            ((stops_0 > 0) & (stops_0 <= length))
            # Start lands in the range of [0, length)
            | ((starts_0 >= 0) & (starts_0 < length))
        )
        straddling_fragments = (starts_0 < 0) & (stops_0 > length)
        return starts_or_stops_overlap_array | straddling_fragments

    def downsampled(self, n=None, random_state=None):
        """
        :param n: number of fragments to keep
        :param random_state: random seed
        :return: a downsampled version of this FragmentArray

        >>> fa = FragmentArray(starts_0=[-1,2,3], stops_0=[3,4,5], length=5, max_frag_len=10)
        >>> fa.n_frags
        3
        >>> fa.downsampled(2, random_state=1)
        FragmentArray(n_frags=2, length=5, starts_0=[-1, 3], stops_0=[3, 5], weights=[1.0, 1.0], first_covered_base_weights=[1.0, 1.0], last_covered_base_weights=[1.0, 1.0], num_cpgs=[0, 0], num_meth_cpgs=[0, 0], max_frag_len=10)
        >>> fa.downsampled(4, random_state=1)
        Traceback (most recent call last):
        ...
        ValueError: n_population should be greater or equal than n_samples, got n_samples > n_population (4 > 3)
        """
        if n is None:
            return self._replace(validate_data=False)

        rng = np.random.default_rng(seed=random_state)
        keep_idxs = rng.choice(self.n_frags, n, replace=False)
        return self.mask(keep_idxs, validate_data=False)

    def downsampled_frag_lens(self, frag_len_acceptance_prbs):
        """Downsample self where the acceptance probability for a fragment is taken from frag_len_acceptance_prbs."""
        frag_len_acceptance_prbs = numpy.asarray(frag_len_acceptance_prbs)
        assert 0 <= frag_len_acceptance_prbs.max() <= 1, frag_len_acceptance_prbs.max()
        assert (
            frag_len_acceptance_prbs.shape[0] == self.max_frag_len
        ), f"Frag len acceptance prbs shape: {frag_len_acceptance_prbs.shape} -- Max Frag Len: {self.max_frag_len}"

        # subtract 1 to account for the fact that valid fragment lengths start at 1, but the array is zero indexed
        prbs = frag_len_acceptance_prbs[self.fragment_lengths - 1]
        mask = numpy.random.rand(len(prbs)) < prbs

        return self.mask(mask)

    def oversampled(self, n=None):
        """
        Takes fragments, returns all of them, plus some oversampled
        :param n: total number of fragments to return
        :return:
        """
        if n is None:
            return self._replace(validate_data=False)

        assert n >= self.n_frags
        keep_idxs = numpy.concatenate(
            (
                numpy.arange(self.n_frags),
                numpy.random.choice(self.n_frags, size=self.n_frags - n, replace=True),
            )
        )
        return self.mask(keep_idxs, validate_data=False)

    def sampled_to_frac(self, frac, random_state=None):
        if frac >= 1.0:
            return self.oversampled(n=int(round(frac * self.n_frags)))
        else:
            return self.downsampled(
                n=int(round(frac * self.n_frags)), random_state=random_state
            )

    def sample_with_replacement(self, n):
        return self.mask(np.random.choice(self.n_frags, n))

    def _shift_boundaries(self, /, left=0, right=0, validate_data=True):
        """Modify the length of self.region without changin fragments.

        This is a utility function so that FragmentArray and RegionFragmentArray can share code.
        """
        new_length = self.length - left + right
        assert new_length >= 0
        return self._replace(length=new_length, validate_data=validate_data)

    def resize_offset(self, new_size: int, region=None) -> int:
        """Return the integer offset for fragment starts/ends when asking for
        a resize from self.length to new_size.
        """
        if region is None:
            start_dummy = max(new_size, self.length)
            region = Region("NA", start_dummy, start_dummy + self.length)

        ori_start = region.start
        new_start = region.resize(new_size).start
        return ori_start - new_start

    def new_starts_stops_mask_for_resize(self, new_size: int):
        start_delta = self.resize_offset(new_size)
        starts_0 = self.starts_0 + start_delta
        stops_0 = self.stops_0 + start_delta
        keep_idxs = self.valid_idxs(starts_0, stops_0, new_size)
        return starts_0, stops_0, keep_idxs

    def _resize(self, new_size: int):
        #  Try (5-2)//2 != (2-5)//2 while int((5-2)/2) == int((2-5)/2) == -1
        # Truncated integer conversion is different from //
        starts_0, stops_0, mask = self.new_starts_stops_mask_for_resize(new_size)
        return self._replace(
            starts_0=starts_0, stops_0=stops_0, validate_data=False
        ).mask(mask, validate_data=False)

    def resize(self, new_size: int):
        return self._resize(new_size)._shift_boundaries(right=new_size - self.length)

    def shift_and_zero_pad(self, shift_amt):
        """Shift all fragments by shift_amt and then remove all fragments that don't overlap self."""
        starts_0 = self.starts_0 + shift_amt
        stops_0 = self.stops_0 + shift_amt
        keep_idxs = self.valid_idxs(starts_0, stops_0, self.length)
        # note that this won't validate because we haven't masked out invalid positions yet
        rv = self._replace(starts_0=starts_0, stops_0=stops_0, validate_data=False)
        return rv.mask(keep_idxs, validate_data=True)

    def zero_pad(self, left_amt=0, right_amt=0):
        if left_amt > 0:
            assert left_amt >= 0
            return self.shift_and_zero_pad(left_amt)._shift_boundaries(left=-left_amt)
        elif right_amt > 0:
            assert right_amt >= 0
            return self._shift_boundaries(right=right_amt)
        else:
            raise ValueError("'left_amt' or 'right_amt' must be set")

    def truncate(self, /, left_amt=0, right_amt=0):
        """Make self smaller by left_amt and/or right_amt"""
        assert left_amt >= 0 and right_amt >= 0
        if left_amt > 0:
            self = self._replace(
                starts_0=self.starts_0 - left_amt,
                stops_0=self.stops_0 - left_amt,
                validate_data=False,
            )._shift_boundaries(left=left_amt, validate_data=False)
            self = self.mask(self.valid_idxs(self.starts_0, self.stops_0, self.length))
        if right_amt > 0:
            self = self.mask(
                self.valid_idxs(self.starts_0, self.stops_0, self.length - right_amt)
            )._shift_boundaries(right=-right_amt)

        return self

    def left_resize(self, new_length):
        """Resize self to new_length by modifying self.start"""
        if new_length == self.length:
            return self
        elif new_length < self.length:
            return self.truncate(left_amt=self.length - new_length)
        elif new_length > self.length:
            return self.zero_pad(left_amt=new_length - self.length)
        else:
            assert False, "Unreachable"

    def right_resize(self, new_length):
        """Resize self to new_length by modifying self.stop"""
        if new_length == self.length:
            return self
        elif new_length < self.length:
            return self.truncate(right_amt=self.length - new_length)
        elif new_length > self.length:
            return self.zero_pad(right_amt=new_length - self.length)
        else:
            assert False, "Unreachable"

    def jitter(self, jitter_value: int, output_length: int):
        return self.shift_and_zero_pad(jitter_value).resize(output_length)

    def reverse_strand(self):
        """Create a copy of self with the strand and fragment matrix reversed"""
        starts_0 = self.length - self.stops_0
        stops_0 = self.length - self.starts_0
        fragment_strands = (
            None
            if self.fragment_strands is None
            else _switch_plus_with_minus_and_minus_with_plus(self.fragment_strands)[
                ::-1
            ]
        )

        return self._replace(
            starts_0=starts_0[::-1],
            stops_0=stops_0[::-1],
            weights=self.weights[::-1],
            fragment_strands=fragment_strands,
            first_covered_base_weights=self.last_covered_base_weights[::-1],
            last_covered_base_weights=self.first_covered_base_weights[::-1],
            num_cpgs=self.num_cpgs[::-1],
            num_converted_cpgs=self.num_converted_cpgs[::-1],
            num_cytosines=self.num_cytosines[::-1],
            num_converted_cytosines=self.num_converted_cytosines[::-1],
            validate_data=False,
            is_flipped=(not self.is_flipped),
        )

    @property
    def dense_array(self) -> numpy.ndarray:
        """
        Returns a dense array who's axes are fragment_length/midpoint over the region
        :return:
        """
        return self.fragment_matrix.dense_array

    def __str__(self):
        return (
            f"FragmentArray(n_frags={self.n_frags}, "
            f"length={self.length}, "
            f"starts_0={self.frag_str(self.starts_0)}, "
            f"stops_0={self.frag_str(self.stops_0)}, "
            f"strand={self.frag_str(self.fragment_strands) if self.fragment_strands is not None else None}, "
            f"weights={self.frag_str(self.weights)}, "
            f"first_covered_base_weights={self.frag_str(self.first_covered_base_weights)}, "
            f"last_covered_base_weights={self.frag_str(self.last_covered_base_weights)}, "
            f"num_cpgs={self.frag_str(self.num_cpgs)}, "
            f"num_converted_cpgs={self.frag_str(self.num_converted_cpgs)}, "
            f"num_cytosines={self.frag_str(self.num_cytosines)}, "
            f"num_converted_cytosines={self.frag_str(self.num_converted_cytosines)}, "
            f"max_frag_len={self.max_frag_len})"
        )

    def __repr__(self):
        return self.__str__()

    @property
    def n_frags(self) -> int:
        return len(self.starts_0)

    @property
    def n_fragments(self) -> int:
        return self.n_frags

    @property
    def shape(self):
        return self.max_frag_len + 1, self.length

    @property
    def fragment_lengths(self) -> numpy.ndarray:
        return self.stops_0 - self.starts_0

    @property
    def lengths(self) -> numpy.ndarray:
        return self.fragment_lengths

    @property
    def last_covered_bases_0(self) -> numpy.ndarray:
        return self.stops_0 - 1

    @property
    def first_covered_bases_0(self) -> numpy.ndarray:
        return self.starts_0

    @property
    def midpoints_0(self) -> numpy.ndarray:
        return self.starts_0 + (self.fragment_lengths // 2)

    @property
    def midpoint_0(self) -> int:
        return self.length // 2

    def subset_fragment_lengths(self, min_frag_len=None, max_frag_len=None):
        mask = numpy.ones(self.n_frags, dtype="bool")
        if min_frag_len is not None:
            mask &= self.fragment_lengths >= min_frag_len
        if max_frag_len is not None:
            mask &= self.fragment_lengths < max_frag_len

        return self.mask(mask)

    def _get_covered_base_array(self, positions_attr, weights_attr, return_sparse):
        def _empty():
            if return_sparse:
                return SparseIntVector([], [], self.shape[1])
            else:
                return numpy.zeros(self.length, dtype=numpy.uint32)

        if self.n_frags == 0: return _empty()

        # The following should (intentionally) remove the straddle case
        mask = (getattr(self, positions_attr) >= 0) & (getattr(self, positions_attr) < self.length)
        x = getattr(self, positions_attr)[mask]

        # deal with the empty fragment array case
        if len(x) == 0: return _empty()

        weights = getattr(self, weights_attr)[mask]
        x = SparseIntVector(x, weights, self.shape[1])
        if return_sparse:
            return x
        else:
            return x.todense()
        pass

    def get_first_covered_base_array(self, return_sparse=False) -> numpy.ndarray:
        """
        1d array of fragment start counts at each position that intersect self.region

        >>> fa = FragmentArray([-1,2,2], [2,3,4], 5, 511)
        >>> fa.first_covered_base_counts
        array([0., 0., 2., 0., 0.])
        """
        return self._get_covered_base_array(positions_attr='first_covered_bases_0', weights_attr='first_covered_base_weights', return_sparse=return_sparse)

    @property
    def first_covered_base_counts(self) -> numpy.ndarray:
        return self.get_first_covered_base_array(return_sparse=False)

    def get_last_covered_base_array(self, return_sparse=False) -> numpy.ndarray:
        """
        1d array of fragment end counts at each position that intersect the region

        ends are the 0-based position of fragment ends (so are 1 less than the half-open "stop" coordinate we
        normall use)

        >>> fa = RegionFragmentArray(starts_0=[-1,2,2], stops_0=[2,4,4], region=Region('chr1', 0, 5), max_frag_len=511)
        >>> fa.last_covered_base_counts
        array([0., 1., 0., 2., 0.])
        """
        return self._get_covered_base_array(positions_attr='last_covered_bases_0', weights_attr='last_covered_base_weights', return_sparse=return_sparse)

    @property
    def last_covered_base_counts(self) -> numpy.ndarray:
        return self.get_last_covered_base_array(return_sparse=False)

    def get_midpoint_coverage_array(self, return_sparse=False) -> numpy.ndarray:
        return self._get_covered_base_array(positions_attr='midpoints_0', weights_attr='weights', return_sparse=return_sparse)

    @property
    def midpoint_covered_base_counts(self) -> numpy.ndarray:
        return self.get_midpoint_coverage_array(return_sparse=False)

    def get_fragment_coverage_array(self) -> numpy.ndarray:
        """
        Get a vector of fragment pileup coverage
        >>> fa = FragmentArray([-1, 4], [3, 7], 5, 511)
        >>> fa.get_fragment_coverage_array()
        array([1, 1, 1, 0, 1], dtype=uint32)

        >>> fa = FragmentArray([-1, 1], [3, 5], 5, 511)
        >>> fa.get_fragment_coverage_array()
        array([1, 2, 2, 1, 1], dtype=uint32)

        Check an empty array
        >>> fa = FragmentArray([], [], 5, 511)
        >>> fa.get_fragment_coverage_array()
        array([0, 0, 0, 0, 0], dtype=uint32)
        """
        coverage_array = numpy.zeros(self.length, dtype=numpy.uint32)

        starts = numpy.maximum(self.starts_0, 0)
        stops = numpy.minimum(self.stops_0, self.length)
        add_at_intervals_inplace(coverage_array, starts, stops, 1)

        return coverage_array

    def build_coverage_counts(self, fl_bands=None, split_strand=True, return_sparse=False):
        if fl_bands is None:
            fl_bands = [(0, self.max_frag_len)]

        res = {}
        strands = ["+", "-"] if split_strand else ['.']
        for strand in strands:
            if strand != '.':
                sub_rfa = self.subset_by_fragment_strand(strand)
            else:
                sub_rfa = self
            for fl in fl_bands:
                sub_sub_rfa = sub_rfa.subset_fragment_lengths(*fl)
                key = (strand, fl, "first")
                assert key not in res
                res[key] = sub_sub_rfa.get_first_covered_base_array(return_sparse=return_sparse)

                key = (strand, fl, "last")
                assert key not in res
                res[key] = sub_sub_rfa.get_last_covered_base_array(return_sparse=return_sparse)

                key = (strand, fl, "midpoint")
                assert key not in res
                res[key] = sub_sub_rfa.get_midpoint_coverage_array(return_sparse=return_sparse)

        res = pandas.Series(res)
        res.index = res.index.set_names(['strand', 'fl_band', 'position'])
        return res.rename("coverage")

    def to_fragment_matrix(self):
        warnings.warn("deprecated, use .fragment_matrix", DeprecationWarning)
        return self.fragment_matrix

    @property
    def arr(self) -> coo_matrix:
        midpoint_mask = (0 <= self.midpoints_0) & (self.midpoints_0 < self.length)
        rows = self.lengths[midpoint_mask]
        cols = self.midpoints_0[midpoint_mask]
        data = self.weights[midpoint_mask]
        arr = coo_matrix((data, (rows, cols)), shape=self.shape)
        return arr

    @property
    def fragment_matrix(self) -> FragmentMatrix:
        # remove fragments who's midpoints are out of bounds
        return FragmentMatrix(self.arr)

    def _subset_or_mask(self, mask, validate_data=True):
        """Keep fragments that satisfy mask.

        Mask can be either boolean array the length of self, or an array of indices
        """
        if self.fragment_strands is not None:
            fragment_strands = self.fragment_strands[mask]
        else:
            fragment_strands = None

        return self._replace(
            starts_0=self.starts_0[mask],
            stops_0=self.stops_0[mask],
            weights=self.weights[mask],
            first_covered_base_weights=self.first_covered_base_weights[mask],
            last_covered_base_weights=self.last_covered_base_weights[mask],
            fragment_strands=fragment_strands,
            num_cpgs=self.num_cpgs[mask],
            num_converted_cpgs=self.num_converted_cpgs[mask],
            num_cytosines=self.num_cytosines[mask],
            num_converted_cytosines=self.num_converted_cytosines[mask],
            validate_data=validate_data,
        )

    def mask(self, mask, validate_data=True):
        return self._subset_or_mask(mask, validate_data=validate_data)

    def subset(self, indices, validate_data=True):
        """Return a subset of self for the fragmnets in indices.

        Indices can be repeated, and we only return a single fragment per cell in arr (even if 'data' is greater)
        """
        return self._subset_or_mask(indices, validate_data=validate_data)

    def subset_by_fragment_strand(self, strand):
        """Return two fragment arrays each containing the fragments on either strand.

        if self.fragment_strands is None, then raise an error
        """
        if self.fragment_strands is None:
            raise ValueError(
                "Can not subset by fragment strand because fragment_strand is not set."
            )

        # If frag strands are strings (should be)
        if len(self.fragment_strands) and isinstance(self.fragment_strands[0], str):
            if strand in [b"+", b"-"]:
                strand = strand.decode()

        elif len(self.fragment_strands) and isinstance(self.fragment_strands[0], bytes):
            if strand in ["+", "-"]:
                strand = strand.encode()

        mask = self.fragment_strands == strand
        return self.subset(mask)

    def drop_duplicate_fragments(self):
        _, indices = np.unique(
            np.array([self.starts_0, self.stops_0]), axis=1, return_index=True
        )
        return self.subset(indices)

    def split_into_nonoverlapping_fas(self, sample_sizes, seed=None):
        """Split into fragment arrays with 'sample_size' distinct fragments in each.
        Returns a list of fragment matrices.
        """
        if sum(sample_sizes) > self.n_fragments or any(x <= 0 for x in sample_sizes):
            raise ValueError(
                f"Fragment array has length {self.n_fragments} -- "
                f"requested samples of {sample_sizes} observations totaling {sum(sample_sizes)} "
                f"must be less than the total number of fragments"
            )
        all_indices = numpy.arange(self.n_fragments, dtype="int32")
        rng = numpy.random.default_rng(seed=seed)
        rng.shuffle(all_indices)
        rv = []
        cum_sample_size = 0
        for sample_size in sample_sizes:
            indices = all_indices[cum_sample_size : cum_sample_size + sample_size]
            rv.append(self.subset(indices))
            cum_sample_size += sample_size
        return rv

    def split_into_k_nonoverlapping_fas(self, sample_size=None, k=2):
        """Split into 'k' fragment arrays with 'sample_size' distinct fragments in each.

        Returns a list of fragment arrays.
        """
        if sample_size is None:
            sample_size = self.n_fragments // k
        return self.split_into_nonoverlapping_fas([sample_size] * k)

    def split_into_2_nonoverlapping_fas(self, sample_size=None, seed=None):
        """Split into 2 fragment arrays with 'sample_size' distinct fragments in the first, and the
           rest in the other fragment array.

        If sample_size is None, split into two equal sizes
        Returns a list of fragment arrays.
        """
        if sample_size is None:
            sample_size = self.n_fragments // 2
        sample_size_2 = self.n_fragments - sample_size
        return self.split_into_nonoverlapping_fas(
            [sample_size, sample_size_2], seed=seed
        )

    def filter_by_methyl(
        self,
        num_cpgs: Union[List, Tuple] = (None, None),
        num_meth_cpgs: Union[List, Tuple] = (None, None),
        pct_meth_cpgs: Union[List, Tuple] = (None, None),
    ):
        """
        Filters by the min and max number of CpGs, max and min number that are methylated, and / or pct methylated
        Intervals are all half-open, except for percent methyl, where 1 includes 100% methylation
        """
        assert False
        assert len(num_cpgs) == len(num_meth_cpgs) == len(pct_meth_cpgs) == 2
        mask = np.ones_like(self.num_cpgs, dtype=bool)
        if num_cpgs[0] is not None:
            mask &= num_cpgs[0] <= self.num_cpgs
        if num_cpgs[1] is not None:
            mask &= self.num_cpgs < num_cpgs[1]
        if num_meth_cpgs[0] is not None:
            mask &= num_meth_cpgs[0] <= self.num_meth_cpgs
        if num_meth_cpgs[1] is not None:
            mask &= self.num_meth_cpgs < num_meth_cpgs[1]
        if pct_meth_cpgs[0] is not None:
            mask &= pct_meth_cpgs[0] <= self.pct_meth_cpgs
        if pct_meth_cpgs[1] is not None:
            mask &= 0 <= self.pct_meth_cpgs
            if pct_meth_cpgs[1] < 1:
                mask &= self.pct_meth_cpgs < pct_meth_cpgs[1]
        return self.mask(mask)

    def vplot(self, sum_pool_by=DEFAULT_VPLOT_SUMPOOL_BY, title=None):
        from fragmentomics_tools.plot.tracks import VplotTrack

        return VplotTrack(
            self.fragment_matrix.dense_array,
            self.plot_region,
            name=title,
            sum_pool_by=sum_pool_by,
        ).plot()

    @classmethod
    def from_frag_length_midpoint_dense_array(cls, dense_array) -> "FragmentArray":
        """
        Converts a dense array of frag_length/midpoint to start/stop sparse array coordinates,
        along with the length of the region and the maximumm fragment length.

        Can be used for converting a FragmentMatrix to a FragmentArray

        :param dense_array: a fragment_length/midpoint dense array
        :returns: starts, stops, vals, length, max_frag_len

        .. warning::
          The fragment_length/midpoint dense array representation loses fragments at the boundaries
          of a region.  This occurs when a fragment overlaps the region, but the midpoint does not.
          If you use this method, you should beware of boundary effects.

        Note that the first row is frag_length == 0, which is invalid, and so does not get represted
        in the FragmentArray
        >>> fa = FragmentArray.from_frag_length_midpoint_dense_array([[0,0],
        ...                                                           [0,1],
        ...                                                           [1,2]])
        >>> fa.n_frags
        3
        >>> fa.starts_0
        array([ 1, -1,  0], dtype=int32)
        >>> fa.stops_0
        array([2, 1, 2], dtype=int32)
        >>> fa.last_covered_bases_0
        array([1, 0, 1], dtype=int32)
        """
        (
            starts_0,
            stops_0,
            vals,
            length,
            max_frag_len,
        ) = frag_len_midpoint_dense_array_to_start_stops(dense_array)
        return cls(starts_0, stops_0, length, max_frag_len, weights=vals)

    def make_tracks(
        self,
        vplot_sum_pool_by=DEFAULT_VPLOT_SUMPOOL_BY,
        vplot_label: str = None,
        vplot_cmap: str = "nipy_spectral_r",
        binding_site_data=None,
        coverage_smoothing_window=20,
        show_coverage: bool = True,
    ) -> "Tracks":
        """
        Creates a Tracks instance for plotting this FragmentMatrix

        :param binding_site_data: a binding site data instance
        """
        from fragmentomics_tools.plot.tracks import (
            VectorTrack,
            VplotTrack,
            MotifTrack,
            Tracks,
            MidpointCoverageTrack,
            CoverageTrack,
        )

        region = self.plot_region

        vlines = []
        tracks = [
            VplotTrack(
                self,
                sum_pool_by=vplot_sum_pool_by,
                name=vplot_label,
                cmap=vplot_cmap,
                region=region,
                vlines=vlines,
            ),
        ]

        if show_coverage:
            tracks.extend(
                [
                    MidpointCoverageTrack(
                        self,
                        height=1,
                        name="midpoint coverage",
                        region=region,
                        vlines=vlines,
                    ),
                    CoverageTrack(
                        self,
                        height=1,
                        coverage_type="fragment",
                        name="sub nucleosome coverage",
                        fraction="sub_nucleosome",
                        smooth_window=coverage_smoothing_window,
                        vlines=vlines,
                    ),
                    CoverageTrack(
                        self,
                        height=1,
                        coverage_type="fragment",
                        name="mono-nucleosome coverage",
                        fraction="single_nucleosome",
                        smooth_window=coverage_smoothing_window,
                        vlines=vlines,
                    ),
                    CoverageTrack(
                        self,
                        height=1,
                        coverage_type="fragment",
                        name="dual-nucleosome coverage",
                        fraction="dual_nucleosome",
                        smooth_window=coverage_smoothing_window,
                        vlines=vlines,
                    ),
                    CoverageTrack(
                        self,
                        height=1,
                        coverage_type="fragment",
                        name="fragment coverage",
                        fraction=None,
                        smooth_window=coverage_smoothing_window,
                        vlines=vlines,
                    ),
                ]
            )

        if binding_site_data is not None:
            tracks += [
                MotifTrack(
                    (
                        binding_site_data.seq_data
                        * (
                            binding_site_data.acc_imp_values
                            + binding_site_data.seq_imp_values
                        )[:, None]
                    ),
                    name="motif",
                    region=binding_site_data.region,
                    vlines=vlines,
                ),
                VectorTrack(
                    binding_site_data.smooth(
                        binding_site_data.values,
                        binding_site_data._pfm_len,
                        mode="same",
                    ),
                    name="smoothed_scores",
                    region=binding_site_data.region,
                    vlines=vlines,
                )
                # commented, but this shows how to plot more binding_site_data attributes
                # NumpyTrack(binding_site_data.acc_data,
                #            label='acc_data',
                #            region=binding_site_data_region),
                # NumpyTrack(binding_site_data.seq_imp_values,
                #            label='seq_imp_values',
                #            region=binding_site_data_region),
            ]
        return Tracks(tracks)

    def plot(self, *args, **kwargs):
        return self.make_tracks(*args, **kwargs).plot()


class RegionFragmentArray(FragmentArray):
    @property
    def plot_region(self):
        """The Region instance to use for plotting

        The reason that we do this is because sometimes we need a "pseudo region" for the plotting code. This
        allows for a unified interface.
        """
        return self.region

    @classmethod
    def init_from_fragment_array(cls, fa: FragmentArray, region: Region):
        """Convenience function to add a region toa  base fragmnet array"""
        assert fa.length == region.length
        return cls(fa.starts_0, fa.stops_0, region, fa.max_frag_len)

    def __init__(
        self,
        starts_0: Union[numpy.ndarray, List],
        stops_0: Union[numpy.ndarray, List],
        region: Region,
        max_frag_len: int,
        validate_data: bool = True,
        fragment_strands: Union[numpy.ndarray, List, None] = None,
        weights: Union[numpy.ndarray, List] = None,
        first_covered_base_weights: Union[numpy.ndarray, List, None] = None,
        last_covered_base_weights: Union[numpy.ndarray, List, None] = None,
        num_cpgs: Union[numpy.ndarray, List, None] = None,
        num_converted_cpgs: Union[numpy.ndarray, List, None] = None,
        num_cytosines: Union[numpy.ndarray, List, None] = None,
        num_converted_cytosines: Union[numpy.ndarray, List, None] = None,
        is_flipped: bool = False,
        gc: Union[numpy.ndarray, None] = None,
    ):
        self.region = region

        init_kwargs = dict(
            starts_0=starts_0,
            stops_0=stops_0,
            max_frag_len=max_frag_len,
            fragment_strands=fragment_strands,
            weights=weights,
            region=region,
            first_covered_base_weights=first_covered_base_weights,
            last_covered_base_weights=last_covered_base_weights,
            num_cpgs=num_cpgs,
            num_converted_cpgs=num_converted_cpgs,
            num_cytosines=num_cytosines,
            num_converted_cytosines=num_converted_cytosines,
            is_flipped=is_flipped,
            gc=gc,
        )

        # assert obj_type_name_matches_class_name_str(region, Region)
        super().__init__(
            length=region.length,
            validate_data=validate_data,
            **{k: v for k, v in init_kwargs.items() if k != "region"},
        )
        # Set self.init_kwargs after super so that we override the parent ones.
        # Make sure that anything which was modified in init gets set here
        for k in init_kwargs.keys():
            init_kwargs[k] = getattr(self, k)
        self.init_kwargs = init_kwargs

    @wraps(FragmentArray.reverse_strand)
    def reverse_strand(self):
        """Create a copy of self with the strand and fragment matrix reversed"""
        return (
            super()
            .reverse_strand()
            ._replace(region=self.region.flip_strand(), validate_data=False)
        )

    def make_data_direction_match_strand(self):
        if self.strand not in ["+", "-"]:
            raise TypeError(
                f"Unrecognized strand '{self.strand}' (are you sure that you want to use this function?)"
            )
        if self.strand == "+" and self.is_flipped:
            return self.reverse_strand()
        elif self.strand == "-" and not self.is_flipped:
            return self.reverse_strand()
        else:
            return self

    @wraps(FragmentArray.resize)
    def resize(self, new_size: int):
        new_region = self.region.resize(new_size)
        self = super()._resize(new_size)
        return self._replace(region=new_region)

    @wraps(FragmentArray.shift_and_zero_pad)
    def shift_and_zero_pad(self, shift_amt):
        shifted_region = self.region.shift(shift_amt)
        return (
            super()
            .shift_and_zero_pad(shift_amt)
            ._replace(region=shifted_region, validate_data=False)
        )

    def _shift_boundaries(self, /, left=0, right=0, validate_data=True):
        """Modify the length of self.region without changin fragments.

        This is a utility function so that FragmentArray and RegionFragmentArray can share code.
        """
        return self._replace(
            region=self.region.left_shift(left).right_shift(right), validate_data=False
        )

    def resize_offset(self, new_size: int) -> int:
        """Return the offset for fragment starts/ends when asking for a resize of this region"""
        return super().resize_offset(new_size, self.region)

    def five_prime_resize(self, new_length):
        """Resize self by modifying the five prime end."""
        # check that we have a valid region
        _new_region = self.region.five_prime_resize(new_length)
        assert self.strand == _new_region.strand
        if self.strand == "+":
            return self.left_resize(new_length)
        elif self.strand == "-":
            return self.right_resize(new_length)
        else:
            assert False, "Unreachable -- should be caught by the region resize"

    def three_prime_resize(self, new_length):
        """Resize self by modifying the three prime end."""
        # check that we have a valid region
        _new_region = self.region.three_prime_resize(new_length)
        assert self.strand == _new_region.strand
        if self.strand == "+":
            return self.right_resize(new_length)
        elif self.strand == "-":
            return self.left_resize(new_length)
        else:
            assert False, "Unreachable -- should be caught by the region resize"

    # def truncate(self, /, left_amt=0, right_amt=0):
    #    # self = self._replace(region=self.region.truncate(left_amt=left_amt, right_amt=right_amt))
    #    return super().truncate(left_amt=left_amt, right_amt=right_amt)

    def subset_by_region(self, subregion):
        new_region = self.region.intersect(subregion)
        if new_region is None:
            raise ValueError(
                "sub_region ({sub_region{) must be a subregion of self.region ({self.region}) to take the subset (ensure that strands match)"
            )

        left_amt = subregion.start - self.region.start
        assert left_amt >= 0 and left_amt < self.region.length

        right_amt = self.region.stop - subregion.stop
        assert right_amt >= 0 and right_amt < self.region.length

        return self.truncate(left_amt=left_amt, right_amt=right_amt)

    def mask_overlapping_fragments(self, mask_regions, expansion=0):
        # fast path when we don't have anuy regions to mask
        if len(mask_regions) == 0:
            return self

        mask = numpy.ones(self.n_fragments, dtype=bool)
        for region in mask_regions:
            region = region.shift(-min(self.start, region.start)).resize(
                region.length + expansion * 2, truncate_when_out_of_bounds=True
            )
            mask[(self.starts_0 >= region.start) & (self.stops_0 <= region.stop)] = 0
        mask = mask.astype(bool)
        return self.subset(mask)

    def __str__(self):
        return (
            f"RegionFragmentArray(n_frags={self.n_frags}, "
            f"region={self.region}, "
            f"length={self.length}, "
            f"starts_0={self.frag_str(self.starts_0)}, "
            f"stops_0={self.frag_str(self.stops_0)}, "
            f"strand={self.frag_str(self.fragment_strands) if self.fragment_strands is not None else None}, "
            f"weights={self.frag_str(self.weights)}, "
            f"first_covered_base_weights={self.frag_str(self.first_covered_base_weights)}, "
            f"last_covered_base_weights={self.frag_str(self.last_covered_base_weights)}, "
            f"num_cpgs={self.frag_str(self.num_cpgs)}, "
            f"num_converted_cpgs={self.frag_str(self.num_converted_cpgs)}, "
            f"num_cytosines={self.frag_str(self.num_cytosines)}, "
            f"num_converted_cytosines={self.frag_str(self.num_converted_cytosines)}, "
            f"max_frag_len={self.max_frag_len})"
        )

    def save(self, fname: Union[str, Path]) -> None:
        assert str(fname).endswith(".rfa.h5"), "Save name needs to end in .rfa.h5"
        with h5py.File(fname, "w") as f:
            for k, val in self.init_kwargs.items():
                if k == "region":
                    r: Region = val
                    r_str = str(r)
                    data = np.chararray((len(r_str),))
                    data[:] = list(r_str)
                elif k == "max_frag_len":
                    data = np.array(val, dtype=int)
                else:
                    data = val
                if data is not None:
                    f.create_dataset(k, data=data)

    @classmethod
    def load(cls, fname: Union[str, Path]) -> "RegionFragmentArray":
        with h5py.File(fname, "r") as f:
            init_kwargs = {}
            for k in f.keys():
                val = f[k][()]
                if k == "region":
                    print(val.tobytes().decode("utf-8"))
                    data: Region = Region.from_region_str(val.tobytes().decode("utf-8"))
                elif k == "max_frag_len":
                    data = val
                else:
                    data = val
                init_kwargs[k] = data
        return cls(**init_kwargs)

    @property
    def midpoint_0(self) -> int:
        return self.region.midpoint - self.region.start

    @property
    def starts(self):
        return self.starts_0 + self.region.start

    @property
    def stops(self):
        return self.stops_0 + self.region.start

    @property
    def midpoints(self):
        return self.midpoints_0 + self.region.start

    @classmethod
    def from_frag_length_midpoint_dense_array(
        cls, dense_array, region
    ) -> "FragmentArray":
        return cls.init_from_fragment_array(
            super().from_frag_length_midpoint_dense_array(dense_array), region
        )

    @property
    def fragment_matrix(self) -> RegionFragmentMatrix:
        # remove fragments whose midpoints are out of bounds
        return RegionFragmentMatrix(self.arr, region=self.region)

    @property
    def chrom(self):
        return self.region.chrom

    @property
    def strand(self):
        return self.region.strand

    @property
    def start(self) -> int:
        return self.region.start

    @property
    def stop(self) -> int:
        return self.region.stop

    def is_minus_strand(self) -> bool:
        return self.region.is_minus_strand()

    @property
    def midpoint(self) -> int:
        return self.region.midpoint

    def __add__(
        self, other: Union["RegionFragmentArray", FragmentArray]
    ) -> Union["RegionFragmentArray", FragmentArray]:
        """
        Adds two FragmentArrays together.  Both FragmentArrays must have the same region length.
            If the regions are identical, the new fragment array with will have the same region.
            If the region are not identical (but have the same length), the result will be downcast
            to a FragmentArray which doesn't have a region.
        """
        global already_warned_diff_region_add
        if "region" not in dir(other) or self.region != other.region:
            assert (
                self.shape == other.shape
            ), f"shape mismatch! {self.shape} != {other.shape}"
            if not already_warned_diff_region_add:
                already_warned_diff_region_add = True
                logger.warning(
                    f"Adding fragment matrices from different regions {self} vs {other}. "
                    "Suppressing future warnings of this type."
                )
            region = None
        else:
            assert self.region == other.region
            region = deepcopy(self.region)

        starts_0 = numpy.concatenate([self.starts_0, other.starts_0])
        stops_0 = numpy.concatenate([self.stops_0, other.stops_0])
        max_frag_len = min(self.max_frag_len, other.max_frag_len)
        weights = numpy.concatenate([self.weights, other.weights])
        first_covered_base_weights = numpy.concatenate(
            [self.first_covered_base_weights, other.first_covered_base_weights]
        )
        last_covered_base_weights = numpy.concatenate(
            [self.last_covered_base_weights, other.last_covered_base_weights]
        )

        fragment_strands = _concat_fragment_strands(
            self.fragment_strands, other.fragment_strands
        )

        num_cpgs = numpy.concatenate([self.num_cpgs, other.num_cpgs])
        num_converted_cpgs = numpy.concatenate([self.num_converted_cpgs, other.num_converted_cpgs])
        num_cytosines = numpy.concatenate([self.num_cytosines, other.num_cytosines])
        num_converted_cytosines = numpy.concatenate([self.num_converted_cytosines, other.num_converted_cytosines])
        if region is None:
            return FragmentArray(
                starts_0=starts_0,
                stops_0=stops_0,
                length=self.length,
                weights=weights,
                first_covered_base_weights=first_covered_base_weights,
                last_covered_base_weights=last_covered_base_weights,
                fragment_strands=fragment_strands,
                num_cpgs=num_cpgs,
                num_converted_cpgs=num_converted_cpgs,
                num_cytosines=num_cytosines,
                num_converted_cytosines=num_converted_cytosines,
                max_frag_len=max_frag_len,
                validate_data=False,
            )
        return RegionFragmentArray(
            starts_0=starts_0,
            stops_0=stops_0,
            max_frag_len=self.max_frag_len,
            weights=weights,
            first_covered_base_weights=first_covered_base_weights,
            last_covered_base_weights=last_covered_base_weights,
            fragment_strands=fragment_strands,
            num_cpgs=num_cpgs,
            num_converted_cpgs=num_converted_cpgs,
            num_cytosines=num_cytosines,
            num_converted_cytosines=num_converted_cytosines,
            region=region,
            validate_data=False,
        )

    def __eq__(self, other: "RegionFragmentArray"):
        """
        >>> fa1 = RegionFragmentArray([-1,2,3], [3,4,500], Region('chr1', 0, 100), 511)
        >>> fa2 = RegionFragmentArray([-1,2,3], [3,4,500], Region('chr1', 0, 100), 511)
        >>> fa3 = RegionFragmentArray([-1,2,4], [3,4,500], Region('chr1', 0, 100), 511)
        >>> fa1 == fa2
        True
        >>> fa1 == fa3
        False
        """
        if type(other) != type(self):
            return False
        if (
            self.region != other.region
            or self.max_frag_len != other.max_frag_len
            or self.n_frags != other.n_frags
        ):
            return False
        self.sort_in_place()
        other.sort_in_place()
        return (
            numpy.all(self.starts_0 == other.starts_0)
            and numpy.all(self.stops_0 == other.stops_0)
            and numpy.allclose(self.weights, other.weights)
            and numpy.allclose(
                self.first_covered_base_weights, other.first_covered_base_weights
            )
            and numpy.allclose(
                self.last_covered_base_weights, other.last_covered_base_weights
            )
            and _fragment_strands_are_equal(
                self.fragment_strands, other.fragment_strands
            )
            and numpy.all(self.num_cpgs == other.num_cpgs)
            and numpy.all(self.num_converted_cpgs == other.num_converted_cpgs)
            and numpy.all(self.num_cytosines == other.num_cytosines)
            and numpy.all(self.num_converted_cytosines == other.num_converted_cytosines)
        )

    @classmethod
    def from_frag_bed(
        cls,
        in_frag_bed: str,
        region: Region,
        min_mapq: int = DEFAULT_MIN_MAPQ,
        max_frag_len: int = DEFAULT_MAX_FRAG_LEN,
    ) -> "RegionFragmentArray":
        """Read an indexed frag bed."""
        # chr1    1079316 1079500 40,60,163,83,45M,45M
        # contig  start   stop    mapq1,mapq2,samflag1,samflag2,cigar1,cigar2
        import pysam
        import io

        with pysam.TabixFile(in_frag_bed) as tabixfile:
            s = io.StringIO(
                "\n".join(
                    (
                        s.replace(",", "\t")
                        for s in tabixfile.fetch(
                            region.chrom, region.start, region.stop
                        )
                    )
                )
            )

        colnames = [
            "contig",
            "start",
            "stop",
            "mapq1",
            "mapq2",
            "sam1",
            "sam2",
            "cigar1",
            "cigar2",
            "drop",
        ]
        df = pandas.read_table(s, names=colnames, usecols=colnames[:-1])

        # filter fragments that are too long or that don't have a high enough mapq score
        df = df.query(
            "stop - start <= @max_frag_len and mapq1 >= @min_mapq and mapq2 >= @min_mapq"
        )

        # make sam1 the first read in the pair
        first_in_pair_sam_flag = np.zeros(df.shape[0], dtype=int) - 1
        second_in_pair_sam_flag = np.zeros(df.shape[0], dtype=int) - 1

        first_in_pair_mask = df.sam1 & 64 > 0
        first_in_pair_sam_flag[first_in_pair_mask] = df.sam1[first_in_pair_mask]
        second_in_pair_sam_flag[~first_in_pair_mask] = df.sam1[~first_in_pair_mask]

        second_in_pair_mask = df.sam2 & 64 > 0
        first_in_pair_sam_flag[second_in_pair_mask] = df.sam2[second_in_pair_mask]
        second_in_pair_sam_flag[~second_in_pair_mask] = df.sam2[~second_in_pair_mask]

        df["sam1"] = first_in_pair_sam_flag
        df["sam2"] = second_in_pair_sam_flag

        strand = np.empty((df.shape[0],), dtype="U1")
        plus_strand_mask = (first_in_pair_sam_flag & 16 == 0) & (
            second_in_pair_sam_flag & 16 > 0
        )
        strand[plus_strand_mask] = "+"
        minus_strand_mask = (first_in_pair_sam_flag & 16 > 0) & (
            second_in_pair_sam_flag & 16 == 0
        )
        strand[minus_strand_mask] = "-"
        df["strand"] = strand

        return cls(
            starts_0=(df.start - region.start),
            stops_0=(df.stop - region.start),
            region=region,
            max_frag_len=max_frag_len,
            validate_data=True,
            fragment_strands=df.strand,
            num_cpgs=None,
            num_converted_cpgs=None,
            num_cytosines=None,
            num_converted_cytosines=None,
        )

    @classmethod
    def from_fragments_h5(
        cls,
        in_fragments_h5: Union[str, FragmentsH5],
        region: Region,
        max_frag_len: int = DEFAULT_MAX_FRAG_LEN,
        generate_weights_callback = None,
        fetch_array_kwargs: dict = None,
        min_mapq: int = None,
        return_gc: bool = None,
    ) -> "RegionFragmentArray":
        """
        :param flip_data_to_match_region_strand: If True, the data is placed on the strand matching the region strand (if set).
            If False, strand is ignored in the region and the data is always on the default reference strand.
        :param include_fragment_strand: If true, include a vector of strands for each fragment.
        :param in_fragments_h5: fragment h5
        :param region: region to query
        :param min_mapq: mapq filter
        :param max_frag_len: max frag len filter
        """
        if isinstance(in_fragments_h5, str):
            fragments_h5 = FragmentsH5(in_fragments_h5, cache_pointers=False)
            do_close = True
        elif str(type(in_fragments_h5)) == str(type(FragmentsH5)) or (
            "FragmentsH5" in str(type(in_fragments_h5))
        ):
            fragments_h5 = in_fragments_h5
            do_close = False
        else:
            raise TypeError(
                f"{type(in_fragments_h5)} must be a str or FragmentsH5 {str(type(FragmentsH5))}"
            )

        return_methyl = fragments_h5.has_methyl
        include_fragment_strand = fragments_h5.has_strand
        if return_gc is None:
            return_gc = fragments_h5.has_gc if hasattr(fragments_h5, 'has_gc') else (generate_weights_callback is not None)

        _fetch_kwargs = dict(
            max_frag_len=max_frag_len,
            return_strand=include_fragment_strand,
            return_methyl=return_methyl,
            return_gc=return_gc,
        )
        if fetch_array_kwargs is not None:
            _fetch_kwargs.update(fetch_array_kwargs)

        # If min_mapq filtering requested, ensure we fetch MAPQ values
        if min_mapq is not None:
            _fetch_kwargs['return_mapqs'] = True

        starts, stops, supp_data = fragments_h5.fetch_array(
            region.chrom,
            region.start,
            region.stop,
            **_fetch_kwargs,
        )

        # Filter by MAPQ if requested
        if min_mapq is not None and 'mapq' in supp_data:
            mapq_raw = supp_data['mapq']
            # MAPQ may have 2 values per fragment (paired-end) — take min per pair
            if mapq_raw.ndim == 2:
                mapq_vals = mapq_raw.min(axis=1)
            elif len(mapq_raw) == 2 * len(starts):
                mapq_vals = numpy.minimum(mapq_raw[::2], mapq_raw[1::2])
            else:
                mapq_vals = mapq_raw.ravel()
            mask = mapq_vals >= min_mapq
            starts = starts[mask]
            stops = stops[mask]
            supp_data = {k: v[mask] for k, v in supp_data.items()}

        if generate_weights_callback is not None:
            weights = generate_weights_callback(starts, stops, supp_data)
        else:
            weights = None

        if return_methyl:
            num_cpgs = supp_data["num_cpgs"]
            num_converted_cpgs = supp_data["num_converted_cpgs"]
            num_cytosines = supp_data["num_cytosines"]
            num_converted_cytosines = supp_data["num_converted_cytosines"]
        else:
            num_cpgs, num_converted_cpgs, num_cytosines, num_converted_cytosines = None, None, None, None

        if include_fragment_strand:
            fragment_strands = supp_data["strand"]
        else:
            fragment_strands = None

        if return_gc:
            gc = supp_data['gc']
        else:
            gc = None

        starts_0 = (starts - region.start)
        stops_0 = (stops - region.start)

        # if the region is on the minus strand then flip the data to be in the
        # correct orientation. (if we *dont* want this to happen, then just pass
        # '.' in as the region's strand)
        if region.is_minus_strand():
            ## TODO -- move this into reverse strand
            # Store into temp variables so we can swap. One depends on other.
            _starts_0 = (region.length - stops_0)[::-1]
            _stops_0 = (region.length - starts_0)[::-1]
            starts_0 = _starts_0
            stops_0 = _stops_0
            if fragment_strands is not None:
                fragment_strands = _switch_plus_with_minus_and_minus_with_plus(
                    fragment_strands
                )
            if return_methyl:
                num_cpgs = num_cpgs[::-1]
                num_converted_cpgs = num_converted_cpgs[::-1]
                num_cytosines = num_cytosines[::-1]
                num_converted_cytosines = num_converted_cytosines[::-1]
            if return_gc:
                gc = gc[::-1]

        if do_close:
            fragments_h5.close()

        rfa = RegionFragmentArray(
            starts_0=starts_0,
            stops_0=stops_0,
            region=region,
            max_frag_len=max_frag_len,
            validate_data=True,
            fragment_strands=fragment_strands,
            num_cpgs=num_cpgs,
            num_converted_cpgs=num_converted_cpgs,
            num_cytosines=num_cytosines,
            num_converted_cytosines=num_converted_cytosines,
            weights=weights,
            is_flipped=region.is_minus_strand(),
            gc=gc,
        )

        return rfa

    @classmethod
    def from_fname(
        cls,
        fname: str,
        region: Region,
        min_mapq: int = DEFAULT_MIN_MAPQ,
        max_frag_len: int = DEFAULT_MAX_FRAG_LEN,
        include_fragment_strand=False,
        flip_data_to_match_region_strand=True,
        background_model: Optional["SeqToEndpointsMultiResModel"] = None,
        min_background_scaling_factor: float = DEFAULT_MIN_SCALING_FACTOR,
        max_background_scaling_factor: float = DEFAULT_MAX_SCALING_FACTOR,
    ):
        assert fname.endswith(
            "h5"
        ), f"Currently only have h5 implemented for fragment array, requested parse of {fname}"
        return cls.from_fragments_h5(
            in_fragments_h5=fname,
            region=region,
            min_mapq=min_mapq,
            max_frag_len=max_frag_len,
            include_fragment_strand=include_fragment_strand,
            flip_data_to_match_region_strand=flip_data_to_match_region_strand,
            background_model=background_model,
            min_background_scaling_factor=min_background_scaling_factor,
            max_background_scaling_factor=max_background_scaling_factor,
        )


def get_start_end_densities(density_matrix):
    # convert density to start/stop coordinates
    (
        density_starts,
        density_stops,
        density_vals,
        length,
        _,
    ) = frag_len_midpoint_dense_array_to_start_stops(density_matrix)
    density_ends = density_stops - 1

    # filter out of bounds starts/ends
    t = pandas.DataFrame(
        dict(
            density_starts=density_starts,
            density_ends=density_ends,
            density_vals=density_vals,
        )
    )
    t = t.query(f"0 <= density_starts < {length} and 0 <= density_ends <= {length}")

    # turn sparse into dense
    msk = (density_starts >= 0) & (density_starts < length)
    start_density = sparse.COO(density_starts[msk], density_vals[msk], length).todense()
    # renormalize in case of out of bounds coordinates
    start_density /= start_density.sum()

    # turn sparse into dense
    msk = (density_ends >= 0) & (density_ends < length)
    end_density = sparse.COO(density_ends[msk], density_vals[msk], length).todense()
    # renormalize in case of out of bounds coordinates
    end_density /= end_density.sum()

    return start_density, end_density


def frag_len_midpoint_dense_array_to_start_stops(dense_array):
    """
    Converts a dense array of frag_length/midpoint to start/stop sparse array coordinates,
    along with the length of the region and the maximumm fragment length.

    By a dense array of frag_length/midpoint's we really mean the normal fragment matrix
    representation.

    Can be used for converting a FragmentMatrix to a FragmentArray

    :param dense_array: a fragment_length/midpoint dense array
    :returns: starts, stops, vals, length, max_frag_len

    .. warning::
      The fragment_length/midpoint dense array representation loses fragments at the boundaries
      of a region.  This occurs when a fragment overlaps the region, but the midpoint does not.

    Note that the first row is frag_length == 0, which is invalid, and so does not get represted
    in the FragmentArray
    >>> starts, stops, vals, length, max_frag_len = frag_len_midpoint_dense_array_to_start_stops([[0,0],
    ...                                                                                           [0,1],
    ...                                                                                           [1,2]])
    >>> starts
    array([ 1, -1,  0], dtype=int32)
    >>> stops
    array([2, 1, 2], dtype=int32)
    >>> vals
    array([1, 1, 2])
    >>> length, max_frag_len
    (2, 2)

    This also works for non-integer dense_arrays (for example, a density)
    >>> starts, stops, vals, length, max_frag_len = frag_len_midpoint_dense_array_to_start_stops([[0, 0],
    ...                                                                                           [0.,.25],
    ...                                                                                           [.2,.55]])
    >>> starts
    array([ 1, -1,  0], dtype=int32)
    >>> stops
    array([2, 1, 2], dtype=int32)
    >>> vals
    array([0.25, 0.2 , 0.55])
    >>> length, max_frag_len
    (2, 2)
    """
    dense_array = numpy.asarray(dense_array)
    arr = coo_matrix(dense_array)

    frag_lens = arr.row
    midpoints = arr.col
    vals = arr.data

    # remove frag_lengths with length 0
    mask = frag_lens > 0
    frag_lens = frag_lens[mask]
    midpoints = midpoints[mask]
    vals = vals[mask]

    starts, stops = Fragment.length_and_midpoint_to_start_and_stop(frag_lens, midpoints)
    max_frag_len = dense_array.shape[0] - 1
    length = dense_array.shape[1]

    return starts, stops, vals, length, max_frag_len


def unpack_starts_and_stop_vals(starts, stops, counts):
    """
    Unpacks duplicate entries of starts and stops, where a duplicate entry is a count greater than 1

    >>> unpack_starts_and_stop_vals([1,2,3], [3,4,5], [1,3,2])
    (array([1, 2, 2, 2, 3, 3], dtype=int32), array([3, 4, 4, 4, 5, 5], dtype=int32))
    >>> unpack_starts_and_stop_vals([1,2,3], [3,4,5], [1,1,1])
    (array([1, 2, 3], dtype=int32), array([3, 4, 5], dtype=int32))
    >>> unpack_starts_and_stop_vals([], [], [])
    (array([], dtype=int32), array([], dtype=int32))
    """
    starts = numpy.asarray(starts)
    stops = numpy.asarray(stops)
    counts = numpy.asarray(counts)

    if len(starts) == 0:
        new_starts = []
        new_stops = []
    else:
        new_starts, new_stops, new_vals = zip(
            *(
                (start, stop, val)
                for start, stop, val in zip(starts, stops, counts)
                for val in range(val)
            )
        )
    return (
        numpy.array(new_starts, dtype=numpy.int32),
        numpy.array(new_stops, dtype=numpy.int32),
    )


def merge_fragment_arrays(ars, make_data_direction_match_strand=True, force_fragment_array=False):
    assert len(ars) > 0
    ars = list(ars)
    regions = list(set(getattr(ar, "region", None) for ar in ars))

    # only flip data if we're merging arrays across multiple regions
    if make_data_direction_match_strand and (
        len(regions) > 1 and regions[0] is not None
    ):
        ars = [
            (ar.make_data_direction_match_strand() if hasattr(ar, "strand") else ar)
            for ar in ars
        ]

    region_length = ars[0].length
    if not all(ar.length == region_length for ar in ars):
        raise ValueError("Can not merge regions of differing lengths")

    starts = numpy.concatenate([ar.starts_0 for ar in ars])
    stops = numpy.concatenate([ar.stops_0 for ar in ars])

    if all(ar.fragment_strands is None for ar in ars):
        fragment_strands = None
    elif all(ar.fragment_strands is not None for ar in ars):
        fragment_strands = numpy.concatenate([ar.fragment_strands for ar in ars])
    else:
        raise ValueError(
            "Can not merge fragment arrays where only some have fragment strands."
        )

    weights = numpy.concatenate([ar.weights for ar in ars])
    first_covered_base_weights = numpy.concatenate(
        [ar.first_covered_base_weights for ar in ars]
    )
    last_covered_base_weights = numpy.concatenate(
        [ar.last_covered_base_weights for ar in ars]
    )
    num_cpgs = numpy.concatenate([ar.num_cpgs for ar in ars])
    num_converted_cpgs = numpy.concatenate([ar.num_converted_cpgs for ar in ars])
    num_cytosines = numpy.concatenate([ar.num_cytosines for ar in ars])
    num_converted_cytosines = numpy.concatenate([ar.num_converted_cytosines for ar in ars])

    if len(regions) == 1 and regions[0] is not None and not force_fragment_array:
        region = regions.pop()
        return RegionFragmentArray(
            starts,
            stops,
            fragment_strands=fragment_strands,
            weights=weights,
            first_covered_base_weights=first_covered_base_weights,
            last_covered_base_weights=last_covered_base_weights,
            max_frag_len=ars[0].max_frag_len,
            region=region,
            num_cpgs=num_cpgs,
            num_converted_cpgs=num_converted_cpgs,
            num_cytosines=num_cytosines,
            num_converted_cytosines=num_converted_cytosines,
        )
    else:
        return FragmentArray(
            starts,
            stops,
            fragment_strands=fragment_strands,
            weights=weights,
            first_covered_base_weights=first_covered_base_weights,
            last_covered_base_weights=last_covered_base_weights,
            max_frag_len=ars[0].max_frag_len,
            length=ars[0].length,
            num_cpgs=num_cpgs,
            num_converted_cpgs=num_converted_cpgs,
            num_cytosines=num_cytosines,
            num_converted_cytosines=num_converted_cytosines,
        )
