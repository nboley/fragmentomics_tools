from typing import Union

import numpy
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.utils._random import sample_without_replacement

from numpy.lib.stride_tricks import as_strided


def view_as_windows(arr_in, window_shape, step=1):
    """Rolling window view of the input n-dimensional array.
    Windows are overlapping views of the input array, with adjacent windows
    shifted by a single row or column (or an index of a higher dimension).
    Parameters
    ----------
    arr_in : ndarray
        N-d input array.
    window_shape : integer or tuple of length arr_in.ndim
        Defines the shape of the elementary n-dimensional orthotope
        (better know as hyperrectangle [1]_) of the rolling window view.
        If an integer is given, the shape will be a hypercube of
        sidelength given by its value.
    step : integer or tuple of length arr_in.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.
    Returns
    -------
    arr_out : ndarray
        (rolling) window view of the input array.   If `arr_in` is
        non-contiguous, a copy is made.
    Notes
    -----
    One should be very careful with rolling views when it comes to
    memory usage.  Indeed, although a 'view' has the same memory
    footprint as its base array, the actual array that emerges when this
    'view' is used in a computation is generally a (much) larger array
    than the original, especially for 2-dimensional arrays and above.
    For example, let us consider a 3 dimensional array of size (100,
    100, 100) of ``float64``. This array takes about 8*100**3 Bytes for
    storage which is just 8 MB. If one decides to build a rolling view
    on this array with a window of (3, 3, 3) the hypothetical size of
    the rolling view (if one was to reshape the view for example) would
    be 8*(100-3+1)**3*3**3 which is about 203 MB! The scaling becomes
    even worse as the dimension of the input array becomes larger.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Hyperrectangle
    Examples
    --------
    >>> import numpy as np
    >>> A = np.arange(4*4).reshape(4,4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> window_shape = (2, 2)
    >>> B = view_as_windows(A, window_shape)
    >>> B[0, 0]
    array([[0, 1],
           [4, 5]])
    >>> B[0, 1]
    array([[1, 2],
           [5, 6]])
    >>> A = np.arange(10)
    >>> A
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> window_shape = (3,)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (8, 3)
    >>> B
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5],
           [4, 5, 6],
           [5, 6, 7],
           [6, 7, 8],
           [7, 8, 9]])
    >>> A = np.arange(5*4).reshape(5, 4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19]])
    >>> window_shape = (4, 3)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (2, 2, 4, 3)
    >>> B  # doctest: +NORMALIZE_WHITESPACE
    array([[[[ 0,  1,  2],
             [ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14]],
            [[ 1,  2,  3],
             [ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15]]],
           [[[ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14],
             [16, 17, 18]],
            [[ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15],
             [17, 18, 19]]]])
    """

    # -- basic checks on arguments
    if not isinstance(arr_in, np.ndarray):
        raise TypeError("`arr_in` must be a numpy ndarray")

    ndim = arr_in.ndim

    if isinstance(window_shape, int):
        window_shape = (window_shape,) * ndim
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, int):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = np.array(arr_in.shape)
    window_shape = np.array(window_shape, dtype=arr_shape.dtype)

    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    # -- build rolling window view
    if not arr_in.flags.contiguous:
        warn(RuntimeWarning("Cannot provide views on a non-contiguous input " "array without copying."))

    arr_in = np.ascontiguousarray(arr_in)

    slices = tuple(slice(None, None, st) for st in step)
    window_strides = np.array(arr_in.strides)

    indexing_strides = arr_in[slices].strides

    win_indices_shape = ((np.array(arr_in.shape) - np.array(window_shape)) // np.array(step)) + 1

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = as_strided(arr_in, shape=new_shape, strides=strides)
    return arr_out


def _is_coo_integer_matrix(arr):
    return type(arr) == coo_matrix and numpy.issubdtype(arr.dtype, numpy.integer)


def dedup_coords(rows, cols):
    """
    >>> rows = [1, 2, 3, 3, 2]
    >>> cols = [4, 5, 6, 6, 5]
    >>> dedup_coords(rows, cols)
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> dedup_coords([], [])
    (array([], dtype=int64), array([], dtype=int64))
    """
    assert len(rows) == len(cols)
    if len(rows) == 0:
        return numpy.array([], dtype=int), numpy.array([], dtype=int)
    return numpy.unique(numpy.array([rows, cols]), axis=1)


def make_ones_coo_arr(arr):
    """Convert a coo_matrix arr so that every '.data' entry is one.

    This does the inverse of coo_arr_arr.sum_duplicates(). Rather than sum the data
    values for records with identical coordinates, this expands records with values
    greater than 1 into multiple records.
    """
    # if arr is empty
    if len(arr.data) == 0:
        return coo_matrix(arr.shape, dtype=arr.dtype)

    # copy to avoid in place mutations of arr caused by methods like arr.max()
    # which will internally call arr.sum_duplicates()
    arr = arr.copy()

    # FIXME get the following block working with tests. The key issue is supporting smooth_by_sumpool
    # if not _is_coo_integer_matrix(arr):
    #     raise TypeError("arr must be a coo_matrix with an integer dtype.")
    if arr.min() < 0:
        raise TypeError("all entries in arr must be positive.")

    if numpy.all(numpy.mod(arr.data, 1) == 0):
        arr = arr.astype(int)
    else:
        raise ValueError("All entries in arr must be whole numbers.")

    row_arrays = []
    col_arrays = []

    for val in range(1, arr.max() + 1):
        indices = arr.data >= val
        row_arrays.append(arr.row[indices])
        col_arrays.append(arr.col[indices])

    rows = numpy.concatenate(row_arrays)
    cols = numpy.concatenate(col_arrays)
    rv = coo_matrix((numpy.ones(len(rows), dtype=int), (rows, cols)), shape=arr.shape)

    return rv


def downsample_coo_matrix(arr, k, random_state=None, with_replacement=False):
    """
    >>> row  = numpy.array([0, 3, 1, 0, 3, 3])
    >>> col  = numpy.array([0, 3, 1, 2, 3, 3])
    >>> data = numpy.array([1, 1, 1, 1, 1, 1])
    >>> arr = coo_matrix((data, (row, col)), shape=(4, 4))
    >>> arr.toarray()
    array([[1, 0, 1, 0],
           [0, 1, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 3]])
    >>> downsample_coo_matrix(arr, 4, random_state=2).toarray()
    array([[0, 0, 1, 0],
           [0, 1, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 2]])
    >>> arr.sum_duplicates()
    >>> downsample_coo_matrix(arr, 4, random_state=2).toarray()
    array([[0, 0, 1, 0],
           [0, 1, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 2]])
    """
    if with_replacement:
        raise NotImplementedError("Sampling with replacement is not implemented yet.")

    if isinstance(arr, csr_matrix):
        arr = arr.tocoo()

    if numpy.all(numpy.mod(arr.data, 1) == 0):
        arr = arr.astype(int)
    else:
        raise ValueError("All entries in arr must be whole numbers.")

    if k is None:
        return arr

    if (arr.data > 1).any():
        arr = make_ones_coo_arr(arr)

    assert (arr.data == 1).all()
    # If all of the data values are one, then we can just subset the indices.
    # This is a fast path for a common scenario
    keep_idxs = sample_without_replacement(len(arr.row), k, random_state=random_state)
    rows = arr.row[keep_idxs]
    cols = arr.col[keep_idxs]
    return coo_matrix((numpy.ones(len(rows), dtype=int), (rows, cols)), shape=arr.shape)


def sum_pool(
    arr: Union[coo_matrix, numpy.array], sum_pool_by: Union[int, tuple, None]
) -> Union[coo_matrix, numpy.array]:
    """
    Sum_pools along given axes

    :param sum_pool_by: The window size to sum_pool over.  Shape must be evenly divisible by the sum_pool_by.  If
      an integer scalar is provided, then both axes of the window will be set to that size.

    >>> sum_pool(numpy.array([1,2,3,4]), 2)
    array([3, 7])
    >>> sum_pool(numpy.array([1,2,3,4]), None)
    array([1, 2, 3, 4])
    """
    if sum_pool_by in [None, 1]:
        return arr
    if len(arr.shape) == 1:
        if isinstance(sum_pool_by, int):
            i = sum_pool_by
        else:
            (i,) = sum_pool_by

        if arr.shape[0] % i != 0:
            raise ValueError(f"row shape: {arr.shape[0]} is not evenly divisible by {i}")

        if isinstance(arr, coo_matrix):
            raise NotImplementedError()
        elif isinstance(arr, numpy.ndarray):
            return view_as_windows(arr, (i,), (i,)).sum(-1)
        else:
            raise TypeError(f"{type(arr)} is an invalid type")
    elif len(arr.shape) == 2:
        if isinstance(sum_pool_by, int):
            i = j = sum_pool_by
        else:
            i, j = sum_pool_by

        if arr.shape[0] % i != 0:
            raise ValueError(f"row shape: {arr.shape[0]} is not evenly divisible by {i}")
        if arr.shape[1] % j != 0:
            raise ValueError(f"row shape: {arr.shape[1]} is not evenly divisible by {j}")

        if isinstance(arr, coo_matrix):
            # This returns identical results as view_as_windows().sum(),
            # but works for a sparse array.
            return coo_matrix(
                (arr.data, (arr.row // i, arr.col // j)), shape=(arr.shape[0] // i, arr.shape[1] // j),
            )
        elif isinstance(arr, numpy.ndarray):
            return view_as_windows(arr, (i, j), (i, j)).sum((2, 3))
        else:
            raise TypeError(f"{type(arr)} is an invalid type")
    else:
        raise NotImplementedError(f"Array has an unsupported number of dimensions: {len(arr.shape)}")


def reverse_sum_pool(arr, sum_pool_by: Union[int, tuple, None], preserve_sum: bool = False):
    """
    Upsamples by performing the inverse operation of a sum_pool()

    :param arr: count or density matrix
    :param sum_pool_by: the kernel to upsample with
    :param preserve_sum: divide the value by the duplication factor prior to copy, this way matrix.sum() is preserved.
    :return:
    >>> reverse_sum_pool([1], None)
    array([1])
    >>> reverse_sum_pool([1], 1)
    array([1])
    >>> reverse_sum_pool([1,2], 2)
    array([1, 1, 2, 2])
    >>> reverse_sum_pool([1,2], 2, preserve_sum=True)
    array([0.5, 0.5, 1. , 1. ])
    >>> reverse_sum_pool([[1]], 1)
    array([[1]])
    >>> reverse_sum_pool([[1]], 2)
    array([[1, 1],
           [1, 1]])
    >>> reverse_sum_pool([[1,2], [3,4]], 2)
    array([[1, 1, 2, 2],
           [1, 1, 2, 2],
           [3, 3, 4, 4],
           [3, 3, 4, 4]])
    """
    arr = numpy.asarray(arr)

    if sum_pool_by in [None, 1]:
        return arr

    if isinstance(sum_pool_by, int):
        # reverse sum_pool_by the same amount in in all dimensions
        sum_pool_by = (sum_pool_by,) * arr.ndim

    assert arr.ndim == len(sum_pool_by), f"{sum_pool_by} should have {arr.ndim} dimensions"
    unpooler = numpy.ones(sum_pool_by, dtype=arr.dtype)
    if preserve_sum:
        dims = numpy.product(unpooler.shape)
        unpooler = unpooler / dims
    return numpy.kron(arr, unpooler)
