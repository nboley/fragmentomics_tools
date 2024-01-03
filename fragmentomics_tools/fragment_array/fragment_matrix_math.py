from typing import Union

import numpy
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.utils._random import sample_without_replacement

from ravel.util.numpy_utils import view_as_windows


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
