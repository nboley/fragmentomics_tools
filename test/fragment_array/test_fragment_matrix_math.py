import os

import numpy
import numpy as np
import pytest
from numpy import array
from scipy.sparse import coo_matrix

from fragmentomics_tools.fragment_array.fragment_matrix_math import (
    sum_pool,
    downsample_coo_matrix,
    make_ones_coo_arr,
)

ROW = np.array([0, 3, 1, 0, 3, 3])
COL = np.array([0, 3, 1, 2, 3, 3])
DATA = np.array([1, 1, 1, 1, 1, 1])


def test_sum_pool_by_on_sparse():
    shape = 64, 512
    for i in range(10):
        for sum_pool_by in [None, 1, 2, 4, 8]:
            dense_arr = np.random.randint(0, 100, shape)
            sparse_arr = coo_matrix(dense_arr)
            # sum_pool with an dense array uses view_as_windows
            # sum_pool with an sparse array manually changes the coordinates
            assert (
                sum_pool(dense_arr, sum_pool_by) == sum_pool(sparse_arr, sum_pool_by)
            ).all()


def test_sum_pool():
    ###############
    # Test 1d array
    ###############

    a = np.array([1, 1, 2, 2])
    assert numpy.array_equal(sum_pool(a, 2), [2, 4])
    assert numpy.array_equal(sum_pool(a, None), [1, 1, 2, 2])
    assert numpy.array_equal(sum_pool(a, 1), [1, 1, 2, 2])

    ###############
    # Test 2d array
    ###############
    a = np.array([[1, 1, 2, 2], [1, 1, 1, 1]])
    assert numpy.array_equal(sum_pool(a, 2), [[4, 6]])
    assert numpy.array_equal(sum_pool(a, None), [[1, 1, 2, 2], [1, 1, 1, 1]])
    assert numpy.array_equal(sum_pool(a, 1), [[1, 1, 2, 2], [1, 1, 1, 1]])

    assert numpy.array_equal(sum_pool(np.concatenate([a, a], 0), 2), [[4, 6], [4, 6]])

    # test specifying non symmetric window
    arr = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    assert numpy.array_equal(sum_pool(arr, (2, 1)), [[5, 7, 9], [17, 19, 21]])

    assert numpy.array_equal(sum_pool(arr, (1, 3)), [[6], [15], [24], [33]])

    with pytest.raises(ValueError):
        sum_pool(arr, (1, 2))


def test_make_ones_coo_arr():
    original_arr = coo_matrix((DATA, (ROW, COL)), shape=(10, 10))
    assert (original_arr.data == 1).all()

    collapsed_arr = original_arr.copy()
    collapsed_arr.sum_duplicates()
    assert (collapsed_arr.data > 1).any()

    new_ones_arr = make_ones_coo_arr(collapsed_arr)
    assert (new_ones_arr.toarray() == original_arr.toarray()).all()
    assert (new_ones_arr.data == 1).all()

    # test input does not get mutated due to some methods like arr.sum() calling .sum_duplicates()
    copied_arr = original_arr.copy()
    make_ones_coo_arr(original_arr)
    assert len(copied_arr.row) == len(original_arr.row)


def test_make_ones_fails_for_non_integer_float():
    try:
        make_ones_coo_arr(
            coo_matrix((np.array(DATA + 0.1, dtype=float), (ROW, COL)), shape=(4, 4))
        )
    except ValueError:
        pass
    else:
        assert False


def test_downsample_coo_matrix():
    arr = coo_matrix((DATA, (ROW, COL)), shape=(4, 4))
    assert (
        arr.toarray() == array([[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 3]])
    ).all()

    assert (
        downsample_coo_matrix(arr, 4, random_state=2).toarray()
        == array([[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]])
    ).all()

    # this is in-place
    arr.sum_duplicates()

    assert (
        downsample_coo_matrix(arr, 4, random_state=2).toarray()
        == array([[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]])
    ).all()


def test_downsample_coo_matrix_fails_for_fraction():
    try:
        arr = coo_matrix((np.array(DATA + 0.1, dtype=float), (ROW, COL)), shape=(4, 4))
        downsample_coo_matrix(arr, 4, random_state=2)
    except ValueError:
        pass
    else:
        assert False
