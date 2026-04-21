import numpy as np

from numcompute.sort_search import binary_search, lexsort_rows, quickselect, sort_array, top_k


def test_sort_array_basic():
    arr = np.array([3, 1, 2])
    assert np.array_equal(sort_array(arr), np.array([1, 2, 3]))


def test_lexsort_rows_by_columns():
    arr = np.array([[2, 1], [1, 2], [1, 1]])
    out = lexsort_rows(arr, keys=[0, 1])
    assert np.array_equal(out, np.array([[1, 1], [1, 2], [2, 1]]))


def test_top_k_largest_sorted():
    arr = np.array([1, 9, 3, 7, 5])
    out = top_k(arr, 3, largest=True)
    assert np.array_equal(out, np.array([9, 7, 5]))


def test_top_k_smallest_indices():
    arr = np.array([8, 2, 4, 1])
    idx = top_k(arr, 2, largest=False, return_indices=True)
    assert np.array_equal(idx, np.array([3, 1]))


def test_binary_search_found_and_not_found():
    arr = np.array([1, 3, 5, 7])
    assert binary_search(arr, 5) == (2, True)
    assert binary_search(arr, 4) == (2, False)


def test_quickselect_kth_smallest():
    arr = np.array([7, 1, 3, 9, 5])
    assert quickselect(arr, 2) == 5
