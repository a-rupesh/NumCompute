"""Sorting, searching, and top-k helpers."""

from __future__ import annotations

import numpy as np


def sort_array(arr, axis=-1, stable=True):
    """
    Sort an array along the specified axis.

    Parameters
    ----------
    arr : array-like
        Input array.
    axis : int, default=-1
        Axis along which to sort.
    stable : bool, default=True
        Whether to use a stable sorting algorithm.

    Returns
    -------
    np.ndarray
        Sorted array.
    """
    arr = np.asarray(arr)
    kind = "stable" if stable else "quicksort"
    return np.sort(arr, axis=axis, kind=kind)


def lexsort_rows(arr, keys=None, ascending=True):
    """
    Sort rows of a 2D array by one or more columns.

    Parameters
    ----------
    arr : array-like of shape (n_rows, n_cols)
        Input 2D array.
    keys : sequence[int] or None
        Column indices to sort by, in priority order.
        If None, uses all columns left-to-right.
    ascending : bool, default=True
        Sort order.

    Returns
    -------
    np.ndarray
        Row-sorted array.
    """
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError("arr must be a 2D array.")
    n_cols = arr.shape[1]
    if keys is None:
        keys = list(range(n_cols))
    if len(keys) == 0:
        raise ValueError("keys must not be empty.")
    keys = [int(k) for k in keys]
    for k in keys:
        if k < -n_cols or k >= n_cols:
            raise ValueError("key index out of bounds.")
    normalized = [(k + n_cols) % n_cols for k in keys]
    lex_keys = tuple(arr[:, k] if ascending else -arr[:, k] for k in reversed(normalized))
    order = np.lexsort(lex_keys)
    return arr[order]


def top_k(arr, k, largest=True, return_indices=False, sorted_output=True):
    """
    Return top-k elements using partial partitioning.

    Parameters
    ----------
    arr : array-like, shape (n,)
        Input 1D array.
    k : int
        Number of elements to select. Must satisfy 1 <= k <= n.
    largest : bool, default=True
        Select largest values if True, else smallest values.
    return_indices : bool, default=False
        Whether to return indices instead of values.
    sorted_output : bool, default=True
        Whether to sort the selected items.

    Returns
    -------
    np.ndarray
        Selected values or indices.
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("arr must be a 1D array.")
    if not isinstance(k, int):
        raise TypeError("k must be an integer.")
    if k < 1 or k > arr.size:
        raise ValueError("k must satisfy 1 <= k <= len(arr).")

    if largest:
        part = np.argpartition(arr, -k)[-k:]
        if sorted_output:
            part = part[np.argsort(arr[part])[::-1]]
    else:
        part = np.argpartition(arr, k - 1)[:k]
        if sorted_output:
            part = part[np.argsort(arr[part])]

    return part if return_indices else arr[part]


def binary_search(arr, target):
    """
    Binary search on a sorted 1D array.

    Returns
    -------
    tuple[int, bool]
        (insertion_index, found)
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("arr must be a 1D sorted array.")
    insertion_index = int(np.searchsorted(arr, target, side="left"))
    found = insertion_index < arr.size and arr[insertion_index] == target
    return insertion_index, bool(found)


def quickselect(arr, k):
    """
    Return the k-th smallest element (0-based) using NumPy partition.
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("arr must be a 1D array.")
    if not isinstance(k, int):
        raise TypeError("k must be an integer.")
    if k < 0 or k >= arr.size:
        raise ValueError("k out of bounds.")
    return np.partition(arr, k)[k]
