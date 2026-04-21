"""Shared helper utilities for NumCompute."""

from __future__ import annotations

import numpy as np


def check_array(arr, ensure_2d=False, dtype=float, allow_nan=True, copy=False):
    """
    Validate and convert input to a NumPy array.

    Parameters
    ----------
    arr : array-like
        Input data.
    ensure_2d : bool, default=False
        Require a 2D array.
    dtype : data-type, default=float
        Desired output dtype.
    allow_nan : bool, default=True
        Whether NaNs are allowed.
    copy : bool, default=False
        Whether to force a copy.

    Returns
    -------
    np.ndarray
        Validated array.
    """
    out = np.asarray(arr, dtype=dtype)
    if copy:
        out = out.copy()
    if ensure_2d and out.ndim != 2:
        raise ValueError("Expected a 2D array.")
    if not allow_nan and np.isnan(out).any():
        raise ValueError("NaN values are not allowed.")
    return out


def sigmoid(x):
    """Compute the element-wise logistic sigmoid."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def relu(x):
    """Compute the element-wise ReLU."""
    x = np.asarray(x, dtype=float)
    return np.maximum(x, 0.0)


def logsumexp(x, axis=None, keepdims=False):
    """
    Compute log(sum(exp(x))) in a numerically stable way.
    """
    x = np.asarray(x, dtype=float)
    xmax = np.max(x, axis=axis, keepdims=True)
    shifted = x - xmax
    summed = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    out = xmax + np.log(summed)
    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def stable_softmax(x, axis=-1):
    """
    Compute softmax in a numerically stable way.
    """
    x = np.asarray(x, dtype=float)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)


def euclidean_distance(a, b):
    """Compute Euclidean distance between two 1D vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b must be 1D arrays.")
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape.")
    return float(np.linalg.norm(a - b))


def pairwise_euclidean(X, Y=None):
    """
    Compute pairwise Euclidean distances between rows.

    If Y is None, computes distances among rows of X.
    """
    X = check_array(X, ensure_2d=True, dtype=float)
    Y = X if Y is None else check_array(Y, ensure_2d=True, dtype=float)
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features.")
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)
    Y_sq = np.sum(Y ** 2, axis=1, keepdims=True).T
    sq_dist = np.maximum(X_sq + Y_sq - 2.0 * X @ Y.T, 0.0)
    return np.sqrt(sq_dist)


def topk_indices(values, k, largest=True):
    """
    Return indices of the top-k largest or smallest values.
    """
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("values must be a 1D array.")
    if not isinstance(k, int):
        raise TypeError("k must be an integer.")
    if k < 1 or k > values.size:
        raise ValueError("k must satisfy 1 <= k <= len(values).")
    if largest:
        idx = np.argpartition(values, -k)[-k:]
        return idx[np.argsort(values[idx])[::-1]]
    idx = np.argpartition(values, k - 1)[:k]
    return idx[np.argsort(values[idx])]


def make_batches(X, batch_size, *, drop_last=False):
    """
    Yield batches from the first axis of an array.

    Parameters
    ----------
    X : array-like
        Input data.
    batch_size : int
        Batch size.
    drop_last : bool, default=False
        Whether to drop the last incomplete batch.

    Yields
    ------
    np.ndarray
        Consecutive slices of X.
    """
    X = np.asarray(X)
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    n = X.shape[0]
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        if drop_last and (stop - start) < batch_size:
            continue
        yield X[start:stop]
