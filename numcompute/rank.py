"""Ranking and percentile helpers."""

from __future__ import annotations

import numpy as np


def rank(data, method="average"):
    """
    Rank 1D data with tie handling.

    Parameters
    ----------
    data : array-like of shape (n,)
        Input values.
    method : {'average', 'dense', 'ordinal'}, default='average'
        Ranking strategy.

    Returns
    -------
    np.ndarray
        Ranks starting at 1.
    """
    x = np.asarray(data)
    if x.ndim != 1:
        raise ValueError("data must be a 1D array.")
    if method not in {"average", "dense", "ordinal"}:
        raise ValueError("method must be 'average', 'dense', or 'ordinal'.")

    order = np.argsort(x, kind="stable")
    sorted_x = x[order]

    if method == "ordinal":
        out = np.empty(x.size, dtype=float)
        out[order] = np.arange(1, x.size + 1, dtype=float)
        return out

    unique_vals, first_idx, counts = np.unique(
        sorted_x, return_index=True, return_counts=True
    )

    if method == "dense":
        dense_ranks = np.arange(1, unique_vals.size + 1, dtype=float)
        repeated = np.repeat(dense_ranks, counts)
        out = np.empty(x.size, dtype=float)
        out[order] = repeated
        return out

    avg_ranks = first_idx + 1 + (counts - 1) / 2.0
    repeated = np.repeat(avg_ranks, counts)
    out = np.empty(x.size, dtype=float)
    out[order] = repeated
    return out


def percentile(data, q, interpolation="linear"):
    """
    Compute percentile(s) of an array.

    Parameters
    ----------
    data : array-like
        Input data.
    q : float or array-like
        Percentile or percentiles in [0, 100].
    interpolation : {'linear', 'lower', 'higher', 'midpoint'}
        Percentile interpolation method.

    Returns
    -------
    scalar or np.ndarray
        Percentile value(s).
    """
    if interpolation not in {"linear", "lower", "higher", "midpoint"}:
        raise ValueError("Unsupported interpolation method.")
    x = np.asarray(data, dtype=float)
    q = np.asarray(q)
    if np.any((q < 0) | (q > 100)):
        raise ValueError("q must be between 0 and 100.")
    return np.percentile(x, q, method=interpolation)
