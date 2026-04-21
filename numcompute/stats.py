"""Descriptive statistics for NumCompute."""

from __future__ import annotations

import numpy as np


def _validate_array(arr):
    x = np.asarray(arr, dtype=float)
    if x.ndim == 0:
        raise ValueError("arr must have at least one dimension.")
    return x


def _validate_axis(x, axis):
    if axis is None:
        return None
    if not isinstance(axis, int):
        raise TypeError("axis must be an integer or None.")
    if axis < -x.ndim or axis >= x.ndim:
        raise ValueError(f"axis must be between {-x.ndim} and {x.ndim - 1}.")
    return axis


def mean(arr, axis=None):
    """Return the arithmetic mean, ignoring NaNs."""
    x = _validate_array(arr)
    axis = _validate_axis(x, axis)
    return np.nanmean(x, axis=axis)


def median(arr, axis=None):
    """Return the median, ignoring NaNs."""
    x = _validate_array(arr)
    axis = _validate_axis(x, axis)
    return np.nanmedian(x, axis=axis)


def std(arr, axis=None, ddof=0):
    """Return the standard deviation, ignoring NaNs."""
    x = _validate_array(arr)
    axis = _validate_axis(x, axis)
    return np.nanstd(x, axis=axis, ddof=ddof)


def min(arr, axis=None):  # noqa: A001
    """Return the minimum value, ignoring NaNs."""
    x = _validate_array(arr)
    axis = _validate_axis(x, axis)
    return np.nanmin(x, axis=axis)


def max(arr, axis=None):  # noqa: A001
    """Return the maximum value, ignoring NaNs."""
    x = _validate_array(arr)
    axis = _validate_axis(x, axis)
    return np.nanmax(x, axis=axis)


def histogram(arr, bins=10, range=None):
    """
    Compute histogram counts and bin edges after dropping NaNs.
    """
    x = _validate_array(arr)
    if not isinstance(bins, int):
        raise TypeError("bins must be an integer.")
    if bins < 1:
        raise ValueError("bins must be positive.")
    flat = x[~np.isnan(x)]
    return np.histogram(flat, bins=bins, range=range)


def quantile(arr, q, axis=None, interpolation="linear"):
    """
    Compute quantiles in [0, 1], ignoring NaNs.
    """
    x = _validate_array(arr)
    axis = _validate_axis(x, axis)
    if interpolation not in {"linear", "lower", "higher", "midpoint"}:
        raise ValueError("Unsupported interpolation method.")
    if np.isscalar(q):
        q_arr = np.asarray([q], dtype=float)
        scalar = True
    else:
        q_arr = np.asarray(q, dtype=float)
        scalar = False
    if np.any((q_arr < 0) | (q_arr > 1)):
        raise ValueError("q must be between 0 and 1.")
    result = np.nanquantile(x, q_arr, axis=axis, method=interpolation)
    if scalar:
        return np.asarray(result)[0]
    return result
