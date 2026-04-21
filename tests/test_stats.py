import numpy as np
import pytest

from numcompute.stats import histogram, max, mean, median, min, quantile, std


def test_mean_basic():
    assert mean(np.array([1, 2, 3], dtype=float)) == 2.0


def test_mean_axis_and_nan():
    arr = np.array([[1, 2, np.nan], [4, 5, 6]], dtype=float)
    out = mean(arr, axis=1)
    assert np.allclose(out, np.array([1.5, 5.0]))


def test_median_basic():
    assert median(np.array([1, 3, 2], dtype=float)) == 2.0


def test_std_matches_numpy():
    arr = np.array([1, 2, 3], dtype=float)
    assert np.isclose(std(arr), np.std(arr))


def test_min_and_max_ignore_nan():
    arr = np.array([1, np.nan, 3], dtype=float)
    assert min(arr) == 1.0
    assert max(arr) == 3.0


def test_histogram_basic():
    counts, bins = histogram(np.array([1, 2, 3], dtype=float), bins=2)
    assert counts.sum() == 3
    assert bins.shape == (3,)


def test_quantile_basic():
    arr = np.array([1, 2, 3], dtype=float)
    assert quantile(arr, 0.5) == 2.0


def test_quantile_axis():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    out = quantile(arr, 0.5, axis=1)
    assert np.allclose(out, np.array([2.0, 5.0]))


def test_invalid_axis_raises():
    with pytest.raises(ValueError):
        mean(np.array([1, 2, 3], dtype=float), axis=2)
