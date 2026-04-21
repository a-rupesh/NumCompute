import numpy as np

from numcompute.rank import percentile, rank


def test_rank_average():
    x = np.array([10, 20, 20, 30])
    out = rank(x, method="average")
    assert np.allclose(out, np.array([1.0, 2.5, 2.5, 4.0]))


def test_rank_dense():
    x = np.array([10, 20, 20, 30])
    out = rank(x, method="dense")
    assert np.allclose(out, np.array([1.0, 2.0, 2.0, 3.0]))


def test_rank_ordinal():
    x = np.array([20, 10, 20])
    out = rank(x, method="ordinal")
    assert np.allclose(out, np.array([2.0, 1.0, 3.0]))


def test_percentile_basic():
    x = np.array([1, 2, 3, 4])
    assert percentile(x, 50) == 2.5
