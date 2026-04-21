import numpy as np
import pytest

from numcompute.utils import (
    check_array,
    euclidean_distance,
    logsumexp,
    make_batches,
    pairwise_euclidean,
    relu,
    sigmoid,
    stable_softmax,
    topk_indices,
)


def test_check_array_2d():
    arr = check_array([[1, 2], [3, 4]], ensure_2d=True)
    assert arr.shape == (2, 2)


def test_check_array_no_nan_raises():
    with pytest.raises(ValueError):
        check_array([1.0, np.nan], allow_nan=False)


def test_sigmoid_values():
    x = np.array([-1000.0, 0.0, 1000.0])
    out = sigmoid(x)
    assert np.allclose(out[1], 0.5)
    assert out[0] < 1e-10
    assert out[2] > 1 - 1e-10


def test_relu_values():
    x = np.array([-1.0, 0.0, 2.0])
    assert np.array_equal(relu(x), np.array([0.0, 0.0, 2.0]))


def test_logsumexp_matches_manual():
    x = np.array([1.0, 2.0, 3.0])
    assert np.isclose(logsumexp(x), np.log(np.sum(np.exp(x))))


def test_stable_softmax_rows_sum_to_one():
    x = np.array([[1.0, 2.0], [1000.0, 1001.0]])
    out = stable_softmax(x, axis=1)
    assert np.allclose(np.sum(out, axis=1), 1.0)


def test_euclidean_distance_basic():
    assert np.isclose(euclidean_distance([0, 0], [3, 4]), 5.0)


def test_pairwise_euclidean_shape():
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    D = pairwise_euclidean(X)
    assert D.shape == (2, 2)
    assert np.allclose(np.diag(D), 0.0)


def test_topk_indices_largest():
    x = np.array([1, 9, 3, 7])
    idx = topk_indices(x, 2)
    assert np.array_equal(idx, np.array([1, 3]))


def test_make_batches():
    X = np.arange(10)
    batches = list(make_batches(X, 4))
    assert len(batches) == 3
    assert np.array_equal(batches[0], np.array([0, 1, 2, 3]))
