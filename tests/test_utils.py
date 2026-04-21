import numpy as np

from numcompute.utils import (
    stable_softmax,
    logsumexp,
    sigmoid,
    relu,
    euclidean_distance,
    pairwise_euclidean,
    topk_indices,
    make_batches,
)


def test_softmax_sums_to_one():
    x = np.array([1000.0, 1001.0, 1002.0])
    s = stable_softmax(x)
    assert np.allclose(np.sum(s), 1.0)


def test_logsumexp_matches_direct_small_values():
    x = np.array([1.0, 2.0, 3.0])
    expected = np.log(np.sum(np.exp(x)))
    assert np.allclose(logsumexp(x), expected)


def test_sigmoid_range():
    x = np.array([-10.0, 0.0, 10.0])
    y = sigmoid(x)
    assert np.all((y >= 0) & (y <= 1))


def test_relu_basic():
    x = np.array([-2.0, 0.0, 3.0])
    assert np.array_equal(relu(x), np.array([0.0, 0.0, 3.0]))


def test_euclidean_distance():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert np.isclose(euclidean_distance(a, b), 5.0)


def test_pairwise_euclidean_shape():
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    D = pairwise_euclidean(X)
    assert D.shape == (2, 2)


def test_topk_indices_largest():
    x = np.array([10, 5, 7, 20])
    idx = topk_indices(x, 2, largest=True)
    assert np.array_equal(x[idx], np.array([20, 10]))


def test_make_batches():
    x = np.arange(10)
    batches = make_batches(x, batch_size=4)
    assert len(batches) == 3
    assert np.array_equal(batches[0], np.array([0, 1, 2, 3]))
