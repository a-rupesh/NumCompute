import numpy as np


def check_array(X, ensure_2d=False, dtype=float):
    """Convert input to a NumPy array and optionally require 2D."""
    X = np.asarray(X, dtype=dtype)
    if ensure_2d and X.ndim != 2:
        raise ValueError("Expected a 2D array")
    return X


def logsumexp(x, axis=None, keepdims=False):
    """Numerically stable log(sum(exp(x)))."""
    x = np.asarray(x, dtype=float)
    xmax = np.max(x, axis=axis, keepdims=True)
    shifted = x - xmax
    summed = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    out = xmax + np.log(summed)
    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def stable_softmax(x, axis=-1):
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=float)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def sigmoid(x):
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def relu(x):
    """Rectified linear unit."""
    x = np.asarray(x, dtype=float)
    return np.maximum(0.0, x)


def euclidean_distance(a, b):
    """Euclidean distance between two 1D vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    return np.sqrt(np.sum((a - b) ** 2))


def pairwise_euclidean(X, Y=None):
    """Pairwise Euclidean distances between rows of X and Y."""
    X = check_array(X, ensure_2d=True, dtype=float)
    Y = X if Y is None else check_array(Y, ensure_2d=True, dtype=float)
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of columns")

    x_sq = np.sum(X ** 2, axis=1, keepdims=True)
    y_sq = np.sum(Y ** 2, axis=1)
    dist_sq = x_sq + y_sq - 2.0 * (X @ Y.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    return np.sqrt(dist_sq)


def topk_indices(values, k, largest=True):
    """Return indices of the top-k or bottom-k values."""
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("values must be a 1D array")
    n = values.size
    if not 1 <= k <= n:
        raise ValueError("k must satisfy 1 <= k <= len(values)")

    if largest:
        idx = np.argpartition(values, -k)[-k:]
        order = np.argsort(values[idx])[::-1]
    else:
        idx = np.argpartition(values, k - 1)[:k]
        order = np.argsort(values[idx])

    return idx[order]


def make_batches(X, batch_size, drop_last=False):
    """Split an array into batches along axis 0."""
    X = np.asarray(X)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    n = len(X)
    batches = []
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        batch = X[start:stop]
        if drop_last and len(batch) < batch_size:
            continue
        batches.append(batch)
    return batches
