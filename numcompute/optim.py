import numpy as np


def _as_1d_float_array(x):
    """Convert input to a 1D NumPy float array."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array")
    return x


def grad(f, x, h=1e-5, method="central"):
    """
    Estimate the gradient of a scalar-valued function using finite differences.

    Parameters
    ----------
    f : callable
        Function mapping a 1D array of shape (n_features,) to a scalar.
    x : array-like of shape (n_features,)
        Point at which the gradient is evaluated.
    h : float, default=1e-5
        Step size for finite differences. Must be positive.
    method : {'central', 'forward'}, default='central'
        Finite-difference scheme.

    Returns
    -------
    np.ndarray of shape (n_features,)
        Estimated gradient.
    """
    x = _as_1d_float_array(x)

    if h <= 0:
        raise ValueError("h must be positive")
    if method not in {"central", "forward"}:
        raise ValueError("method must be 'central' or 'forward'")

    grad_out = np.empty_like(x, dtype=float)

    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = h

        if method == "central":
            grad_out[i] = (f(x + step) - f(x - step)) / (2.0 * h)
        else:
            grad_out[i] = (f(x + step) - f(x)) / h

    return grad_out


def jacobian(F, x, h=1e-5, method="central"):
    """
    Estimate the Jacobian of a vector-valued function using finite differences.

    Parameters
    ----------
    F : callable
        Function mapping a 1D array of shape (n_features,) to a scalar or
        1D array of shape (n_outputs,).
    x : array-like of shape (n_features,)
        Point at which the Jacobian is evaluated.
    h : float, default=1e-5
        Step size for finite differences. Must be positive.
    method : {'central', 'forward'}, default='central'
        Finite-difference scheme.

    Returns
    -------
    np.ndarray of shape (n_outputs, n_features)
        Estimated Jacobian. For scalar output, shape is (1, n_features).
    """
    x = _as_1d_float_array(x)

    if h <= 0:
        raise ValueError("h must be positive")
    if method not in {"central", "forward"}:
        raise ValueError("method must be 'central' or 'forward'")

    y0 = np.atleast_1d(np.asarray(F(x), dtype=float))
    if y0.ndim != 1:
        raise ValueError("F(x) must return a scalar or 1D array")

    J = np.empty((y0.size, x.size), dtype=float)

    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = h

        if method == "central":
            diff = (np.atleast_1d(np.asarray(F(x + step), dtype=float)) -
                    np.atleast_1d(np.asarray(F(x - step), dtype=float))) / (2.0 * h)
        else:
            diff = (np.atleast_1d(np.asarray(F(x + step), dtype=float)) - y0) / h

        if diff.shape != y0.shape:
            raise ValueError("F must return outputs with consistent shape")
        J[:, i] = diff

    return J
