import numpy as np

from numcompute.optim import grad, jacobian


def test_grad_central_quadratic():
    f = lambda x: x[0] ** 2 + 3 * x[1] ** 2
    x = np.array([2.0, -1.0])
    g = grad(f, x, method="central")
    expected = np.array([4.0, -6.0])
    assert np.allclose(g, expected, atol=1e-4)


def test_grad_forward_quadratic():
    f = lambda x: x[0] ** 2 + x[1] ** 2
    x = np.array([1.5, -2.0])
    g = grad(f, x, method="forward")
    expected = np.array([3.0, -4.0])
    assert np.allclose(g, expected, atol=1e-3)


def test_jacobian_vector_function():
    F = lambda x: np.array([x[0] + x[1], x[0] * x[1]])
    x = np.array([2.0, 3.0])
    J = jacobian(F, x)
    expected = np.array([[1.0, 1.0], [3.0, 2.0]])
    assert np.allclose(J, expected, atol=1e-4)
