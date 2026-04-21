import numpy as np
from numcompute.optim import grad, jacobian

# Scalar function: f(x) = x1^2 + x2^2
def f(x):
    return x[0] ** 2 + x[1] ** 2

# Vector function:
# F(x) = [x1 + x2, x1 * x2]
def F(x):
    return np.array([
        x[0] + x[1],
        x[0] * x[1]
    ])

x = np.array([3.0, 4.0])

g = grad(f, x)
J = jacobian(F, x)

print("Gradient:", g)
print("Jacobian:")
print(J)