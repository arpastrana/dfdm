#!/usr/bin/env python3

import jax.numpy as np

from jax import grad
from jax import random

key = random.PRNGKey(1)

data_points, data_dimension = 100, 10

# Generate X and w, then set y = Xw + ϵ
X = random.normal(key, (data_points, data_dimension))

true_w = random.normal(key, (data_dimension,))
y = X.dot(true_w) + 0.1 * random.normal(key, (data_points,))

def make_squared_error(X, y):
    def squared_error(w):
        return np.sum(np.power(X.dot(w) - y, 2)) / X.shape[0]
    return squared_error

# Now use autograd!
grad_loss = grad(make_squared_error(X, y))

# V rough gradient descent routine. don't use this for a real problem.
w_grad = np.zeros(data_dimension)
epsilon = 0.1
iterations = 100
for _ in range(iterations):
    w_grad = w_grad - epsilon * grad_loss(w_grad)

print(w_grad)

# Linear algebra! The Moore-Penrose pseudoinverse: (X^TX)^{-1}X^T.
w_linalg = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
print(w_linalg)
