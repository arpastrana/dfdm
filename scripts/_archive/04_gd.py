#!/usr/bin/env python3
from jax import grad
from jax import vmap

import jax.numpy as np
import numpy as onp


def addition(a, b):
    """
    """
    return a + b


def parabola(x, y):
    """
    """
    # return x ** 2 + y
    return addition(np.square(x), y)
    # return np.square(x) + y  # must return a scalar


def other_parabola(x, y):
    """
    """
    a = np.square(x) + y
    return 1.0


# parabola = other_parabola
# test
w = np.array(50.0)  # initial value
# w = np.array([50.0, 5.0])

offset = 5

grad_parabola = grad(parabola)
# jax.vmap(jax.grad(loss), in_axes=(None, 0, 0), out_axes=0))
# grad_parabola = vmap(grad_parabola, in_axes=(0, None))

lr = 0.1
iterations = 100

for _ in range(iterations):
    w = w - lr * grad_parabola(w, offset)

print(f"min grad: {w}")
print(f"y optimal: {parabola(w, )}")
