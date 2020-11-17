#!/usr/bin/env python3

import matplotlib.pyplot as plt

import jax.numpy as np
from jax import grad, vmap


fig, ax = plt.subplots(figsize=(10, 7))

x = np.linspace(-4, 4, 1000)

my_func = np.tanh
ax.plot(x, my_func(x))

for _ in range(4):
    my_func = grad(my_func)
    ax.plot(x, vmap(my_func)(x))

plt.show()
