#!/usr/bin/env python3

import jax.numpy as np

from jax.ops import index_update
from jax.ops import index

from compas.numerical import connectivity_matrix


__all__ = ["force_equilibrium"]


def force_equilibrium(q, edges, xyz, free, fixed, loads):
    """
    Jax-based force density method.
    """
    c_matrix = connectivity_matrix(edges, "list")
    c_matrix = np.array(c_matrix)

    c_free = c_matrix[:, free]
    c_fixed = c_matrix[:, fixed]
    c_free_t = np.transpose(c_free)

    q_matrix = np.diagflat(q)

    A = c_free_t @ q_matrix @ c_free
    b = loads[free, :] - c_free_t @ q_matrix @ c_fixed @ xyz[fixed, :]
    x = np.linalg.solve(A, b)

    xyz = index_update(xyz, index[free, :], x)

    return xyz
