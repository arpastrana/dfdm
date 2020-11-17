#!/usr/bin/env python3

import jax.numpy as np

from jax.ops import index_update
from jax.ops import index

from compas.numerical import connectivity_matrix




__all__ = ["ForceDensity", "force_equilibrium"]


class ForceDensity():
    """
    A callable-object version of the force density method.
    """
    def __call__(self, q, network):
        """
        Do FD directly from information pertaining a network.
        """
        params = [np.array(param) for param in network.fd_parameters()]
        xyz, lengths, forces, residuals = force_equilibrium(q, *params)
        fd_state = {"xyz": xyz, "lengths": lengths, "forces": forces, "residuals": residuals}
        return fd_state


def force_equilibrium(q, edges, xyz, free, fixed, loads):
    """
    Jax-based force density method.
    """
    c_matrix = connectivity_matrix(edges, "list")
    c_matrix = np.array(c_matrix)
    c_matrix_t = np.transpose(c_matrix)

    c_free = c_matrix[:, free]
    c_fixed = c_matrix[:, fixed]
    c_free_t = np.transpose(c_free)

    q_matrix = np.diag(q)

    A = c_free_t @ q_matrix @ c_free
    b = loads[free, :] - c_free_t @ q_matrix @ c_fixed @ xyz[fixed, :]
    x = np.linalg.solve(A, b)

    xyz = index_update(xyz, index[free, :], x)
    lengths = np.linalg.norm(c_matrix @ xyz, axis=1)
    forces = q * lengths
    residuals = loads - c_matrix_t @ q_matrix @ c_matrix @ xyz

    return xyz, lengths, forces, residuals
