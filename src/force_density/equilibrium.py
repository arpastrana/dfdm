import autograd.numpy as anp
import numpy as np

from compas.numerical import connectivity_matrix


__all__ = ["ForceDensity", "force_equilibrium"]


class ForceDensity:
    """
    A callable-object version of the force density method.
    """
    def __call__(self, q, network):
        """
        Do FD directly from information pertaining a network.
        """
        # q = anp.array(network.force_densities())
        params = [anp.array(param) for param in network.fd_parameters()]
        xyz, lengths, forces, residuals = force_equilibrium(q, *params)
        fd_state = {"xyz": xyz, "lengths": lengths, "forces": forces, "residuals": residuals}
        return fd_state


def force_equilibrium(q, edges, xyz, free, fixed, loads):
    """
    Compute a state of static equilibrium using the force density method.
    """
    # Immutable stuff
    c_matrix = connectivity_matrix(edges, "array")
    c_matrix_t = anp.transpose(c_matrix)

    c_fixed = c_matrix[:, fixed]
    c_free = c_matrix[:, free]
    c_free_t = anp.transpose(c_free)

    # Mutable stuff
    q_matrix = anp.diag(q)

    # solve equilibrium after solving a linear system of equations
    A = c_free_t @ q_matrix @ c_free
    b = loads[free, :] - c_free_t @ q_matrix @ c_fixed @ xyz[fixed, :]
    xyz_free = anp.linalg.solve(A, b)

    # what we want with regular numpy
    # xyz[free, :] = xyz_free
    # xyz = xyz.at[free, :].set(x)  # what we cann do, butt only works with JAX

    # xyz -> workaround for in-place assignment with autograd
    indices = {key: idx for idx, key in enumerate(free.tolist() + fixed.tolist())}
    indices = [v for k, v in sorted(indices.items(), key=lambda item: item[0])]
    xyz = anp.concatenate((xyz_free, xyz[fixed, :]))[indices]

    # compute additional things for equilibrium state
    lengths = anp.linalg.norm(c_matrix @ xyz, axis=1)  # shape (n_edges, )
    forces = q * lengths  # TODO: is there a bug in forces?
    residuals = loads - c_matrix_t @ q_matrix @ c_matrix @ xyz

    return xyz, lengths, forces, residuals
