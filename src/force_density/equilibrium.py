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
        params = [anp.array(param) for param in network.fd_parameters()]
        xyz, lengths, forces, residuals = force_equilibrium(q, *params)
        fd_state = {"xyz": xyz, "lengths": lengths, "forces": forces, "residuals": residuals}
        return fd_state


def force_equilibrium_2(q, edges, xyz, free, fixed, loads):
    """
    JAX-based force density method.
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

    xyz[free, :] = x
    # xyz = index_update(xyz, index[free, :], x)
    xyz = xyz.at[free, :].set(x)

    lengths = anp.linalg.norm(c_matrix @ xyz, axis=1)
    forces = q * lengths  # there is a bug in forces?
    residuals = loads - c_matrix_t @ q_matrix @ c_matrix @ xyz

    return xyz, lengths, forces, residuals


def force_equilibrium(q, edges, xyz, free, fixed, loads):
    """
    JAX-based force density method.
    """
    c_matrix = connectivity_matrix(edges, "list")
    c_matrix = anp.array(c_matrix)

    c_matrix_t = anp.transpose(c_matrix)

    c_free = c_matrix[:, free]
    c_fixed = c_matrix[:, fixed]
    c_free_t = anp.transpose(c_free)

    q_matrix = anp.diag(q)

    A = c_free_t @ q_matrix @ c_free
    b = loads[free, :] - c_free_t @ q_matrix @ c_fixed @ xyz[fixed, :]
    x = anp.linalg.solve(A, b)

    # xyz = anp.copy(xyz)

    try:
        xyz[free, :] = x
    except:
        xyz[free, :] = x._value
    #
    # mask_fixed = np.zeros_like(xyz)
    # mask_fixed[fixed, :] = 1
    # xyz = mask_fixed * xyz
    # xyz = xyz + x

    # xyz = xyz * mask_free
    # print(xyz)

    # xyz = anp.array(xyz, copy=True)
    # xyz[free] = x

    # lengths = anp.linalg.norm(c_matrix @ xyz, axis=1)  # shape (n_edges, )
    # forces = q * lengths  # there is a bug in forces?
    # residuals = loads - c_matrix_t @ q_matrix @ c_matrix @ xyz

    lengths = anp.linalg.norm(c_matrix @ xyz, axis=1)  # shape (n_edges, )
    forces = q * lengths  # there is a bug in forces?
    residuals = loads - c_matrix_t @ q_matrix @ c_matrix @ xyz

    return xyz, lengths, forces, residuals
