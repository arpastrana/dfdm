import autograd.numpy as np

from dfdm.equilibrium.state import EquilibriumState
from dfdm.equilibrium.structure import EquilibriumStructure


# ==========================================================================
# Equilibrium model
# ==========================================================================


class EquilibriumModel:
    """
    The calculator.
    """
    def __init__(self, network):
        self.structure = EquilibriumStructure(network)

    def _edges_lengths(self, xyz):
        connectivity = self.structure.connectivity
        return np.linalg.norm(connectivity @ xyz, axis=1)

    def _edges_forces(self, q, lengths):
        # TODO: is there a bug in edge forces?
        return q * lengths

    def _nodes_residuals(self, q, loads, xyz):
        connectivity = self.structure.connectivity
        return loads - np.transpose(connectivity) @ np.diag(q) @ connectivity @ xyz

    def _nodes_positions(self, q, loads, xyz):
        # convenience shorthands
        connectivity = self.structure.connectivity
        free = self.structure.free_nodes
        fixed = self.structure.fixed_nodes

        # Immutable stuff
        c_matrix = connectivity
        c_fixed = c_matrix[:, fixed]
        c_free = c_matrix[:, free]
        c_free_t = np.transpose(c_free)

        # Mutable stuff
        q_matrix = np.diag(q)

        # solve equilibrium after solving a linear system of equations
        A = c_free_t @ q_matrix @ c_free
        b = loads[free, :] - c_free_t @ q_matrix @ c_fixed @ xyz[fixed, :]
        xyz_free = np.linalg.solve(A, b)

        # syntactic sugar
        xyz_fixed = xyz[fixed, :]

        # NOTE: free fixed indices sorted by enumeration
        indices = self.structure.freefixed_nodes

        # NOTE: concatenation is a workaround specific to autograd
        return np.concatenate((xyz_free, xyz_fixed))[indices]

    def __call__(self, q, loads, xyz):
        """
        Compute an equilibrium state using the force density method.
        """
        xyz_eq = self._nodes_positions(q, loads, xyz)
        residuals = self._nodes_residuals(q, loads, xyz_eq)
        lengths = self._edges_lengths(xyz_eq)
        forces = self._edges_forces(q, lengths)

        return EquilibriumState(xyz=xyz_eq,
                                residuals=residuals,
                                lengths=lengths,
                                forces=forces,
                                force_densities=q)
