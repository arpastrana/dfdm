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
        self.loads = np.asarray(list(network.nodes_loads()), dtype=np.float64)
        self.xyz_fixed = np.asarray([network.node_coordinates(node) for node in network.nodes_fixed()], dtype=np.float64)

    def _edges_vectors(self, xyz):
        connectivity = self.structure.connectivity
        return connectivity @ xyz

    def _edges_lengths(self, vectors):
        return np.linalg.norm(vectors, axis=1)

    def _edges_forces(self, q, lengths):
        # TODO: is there a bug in edge forces?
        return q * lengths

    def _nodes_residuals(self, q, vectors):
        connectivity = self.structure.connectivity
        return self.loads - np.transpose(connectivity) @ np.diag(q) @ vectors

    def _nodes_free_positions(self, q):
        # convenience shorthands
        free = self.structure.free_nodes
        fixed = self.structure.fixed_nodes
        loads = self.loads
        xyz_fixed = self.xyz_fixed

        # Immutable stuff
        c_matrix = self.structure.connectivity
        c_fixed = c_matrix[:, fixed]
        c_free = c_matrix[:, free]
        c_free_t = np.transpose(c_free)

        # Mutable stuff
        q_matrix = np.diag(q)

        # solve equilibrium after solving a linear system of equations
        A = c_free_t @ q_matrix @ c_free
        b = loads[free, :] - c_free_t @ q_matrix @ c_fixed @ xyz_fixed
        return np.linalg.solve(A, b)

    def _nodes_positions(self, xyz_free):
        # NOTE: free fixed indices sorted by enumeration
        xyz_fixed = self.xyz_fixed
        indices = self.structure.freefixed_nodes
        return np.concatenate((xyz_free, xyz_fixed))[indices, :]

    def __call__(self, q):
        """
        Compute an equilibrium state using the force density method.
        """
        xyz = self._nodes_free_positions(q)
        xyz_eq = self._nodes_positions(xyz)
        vectors = self._edges_vectors(xyz_eq)
        residuals = self._nodes_residuals(q, vectors)
        lengths = self._edges_lengths(vectors)
        forces = self._edges_forces(q, lengths)

        return EquilibriumState(xyz=xyz_eq,
                                residuals=residuals,
                                vectors=vectors,
                                lengths=lengths,
                                forces=forces,
                                force_densities=q)
