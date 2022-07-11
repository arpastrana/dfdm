from dataclasses import dataclass

import autograd.numpy as np

from compas.numerical import connectivity_matrix


# ==========================================================================
# Initial parameters
# ==========================================================================

class EquilibriumModel:
    def __init__(self, network):
        self._network = network

        self._connectivity = None

        self._free_nodes = None
        self._fixed_nodes = None
        self._freefixed_nodes = None

        self._node_index = None
        self._edge_index = None
        self._freefixed_index = None

        # self._ordering_nodes = None
        # self._ordering_edges = None
        # self._ordering_free_fixed = None

    @property
    def network(self):
        """
        A COMPAS network.
        """
        return self._network

    @property
    def node_index(self):
        """
        A dictionary between node keys and their enumeration indices.
        """
        if not self._node_index:
            self._node_index = self.network.key_index()
        return self._node_index

    @property
    def edge_index(self):
        """
        A dictionary between edge keys and their enumeration indices.
        """
        if not self._edge_index:
            self._edge_index = self.network.uv_index()
        return self._edge_index

    @property
    def connectivity(self):
        """
        The connectivity of the network encoded as a branch-node list of lists.
        """
        if not self._connectivity:
            node_idx = self.node_index
            edges = [(node_idx[u], node_idx[v]) for u, v in self.network.edges()]
            self._connectivity = connectivity_matrix(edges, "list")
        return self._connectivity

    @property
    def free_nodes(self):
        """
        Returns a list with the keys of the anchored nodes.
        """
        if not self._free_nodes:
            self._free_nodes = [self.node_index[node] for node in self.network.free_nodes()]
        return self._free_nodes

    @property
    def fixed_nodes(self):
        """
        Returns a list with the keys of the anchored nodes.
        """
        if not self._fixed_nodes:
            self._fixed_nodes = [self.node_index[node] for node in self.network.supports()]
        return self._fixed_nodes

    @property
    def freefixed_nodes(self):
        """
        A list with the node keys of all the edges sorted by their node index.
        TODO: this method must be more transparent / clearer.
        """
        if not self._freefixed_nodes:
            freefixed_nodes = self.free_nodes + self._fixed_nodes
            indices = {node: index for index, node in enumerate(freefixed_nodes)}
            sorted_indices = []
            for _, index in sorted(indices.items(), key=lambda item: item[0]):
                sorted_indices.append(index)
            self._freefixed_nodes = sorted_indices

        return self._freefixed_nodes

# ==========================================================================
# Initial parameters
# ==========================================================================


class EquilibriumSolver:
    """
    The calculator.
    """
    def __init__(self, network):
        model = EquilibriumModel(network)
        self.model = model
        self.structure = model

    def edge_lengths(self, xyz):
        connectivity = self.structure.connectivity
        return np.linalg.norm(connectivity @ xyz, axis=1)

    def edge_forces(self, q, lengths):
        # TODO: is there a bug in edge forces?
        return q * lengths

    def node_residuals(self, q, loads, xyz):
        connectivity = self.structure.connectivity
        return loads - np.transpose(connectivity) @ np.diag(q) @ connectivity @ xyz

    def node_positions(self, q, loads, xyz):
        # convenience shorthands
        connectivity = self.structure.connectivity
        free = self.structure.free_nodes
        fixed = self.structure.fixed_nodes

        # Immutable stuff
        c_matrix = np.array(connectivity, dtype=np.int64)
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

        # TODO: free fixed indices sorted by enumeration - needs clearer solution
        indices = self.structure.freefixed_nodes

        # NOTE: concatenation is a workaround specific to autograd
        return np.concatenate((xyz_free, xyz_fixed))[indices]

    def __call__(self, q, loads, xyz):
        """
        Compute an equilibrium state using the force density method.
        """
        xyz_eq = self.node_positions(q, loads, xyz)
        residuals = self.node_residuals(q, loads, xyz_eq)
        lengths = self.edge_lengths(xyz_eq)
        forces = self.edge_forces(q, lengths)

        return EquilibriumState(xyz=xyz_eq, residuals=residuals, lengths=lengths, forces=forces)

# ==========================================================================
# Initial parameters
# ==========================================================================


# TODO: A method that reindexes state arrays to match network indexing

@dataclass
class EquilibriumState:
    xyz: np.ndarray
    residuals: np.ndarray
    lengths: np.ndarray
    forces: np.ndarray


# ==========================================================================
# Initial parameters
# ==========================================================================

def fdm(network):

    # get parameters
    q = np.array(network.force_densities(), dtype=np.float64)
    loads = np.array(list(network.node_loads()), dtype=np.float64)
    xyz = np.array(list(network.node_xyz()), dtype=np.float64)

    # compute static equilibrium
    # model = EquilibriumModel(network)
    solver = EquilibriumSolver(network)  # model can be instantiated in solver
    eq_state = solver(q, loads, xyz)

    # update equilibrium state in network copy
    return updated_network(network, eq_state)  # Network.update(eqstate)

# ==========================================================================
# Initial parameters
# ==========================================================================

def updated_network(network, eq_state):
    """
    Update in-place the attributes of a network with an equilibrium state.
    TODO: to be extra sure, the node-index and edge-index mappings should be handled
    by EquilibriumModel/EquilibriumStructure
    """
    network = network.copy()

    xyz = eq_state.xyz.tolist()
    lengths = eq_state.lengths.tolist()
    residuals = eq_state.residuals.tolist()
    forces = eq_state.forces.tolist()

    # update q values and lengths on edges
    for idx, edge in enumerate(network.edges()):
        network.edge_attribute(edge, name="length", value=lengths[idx])
        network.edge_attribute(edge, name="force", value=forces[idx])

    # update residuals on nodes
    for idx, node in enumerate(network.nodes()):
        for name, value in zip("xyz", xyz[idx]):
            network.node_attribute(node, name=name, value=value)

        for name, value in zip(["rx", "ry", "rz"], residuals[idx]):
            network.node_attribute(node, name=name, value=value)

    return network

# ==========================================================================
# Constrained fdm
# ==========================================================================


def constrained_fdm(network, optimizer, loss, goals, bounds, maxiter, tol):

    # optimizer works
    q = optimizer.minimize(network, loss, goals, bounds, maxiter, tol)

    # get parameters
    loads = np.array(list(network.node_loads()), dtype=np.float64)
    xyz = np.array(list(network.node_xyz()), dtype=np.float64)

    # compute static equilibrium
    solver = EquilibriumSolver(network)  # model can be instantiated in solver
    eq_state = solver(q, loads, xyz)

    # update equilibrium state in network copy
    return updated_network(network, eq_state)
