import autograd.numpy as np

from compas.datastructures import network_adjacency_matrix
from compas.numerical import connectivity_matrix


# ==========================================================================
# Structure
# ==========================================================================

class EquilibriumStructure:
    def __init__(self, network):
        self._network = network

        self._connectivity = None
        self._adjacency = None

        self._free_nodes = None
        self._fixed_nodes = None
        self._freefixed_nodes = None

        self._node_index = None
        self._edge_index = None

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
    def adjacency(self):
        """
        The adjacency of the network encoded as a 2D matrix.
        """
        if self._adjacency is None:
            self._adjacency = np.array(network_adjacency_matrix(self.network, "list"), dtype=np.int32)
        return self._adjacency

    @property
    def connectivity(self):
        """
        The connectivity of the network encoded as a branch-node matrix.
        """
        if self._connectivity is None:
            node_idx = self.node_index
            edges = [(node_idx[u], node_idx[v]) for u, v in self.network.edges()]
            self._connectivity = np.array(connectivity_matrix(edges, "list"), dtype=np.int32)
        return self._connectivity

    @property
    def free_nodes(self):
        """
        Returns a list with the keys of the anchored nodes.
        """
        if not self._free_nodes:
            self._free_nodes = [self.node_index[node] for node in self.network.nodes_free()]
        return self._free_nodes

    @property
    def fixed_nodes(self):
        """
        Returns a list with the keys of the anchored nodes.
        """
        if not self._fixed_nodes:
            self._fixed_nodes = [self.node_index[node] for node in self.network.nodes_supports()]
        return self._fixed_nodes

    @property
    def freefixed_nodes(self):
        """
        A list with the node keys of all the nodes sorted by their node index.
        TODO: this method must be refactored to be more transparent.
        """
        if not self._freefixed_nodes:
            freefixed_nodes = self.free_nodes + self._fixed_nodes
            indices = {node: index for index, node in enumerate(freefixed_nodes)}
            sorted_indices = []
            for _, index in sorted(indices.items(), key=lambda item: item[0]):
                sorted_indices.append(index)
            self._freefixed_nodes = sorted_indices

        return self._freefixed_nodes
