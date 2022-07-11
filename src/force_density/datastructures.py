"""
A catalogue of force networks.
"""
from compas.datastructures import Network


__all__ = ["CompressionNetwork"]


class CompressionNetwork(Network):
    """
    A compression-only structural network.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize some values at instantiation.
        """
        super(CompressionNetwork, self).__init__(*args, **kwargs)

        self.name = "Funicular Network"

        self.update_default_node_attributes({"x": 0.0,
                                             "y": 0.0,
                                             "z": 0.0,
                                             "px": 0.0,
                                             "py": 0.0,
                                             "pz": 0.0,
                                             "rx": 0.0,
                                             "ry": 0.0,
                                             "rz": 0.0,
                                             "is_support": False})

        self.update_default_edge_attributes({"q": 0.0,
                                             "length": 0.0,
                                             "force": 0.0})

    def node_xyz(self, xyz=None, keys=None):
        return self.nodes_xyz(xyz, keys)

    def nodes_xyz(self, xyz=None, keys=None):
        """
        Gets or sets the node coordinates.
        """
        if xyz is None:
            return self.nodes_attributes(names="xyz", keys=keys)

        if not keys:
            keys = self.nodes()

        for key, values in zip(keys, xyz):
            self.node_attributes(key, names="xyz", values=values)

    def free_nodes(self):
        """
        The keys of the nodes where there is no support assigned.
        """
        return self.nodes_where({"is_support": False})

    def supports(self, keys=None):
        """
        Gets or sets the node keys where a support has been assigned.

        TODO: Currently ambiguous method!
        """
        if keys is None:
            return self.nodes_where({"is_support": True})

        return self.nodes_attribute(name="is_support", value=True, keys=keys)

    def force_densities(self, value=None, keys=None):
        """
        Gets or sets the force densities on a list of edges.
        """
        return self.edges_attribute(name="q", value=value, keys=keys)

    def force_density(self, key, value=None):
        """
        Gets or sets the force density on a single edge.
        """
        return self.edge_attribute(name="q", value=value, key=key)

    def node_loads(self, load=None, keys=None):
        """
        Gets or sets a load to the nodes of the network.
        """
        return self.nodes_attributes(names=("px", "py", "pz"), values=load, keys=keys)

    def applied_load(self, load=None, keys=None):
        """
        Gets or sets a load to the nodes of the network.
        """
        return self.nodes_attributes(names=("px", "py", "pz"), values=load, keys=keys)

    def residual_forces(self, keys=None):
        """
        Gets the residual forces of the nodes of the network.
        """
        return self.nodes_attributes(names=("rx", "ry", "rz"), keys=keys)

    def residual_force(self, key):
        """
        Gets the residual force of a single node of the network.
        """
        return self.node_attributes(key=key, names=("rx", "ry", "rz"))

    def edge_forces(self, keys=None):
        """
        Gets the forces at the edges of the network.
        """
        return self.edges_attribute(keys=keys, name="force")

    def edge_force(self, key):
        """
        Gets the forces at a single edge the network.
        """
        return self.edge_attribute(key=key, name="force")

    def cantilevered_nodes(self):
        """
        Gets the keys of all the support-free leaf nodes.
        """
        for key in set(self.leaves()) - set(self.supports()):
            yield key

    def cantilevered_edges(self):
        """
        Gets the keys of the edges which are connected only to another edge.
        """
        for node in self.cantilevered_nodes():
            edges = self.connected_edges(node)
            if len(edges) == 1:
                yield edges.pop()

    def non_cantilevered_edges(self):
        """
        Gets the keys of the edges which are connected to more than one edge.
        """
        for edge in set(self.edges()) - set(self.cantilevered_edges()):
            yield edge
