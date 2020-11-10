#!/usr/bin/env python3

"""
Best-fit of the Force Density Method using Automatic Differentiation.
"""

from abc import ABC
from abc import abstractmethod

import jax.numpy as np

from jax import grad
from jax import vmap

from jax.ops import index_update
from jax.ops import index

from compas.datastructures import Network

from compas.numerical import connectivity_matrix

from compas.geometry import Line

from compas.utilities import rgb_to_hex

from compas_viewers.objectviewer import ObjectViewer


def fd(q, edges, xyz, free, fixed, loads):
    """
    Jax-based force density method
    """

    c_matrix = connectivity_matrix(edges, "list")  # rtype=csr?
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


def reference_xyz(xyz, keys):
    """
    Gets the xyz coordinates of the reference nodes of a network.
    """
    return xyz[keys, 2].reshape(-1, 1)
    # return xyz[keys, :].reshape(-1, 3)


def target_xyz(network, keys):
    """
    Fabricates the xyz coordinates of the target nodes of a network.
    """
    target_points = [network.node_coordinates(node) for node in keys]
    return np.array(target_points)[:, 2].reshape((-1, 1))
    # return np.array(target_points).reshape((-1, 3))


def loss(q, targets, edges, xyz, free, fixed, loads):
    """
    A toy loss function.
    """
    xyz = fd(q, edges, xyz, free, fixed, loads)
    zn = reference_xyz(xyz, free)

    return np.sum(np.square(zn - targets))


class Loss(ABC):
    """
    The base class for all loss functions.
    """
    @abstractmethod
    def __call__(self):
        """
        Callable loss object
        """
        return


class FunicularNetwork(Network):
    """
    A compression-only structural network.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize some values at instantiation.
        """
        super(FunicularNetwork, self).__init__(*args, **kwargs)
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

        self.name = "Funicular Network"

    def nodes_xyz(self, xyz=None, keys=None):
        """
        Gets or sets the node coordinates.
        """
        return self.nodes_attributes(names="xyz", values=xyz, keys=keys)

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

    def connectivity_matrix(self, rtype="csr"):
        """
        The connectivity matrix of the edges of the network.
        """
        return connectivity_matrix(self.edges(), rtype)

    def force_densities(self, value=None, keys=None):
        """
        Gets or sets the force densities on a list of edges.
        """
        return self.edges_attribute(name="q", value=value, keys=keys)

    def sorted_nodes(self):
        """
        Yield all the sorted keys of the nodes of the network. One at a time.
        """
        for key in sorted(self.nodes()):
            yield key

    def applied_load(self, load=None, keys=None):
        """
        Gets or sets a load to the nodes of the network.
        """
        return self.nodes_attributes(names=("px", "py", "pz"), values=load, keys=keys)


if __name__ == "__main__":

    from time import time


    HERE = "/Users/arpj/princeton/phd/projects/light_vault/io/central_arch_light_vault.json"

    gradient = True
    verbose = False
    iters = 50 # 5000
    lr = 1.0 # 0.1, 1.0, 2.5, 5.0  # cross validation of lambda! sensitive here

    pz = -0.01795 # netwons - TODO: update as position changes?

    q_0 = -1.5  # -2.5

    # ==========================================================================
    # Network
    # ==========================================================================

    # load network
    arch = FunicularNetwork.from_json(HERE)

    # load network
    target_arch = arch.copy()

    # print some info out
    print(f"Funicular network # edges: {arch.number_of_edges()}")

    # ==========================================================================
    # Boundary Conditions
    # ==========================================================================

    # the z lower-most two nodes
    f_z_lowest = lambda x: arch.node_attribute(x, "z")
    fixed = sorted([node for node in arch.nodes()], key=f_z_lowest, reverse=True)
    fixed = fixed[-2:]

    # add supports
    arch.supports(fixed)

    # set initial q to all nodes - TODO: how to find best initial value?
    arch.force_densities(q_0)

    # set initial point loads to all nodes of the network
    arch.applied_load([0.0, 0.0, pz])

    # ==========================================================================
    # Force Density - Extract data
    # ==========================================================================

    sorted_nodes = list(arch.sorted_nodes())

    xyz = arch.nodes_xyz(keys=sorted_nodes)
    loads = arch.applied_load(keys=sorted_nodes)

    # free = list(arch.free_nodes())
    # fixed = list(arch.supports())
    fixed = [v for v in sorted_nodes if arch.node[v].get("is_support")]
    free = [v for v in sorted_nodes if not arch.node[v].get("is_support")]

    edges = list(arch.edges())
    q = arch.force_densities(keys=edges)

    # ==========================================================================
    # Target points
    # ==========================================================================

    sn = target_xyz(arch, free)

    # ==========================================================================
    # Force Density - Extract data
    # ==========================================================================

    xyz = np.array(xyz).reshape((-1, 3))
    q = np.array(q).reshape((-1, 1))
    loads = np.array(loads).reshape((-1, 3))

    # ==========================================================================
    # Force Density
    # ==========================================================================

    xyz = fd(q, edges, xyz, free, fixed, loads)

    # ==========================================================================
    # Update Geometry
    # ==========================================================================

    # arch.nodes_xyz(xyz.tolist(), keys=sorted_nodes)
    xyz_temp = xyz.tolist()
    for node in arch.nodes():
        arch.node_attributes(node, names="xyz", values=xyz_temp[node])

    # ==========================================================================
    # Gradient descent?
    # ==========================================================================

    grad_loss = grad(loss)

    if gradient:

        errors = []
        start_time = time()

        for k in range(iters):

            error = loss(q, sn, edges, xyz, free, fixed, loads)
            errors.append(error)

            if verbose:
                print("Iteration: {} \t Loss: {}".format(k, error))

            q_grad = grad_loss(q, sn, edges, xyz, free, fixed, loads)
            q = q - lr * q_grad

            # do fd and update network
            xyz = fd(q, edges, xyz, free, fixed, loads)

        # print out
        print("Output loss in {} iterations: {}".format(iters, errors[-1]))
        print("Elapsed time: {} seconds".format(time() - start_time))

        # update network coordinates after gradient descent
        xyz_temp = xyz.tolist()
        for node in arch.nodes():
            arch.node_attributes(node, names="xyz", values=xyz_temp[node])

    # ==========================================================================
    # Viewer
    # ==========================================================================

    viewer = ObjectViewer()
    network_viz = arch
    t_network_viz = target_arch

    # blue is target, red is subject
    viewer.add(network_viz, settings={'edges.color': rgb_to_hex((255, 0, 0)),
                                      'edges.width': 2,
                                      'opacity': 0.7,
                                      'vertices.size': 10,
                                      'vertices.on': False,
                                      'edges.on': True})

    viewer.add(t_network_viz, settings={'edges.color': rgb_to_hex((0, 0, 255)),
                                        'edges.width': 1,
                                        'opacity': 0.5,
                                        'vertices.size': 10,
                                        'vertices.on': False,
                                        'edges.on': True})

    # draw lines betwen subject and target nodes
    for node in network_viz.nodes():
        pt = network_viz.node_coordinates(node)
        target_pt = t_network_viz.node_coordinates(node)
        viewer.add(Line(target_pt, pt))

    # draw supports
    supports_network = Network()
    for node in fixed:
        x, y, z = network_viz.node_coordinates(node)
        supports_network.add_node(node, x=x, y=y, z=z)

    viewer.add(supports_network, settings={
        'vertices.size': 10,
        'vertices.on': True,
        'edges.on': False
    })

    # show le crème
    # viewer.show()
