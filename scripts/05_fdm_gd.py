#!/usr/bin/env python3

"""
Best-fit of the Force Density Method using Automatic Differentiation.
"""

# optimization stuff
import jax.numpy as np
from jax import grad

# visualization matters
from compas.datastructures import Network
from compas.geometry import Line
from compas.utilities import rgb_to_hex
from compas_viewers.objectviewer import ObjectViewer

# force equilibrium
from force_density.network import CompressionNetwork
from force_density.equilibrium import force_equilibrium
from force_density.losses import mean_squared_error


def target_xyz(network, keys):
    """
    Fabricates the xyz coordinates of the target nodes of a network.
    """
    target_points = [network.node_coordinates(node) for node in keys]
    # return np.array(target_points)[:, 2].reshape((-1, 1))
    return np.array(target_points).reshape((-1, 3))


if __name__ == "__main__":

    from time import time


    HERE = "/Users/arpj/princeton/phd/projects/light_vault/io/central_arch_light_vault.json"

    gradient = True
    verbose = False
    iters = 1000 # 1000
    lr = 0.1 # 0.1, 1.0, 2.5, 5.0  # cross validation of lambda! sensitive here

    pz = -0.01795 # netwons - TODO: update as position changes?

    q_0 = -2.5  # -2.5
    brick_length = 0.123  # m
    q_0_cantilever = pz / brick_length

    extra_support = None

    # ==========================================================================
    # Network
    # ==========================================================================

    # load network
    arch = CompressionNetwork.from_json(HERE)

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
    arch.force_densities(q_0, keys=arch.non_cantilevered_edges())

    # set initial q to cantilevered edges
    arch.force_densities(q_0_cantilever, keys=arch.cantilevered_edges())

    # set initial point loads to all nodes of the network
    arch.applied_load([0.0, 0.0, pz])

    # add extra supports
    if extra_support:
        arch.supports([extra_support])

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

    xyz = force_equilibrium(q, edges, xyz, free, fixed, loads)

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

    grad_loss = grad(mean_squared_error)

    if gradient:

        print("Optimization started...")

        errors = []
        start_time = time()

        for k in range(iters):

            error = mean_squared_error(q, sn, edges, xyz, free, fixed, loads)
            errors.append(error)

            if verbose:
                print("Iteration: {} \t Loss: {}".format(k, error))

            q_grad = grad_loss(q, sn, edges, xyz, free, fixed, loads)
            q = q - lr * q_grad

            # do fd and update network
            xyz = force_equilibrium(q, edges, xyz, free, fixed, loads)

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
    viewer.show()
