#!/usr/bin/env python3

"""
Equilibrate the force network using the traditional force density method.
"""

# filepath stuff
import os

# visualization matters
from compas.datastructures import Network
from compas.geometry import Line
from compas.geometry import add_vectors
from compas.geometry import scale_vector
from compas.geometry import length_vector
from compas.utilities import rgb_to_hex

from compas_viewers.objectviewer import ObjectViewer

# force equilibrium
from force_density import JSON

from force_density.equilibrium import ForceDensity

from force_density.network import CompressionNetwork

from force_density.losses import SquaredError

from force_density.goals import LineGoal

from force_density.optimization import Optimizer

# ==========================================================================
# Initial parameters
# ==========================================================================

JSON_IN = os.path.abspath(os.path.join(JSON, "compression_network.json"))
JSON_OUT = os.path.abspath(os.path.join(JSON, "compression_network_opt.json"))

export_json = True
view = False

ub = -0.01795 / 0.123  # point load / brick length
method = "SLSQP"
maxiter = 200
tol = 1e-9
scale = 5

# ==========================================================================
# Load Network with boundary conditions from JSON
# ==========================================================================

network = CompressionNetwork.from_json(JSON_IN)
reference_network = network.copy()

loss_f = SquaredError()

# ==========================================================================
# Create goals
# ==========================================================================

node_goals = []
edge_goals = []

for idx, edge in enumerate(network.edges()):
    target_length = reference_network.edge_length(*edge)
    edge_goals.append(LineGoal(idx, target_length))

# ==========================================================================
# Optimization
# ==========================================================================

optimizer = Optimizer(network, node_goals, edge_goals)
q_opt = optimizer.solve_scipy(loss_f, ub, method, maxiter, tol)

# ==========================================================================
# Force Density
# ==========================================================================

fd = ForceDensity()
fd_opt = fd(q_opt, network)

# ==========================================================================
# Update Geometry
# ==========================================================================

xyz_opt = fd_opt["xyz"].tolist()
length_opt = fd_opt["lengths"].tolist()
res_opt = fd_opt["residuals"].tolist()

# update xyz coordinates on nodes
network.nodes_xyz(xyz_opt)

# update q values and lengths on edges
for idx, edge in enumerate(network.edges()):
    network.edge_attribute(edge, name="q", value=q_opt[idx])
    network.edge_attribute(edge, name="length", value=length_opt[idx])

# update residuals on nodes
for idx, node in enumerate(network.nodes()):
    for name, value in zip(["rx", "ry", "rz"], res_opt[idx]):
        network.node_attribute(node, name=name, value=value)

# ==========================================================================
# Export new JSON file for further operations
# ==========================================================================

if export_json:
    network.to_json(JSON_OUT)
    print("Exported network to: {}".format(JSON_OUT))

# ==========================================================================
# Viewer
# ==========================================================================

if view:

    viewer = ObjectViewer()
    network_viz = network
    t_network_viz = reference_network

    # blue is target, red is subject
    viewer.add(network_viz, settings={'edges.color': rgb_to_hex((255, 0, 0)),
                                      'vertices.size': 10,
                                      'edges.width': 2,
                                      'opacity': 0.7,
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

        residual = network_viz.node_attributes(node, names=["rx", "ry", "rz"])

        if length_vector(residual) < 0.001:
            continue

        residual_line = Line(pt, add_vectors(pt, scale_vector(residual, scale)))
        viewer.add(residual_line)

    # draw supports
    supports_network = Network()
    for node in network_viz.supports():
        x, y, z = network_viz.node_coordinates(node)
        supports_network.add_node(node, x=x, y=y, z=z)

    viewer.add(supports_network, settings={'vertices.size': 10,
                                           'vertices.on': True,
                                           'edges.on': False})

    # show le crème
    viewer.show()
