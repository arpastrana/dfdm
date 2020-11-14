#!/usr/bin/env python3

"""
Equilibrate the force network using the traditional force density method.
"""

# filepath stuff
import os

# visualization matters
from compas.datastructures import Network
from compas.geometry import Line
from compas.utilities import rgb_to_hex

from compas_viewers.objectviewer import ObjectViewer

# force equilibrium
from force_density import JSON

from force_density.equilibrium import ForceDensity

from force_density.network import CompressionNetwork

from force_density.losses import MeanSquaredError
from force_density.losses import MeanSquaredErrorGoals

from force_density.goals import PointGoal

from force_density.optimization import Optimizer

# ==========================================================================
# Initial parameters
# ==========================================================================

JSON_IN = os.path.abspath(os.path.join(JSON, "compression_network.json"))
JSON_OUT = os.path.abspath(os.path.join(JSON, "compression_network_opt.json"))

export_json = True

verbose = False
iters = 10 # 1000
lr = 0.1 # 0.1, 1.0, 2.5, 5.0  # cross validation of lambda! sensitive here

# ==========================================================================
# Load Network with boundary conditions from JSON
# ==========================================================================

network = CompressionNetwork.from_json(JSON_IN)
reference_network = network.copy()

loss_f = MeanSquaredErrorGoals()

# ==========================================================================
# Create goals
# ==========================================================================

goals = []
for node in network.free_nodes():
    target_xyz = reference_network.node_coordinates(node)
    goals.append(PointGoal(node, target_xyz))

# ==========================================================================
# Optimization
# ==========================================================================

optimizer = Optimizer(network, goals)
q_opt = optimizer.solve(lr, iters, loss_f)

# ==========================================================================
# Update network xyz coordinates
# ==========================================================================

fd = ForceDensity()
xyz_opt = fd(q_opt, network)
network.nodes_xyz(xyz_opt.tolist())

# ==========================================================================
# Export new JSON file for further operations
# ==========================================================================

if export_json:
    network.to_json(JSON_OUT)
    print("Exported network to: {}".format(JSON_OUT))

# ==========================================================================
# Viewer
# ==========================================================================

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

# draw supports
supports_network = Network()
for node in network_viz.supports():
    x, y, z = network_viz.node_coordinates(node)
    supports_network.add_node(node, x=x, y=y, z=z)

viewer.add(supports_network, settings={
    'vertices.size': 10,
    'vertices.on': True,
    'edges.on': False
})

# show le crème
viewer.show()
