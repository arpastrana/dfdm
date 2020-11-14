#!/usr/bin/env python3

"""
Equilibrate the force network using the traditional force density method.
"""

# filepath stuff
import os

# numpy
import jax.numpy as np

# visualization matters
from compas.datastructures import Network
from compas.geometry import Line
from compas.utilities import rgb_to_hex

from compas_viewers.objectviewer import ObjectViewer

# force equilibrium
from force_density import JSON

from force_density.equilibrium import force_equilibrium

from force_density.network import CompressionNetwork

from force_density.losses import MeanSquaredError

from force_density.goals import PointGoal

from force_density.optimization import Optimizer

# ==========================================================================
# Initial parameters
# ==========================================================================

JSON_IN = os.path.abspath(os.path.join(JSON, "compression_network.json"))
JSON_OUT = os.path.abspath(os.path.join(JSON, "compression_network_opt.json"))

export_json = True

verbose = False
iters = 1000 # 1000
lr = 0.1 # 0.1, 1.0, 2.5, 5.0  # cross validation of lambda! sensitive here

loss_f = MeanSquaredError()

# ==========================================================================
# Load Network with boundary conditions from JSON
# ==========================================================================

network = CompressionNetwork.from_json(JSON_IN)
reference_network = network.copy()

# ==========================================================================
# Create goals
# ==========================================================================

goals = [reference_network.node_coordinates(n) for n in network.free_nodes()]
goals = np.array(goals)

# ==========================================================================
# Optimization
# ==========================================================================

optimizer = Optimizer(network, goals)
q_opt, xyz_opt = optimizer.solve(lr, iters, loss_f)

# ==========================================================================
# Update network xyz coordinates
# ==========================================================================

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
