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
from force_density.network import CompressionNetwork
from force_density.equilibrium import force_equilibrium

# ==========================================================================
# Initial parameters
# ==========================================================================

JSON_IN = os.path.abspath(os.path.join(JSON, "compression_network.json"))
JSON_OUT = os.path.abspath(os.path.join(JSON, "compression_network_eq.json"))

export_json = True

# ==========================================================================
# Load Network with boundary conditions from JSON
# ==========================================================================

network = CompressionNetwork.from_json(JSON_IN)
reference_network = network.copy()

# ==========================================================================
# Force Density - Extract data
# ==========================================================================

# node key: index mapping
k_i = network.key_index()

# find supports
fixed = [k_i[key] for key in network.supports()]

# find free nodes
free = [k_i[key] for key in network.free_nodes()]

# edges
edges = [(k_i[u], k_i[v]) for u, v in network.edges()]

# node coordinates
xyz = np.array(network.nodes_xyz())

# force densities
q = np.array(network.force_densities())

# forces
loads = np.array(network.applied_load())

# ==========================================================================
# Force Density
# ==========================================================================

xyz = force_equilibrium(q, edges, xyz, free, fixed, loads)

# ==========================================================================
# Update Geometry
# ==========================================================================

network.nodes_xyz(xyz.tolist())

# ==============================================================================
# Export new JSON file for further processing
# ==============================================================================

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
