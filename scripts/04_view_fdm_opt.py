#!/usr/bin/env python3

"""
Visualize an optimized network.
"""
from math import fabs

# filepath stuff
import os

# visualization matters
from compas.datastructures import Network
from compas.geometry import Line
from compas.geometry import scale_vector
from compas.geometry import add_vectors
from compas.geometry import length_vector
from compas.geometry import subtract_vectors
from compas.geometry import distance_point_point
from compas.utilities import rgb_to_hex

from compas_viewers.objectviewer import ObjectViewer

# force equilibrium
from force_density import JSON

from force_density.network import CompressionNetwork


# ==========================================================================
# Initial parameter
# ==========================================================================

JSON_REF = os.path.abspath(os.path.join(JSON, "arch_state_17.json"))
JSON_IN = os.path.abspath(os.path.join(JSON, "arch_state_17/33/compression_network_opt.json"))

scale = 5.0

# ==========================================================================
# Load Network with boundary conditions from JSON
# ==========================================================================

network = CompressionNetwork.from_json(JSON_IN)
reference_network = CompressionNetwork.from_json(JSON_REF)

# compare lengths
error = 0.0
for edge in network.edges():

    length = network.edge_length(*edge)
    target_length = reference_network.edge_length(*edge)
    difference = fabs(length - target_length)
    error_length = (difference)**2

    relative_difference = difference / target_length
    if relative_difference > 0.01:
        print(f"Edge {edge} relative difference: {round(100 * relative_difference, 2)} %")

    error += error_length

print(f"Squared error: {error}")


# ==========================================================================
# Viewer
# ==========================================================================

viewer = ObjectViewer()
network_viz = network
t_network_viz = reference_network

# blue is target, red is subject
viewer.add(network_viz, settings={'edges.color': rgb_to_hex((255, 0, 0)),
                                  'edges.width': 2,
                                  'vertices.size': 10,
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
distance_error = 0.0
for node in network_viz.nodes():

    pt = network_viz.node_coordinates(node)
    target_pt = t_network_viz.node_coordinates(node)
    distance_error += distance_point_point(pt, target_pt)

    viewer.add(Line(target_pt, pt))

    residual = network_viz.node_attributes(node, names=["rx", "ry", "rz"])

    if length_vector(residual) < 0.001:
        continue

    residual_line = Line(pt, add_vectors(pt, scale_vector(residual, scale)))
    viewer.add(residual_line)

print(f"Nodes distance error: {distance_error}")

# draw supports
supports_network = Network()
for node in network_viz.supports():
    x, y, z = network_viz.node_coordinates(node)
    supports_network.add_node(node, x=x, y=y, z=z)

viewer.add(supports_network, settings={
    'vertices.size': 20,
    'vertices.on': True,
    'edges.on': False
})

# show le crème
viewer.show()
