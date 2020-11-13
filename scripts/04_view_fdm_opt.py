#!/usr/bin/env python3

"""
Visualize an optimized network.
"""

# filepath stuff
import os

# visualization matters
from compas.datastructures import Network
from compas.geometry import Line
from compas.geometry import scale_vector
from compas.geometry import add_vectors
from compas.geometry import normalize_vector
from compas.geometry import subtract_vectors
from compas.utilities import rgb_to_hex

from compas_viewers.objectviewer import ObjectViewer

# force equilibrium
from force_density import JSON

from force_density.network import CompressionNetwork


# ==========================================================================
# Initial parameters
# ==========================================================================

JSON_REF = os.path.abspath(os.path.join(JSON, "compression_network.json"))
JSON_IN = os.path.abspath(os.path.join(JSON, "compression_network_opt.json"))

scale = 1.0

# ==========================================================================
# Load Network with boundary conditions from JSON
# ==========================================================================

network = CompressionNetwork.from_json(JSON_IN)
reference_network = CompressionNetwork.from_json(JSON_REF)

# ==========================================================================
# Exaggerate deformation
# ==========================================================================

for node in network.free_nodes():

    reference = network.node_coordinates(node)
    target = reference_network.node_coordinates(node)
    deformation_vector = subtract_vectors(target, reference)
    new_xyz = add_vectors(reference, scale_vector(deformation_vector, scale))

    network.node_attributes(key=node, names="xyz", values=new_xyz)

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
