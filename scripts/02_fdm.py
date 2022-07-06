#!/usr/bin/env python3

"""
Equilibrate the force network using the traditional force density method.
"""

# filepath stuff
import os

# numpy
import autograd.numpy as np

# visualization matters
from compas.colors import Color
from compas.datastructures import Network
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import length_vector
from compas.geometry import add_vectors
from compas.geometry import scale_vector
from compas.utilities import rgb_to_hex

from compas_view2.app import App

# force equilibrium
from force_density import JSON
from force_density.network import CompressionNetwork
from force_density.equilibrium import ForceDensity

# ==========================================================================
# Initial parameters
# ==========================================================================

JSON_IN = os.path.abspath(os.path.join(JSON, "compression_network.json"))
JSON_OUT = os.path.abspath(os.path.join(JSON, "compression_network_eq.json"))

export_json = False

# ==========================================================================
# Load Network with boundary conditions from JSON
# ==========================================================================

network = CompressionNetwork.from_json(JSON_IN)
reference_network = network.copy()
print(f"Num nodes = {network.number_of_nodes()}, num edges = {network.number_of_edges()}")

# ==========================================================================
# Force Density - Extract data
# ==========================================================================

q = np.array(network.force_densities())

# ==========================================================================
# Force Density
# ==========================================================================

fd_state = ForceDensity()(q, network)

# ==========================================================================
# Update Geometry
# ==========================================================================

xyz = fd_state["xyz"].tolist()
length = fd_state["lengths"].tolist()
res = fd_state["residuals"].tolist()

# update xyz coordinates on nodes
network.nodes_xyz(xyz)

# update q values and lengths on edges
for idx, edge in enumerate(network.edges()):
    network.edge_attribute(edge, name="length", value=length[idx])

# update residuals on nodes
for idx, node in enumerate(network.nodes()):
    for name, value in zip(["rx", "ry", "rz"], res[idx]):
        network.node_attribute(node, name=name, value=value)

# ==============================================================================
# Export new JSON file for further processing
# ==============================================================================

if export_json:
    network.to_json(JSON_OUT)
    print("Exported network to: {}".format(JSON_OUT))

# ==========================================================================
# Viewer
# ==========================================================================

# viewer = App(width=1600, height=900)

# # equilibrated arch
# viewer.add(network,
#            show_vertices=True,
#            pointsize=12.0,
#            show_edges=True,
#            linecolor=Color.teal(),
#            linewidth=4.0)

# # reference arch
# viewer.add(reference_network, show_points=False)

# # draw supports
# for node in network.supports():
#     x, y, z = network.node_coordinates(node)
#     viewer.add(Point(x, y, z), color=Color.green(), size=20)

# # show le crème
# viewer.show()
