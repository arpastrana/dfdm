"""
Solve a constrained force density problem using gradient based optimization.
"""

# filepath stuff
import os

# visualization matters
from compas.colors import Color
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import add_vectors
from compas.geometry import length_vector
from compas_view2.app import App

# force equilibrium
from force_density import JSON
from force_density.datastructures import CompressionNetwork
from force_density.equilibrium import fdm

# ==========================================================================
# Initial parameters
# ==========================================================================

view = True
JSON_IN = os.path.abspath(os.path.join(JSON, "compression_network.json"))

# ==========================================================================
# Load Network with boundary conditions from JSON
# ==========================================================================

network_start = CompressionNetwork.from_json(JSON_IN)

# ==========================================================================
# Re-run force density to update model
# ==========================================================================

network = fdm(network_start)

# ==========================================================================
# Viewer
# ==========================================================================

if view:
    viewer = App(width=1600, height=900)

    viewer.add(network,
               show_vertices=False,
               show_edges=True,
               linecolor=Color.pink(),
               linewidth=4.0)

    # reference arch
    viewer.add(network_start, show_points=False, linewidth=4.0)

    # draw lines betwen subject and target nodes
    for node in network.nodes():

        pt = network_start.node_coordinates(node)
        target_pt = network.node_coordinates(node)
        viewer.add(Line(target_pt, pt))

        # draw reaction forces
        residual = network.node_attributes(node, names=["rx", "ry", "rz"])

        if length_vector(residual) < 0.001:
            continue

        residual_line = Line(pt, add_vectors(pt, residual))
        viewer.add(residual_line, color=Color.purple())

    # draw supports
    for node in network.supports():
        x, y, z = network.node_coordinates(node)
        viewer.add(Point(x, y, z), color=Color.green(), size=20)

    # show le crème
    viewer.show()
