"""
Solve a constrained force density problem using gradient based optimization.
"""

# filepath stuff
import os

# visualization matters
from compas.colors import Color
from compas.datastructures import Network
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import add_vectors
from compas.geometry import scale_vector
from compas.geometry import length_vector

from compas_view2.app import App

# force equilibrium
from force_density import JSON
from force_density.equilibrium import ForceDensity
from force_density.network import CompressionNetwork
from force_density.losses import SquaredError
from force_density.goals import LengthGoal
from force_density.optimization import Optimizer

# ==========================================================================
# Initial parameters
# ==========================================================================

view = True
JSON_IN = os.path.abspath(os.path.join(JSON, "compression_network.json"))

# ==========================================================================
# Load Network with boundary conditions from JSON
# ==========================================================================

network = CompressionNetwork.from_json(JSON_IN)
reference_network = network.copy()

# ==========================================================================
# Re-run force density to update model
# ==========================================================================

from force_density.equilibrium import fdm
new_network = fdm(network)

# ==========================================================================
# Re-run force density to update model
# ==========================================================================

q = network.force_densities()
fd = ForceDensity()(q, network)

# ==========================================================================
# Update geometry
# ==========================================================================

xyz = fd["xyz"].tolist()
lengths = fd["lengths"].tolist()
residuals = fd["residuals"].tolist()

# update xyz coordinates on nodes
network.nodes_xyz(xyz)

# update q values and lengths on edges
for idx, edge in enumerate(network.edges()):
    network.edge_attribute(edge, name="length", value=lengths[idx])

# update residuals on nodes
for idx, node in enumerate(network.nodes()):
    for name, value in zip(["rx", "ry", "rz"], residuals[idx]):
        network.node_attribute(node, name=name, value=value)

# ==========================================================================
# Viewer
# ==========================================================================

if view:
    viewer = App(width=1600, height=900)

    viewer.add(new_network,
               show_vertices=False,
               show_edges=True,
               linecolor=Color.pink(),
               linewidth=4.0)

    # equilibrated arch
    viewer.add(network,
               show_vertices=True,
               pointsize=12.0,
               show_edges=True,
               linecolor=Color.teal(),
               linewidth=4.0)

    # reference arch
    viewer.add(reference_network, show_points=False, linewidth=4.0)

    # draw lines betwen subject and target nodes
    for node in network.nodes():

        pt = network.node_coordinates(node)
        target_pt = reference_network.node_coordinates(node)
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
