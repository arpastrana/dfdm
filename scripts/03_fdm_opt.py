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
from force_density.losses import squared_error
from force_density.goals import LengthGoal
from force_density.goals import PointGoal
from force_density.optimization import SLSQP
from force_density.equilibrium import constrained_fdm

# ==========================================================================
# Initial parameters
# ==========================================================================

JSON_IN = os.path.abspath(os.path.join(JSON, "compression_network.json"))
JSON_OUT = os.path.abspath(os.path.join(JSON, "compression_network_opt_2.json"))

export_json = False
view = True

# ==========================================================================
# Load Network with boundary conditions from JSON
# ==========================================================================

network = CompressionNetwork.from_json(JSON_IN)
reference_network = network.copy()

# ==========================================================================
# Define goals
# ==========================================================================

goals = []
parameters = []
constraints = []

goals.append(PointGoal(node_key=0, point=network.node_coordinates(0)))
for edge in network.edges():
    target_length = reference_network.edge_length(*edge)
    goals.append(LengthGoal(edge, target_length))  # length goal

# ==========================================================================
# Optimization
# ==========================================================================

network = constrained_fdm(network,
                          optimizer=SLSQP(),
                          loss=squared_error,
                          goals=goals,
                          bounds=(None, -0.01795/0.123),
                          maxiter=200,
                          tol=1e-9)

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
    viewer = App(width=1600, height=900)

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
