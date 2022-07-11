"""
Solve a constrained force density problem using gradient based optimization.
"""

# filepath stuff
import os

# differentiable numpy
import autograd.numpy as np

# geometry processing
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import add_vectors
from compas.geometry import length_vector

# visualization mattters
from compas.colors import Color
from compas_view2.app import App

# differentiable fdm
from force_density import JSON

from force_density.datastructures import CompressionNetwork  # datastructures

from force_density.losses import squared_loss
from force_density.losses import squared_loss

from force_density.equilibrium import constrained_fdm

from force_density.goals import LengthGoal
from force_density.goals import PointGoal

from force_density.optimization import SLSQP
from force_density.optimization import BFGS

# ==========================================================================
# Initial parameters
# ==========================================================================

view = True
JSON_IN = os.path.abspath(os.path.join(JSON, "compression_network.json"))

# ==========================================================================
# Load Network with boundary conditions from JSON
# ==========================================================================

network = CompressionNetwork.from_json(JSON_IN)

# ==========================================================================
# Define goals
# ==========================================================================

goals = []

goals.append(PointGoal(node_key=0, point=network.node_coordinates(0)))

for edge in network.edges():
    target_length = network.edge_length(*edge)
    goals.append(LengthGoal(edge, target_length))

# ==========================================================================
# Optimization
# ==========================================================================

constrained_network = constrained_fdm(network,
                                      optimizer=SLSQP(),
                                      loss=squared_loss,
                                      goals=goals,
                                      bounds=(-np.inf, 0.0),
                                      maxiter=200,
                                      tol=1e-9)

# ==========================================================================
# Visualization
# ==========================================================================

if view:
    viewer = App(width=1600, height=900)

    # equilibrated arch
    viewer.add(constrained_network,
               show_vertices=True,
               pointsize=12.0,
               show_edges=True,
               linecolor=Color.teal(),
               linewidth=4.0)

    # reference arch
    viewer.add(network, show_points=False, linewidth=4.0)

    # draw lines betwen subject and target nodes
    for node in constrained_network.nodes():

        pt = network.node_coordinates(node)
        target_pt = constrained_network.node_coordinates(node)
        viewer.add(Line(target_pt, pt))

        # draw reaction forces
        residual = constrained_network.node_attributes(node, names=["rx", "ry", "rz"])

        if length_vector(residual) < 0.001:
            continue

        residual_line = Line(pt, add_vectors(pt, residual))
        viewer.add(residual_line, color=Color.purple())

    # draw supports
    for node in constrained_network.supports():
        x, y, z = constrained_network.node_coordinates(node)
        viewer.add(Point(x, y, z), color=Color.green(), size=20)


    # show le crème
    viewer.show()
