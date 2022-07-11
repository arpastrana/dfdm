"""
Solve a constrained force density problem using gradient based optimization.
"""
import autograd.numpy as np

# compas
from compas.colors import Color
from compas.geometry import Line
from compas.geometry import Vector
from compas.geometry import Point
from compas.geometry import Polyline
from compas.geometry import add_vectors
from compas.geometry import length_vector
from compas.utilities import geometric_key

# visualization
from compas_view2.app import App

# static equilibrium
from force_density.equilibrium import constrained_fdm

from force_density.datastructures import CompressionNetwork
from force_density.losses import squared_loss
from force_density.goals import LengthGoal
from force_density.goals import ResidualForceGoal
from force_density.goals import ResidualVectorGoal
from force_density.optimization import SLSQP

# ==========================================================================
# Initial parameters
# ==========================================================================

arch_length = 5.0
num_segments = 10
q_init = -1
pz = -0.1

# ==========================================================================
# Create the geometry of a catenary curve
# ==========================================================================

start = [0.0, 0.0, 0.0]
end = add_vectors(start, [arch_length, 0.0, 0.0])
curve = Polyline([start, end])
points = curve.divide_polyline(num_segments)
lines = Polyline(points).lines

# ==========================================================================
# Create arch
# ==========================================================================

network = CompressionNetwork.from_lines(lines)

# ==========================================================================
# Define structural system
# ==========================================================================

# assign supports
gkeys = [geometric_key(point) for point in [start, end]]
supports = [node for node in network.nodes() if geometric_key(network.node_coordinates(node)) in gkeys]
network.supports(keys=supports)

# set initial q to all nodes
network.force_densities(q_init, keys=None)

# set initial point loads to all nodes of the network
network.applied_load([0.0, 0.0, pz], keys=[node for node in network.nodes() if node not in supports])

# ==========================================================================
# Define goals
# ==========================================================================

goals = []

key_index = network.key_index()
index = key_index.get(0)
# goals.append(ResidualForceGoal(index, 1.0))
goals.append(ResidualVectorGoal(index, [-1.0, 0.0, -1.0]))

for edge in network.edges():
    target_length = network.edge_length(*edge)
    goals.append(LengthGoal(edge, target_length))  # length goal

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


for node in constrained_network.nodes():

    pt = constrained_network.node_coordinates(node)

    # draw lines betwen subject and target nodes
    target_pt = network.node_coordinates(node)
    viewer.add(Line(target_pt, pt))

    # draw residual forces
    residual = constrained_network.node_attributes(node, names=["rx", "ry", "rz"])

    if length_vector(residual) < 0.001:
        continue

    print(node, residual, length_vector(residual))
    residual_line = Line(pt, add_vectors(pt, residual))
    viewer.add(residual_line,
               linewidth=2.0,
               color=Color.purple())

    # viewer.add(Vector(*residual),
    #            color=Color.black(),
    #            position=pt
    #            )

# draw applied loads
for node in constrained_network.nodes():
    pt = constrained_network.node_coordinates(node)
    load = network.node_attributes(node, names=["px", "py", "pz"])
    viewer.add(Line(pt, add_vectors(pt, load)),
               linewidth=2.0,
               color=Color.green().darkened())

# draw supports
for node in constrained_network.supports():

    x, y, z = constrained_network.node_coordinates(node)
    viewer.add(Point(x, y, z), color=Color.green(), size=20)


# show le crème
viewer.show()
