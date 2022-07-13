"""
Solve a constrained force density problem using gradient based optimization.
"""
import autograd.numpy as np

# compas
from compas.colors import Color
from compas.colors import ColorMap
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
from force_density.datastructures import CompressionNetwork

from force_density.equilibrium import fdm
from force_density.equilibrium import constrained_fdm

from force_density.goals import LengthGoal
from force_density.goals import LineGoal
from force_density.goals import ResidualForceGoal
from force_density.goals import ResidualVectorGoal
from force_density.goals import ResidualDirectionGoal

from force_density.losses import squared_loss
from force_density.optimization import SLSQP

# ==========================================================================
# Initial parameters
# ==========================================================================

length_arch = 5.0
num_segments = 10
q_init = -1
pz = -0.1
start = [0.0, 0.0, 0.0]

# ==========================================================================
# Create the geometry of a catenary curve
# ==========================================================================

points = []
length_segment = length_arch / num_segments
for i in range(num_segments + 1):
    point = add_vectors(start, [i * length_segment, 0.0, 0.0])
    points.append(point)

# ==========================================================================
# Create arch
# ==========================================================================

network = CompressionNetwork()

for idx, point in enumerate(points):
    x, y, z = point
    network.add_node(idx, x=x, y=y, z=z)

for u, v in zip(range(0, num_segments), range(1, num_segments + 1)):
    network.add_edge(u, v)

# ==========================================================================
# Define structural system
# ==========================================================================

# define supports
network.node_support(0)
network.node_support(num_segments)

# apply loads to unsupported nodes
for node in network.nodes_free():
    network.node_load(node, load=[0.0, 0.0, pz])

# set initial q to all nodes
for edge in network.edges():
    network.edge_forcedensity(edge, q_init)

# ==========================================================================
# Run the force density method
# ==========================================================================

# eq_network = fdm(network)

# ==========================================================================
# Visualization
# ==========================================================================

viewer = App(width=1600, height=900)

# reference arch
viewer.add(network, show_points=False, linewidth=4.0)

# color map
cmap = ColorMap.from_color(Color.teal(), rangetype="dark")
cmap = ColorMap.from_mpl("viridis")

# ==========================================================================
# Define goals
# ==========================================================================

constrained_networks = []
vertical_comps = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]

for idx, vertical_comp in enumerate(vertical_comps):
    vertical_comp = vertical_comp * -1.0
#   define goals
#   solve constrained_fdm
#   constrained_networks.append(constrained_network)

    goals = []
    goals.append(ResidualDirectionGoal(0, vector=[-1.0, 0.0, vertical_comp]))
    goals.append(ResidualDirectionGoal(num_segments, vector=[1.0, 0.0, vertical_comp]))

    # for edge in network.edges():
    #     target_length = network.edge_length(*edge)
    #     goals.append(LengthGoal(edge, target_length))

    # for node in network.nodes_free():
    #     start = network.node_coordinates(node)
    #     end = add_vectors(start, [0.0, 0.0, 10.0])
    #     goal = LineGoal(node, line=(start, end))
    #     goals.append(goal)

# ==========================================================================
# Optimization
# ==========================================================================

    print()
    constrained_network = constrained_fdm(network,
                                          optimizer=SLSQP(),
                                          loss=squared_loss,
                                          goals=goals,
                                          bounds=(-np.inf, 0.0),
                                          maxiter=200,
                                          tol=1e-9)

    constrained_networks.append(constrained_networks)


# ==========================================================================
# Visualization
# ==========================================================================

    # equilibrated arch
    color = cmap(1.0 - idx / len(vertical_comps))
    viewer.add(constrained_network,
               show_vertices=True,
               pointsize=12.0,
               show_edges=True,
               linecolor=color,
               linewidth=5.0)

    for node in constrained_network.nodes():

        pt = constrained_network.node_coordinates(node)

        # draw lines betwen subject and target nodes
        target_pt = network.node_coordinates(node)
        viewer.add(Line(target_pt, pt))

        # draw residual forces
        residual = constrained_network.node_attributes(node, names=["rx", "ry", "rz"])

        if length_vector(residual) < 0.001:
            continue

        # print(node, residual, length_vector(residual))
        residual_line = Line(pt, add_vectors(pt, residual))
        viewer.add(residual_line,
                   linewidth=2.0,
                   color=color.darkened())  # Color.purple()

    # draw applied loads
    for node in constrained_network.nodes():
        pt = constrained_network.node_coordinates(node)
        load = network.node_attributes(node, names=["px", "py", "pz"])
        viewer.add(Line(pt, add_vectors(pt, load)),
                   linewidth=4.0,
                   color=Color.green().darkened())

    # draw supports
    for node in constrained_network.supports():

        x, y, z = constrained_network.node_coordinates(node)
        viewer.add(Point(x, y, z), color=Color.green(), size=20)


# show le crème
viewer.show()
