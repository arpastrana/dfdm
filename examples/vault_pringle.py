"""
Solve a constrained force density problem using gradient based optimization.
"""
from math import fabs
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
from force_density.goals import PlaneGoal
from force_density.goals import ResidualForceGoal
from force_density.goals import ResidualVectorGoal
from force_density.goals import ResidualDirectionGoal

from force_density.losses import squared_loss
from force_density.losses import l2_loss
from force_density.optimization import SLSQP

# ==========================================================================
# Initial parameters
# ==========================================================================

length_vault = 6.0
width_vault = 3.0

num_u = 10
num_v = 5

q_init = -0.25
pz = -0.1

rz_min = 0.25
rz_max = 2.0

# ==========================================================================
# Instantiate compression network
# ==========================================================================

network = CompressionNetwork()

# ==========================================================================
# Create the base geometry of the vault
# ==========================================================================

xyz_origin = [0.0, 0.0, 0.0]
length_u = length_vault / (num_u - 1)
length_v = width_vault / (num_v - 1)

arches = []
long_edges = []
for i in range(num_v):

    arch = []
    start = add_vectors(xyz_origin, [0.0, i * length_v, 0.0, 0.0])

    for j in range(num_u):

        x, y, z = add_vectors(start, [j * length_u, 0.0, 0.0])
        node = network.add_node(x=x, y=y, z=z)
        arch.append(node)

    arches.append(arch)

    a = i * num_u
    b = a + num_u - 1
    for u, v in zip(range(a, b), range(a + 1, b + 1)):
        edge = network.add_edge(u, v)
        long_edges.append(edge)

cross_edges = []
for i in range(1, num_u - 1):
    seq = []
    for arch in arches:
        seq.append(arch[i])
    for u, v in zip(seq[:-1], seq[1:]):
        edge = network.add_edge(u, v)
        cross_edges.append(edge)

# ==========================================================================
# Define structural system
# ==========================================================================

# define supports
for arch in arches:
    network.node_support(arch[0])
    network.node_support(arch[-1])

# apply loads to unsupported nodes
for node in network.nodes_free():
    network.node_load(node, load=[0.0, 0.0, pz])

# set initial q to all nodes
for edge in network.edges():
    network.edge_forcedensity(edge, q_init)

# ==========================================================================
# Create a target distribution of residual force magnitudes
# ==========================================================================

# create linear range of reaction force goals
assert num_v % 2 != 0
num_steps = (num_v - 1) / 2.0
step_size = (rz_max - rz_min) / num_steps

rzs = []
for i in range(int(num_steps) + 1):
    rzs.append(rz_min + i * step_size)

rzs = rzs + rzs[0:-1][::-1]

# ==========================================================================
# Define goals
# ==========================================================================

goals = []
for rz, arch in zip(rzs, arches):
    goals.append(ResidualForceGoal(arch[0], force=rz))
    goals.append(ResidualForceGoal(arch[-1], force=rz))

for edge in cross_edges:
    target_length = network.edge_length(*edge)
    goals.append(LengthGoal(edge, 0.75 * target_length))

for node in network.nodes_free():
    origin = network.node_coordinates(node)
    normal = [1.0, 0.0, 0.0]
    goal = PlaneGoal(node, plane=(origin, normal))
    goals.append(goal)


# ==========================================================================
# Optimization
# ==========================================================================

c_network = constrained_fdm(network,
                            optimizer=SLSQP(),
                            loss=squared_loss,  # squared_loss
                            goals=goals,
                            bounds=(-5.0, -0.1),
                            maxiter=200,
                            tol=1e-9)

# ==========================================================================
# Print out stats
# ==========================================================================

counter = 0
for edge in c_network.edges():
    force = c_network.edge_force(edge)
    if force > 0.0:
        counter += 1
        print(f"Tension force on edge {edge}: {round(force, 2)}")

ratio = counter / c_network.number_of_edges()
print(f"Ratio of edges in tension: {round(100.0 * ratio, 2)}")
print()

fds = [c_network.edge_forcedensity(edge) for edge in c_network.edges()]
print(f"Force density stats. Min: {round(min(fds), 2)}. Max: {round(max(fds), 2)}. Mean: {round(sum(fds) / len(fds), 2)}")

# ==========================================================================
# Visualization
# ==========================================================================

viewer = App(width=1600, height=900, show_grid=False)

# reference arch
viewer.add(network, show_points=True, linewidth=2.0, color=Color.grey().darkened())

# edges color map
cmap = ColorMap.from_mpl("viridis")

fds = [fabs(c_network.edge_forcedensity(edge)) for edge in c_network.edges()]
colors = {}
for edge in c_network.edges():
    fd = fabs(c_network.edge_forcedensity(edge))
    ratio = (fd - min(fds)) / (max(fds) - min(fds))
    colors[edge] = cmap(ratio)

# optimized network
viewer.add(c_network,
           show_vertices=True,
           pointsize=12.0,
           show_edges=True,
           linecolors=colors,
           linewidth=5.0)

for node in c_network.nodes():

    pt = c_network.node_coordinates(node)

    # draw lines betwen subject and target nodes
    target_pt = network.node_coordinates(node)
    viewer.add(Line(target_pt, pt), linewidth=1.0, color=Color.grey().lightened())

    # draw residual forces
    residual = c_network.node_attributes(node, names=["rx", "ry", "rz"])

    if length_vector(residual) < 0.001:
        continue

    # print(node, residual, length_vector(residual))
    residual_line = Line(pt, add_vectors(pt, residual))
    viewer.add(residual_line,
               linewidth=4.0,
               color=Color.pink())  # Color.purple()

# draw applied loads
for node in c_network.nodes():
    pt = c_network.node_coordinates(node)
    load = network.node_attributes(node, names=["px", "py", "pz"])
    viewer.add(Line(pt, add_vectors(pt, load)),
               linewidth=4.0,
               color=Color.green().darkened())

# draw supports
for node in c_network.supports():

    x, y, z = c_network.node_coordinates(node)
    viewer.add(Point(x, y, z), color=Color.green(), size=20)


# show le crème
viewer.show()