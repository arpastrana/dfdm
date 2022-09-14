"""
Solve a constrained force density problem using gradient-based optimization.
"""
import numpy as np
from math import fabs
from math import radians
from math import sqrt

# compas
from compas.colors import Color
from compas.colors import ColorMap
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import add_vectors
from compas.geometry import length_vector
from compas.geometry import subtract_vectors
from compas.geometry import cross_vectors
from compas.geometry import rotate_points
from compas.geometry import scale_vector
from compas.geometry import Polygon
from compas.geometry import offset_polygon
from compas.utilities import pairwise

# visualization
from compas_view2.app import App

# static equilibrium
from dfdm.datastructures import FDNetwork

from dfdm.equilibrium import fdm
from dfdm.equilibrium import constrained_fdm
from dfdm.equilibrium import EquilibriumModel

from dfdm.goals import EdgeVectorAngleGoal
from dfdm.goals import EdgeDirectionGoal
from dfdm.goals import EdgeLengthGoal
from dfdm.goals import NodeLineGoal
from dfdm.goals import NodePlaneGoal
from dfdm.goals import NodeResidualForceGoal
from dfdm.goals import NetworkLoadPathGoal

from dfdm.constraints import EdgeVectorAngleConstraint
from dfdm.constraints import NetworkEdgesLengthConstraint
from dfdm.constraints import NetworkEdgesForceConstraint

from dfdm.losses import PredictionError
from dfdm.losses import SquaredError
from dfdm.losses import MeanSquaredError
from dfdm.losses import L2Regularizer
from dfdm.losses import Loss

from dfdm.optimization import SLSQP
from dfdm.optimization import TrustRegionConstrained
from dfdm.optimization import OptimizationRecorder


# ==========================================================================
# Initial parameters
# ==========================================================================

name = "dome"

# geometric parameters
diameter = 1.0
num_sides = 16
num_rings = 24
offset_distance = 0.02  # ring offset

# initial form-finding parameters
q0_ring = -2.0  # starting force density for ring (hoop) edges
q0_cross = -0.5  # starting force density for the edges transversal to the rings
pz = -0.1  # z component of the applied load

# optimization
optimizer = SLSQP
maxiter = 1000
tol = 1e-6

# parameter bounds
qmin = -200.0
qmax = -0.001

# goal length
length_target = 0.03

# goal vector, angle
angle_vector = [0.0, 0.0, 1.0]  # reference vector to compute angle to in constraint
angle_min = 10.0  # angle constraint, lower bound
angle_max = 45.0  # angle constraint, upper bound

# ==========================================================================
# Instantiate a force density network
# ==========================================================================

network = FDNetwork()

# ==========================================================================
# Create the base geometry of the dome
# ==========================================================================

polygon = Polygon.from_sides_and_radius_xy(num_sides, diameter / 2.).points

rings = []
for i in range(num_rings + 1):
    polygon = offset_polygon(polygon, offset_distance)
    nodes = [network.add_node(x=x, y=y, z=z) for x, y, z in polygon]
    rings.append(nodes)

edges_rings = []
for ring in rings[1:]:
    for u, v in pairwise(ring + ring[:1]):
        edge = network.add_edge(u, v)
        edges_rings.append(edge)

crosses = []
edges_cross = []
for i in range(num_sides):

    radial = []
    for ring in rings:
        radial.append(ring[i])
    crosses.append(radial)

    for u, v in pairwise(radial):
        edge = network.add_edge(u, v)
        edges_cross.append(edge)

edges_cross_rings = []
for rings_pair in pairwise(rings):
    cross_ring = []
    for edge in zip(*rings_pair):
        cross_ring.append(edge)
    edges_cross_rings.append(cross_ring)

# ==========================================================================
# Define structural system
# ==========================================================================

# define supports
for node in rings[0]:
    network.node_support(node)

# apply loads to unsupported nodes
for node in network.nodes_free():
    network.node_load(node, load=[0.0, 0.0, pz])

# set initial q to all edges
q0_scale = sqrt(network.number_of_edges()) / 2.

for edge in edges_rings:
    network.edge_forcedensity(edge, q0_ring)

for i, cross_ring in enumerate(edges_cross_rings):
    for edge in cross_ring:
        network.edge_forcedensity(edge, q0_cross * q0_scale * (num_rings - i))

# ==========================================================================
# Store network
# ==========================================================================

networks = {"start": network}

# ==========================================================================
# Create loss function with soft goals
# ==========================================================================

goals = []

# edge length goal
for cross_ring in edges_cross_rings[:]:
    for edge in cross_ring:
        goal = EdgeLengthGoal(edge, target=length_target, weight=1.)
        goals.append(goal)

# edge vector goal
vector_edges = []
for i, cross_ring in enumerate(edges_cross_rings):

    angle = ((i + 1) / len(edges_cross_rings)) * angle_max

    print(f"Edges ring {i + 1}/{len(edges_cross_rings)}. Angle goal: {angle}")

    for u, v in cross_ring:

        edge = (u, v)
        xyz = network.node_coordinates(u)  # xyz of first node, assumes it is the lowermost
        normal = cross_vectors(network.edge_vector(u, v), angle_vector)
        end = rotate_points([angle_vector], -radians(angle), axis=normal, origin=xyz).pop()
        vector = subtract_vectors(end, xyz)

        goal = EdgeDirectionGoal(edge, target=vector, weight=1.)

        goals.append(goal)
        vector_edges.append((vector, edge))


loss = Loss(SquaredError(goals=goals))


# ==========================================================================
# Form-finding sweep
# ==========================================================================

sweep_configs = [{"name": "eq",
                  "method": fdm,
                  "msg": "\n*Form found network*",
                  "save": True},
                 {"name": "eq_g",
                 "method": constrained_fdm,
                  "msg": "\n*Constrained form found network. No constraints*",
                  "save": True},
                 ]

# ==========================================================================
# Print out stats
# ==========================================================================

for config in sweep_configs:

    fofin_method = config["method"]

    print()
    print(config["msg"])

    if fofin_method == fdm:
        network = fofin_method(network)
    else:
        network = fofin_method(network,
                               optimizer=optimizer(),
                               bounds=(qmin, qmax),
                               loss=loss,
                               constraints=config.get("constraints"),
                               maxiter=maxiter)

    # store network
    if config["save"]:
        networks[config["name"]] = network

    # Report stats
    q = list(network.edges_forcedensities())
    f = list(network.edges_forces())
    l = list(network.edges_lengths())

    fields = [q, f, l]
    field_names = ["FDs", "Forces", "Lengths"]

    print(f"Load path: {round(network.loadpath(), 3)}")
    for name, vals in zip(field_names, fields):

        minv = round(min(vals), 3)
        maxv = round(max(vals), 3)
        meanv = round(sum(vals) / len(vals), 3)
        print(f"{name}\t\tMin: {minv}\tMax: {maxv}\tMean: {meanv}")

# ==========================================================================
# Visualization
# ==========================================================================

viewer = App(width=1600, height=900, show_grid=False)

# add all networks except the last one
networks = list(networks.values())
for i, network in enumerate(networks):
    if i == (len(networks) - 1):
        continue
    viewer.add(network, show_points=True, linewidth=1.0, color=Color.grey().darkened(i * 10))

network0 = networks[0]
if len(networks) > 1:
    c_network = networks[-1]  # last network is colored
else:
    c_network = networks[0]

for vector, edge in vector_edges:
    u, v = edge
    xyz = c_network.node_coordinates(u)
    viewer.add(Line(xyz, add_vectors(xyz, scale_vector(vector, 0.1))))

# plot the last network
# edges color map
cmap = ColorMap.from_mpl("viridis")

fds = [fabs(c_network.edge_forcedensity(edge)) for edge in c_network.edges()]
colors = {}
for edge in c_network.edges():
    fd = fabs(c_network.edge_forcedensity(edge))
    try:
        ratio = (fd - min(fds)) / (max(fds) - min(fds))
    except ZeroDivisionError:
        ratio = 1.
    colors[edge] = cmap(ratio)

# optimized network
viewer.add(c_network,
           show_vertices=True,
           pointsize=20.0,
           show_edges=True,
           linecolors=colors,
           linewidth=5.0)

for node in c_network.nodes():

    pt = c_network.node_coordinates(node)

    # draw lines betwen subject and target nodes
    # target_pt = network0.node_coordinates(node)
    # viewer.add(Line(target_pt, pt), linewidth=1.0, color=Color.grey().lightened())

    # draw residual forces
    residual = c_network.node_residual(node)

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
    load = c_network.node_load(node)
    viewer.add(Line(pt, add_vectors(pt, load)),
               linewidth=4.0,
               color=Color.green().darkened())

# draw supports
for node in c_network.nodes_supports():
    x, y, z = c_network.node_coordinates(node)
    viewer.add(Point(x, y, z), color=Color.green(), size=30)

# show le crème
# viewer.zoom_extents()
viewer.show()
