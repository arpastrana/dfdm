"""
Solve a constrained force density problem using gradient-based optimization.
"""
import numpy as np
from math import fabs

# compas
from compas.colors import Color
from compas.colors import ColorMap
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import add_vectors
from compas.geometry import length_vector
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

from dfdm.goals import LineGoal
from dfdm.goals import LengthGoal
from dfdm.goals import PlaneGoal
from dfdm.goals import ResidualForceGoal
from dfdm.goals import NetworkLoadPathGoal

from dfdm.constraints import EdgeVectorAngleConstraint
from dfdm.constraints import NetworkEdgesLengthConstraint

from dfdm.losses import PredictionError
from dfdm.losses import Loss
from dfdm.losses import SquaredError

from dfdm.optimization import SLSQP
from dfdm.optimization import TrustRegionConstrained
from dfdm.optimization import OptimizationRecorder

# ==========================================================================
# Initial parameters
# ==========================================================================

name = "dome"

# geometric parameters
diameter = 1.0
num_sides = 8
num_rings = 5
distance = 0.05  # hoop offset

# initial form-finding parameters
q0_ring = -2.0  # starting force density for ring (hoop) edges
q0_cross = -0.5  # starting force density for the edges transversal to the rings
pz = -0.1  # z component of the applied load

# optimization
optimizer = SLSQP
maxiter = 1000
tol = 1e-6

# parameter bounds
qmin = -10.0
qmax = -0.01

# constraint angle
angle_vector = [0.0, 0.0, 1.0]  # reference vector to compute angle to in constraint
angle_min = 10.0  # angle constraint, lower bound
angle_max = 30.0  # angle constraint, upper bound

# constraint length
length_min = distance
length_max = distance * 2

# load path
alpha_lp = 0.1

# output
record = False
export = False

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
    polygon = offset_polygon(polygon, distance)
    nodes = [network.add_node(x=x, y=y, z=z) for x, y, z in polygon]
    rings.append(nodes)

edges_rings = []
for ring in rings[1:]:
    for u, v in pairwise(ring + ring[:1]):
        edge = network.add_edge(u, v)
        edges_rings.append(edge)

edges_cross = []
for i in range(num_sides):
    seq = []
    for ring in rings:
        seq.append(ring[i])

    for u, v in pairwise(seq):
        edge = network.add_edge(u, v)
        edges_cross.append(edge)

# ==========================================================================
# Define structural system
# ==========================================================================

# define supports
for node in rings[0]:
    network.node_support(node)

# apply loads to unsupported nodes
for node in network.nodes_free():
    network.node_load(node, load=[0.0, 0.0, pz])

# set initial q to all nodes
for edge in edges_rings:
    network.edge_forcedensity(edge, q0_ring)

for edge in edges_cross:
    network.edge_forcedensity(edge, q0_cross)

# ==========================================================================
# Store network
# ==========================================================================

networks = {"start": network}

# ==========================================================================
# Create loss function with soft goals
# ==========================================================================

goals = []

# horizontal projection goal
# for node in network.nodes_free():
#     xyz = network.node_coordinates(node)
#     line = Line(xyz, add_vectors(xyz, [0.0, 0.0, 1.0]))
#     goal = LineGoal(node, target=line)
#     goals.append(goal)

# loss = Loss(PredictionError(goals=goals))

# edge length goal
for edge in edges_cross:
    length = network.edge_length(*edge)
    goal = LengthGoal(edge, target=length)
    goals.append(goal)

# loss = Loss(SquaredError(goals=goals), PredictionError([NetworkLoadPathGoal()], alpha=alpha_lp))
loss = Loss(SquaredError(goals=goals))

# ==========================================================================
# Create constraints
# ==========================================================================

constraints_angles = []
for edge in edges_cross:
    constraint = EdgeVectorAngleConstraint(edge,
                                           vector=angle_vector,
                                           bound_low=angle_min,
                                           bound_up=angle_max)
    constraints_angles.append(constraint)

# constraints_length = [NetworkEdgesLengthConstraint(bound_low=length_min, bound_up=length_max)]
# constraints = constraints_angles + constraints_length

constraints = constraints_angles

# ==========================================================================
# Form-finding sweep
# ==========================================================================

sweep_configs = [{"name": "eq",
                  "method": fdm,
                  "msg": "\n*Form found network*",
                  "save": False},
                 # {"name": "eq_g",
                 # "method": constrained_fdm,
                 #  "msg": "\n*Constrained form found network. No constraints*",
                 #  "save": True},
                 {"name": "eq_g_c",
                  "method": constrained_fdm,
                  "msg": "\n*Constrained form found network. With Constraints*",
                  "save": True,
                  "constraints": constraints}
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

    if constraints_angles:
        model = EquilibriumModel(network)
        q = np.array(network.edges_forcedensities())
        eqstate = model(q)
        a = [constraint.constraint(eqstate, model) for constraint in constraints_angles]
        fields.append(a)
        field_names.append("Angles")

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

# network = networks["start"]
# c_network = networks["eq_g_c"]

# reference network
for i, network in enumerate(networks.values()):
    viewer.add(network, show_points=True, linewidth=1.0, color=Color.grey().darkened(i * 10))
    c_network = network  # last network is colored

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
    # target_pt = network.node_coordinates(node)
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
