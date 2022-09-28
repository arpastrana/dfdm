"""
Solve a constrained force density problem using gradient-based optimization.
"""

import os

from math import fabs
from math import radians
from math import pi, cos, sin, atan

from random import random

import numpy as np

# compas
from compas.colors import Color
from compas.colors import ColorMap
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import Vector
from compas.geometry import add_vectors
from compas.geometry import length_vector
from compas.geometry import subtract_vectors
from compas.geometry import cross_vectors
from compas.geometry import rotate_points
from compas.geometry import scale_vector
from compas.geometry import Polygon
from compas.geometry import offset_polygon
from compas.geometry import discrete_coons_patch
from compas.datastructures import Mesh
from compas.datastructures import mesh_weld
from compas.utilities import pairwise

from compas_singular.datastructures import CoarseQuadMesh
# from compas_quad.datastructures import CoarseQuadMesh

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
from dfdm.constraints import NodeNormalAngleConstraint
from dfdm.constraints import NodeCurvatureConstraint
from dfdm.constraints import NetworkEdgesLengthConstraint
from dfdm.constraints import NetworkEdgesForceConstraint

from dfdm.losses import PredictionError
from dfdm.losses import SquaredError
from dfdm.losses import MeanSquaredError
from dfdm.losses import L2Regularizer
from dfdm.losses import Loss

from dfdm.optimization import SLSQP
from dfdm.optimization import BFGS
from dfdm.optimization import TrustRegionConstrained
from dfdm.optimization import OptimizationRecorder


# ==========================================================================
# Initial parameters
# ==========================================================================

model_name = "pillow"

# geometric parameters
l1, l2 = 10.0, 10.0
divisions = 6

# initial form-finding parameters
q0, dq = -2.0, 0.1  # starting average force density and random deviation
pz = -100.0  # z component of the total applied load

# optimization
optimizer = SLSQP
maxiter = 1000
tol = 1e-3

# parameter bounds
qmin = None
qmax = None

# goal horizontal projection
add_horizontal_projection_goal = True
weight_horizontal_projection = 1.0

# goal load path
add_load_path_goal = False
normalise_by_edge_number = True
weight_load_path = 0.01

# goal edge length
add_edge_length_goal = False
weight_edge_length = 1.0

# constraint normal angle
add_node_normal_angle_constraint = False
angle_vector = [0.0, 0.0, 1.0]  # reference vector to compute angle to in constraint
angle_min = pi/2.0 - atan(0.75)
angle_max = pi/2.0
print(angle_min, angle_max)

# constraint length
add_edge_length_constraint = True
ratio_length_min = 0.5
ratio_length_max = 3.0

# constraint force
add_edge_force_constraint = True
force_min = -100.0
force_max = -1.0

# constraint angle (in radians)
add_normal_angle_constraint = False
angle_min = pi / 2.0 - atan(0.65)
angle_max = pi/ 2.0

# constraint curvature
add_curvature_constraint = True
crv_min = -100.0
crv_max = -0.2

export = True
view = False

# ==========================================================================
# Create base geometry
# ==========================================================================

vertices = [[l1, 0.0, 0.0], [l1, l2, 0.0], [0.0, l2, 0.0], [0.0, 0.0, 0.0]]
faces = [[0, 1, 2, 3]]
coarse = CoarseQuadMesh.from_vertices_and_faces(vertices, faces)
coarse.collect_strips()
coarse.set_strips_density(divisions)
coarse.densification()
mesh = coarse.get_quad_mesh()

(vertices, faces), edges = mesh.to_vertices_and_faces(), mesh.edges()
network = FDNetwork.from_nodes_and_edges(vertices, edges)

# ==========================================================================
# Define structural system
# ==========================================================================

# define supports
for key in network.nodes():
    if mesh.is_vertex_on_boundary(key):
        # x, y, z = mesh.vertex_coordinates(key)
        # if abs(x) < tol or abs(x - l1) < tol or abs(y) < tol:
        network.node_support(key)

# set initial q to all edges
for edge in network.edges():
    q = q0 + dq * (random() - 0.5)
    network.edge_forcedensity(edge, q)

networks = {'input': network}

# ==========================================================================
# Initial form finding - no external loads
# ==========================================================================

print('')
networks['unloaded'] = fdm(network)

# ==========================================================================
# Initial form finding - loaded
# ==========================================================================

# apply loads
mesh_area = mesh.area()
for key in network.nodes():
    network.node_load(key, load=[0.0, 0.0, pz * mesh.vertex_area(key) / mesh_area])

print('')
networks['loaded'] = fdm(network)

# ==========================================================================
# Create loss function with soft goals
# ==========================================================================

goals = []

# horizontal projection goal
if add_horizontal_projection_goal:
    print('Horizontal projection goal')
    for node in network.nodes_free():
        xyz = network.node_coordinates(node)
        line = Line(xyz, add_vectors(xyz, [0.0, 0.0, 1.0]))
        goal = NodeLineGoal(node, target=line, weight=weight_horizontal_projection)
        goals.append(goal)

# load path goal
if add_load_path_goal:
    if normalise_by_edge_number:
        weight_loadpath /= mesh.number_of_edges()
    goals.append(NetworkLoadPathGoal(target=0.0, weight=weight_load_path))

# edge length goal
if add_edge_length_goal:
    network2 = networks['loaded']
    goals += [EdgeLengthGoal(edge, network2.edge_length(*edge), weight=weight_edge_length) for edge in network.edges()]

loss = Loss(SquaredError(goals=goals))

# ==========================================================================
# Create constraints
# ==========================================================================

constraints = []

if add_normal_angle_constraint:
    print('Normal angle constraint w.r.t vertical between {} and {} degrees'.format(round(angle_min / pi * 180, 1), round(angle_max / pi * 180, 1)))
    for key in network.nodes():
        if not mesh.is_vertex_on_boundary(key):
            polygon = mesh.vertex_neighbors(key, ordered=True)
            constraints.append(NodeNormalAngleConstraint(key, polygon, [0.0, 0.0, 1.0], bound_low=angle_min, bound_up=angle_max))

if add_edge_length_constraint:
    average_length = np.mean([network.edge_length(*edge) for edge in network.edges()])
    length_min = ratio_length_min * average_length
    length_max = ratio_length_max * average_length
    print('Edge length constraint between {} and {}'.format(round(length_min, 2), round(length_max, 2)))
    constraints.append(NetworkEdgesLengthConstraint(bound_low=length_min, bound_up=length_max))

if add_edge_force_constraint:
    print('Edge force constraint between {} and {}'.format(round(force_min, 2), round(force_max, 2)))
    constraints.append(NetworkEdgesForceConstraint(bound_low=force_min, bound_up=force_max))

if add_curvature_constraint:
    polyedge0 = mesh.collect_polyedge(*mesh.edges_on_boundary()[0])
    n = len(polyedge0)
    i = int(n / 2)
    u0, v0 = polyedge0[i -1 : i + 1]
    if mesh.halfedge[u0][v0] is None:
        u0, v0 = v0, u0
    u, v = mesh.halfedge_after(u0, v0)
    polyedge = mesh.collect_polyedge(u, v)
    subpolyedge = polyedge[1:-1]
    print('Node curvature constraint between {} and {} on {} nodes'.format(round(crv_min, 2), round(crv_max, 2), len(subpolyedge)))
    for key in subpolyedge:
        polygon = mesh.vertex_neighbors(key, ordered=True)
        constraints.append(NodeCurvatureConstraint(key, polygon, bound_low=crv_min, bound_up=crv_max))

# ==========================================================================
# Form finding
# ==========================================================================

networks['free'] = fdm(network)

networks['uncstr_opt'] = constrained_fdm(network,
                                         optimizer=optimizer(),
                                         bounds=(qmin, qmax),
                                         loss=loss,
                                         maxiter=maxiter)

networks['cstr_opt'] = constrained_fdm(network,
                                       optimizer=optimizer(),
                                       bounds=(qmin, qmax),
                                       loss=loss,
                                       constraints=constraints,
                                       maxiter=maxiter)

# ==========================================================================
# Print and export results
# ==========================================================================

for network_name, network in networks.items():

    print()
    print("Design {}".format(network_name))

    print(f"Load path: {round(network.loadpath(), 3)}")

    q = list(network.edges_forcedensities())
    f = list(network.edges_forces())
    l = list(network.edges_lengths())

    data = {'Force densities': q, 'Forces': f, 'Lengths': l}

    if constraint_normals:
        model = EquilibriumModel(network)
        q = np.array(network.edges_forcedensities())
        eqstate = model(q)
        a = [constraint.constraint(eqstate, model) for constraint in constraint_normals]
        data['Normal angles'] = a

    for name, values in data.items():
        minv = round(min(values), 3)
        maxv = round(max(values), 3)
        meanv = round(np.mean(values), 3)
        print(f"{name}\t\tMin: {minv}\tMax: {maxv}\tMean: {meanv}")

    if export:
        HERE = os.path.dirname(__file__)
        FILE_OUT = os.path.join(HERE, "../data/json/{}_{}.json".format(model_name, network_name))
        network.to_json(FILE_OUT)
        print("Design {} exported to".format(network_name), FILE_OUT)


viewer = App(width=1600, height=900, show_grid=False)

# add all networks except the last one
networks = list(networks.values())
for i, network in enumerate(networks):
    if i == (len(networks) - 1):
        continue
    viewer.add(network, show_points=False, linewidth=1.0, color=Color.grey().darkened(i * 10))

network0 = networks[0]
if len(networks) > 1:
    c_network = networks[-1]  # last network is colored
else:
    c_network = networks[0]

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
viewer.show()
