"""
Solve a constrained force density problem using gradient-based optimization.
"""

import os
import numpy as np
from math import fabs
from math import radians
from math import pi, cos, sin, atan

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

# from compas_singular.datastructures import CoarseQuadMesh
from compas_quad.datastructures import CoarseQuadMesh

# visualization
# from compas_view2.app import App

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
divisions = 8

# initial form-finding parameters
q0 = -2.0  # starting force density
pz = -1.0  # z component of the applied load

# optimization
optimizer = SLSQP
maxiter = 1000
tol = 1e-3

# parameter bounds
qmin = None
qmax = None

# goal horizontal projection
add_horizontal_projection_goal = True

# constraint length
add_edge_length_constraint = True
ratio_length_min = 0.8
ratio_length_max = 1.2

# constraint force
add_edge_force_constraint = True
force_min = -100.0
force_max = 0.0

export = True
view = False

# ==========================================================================
# Instantiate a force density network
# ==========================================================================

network = FDNetwork()

# ==========================================================================
# Create the base geometry of the dome
# ==========================================================================

vertices = [[l1, 0.0, 0.0], [l1, l2, 0.0], [0.0, l2, 0.0], [0.0, 0.0, 0.0]]
faces = [[0, 1, 2, 3]]
coarse = CoarseQuadMesh.from_vertices_and_faces(vertices, faces)
coarse.collect_strips()
coarse.set_strips_density(divisions)
coarse.densification()
mesh = coarse.get_quad_mesh()

vertices, faces = mesh.to_vertices_and_faces()
edges = mesh.edges()
network = FDNetwork.from_nodes_and_edges(vertices, edges)

# ==========================================================================
# Define structural system
# ==========================================================================

# define supports
for key in network.nodes():
    if mesh.is_vertex_on_boundary(key):
        network.node_support(key)

# apply loads
for key in network.nodes():
    network.node_load(key, load=[0.0, 0.0, pz])

# set initial q to all edges
for edge in network.edges():
    network.edge_forcedensity(edge, q0)

networks = {'input': network}

# ==========================================================================
# Create loss function with soft goals
# ==========================================================================

goals = []

# horizontal projection goal
if add_horizontal_projection_goal:
    for node in network.nodes_free():
        xyz = network.node_coordinates(node)
        line = Line(xyz, add_vectors(xyz, [0.0, 0.0, 1.0]))
        goal = NodeLineGoal(node, target=line)
        goals.append(goal)

# goals.append(NetworkLoadPathGoal(target=0.0, weight=1.0))

loss = Loss(SquaredError(goals=goals))

# ==========================================================================
# Create constraints
# ==========================================================================

constraints = []

for key in network.nodes():
    if not mesh.is_vertex_on_boundary(key):
        polygon = mesh.vertex_neighbors(key, ordered=True)
        constraints.append(NodeNormalAngleConstraint(key, polygon, [0.0, 0.0, 1.0], bound_low=pi/2.0-atan(0.65), bound_up=pi/2.0))

if add_edge_length_constraint:
    average_length = np.mean([network.edge_length(*edge) for edge in network.edges()])
    length_min = ratio_length_min * average_length
    length_max = ratio_length_max * average_length
    constraints.append(NetworkEdgesLengthConstraint(bound_low=length_min, bound_up=length_max))

if add_edge_force_constraint:
    constraints.append(NetworkEdgesForceConstraint(bound_low=force_min, bound_up=force_max))

# ==========================================================================
# Form-finding
# ==========================================================================

networks['free'] = fdm(network.copy())

networks['uncstr_opt'] = constrained_fdm(network.copy(),
                          optimizer=optimizer(),
                          bounds=(qmin, qmax),
                          loss=loss,
                          constraints=None,
                          maxiter=maxiter)

networks['cstr_opt'] = constrained_fdm(network.copy(),
                          optimizer=optimizer(),
                          bounds=(qmin, qmax),
                          loss=loss,
                          constraints=constraints,
                          maxiter=maxiter)

for network_name, network in networks.items():

    print("\n Design {}".format(network_name))

    print(f"Load path: {round(network.loadpath(), 3)}")

    q = list(network.edges_forcedensities())
    f = list(network.edges_forces())
    l = list(network.edges_lengths())

    data = {'Force densities': q, 'Forces': f, 'Lengths': l}

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
