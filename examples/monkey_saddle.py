import os

from compas_singular.datastructures import CoarseQuadMesh

from compas.topology import dijkstra_path

from dfdm.datastructures import ForceDensityNetwork
from dfdm.equilibrium import fdm, constrained_fdm
from dfdm.optimization import SLSQP
from dfdm.goals import LengthGoal, ResidualForceGoal
from dfdm.losses import squared_loss

from compas.utilities import pairwise

# ==========================================================================
# Parameters
# ==========================================================================

n = 3  # densification of coarse mesh

px, py, pz = 0.0, 0.0, -1.0  # loads at each node
qmin, qmax = -100.0, -0.01  # min and max force densities
rmin, rmax = 2.0, 10.0  # min and max reaction forces
r_exp = 1.0  # reaction force variation exponent
factor_edgelength = 1.1  # edge length factor

weight_residual= 100.0  # weight for residual force goal in optimisation
weight_length = 1.0  # weight for edge length goal in optimisation

maxiter = 200  # solver maximum iterations
tol = 1e-9  # solver tolerance

export = False  # export result to JSON

# ==========================================================================
# Import coarse mesh
# ==========================================================================

HERE = os.path.dirname(__file__)
FILE_IN = os.path.join(HERE, '../data/json/monkey_saddle.json')
mesh = CoarseQuadMesh.from_json(FILE_IN)

print('Initial coarse mesh:', mesh)

# ==========================================================================
# Densify coarse mesh
# ==========================================================================

mesh.collect_strips()
mesh.set_strips_density(n)
mesh.densification()
mesh = mesh.get_quad_mesh()
mesh.collect_polyedges()

print('Densified mesh:', mesh)

# ==========================================================================
# Define support conditions
# ==========================================================================

supports = []
polyedge2length = {}
for pkey, polyedge in mesh.polyedges(data=True):
    if mesh.is_vertex_on_boundary(polyedge[0]) and mesh.is_vertex_on_boundary(polyedge[1]):
        polyedge2length[tuple(polyedge)] = sum([mesh.edge_length(u, v) for u, v in pairwise(polyedge)])

n = sum(polyedge2length.values()) / len(polyedge2length)
for polyedge, length in polyedge2length.items():
    if length < n:
        supports += polyedge

supports = set(supports)

print('Number of supported nodes:', len(supports))

# ==========================================================================
# Compute assembly sequence (simplified)
# ==========================================================================

steps = {}
corners = set([vkey for vkey in mesh.vertices() if mesh.vertex_degree(vkey) == 2])
adjacency = mesh.adjacency
weight = {(u, v): 1.0 for u in adjacency for v in adjacency[u]}

for vkey in supports:
    if vkey in corners:
        steps[vkey] = 0
    else:
        steps[vkey] = min([len(dijkstra_path(adjacency, weight, vkey, corner)) - 1 for corner in corners])

max_step = max(steps.values())
steps = {vkey: max_step - step for vkey, step in steps.items()}

# ==========================================================================
# Define structural system
# # ==========================================================================

nodes = [mesh.vertex_coordinates(vkey) for vkey in mesh.vertices()]
edges = [(u, v) for u, v in mesh.edges() if u not in supports or v not in supports]
network0 = ForceDensityNetwork.from_nodes_and_edges(nodes, edges)

# data
network0.nodes_supports(supports)
network0.nodes_loads([px, py, pz], keys=network0.nodes())
network0.edges_forcedensities(q=-1.0)

# ==========================================================================
# Define goals
# ==========================================================================

# goals
goals = []

# edge lengths
for edge in network0.edges():
    current_length = network0.edge_length(*edge)
    goal = LengthGoal(edge, factor_edgelength * current_length, weight=weight_length)
    goals.append(goal)

# reaction forces
for key in network0.nodes_supports():
    step = steps[key]
    reaction = (1 - step / max_step) ** r_exp * (rmax - rmin) + rmin
    goals.append(ResidualForceGoal(key, reaction, weight=weight_residual))

# ==========================================================================
# Solve constrained form-finding problem
# ==========================================================================

network = constrained_fdm(network0,
			  optimizer=SLSQP(),
			  loss=squared_loss,
			  goals=goals,
			  bounds=(qmin, qmax),
			  maxiter=maxiter,
			  tol=tol)

# ==========================================================================
# Report stats
# ==========================================================================

q = [network.edge_forcedensity(edge) for edge in network.edges()]
print('Min and max edge force densities: {} and {}'.format(round(min(q), 3), round(max(q), 3)))
f = [network.edge_force(edge) for edge in network.edges()]
print('Min and max edge forces: {} and {}'.format(round(min(f), 3), round(max(f), 3)))
l = [network.edge_length(*edge) for edge in network.edges()]
print('Min and max edge lengths: {} and {}'.format(round(min(l), 3), round(max(l), 3)))

# ==========================================================================
# Export JSON
# ==========================================================================

if export:
    FILE_OUT = os.path.join(HERE, '../data/json/monkey_saddle_form_found.json')
    network.to_json(FILE_OUT)
    print('Form found design exported to', FILE_OUT)
