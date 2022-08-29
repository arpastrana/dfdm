# the essentials
import os
from math import fabs
import matplotlib.pyplot as plt

# compas
from compas.colors import Color
from compas.colors import ColorMap
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import add_vectors
from compas.geometry import length_vector
from compas.topology import dijkstra_path
from compas.utilities import pairwise

# pattern-making
from compas_singular.datastructures import CoarseQuadMesh

# visualization
from compas_view2.app import App

# force density
from dfdm.datastructures import FDNetwork
from dfdm.equilibrium import EquilibriumModel
from dfdm.equilibrium import constrained_fdm, fdm
from dfdm.optimization import SLSQP, BFGS
from dfdm.optimization import OptimizationRecorder
from dfdm.goals import LengthGoal
from dfdm.goals import ResidualForceGoal
from dfdm.losses import MeanSquaredErrorLoss
from dfdm.losses import SquaredErrorLoss
from dfdm.regularizers import L2Regularizer


# ==========================================================================
# Parameters
# ==========================================================================

name = "monkey_saddle"

n = 3  # densification of coarse mesh

px, py, pz = 0.0, 0.0, -1.0  # loads at each node
qmin, qmax = -20.0, -0.001  # min and max force densities
rmin, rmax = 2.0, 10.0  # min and max reaction forces
r_exp = 1.0  # reaction force variation exponent
factor_edgelength = 1.0  # edge length factor

weight_residual = 100.0  # weight for residual force goal in optimisation
weight_length = 1.0  # weight for edge length goal in optimisation

alpha = 0.1  # scale of the L2 regularization term in the loss function

maxiter = 500  # optimizer maximum iterations
tol = 1e-3  # optimizer tolerance

record = True  # True to record optimization history of force densities
export = False  # export result to JSON

# ==========================================================================
# Import coarse mesh
# ==========================================================================

HERE = os.path.dirname(__file__)
FILE_IN = os.path.abspath(os.path.join(HERE, f"../data/json/{name}.json"))
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

print("Densified mesh:", mesh)

# ==========================================================================
# Define support conditions
# ==========================================================================

polyedge2length = {}
for pkey, polyedge in mesh.polyedges(data=True):
    if mesh.is_vertex_on_boundary(polyedge[0]) and mesh.is_vertex_on_boundary(polyedge[1]):
        length = sum([mesh.edge_length(u, v) for u, v in pairwise(polyedge)])
        polyedge2length[tuple(polyedge)] = length

supports = []
n = sum(polyedge2length.values()) / len(polyedge2length)
for polyedge, length in polyedge2length.items():
    if length < n:
        supports += polyedge

supports = set(supports)

print("Number of supported nodes:", len(supports))

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
        len_dijkstra = []
        for corner in corners:
            len_dijkstra.append(len(dijkstra_path(adjacency, weight, vkey, corner)) - 1)
        steps[vkey] = min(len_dijkstra)

max_step = max(steps.values())
steps = {vkey: max_step - step for vkey, step in steps.items()}

# ==========================================================================
# Define structural system
# ==========================================================================

nodes = [mesh.vertex_coordinates(vkey) for vkey in mesh.vertices()]
edges = [(u, v) for u, v in mesh.edges() if u not in supports or v not in supports]
network0 = FDNetwork.from_nodes_and_edges(nodes, edges)

print("FD network:", network0)

# data
network0.nodes_supports(supports)
network0.nodes_loads([px, py, pz], keys=network0.nodes())
network0.edges_forcedensities(q=-1.0)

# ==========================================================================
# Export FD network with problem definition
# ==========================================================================

if export:
    FILE_OUT = os.path.join(HERE, f"../data/json/{name}_base.json")
    network0.to_json(FILE_OUT)
    print("Problem definition exported to", FILE_OUT)

# ==========================================================================
# Form-find network
# ==========================================================================

network0 = fdm(network0)

# ==========================================================================
# Define goals
# ==========================================================================

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
# Combine error function and regularizer into custom loss function
# ==========================================================================

squared_error = SquaredErrorLoss(goals)
regularizer = L2Regularizer(alpha)
# loss = LossContainer(terms=(squared_error, regularizer))

class SquaredErrorRegularized():
    """
    A user-defined loss function.

    A valid loss function is in terms of the force densities `q`, and the
    goals' predictions, targets and weights. This loss function *must* have
    `predictions`, `targets`, `weights` and `force_densities` as arguments
    in its signature. Not all the arguments have to be used in this function.

    Note
    ----
    This loss is equivalent to dfdm.losses.squared_loss, but here
    we recreate it to illustrate how the custom loss function API works.
    """
    def __call__(self, eqstate, model):
        return squared_error(eqstate, model) + regularizer(eqstate, model)

squared_error_reg = SquaredErrorRegularized()

# ==========================================================================
# Solve constrained form-finding problem
# ==========================================================================

recorder = None
if record:
    recorder = OptimizationRecorder()


network = constrained_fdm(network0,
                          optimizer=SLSQP(),
                          loss=squared_error_reg,
                          bounds=(qmin, qmax),
                          maxiter=maxiter,
                          tol=tol,
                          callback=recorder)

# ==========================================================================
# Export optimization history
# ==========================================================================

if record and export:
    FILE_OUT = os.path.join(HERE, f"../data/json/{name}_history.json")
    recorder.to_json(FILE_OUT)
    print("Optimization history exported to", FILE_OUT)

# ==========================================================================
# Plot loss components
# ==========================================================================

if record:
    model = EquilibriumModel(network)
    fig = plt.figure(dpi=150)
    for loss in (squared_error, regularizer, squared_error_reg):
        y = []
        for q in recorder.history:
            eqstate = model(q)
            error = loss(eqstate, model)
            y.append(error)
        plt.plot(y, label=loss.__class__.__name__)

    plt.xlabel("Optimization iterations")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.show()

# ==========================================================================
# Export JSON
# ==========================================================================

if export:
    FILE_OUT = os.path.join(HERE, f"../data/json/{name}_optimized.json")
    network.to_json(FILE_OUT)
    print("Form found design exported to", FILE_OUT)

# ==========================================================================
# Report stats
# ==========================================================================

q = [network.edge_forcedensity(edge) for edge in network.edges()]
f = [network.edge_force(edge) for edge in network.edges()]
l = [network.edge_length(*edge) for edge in network.edges()]

for name, vals in zip(("FDs", "Forces", "Lengths"), (q, f, l)):

    minv = round(min(vals), 3)
    maxv = round(max(vals), 3)
    meanv = round(sum(vals) / len(vals), 3)
    print(f"{name}\t\tMin: {minv}\tMax: {maxv}\tMean: {meanv}")

# ==========================================================================
# Visualization
# ==========================================================================

viewer = App(width=1600, height=900, show_grid=False)

# modify view
viewer.view.camera.zoom(-35)  # number of steps, negative to zoom out
viewer.view.camera.rotation[2] = 0.0  # set rotation around z axis to zero

# reference network
viewer.add(network0, show_points=False, linewidth=1.0, color=Color.grey().darkened())

# edges color map
cmap = ColorMap.from_mpl("viridis")

fds = [fabs(network.edge_forcedensity(edge)) for edge in network.edges()]
colors = {}
for edge in network.edges():
    fd = fabs(network.edge_forcedensity(edge))
    ratio = (fd - min(fds)) / (max(fds) - min(fds))
    colors[edge] = cmap(ratio)

# optimized network
viewer.add(network,
           show_vertices=True,
           pointsize=12.0,
           show_edges=True,
           linecolors=colors,
           linewidth=5.0)

for node in network.nodes():

    pt = network.node_coordinates(node)

    # draw residual forces
    residual = network.node_residual(node)

    if length_vector(residual) < 0.001:
        continue

    # print(node, residual, length_vector(residual))
    residual_line = Line(pt, add_vectors(pt, residual))
    viewer.add(residual_line,
               linewidth=4.0,
               color=Color.pink())  # Color.purple()

# draw applied loads
for node in network.nodes():
    pt = network.node_coordinates(node)
    load = network.node_load(node)
    viewer.add(Line(pt, add_vectors(pt, load)),
               linewidth=4.0,
               color=Color.green().darkened())

# draw supports
for node in network.nodes_supports():
    x, y, z = network.node_coordinates(node)
    viewer.add(Point(x, y, z), color=Color.green(), size=20)

# show le crème
viewer.show()
