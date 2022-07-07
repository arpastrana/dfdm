"""
Solve a constrained force density problem using gradient based optimization.
"""

# filepath stuff
import os

# visualization matters
from compas.colors import Color
from compas.datastructures import Network
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import add_vectors
from compas.geometry import scale_vector
from compas.geometry import length_vector

from compas_view2.app import App

# force equilibrium
from force_density import JSON
from force_density.equilibrium import ForceDensity
from force_density.network import CompressionNetwork
from force_density.losses import SquaredError
from force_density.goals import LengthGoal
from force_density.optimization import Optimizer

# ==========================================================================
# Initial parameters
# ==========================================================================

JSON_IN = os.path.abspath(os.path.join(JSON, "compression_network.json"))
JSON_OUT = os.path.abspath(os.path.join(JSON, "compression_network_opt_2.json"))

export_json = False
view = True

# ==========================================================================
# Load Network with boundary conditions from JSON
# ==========================================================================

network = CompressionNetwork.from_json(JSON_IN)
reference_network = network.copy()

# ==========================================================================
# Create goals
# ==========================================================================

edge_goals = []
for idx, edge in enumerate(network.edges()):
    target_length = reference_network.edge_length(*edge)
    edge_goals.append(LengthGoal(idx, target_length))  # length goal
    # edge_goals.add_goal(LengthGoal(edge_key, target_length, weight))

# ==========================================================================
# Define optimization parameters
# Q - Force densities (select subset)
# P - Applied loads (Future)
# ==========================================================================

# ==========================================================================
# Define constraints - future
# ==========================================================================


# ==========================================================================
# Optimization
# ==========================================================================

optimizer = Optimizer(network, node_goals=[], edge_goals=edge_goals)
q_opt = optimizer.solve_scipy(loss_f=SquaredError(),
                              ub=-0.01795 / 0.123,  # upper bound for q = point load / brick length
                              method="SLSQP",
                              maxiter=200,
                              tol=1e-6)

# q_opt = optimizer.solve_scipy(loss_f=SquaredError(),
#                               ub=-0.01795 / 0.123,
#                               method="SLSQP",
#                               maxiter=200,
#                               tol=1e-6)

'''
form = static_equilibrium(network)
form = constrained_equilibrium(network, goals, constraints, bounds, method, iter, tol)
form = constrained_equilibrium(nework, loss, optimizer, maxiter, tol)
optimizer = Optimizer("method", maxiter, tol) or SLSQP(parameters, goals, constraints)
optimizer.add_parameter()
sort optimizable parameters with a mask matrix

TODO: how to guarantee ordering of nodes and edges?

q = np.array(network.force_densities())   # shape = (n_edges, )
P = np.array(network.loads())  # shape = (n_nodes, 3)

# store ordering of q and P

# combine into one long vector of optimization parameters, but preserve order
x = np.concatenate([q, np.ravel(P)])

# loss
def loss(model, x, y):
    y_pred = model(x)
    return anp.mean((y - y_pred) ** 2)

def loss(y, y_pred):
    return anp.mean((y - y_pred) ** 2)

def loss_for_grad(x, y, loss):
    y_pred = model(x)  # equilibrium model
    return loss(y, y_pred)
'''

# ==========================================================================
# Re-run force density to update model
# ==========================================================================

fd_opt = ForceDensity()(q_opt, network)

# ==========================================================================
# Update geometry
# ==========================================================================

xyz_opt = fd_opt["xyz"].tolist()
length_opt = fd_opt["lengths"].tolist()
res_opt = fd_opt["residuals"].tolist()

# update xyz coordinates on nodes
network.nodes_xyz(xyz_opt)

# update q values and lengths on edges
for idx, edge in enumerate(network.edges()):
    network.edge_attribute(edge, name="q", value=q_opt[idx])
    network.edge_attribute(edge, name="length", value=length_opt[idx])

# update residuals on nodes
for idx, node in enumerate(network.nodes()):
    for name, value in zip(["rx", "ry", "rz"], res_opt[idx]):
        network.node_attribute(node, name=name, value=value)

# ==========================================================================
# Export new JSON file for further operations
# ==========================================================================

if export_json:
    network.to_json(JSON_OUT)
    print("Exported network to: {}".format(JSON_OUT))

# ==========================================================================
# Viewer
# ==========================================================================

if view:
    viewer = App(width=1600, height=900)

    # equilibrated arch
    viewer.add(network,
               show_vertices=True,
               pointsize=12.0,
               show_edges=True,
               linecolor=Color.teal(),
               linewidth=4.0)

    # reference arch
    viewer.add(reference_network, show_points=False, linewidth=4.0)

    # draw lines betwen subject and target nodes
    for node in network.nodes():

        pt = network.node_coordinates(node)
        target_pt = reference_network.node_coordinates(node)
        viewer.add(Line(target_pt, pt))

        # draw reaction forces
        residual = network.node_attributes(node, names=["rx", "ry", "rz"])

        if length_vector(residual) < 0.001:
            continue

        residual_line = Line(pt, add_vectors(pt, residual))
        viewer.add(residual_line, color=Color.purple())

    # draw supports
    for node in network.supports():
        x, y, z = network.node_coordinates(node)
        viewer.add(Point(x, y, z), color=Color.green(), size=20)


    # show le crème
    viewer.show()