"""
Solve a constrained force density problem using gradient based optimization.
"""
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
from force_density.equilibrium import ForceDensity
from force_density.network import CompressionNetwork
from force_density.losses import SquaredError
from force_density.goals import LengthGoal
from force_density.goals import ResidualForceGoal
from force_density.goals import ResidualVectorGoal
from force_density.optimization import Optimizer

# ==========================================================================
# Initial parameters
# ==========================================================================

arch_length = 5.0
num_segments = 10
q_init = -1
pz = -0.1

# ==========================================================================
# Create a catenary curve
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
# create a copy of the intial network for visualization
reference_network = network.copy()

# ==========================================================================
# Define structural system
# ==========================================================================

# assign supports
gkeys = [geometric_key(point) for point in [start, end]]
support_indices = [node for node in network.nodes() if geometric_key(network.node_coordinates(node)) in gkeys]
network.supports(keys=support_indices)

# set initial q to all nodes
network.force_densities(q_init, keys=None)

# set initial point loads to all nodes of the network
network.applied_load([0.0, 0.0, pz])

# ==========================================================================
# Define goals
# ==========================================================================

edge_goals = []
node_goals = []

key_index = network.key_index()
index = key_index.get(0)
# node_goals.append(ResidualForceGoal(index, 2.0))
# node_goals.append(ResidualVectorGoal(index, [-1.0, 0.0, -1.0]))

for idx, edge in enumerate(network.edges()):
    target_length = reference_network.edge_length(*edge)
    edge_goals.append(LengthGoal(idx, target_length))  # length goal

# ==========================================================================
# Optimization
# ==========================================================================

optimizer = Optimizer(network, node_goals=node_goals, edge_goals=edge_goals)
q_opt = optimizer.solve_scipy(loss_f=SquaredError(),
                              ub=-0.01795 / 0.123,  # upper bound for q = point load / brick length
                              method="SLSQP",
                              maxiter=500,
                              tol=1e-9)

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
    # network.edge_attribute(edge, name="q", value=q_opt[idx])
    network.edge_attribute(edge, name="length", value=length_opt[idx])

# update residuals on nodes
for idx, node in enumerate(network.nodes()):
    for name, value in zip(["rx", "ry", "rz"], res_opt[idx]):
        network.node_attribute(node, name=name, value=value)

# ==========================================================================
# Viewer
# ==========================================================================

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

for node in network.nodes():

    pt = network.node_coordinates(node)

    # draw lines betwen subject and target nodes
    # target_pt = reference_network.node_coordinates(node)
    # viewer.add(Line(target_pt, pt))

    # draw residual forces
    residual = network.node_attributes(node, names=["rx", "ry", "rz"])

    if length_vector(residual) < 0.001:
        continue

    print(node, residual, length_vector(residual))
    residual_line = Line(pt, add_vectors(pt, residual))
    viewer.add(residual_line,
               linewidth=2.0,
               color=Color.purple())

# draw applied loads
for node in network.nodes():
    pt = network.node_coordinates(node)

    load = network.node_attributes(node, names=["px", "py", "pz"])

    viewer.add(Line(pt, add_vectors(pt, load)),
               linewidth=2.0,
               color=Color.green().darkened())

# draw supports
for node in network.supports():

    x, y, z = network.node_coordinates(node)
    viewer.add(Point(x, y, z), color=Color.green(), size=20)


# show le crème
viewer.show()
