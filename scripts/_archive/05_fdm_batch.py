#!/usr/bin/env python3

"""
Create a force network and run autograd optimization in batch mode.
"""

import argparse

from force_density.network import CompressionNetwork

from force_density.equilibrium import ForceDensity

from force_density.losses import SquaredError

from force_density.goals import LineGoal

from force_density.optimization import Optimizer


# ==========================================================================
# Parse input arguments
# ==========================================================================

my_parser = argparse.ArgumentParser()

my_parser.add_argument('json_in', type=str)
my_parser.add_argument('json_network_out', type=str)
my_parser.add_argument('json_opt_out', type=str)

my_parser.add_argument('extra_support', type=int)
my_parser.add_argument('pz', type=float)
my_parser.add_argument('q0', type=float)
my_parser.add_argument('brick_length', type=float)

my_parser.add_argument('opt_method', type=str)
my_parser.add_argument('max_iters', type=int)
my_parser.add_argument('tol', type=float)

args = my_parser.parse_args()

# ==========================================================================
# Initial parameters
# ==========================================================================

JSON_IN = args.json_in
JSON_NETWORK_OUT = args.json_network_out
JSON_OPT_OUT = args.json_opt_out

# initial force density parameters
extra_support = args.extra_support
pz = args.pz # -0.01795 # netwons - TODO: update as position changes?
q_0 = args.q0 # -1.5  # -2.5
brick_length = args.brick_length# 0.123  # m
q_0_cantilever = pz / brick_length

method = args.opt_method  # "SLSQP"
maxiter = args.max_iters  # 200
tol = args.tol  # 1e-9
ub = pz / brick_length  # point load / brick length

# ==========================================================================
# Load a Network from JSON
# ==========================================================================

network = CompressionNetwork.from_json(JSON_IN)
reference_network = network.copy()

# ==========================================================================
# Boundary Conditions
# ==========================================================================

# the first and last nodes
sorted_nodes = sorted(list(network.nodes()))
fixed = [sorted_nodes[-1], sorted_nodes[0]]

# add supports
network.supports(fixed)

# set initial q to all nodes - TODO: how to find best initial value?
network.force_densities(q_0, keys=network.non_cantilevered_edges())

# set initial q to cantilevered edges
network.force_densities(q_0_cantilever, keys=network.cantilevered_edges())

# set initial point loads to all nodes of the network
network.applied_load([0.0, 0.0, pz])

# add extra supports
if extra_support is not None:
    network.supports([extra_support])

# ==============================================================================
# Export new JSON file for further processing
# ==============================================================================

network.to_json(JSON_NETWORK_OUT)

# ==========================================================================
# Initialize loss
# ==========================================================================

loss_f = SquaredError()

# ==========================================================================
# Create goals
# ==========================================================================

node_goals = []
edge_goals = []

for idx, edge in enumerate(network.edges()):
    target_length = reference_network.edge_length(*edge)
    edge_goals.append(LineGoal(idx, target_length))

# ==========================================================================
# Optimization
# ==========================================================================

optimizer = Optimizer(network, node_goals, edge_goals)
q_opt = optimizer.solve_scipy(loss_f, ub, method, maxiter, tol)

# ==========================================================================
# Force Density
# ==========================================================================

fd = ForceDensity()
fd_opt = fd(q_opt, network)

# ==========================================================================
# Update Geometry
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

network.to_json(JSON_OPT_OUT)
print("Exported network to: {}".format(JSON_OPT_OUT))
