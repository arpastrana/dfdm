#!/usr/bin/env python3

"""
Create a force network.
"""

# filepath stuff
import os

# directory to export json files
from force_density import JSON

# force equilibrium
from force_density.network import CompressionNetwork


# ==========================================================================
# Initial parameters
# ==========================================================================

JSON_IN = "/Users/arpj/princeton/phd/projects/light_vault/io/central_arch_cantilevered_light_vault.json"
JSON_OUT = os.path.abspath(os.path.join(JSON, "compression_network.json"))

# initial force density parameters
pz = -0.01795 # netwons - TODO: update as position changes?
q_0 = -1.5  # -2.5
brick_length = 0.123  # m
q_0_cantilever = pz / brick_length
print(q_0_cantilever)

# additional intermediate support
extra_support = 13

export_json = True

# ==========================================================================
# Load a Network from JSON
# ==========================================================================

# load network
network = CompressionNetwork.from_json(JSON_IN)

# print some info out
print(f"Funicular network # edges: {network.number_of_edges()}")

# ==========================================================================
# Boundary Conditions
# ==========================================================================

# the z lower-most two nodes
z_val = lambda x: network.node_attribute(x, "z")
z_nodes = sorted([node for node in network.nodes()], key=z_val)
fixed = z_nodes[:2] # + z_nodes[-1:]

# add supports
network.supports(fixed)

# set initial q to all nodes - TODO: how to find best initial value?
network.force_densities(q_0, keys=network.non_cantilevered_edges())

# set initial q to cantilevered edges
network.force_densities(q_0_cantilever, keys=network.cantilevered_edges())

# set initial point loads to all nodes of the network
network.applied_load([0.0, 0.0, pz])

# add extra supports
if extra_support:
    network.supports([extra_support])

# ==============================================================================
# Export new JSON file for further processing
# ==============================================================================

if export_json:
    network.to_json(JSON_OUT)
    print("Exported network to: {}".format(JSON_OUT))
