"""
Generate CSV files with statistics of the nodes and edges of a Network.
This is relative to a reference base network.
"""
# parser
import argparse

# writers
import csv

# math always helps
from math import fabs

# filepath stuff
import os

# containers
from collections import defaultdict

# geometric operations
from compas.geometry import length_vector
from compas.geometry import distance_point_point

# force equilibrium
from force_density import JSON

from force_density.network import CompressionNetwork

# ==========================================================================
# Parse input arguments
# ==========================================================================

my_parser = argparse.ArgumentParser()

my_parser.add_argument('json_ref', type=str)
my_parser.add_argument('json_target', type=str)
my_parser.add_argument('csv_node_out', type=str)
my_parser.add_argument('csv_edge_out', type=str)

my_parser.add_argument('node_shift_back', type=int)
my_parser.add_argument('round_prec', type=int)

args = my_parser.parse_args()

# ==========================================================================
# Initial parameters
# ==========================================================================

json_ref = args.json_ref
json_in = args.json_target

csv_node_out = args.csv_node_out
csv_edge_out = args.csv_edge_out

node_shift_back = args.node_shift_back
round_prec = args.round_prec

# ==========================================================================
# Load Networks from JSON
# ==========================================================================

reference_network = CompressionNetwork.from_json(json_ref)
network = CompressionNetwork.from_json(json_in)

# ==========================================================================
# Similarity check
# ==========================================================================

# assumes node keys are equal in both networks

# checks number of nodes is equal
assert network.number_of_nodes() == reference_network.number_of_nodes()

# check number of supports is the same
# common_supports = set(network.supports()) - set(reference_network.supports())
# assert len(common_supports) > 0.0
assert len(list(network.supports())) == len(list(reference_network.supports()))

# check that supports are in the same position of both networks
tol = 1e-6
for node in reference_network.supports():
    point_a = reference_network.node_coordinates(node)
    point_b = network.node_coordinates(node)

    if distance_point_point(point_a, point_b) > tol:
        raise ValueError(f"Positions of support at node {node} do not match!")

# ==========================================================================
# Compute node statistics
# ==========================================================================

# CSV column headers
nodes_columns = ["node_key",
                 "xref (m)",
                 "yref (m)",
                 "zref (m)",
                 "x (m)",
                 "y (m)",
                 "z (m)",
                 "distance (m)",
                 "rx (kN)",
                 "ry (kN)",
                 "rz (kN)",
                 "rforce (kN)"]

# data collector
nodes_data = []

for node in sorted(list(reference_network.nodes())):

    # default values are set to -1
    node_data = defaultdict(lambda: -1)

    # extract xyz coordinates
    xyz_ref = reference_network.node_coordinates(node)
    xyz = network.node_coordinates(node)

    # compute distance between points
    distance = distance_point_point(xyz, xyz_ref)

    # query residuals
    r_vector = network.residual_force(node)
    r_force = length_vector(r_vector)

    # parse nodal ref coordinates
    for name, coord in zip(["{}ref (m)".format(a) for a in "xyz"], xyz_ref):
        node_data[name] = coord

    # parse nodal coordinates
    for name, coord in zip(["{} (m)".format(a) for a in "xyz"], xyz):
        node_data[name] = coord

    # parse distance between nodes
    node_data["distance (m)"] = distance

    # parse residuals
    for name, comp in zip(["rx (kN)", "ry (kN)", "rz (kN)"], r_vector):
        node_data[name] = comp

    node_data["rforce (kN)"] = r_force

    # parse node key
    # shift nodes back for compatibility with 3rd-party numbering
    node = int(node - node_shift_back)
    node_data["node_key"] = node

    # round stuff
    for key, value in node_data.items():
        if isinstance(value, float):
            node_data[key] = round(value, round_prec)

    # append to nodes data
    nodes_data.append(node_data)

# ==========================================================================
# Compute edge statistics
# ==========================================================================

# CSV column headers
edges_columns = ["u_key",
                 "v_key",
                 "force (kN)",
                 "length (m)",
                 "length_ref (m)",
                 "length_diff (%)"]

# data collector
edges_data = []

# sort according to first node key on every eddge
for edge in sorted(list(reference_network.edges()), key=lambda x: x[0]):

    # defaults value is -1
    edge_data = defaultdict(lambda: -1)

    # fetch length
    length = network.edge_length(*edge)
    edge_data["length (m)"] = length

    # fetch force
    q = network.force_density(edge)
    edge_data["force (kN)"] = q * length

    # fetch reference length
    length_ref = reference_network.edge_length(*edge)
    edge_data["length_ref (m)"] = length_ref

    # calculate length difference
    length_diff = fabs(length - length_ref) / length_ref
    edge_data["length_diff (%)"] = length_diff * 100.0  # *100 to make it %

    # parse edge key
    # shift back edge numbering
    edge = [int(u - node_shift_back) for u in edge]
    edge_data["u_key"] = edge[0]
    edge_data["v_key"] = edge[1]

    # round stuff
    for key, value in edge_data.items():
        if isinstance(value, float):
            edge_data[key] = round(value, round_prec)

    edges_data.append(edge_data)

# ==========================================================================
# Write CSV data
# ==========================================================================

try:
    # node data
    with open(csv_node_out, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=nodes_columns)
        writer.writeheader()
        for data in nodes_data:
            writer.writerow(data)

    # edge data
    with open(csv_edge_out, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=edges_columns)
        writer.writeheader()
        for data in edges_data:
            writer.writerow(data)

except ValueError:
    print("Value error in data", data)

except IOError:
    print("I/O error")
