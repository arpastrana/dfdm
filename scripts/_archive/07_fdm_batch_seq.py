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

# numpy
import numpy as np

# containers
from collections import defaultdict

# geometric operations
from compas.geometry import length_vector
from compas.geometry import distance_point_point

# force equilibrium
from force_density.network import CompressionNetwork

# ==========================================================================
# Parse input arguments
# ==========================================================================

my_parser = argparse.ArgumentParser()

my_parser.add_argument('json_ref', type=str)
my_parser.add_argument('json_target', type=str)
my_parser.add_argument('csv_seq_out', type=str)

my_parser.add_argument('node_shift_back', type=int)
my_parser.add_argument('round_prec', type=int)

my_parser.add_argument('ref_supports', type=str)

args = my_parser.parse_args()

# ==========================================================================
# Initial parameters
# ==========================================================================

json_ref = args.json_ref
json_in = args.json_target

csv_seq_out = args.csv_seq_out

node_shift_back = args.node_shift_back
round_prec = args.round_prec

ref_supports = [int(x) for x in args.ref_supports.split(",")]

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
assert len(list(network.supports())) == len(list(reference_network.supports()))

# check that supports are in the same position of both networks
tol = 1e-6
for node in reference_network.supports():
    point_a = reference_network.node_coordinates(node)
    point_b = network.node_coordinates(node)

    if distance_point_point(point_a, point_b) > tol:
        raise ValueError(f"Positions of support at node {node} do not match!")

# check min number of supports is met
msg = "***Number of supports don't match. Skipping this experiment.***"
assert len(list(network.supports())) > len(ref_supports), msg

# ==========================================================================
# Compute sequence sweep statistics
# ==========================================================================

# CSV column headers
seq_columns = ["node_key",
               "distance (m)",
               "opt_error_mean (%)",
               "opt_error_std (%)",
               "rforce_node_key (kN)"]

# append reference supports at end, sorted and shifted back
for key in sorted(ref_supports):
    seq_columns.append("rforce_{} (kN)".format(int(key - node_shift_back)))

# default values are set to -1
seq_data = defaultdict(lambda: None)

# shift nodes back for compatibility with 3rd-party numbering
node_key = set(network.supports()) - set(ref_supports)
seq_data["node_key"] = int(list(node_key).pop() - node_shift_back)

# ==========================================================================
# Calculate sum of distance to nodes
# ==========================================================================

distance = 0.0
for node in sorted(list(reference_network.nodes())):

    # extract xyz coordinates
    xyz_ref = reference_network.node_coordinates(node)
    xyz = network.node_coordinates(node)

    # compute distance between points
    dist = distance_point_point(xyz, xyz_ref)
    distance += dist

seq_data["distance (m)"] = distance

# ==========================================================================
# Query residuals
# ==========================================================================

for node in reference_network.supports():
    r_vector = network.residual_force(node)
    r_force = length_vector(r_vector)

    shifted_node = int(node - node_shift_back)
    r_tag = "rforce_node_key (kN)"
    if node in ref_supports:
        r_tag = "rforce_{} (kN)".format(shifted_node)

    seq_data[r_tag] = r_force

# ==========================================================================
# Compute edge statistics
# ==========================================================================

length_differences = []

for edge in reference_network.edges():
    length = network.edge_length(*edge)
    length_ref = reference_network.edge_length(*edge)
    length_diff = fabs(length - length_ref) / length_ref
    length_differences.append(length_diff * 100.0)  # * 100 to make it %

# calculate mean
seq_data["opt_error_mean (%)"] = np.mean(length_differences)

# calculate std dev
seq_data["opt_error_std (%)"] = np.std(length_differences)

# ==========================================================================
# Write CSV data
# ==========================================================================

# round stuff
for key, value in seq_data.items():
    if isinstance(value, float):
        seq_data[key] = round(value, round_prec)

try:
    # sequence data
    file_exists = os.path.isfile(csv_seq_out)

    with open(csv_seq_out, 'a+') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=seq_columns)

        if not file_exists:
            writer.writeheader()

        writer.writerow(seq_data)

except ValueError:
    print("Value error in data", seq_data)

except IOError:
    print("I/O error")
