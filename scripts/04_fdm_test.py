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

view = False
JSON_IN = os.path.abspath(os.path.join(JSON, "compression_network.json"))

# ==========================================================================
# Load Network with boundary conditions from JSON
# ==========================================================================

network = CompressionNetwork.from_json(JSON_IN)
reference_network = network.copy()


edges, xyz, free, fixed, loads = network.fd_parameters()


from force_density.equilibrium import EquilibriumModel
model = EquilibriumModel(network)

from compas.numerical import connectivity_matrix
import numpy as np
assert np.allclose(np.array(model.connectivity), connectivity_matrix(edges, "array"))
assert free == model.free_nodes
assert fixed == model.fixed_nodes

print("Done!")
