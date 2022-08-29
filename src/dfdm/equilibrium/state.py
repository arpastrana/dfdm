from dataclasses import dataclass

import autograd.numpy as np


# ==========================================================================
# Equilibrium state
# ==========================================================================

# TODO: A method that reindexes state arrays to match network indexing
@dataclass
class EquilibriumState:
    xyz: np.ndarray
    residuals: np.ndarray
    lengths: np.ndarray
    forces: np.ndarray
    force_densities: np.ndarray
