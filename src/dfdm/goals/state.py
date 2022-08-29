from dataclasses import dataclass

import autograd.numpy as np


@dataclass
class GoalState:
    prediction: np.ndarray
    target: np.ndarray
    weight: np.ndarray
