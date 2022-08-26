from dataclasses import dataclass

import autograd.numpy as np


@dataclass
class GoalState:
    name: str
    target: np.ndarray
    prediction: np.ndarray
    weight: np.ndarray
