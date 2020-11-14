#!/usr/bin/env python3

from abc import ABC
from abc import abstractmethod

import jax.numpy as np

from force_density.equilibrium import force_equilibrium


__all__ = ["mean_squared_error", "MeanSquaredError"]


def reference_xyz(xyz, keys):
    """
    Gets the xyz coordinates of the reference nodes of a network.
    """
    # return xyz[keys, 2].reshape(-1, 1)
    return xyz[keys, :].reshape(-1, 3)


def mean_squared_error(q, targets, edges, xyz, free, fixed, loads):
    """
    A toy loss function.
    """
    xyz = force_equilibrium(q, edges, xyz, free, fixed, loads)
    zn = reference_xyz(xyz, free)

    return np.sum(np.square(zn - targets))  # mse - l2


class Loss(ABC):
    """
    The base class for all loss functions.
    """
    @abstractmethod
    def __call__(self):
        """
        Callable loss object
        """
        return

class MeanSquaredError(Loss):
    """
    The mean squared error loss
    """
    def __call__(self, q, targets, edges, xyz, free, fixed, loads):
        """
        Execute this.
        """
        xyz = force_equilibrium(q, edges, xyz, free, fixed, loads)
        zn = reference_xyz(xyz, free)
        return np.sum(np.square(zn - targets))  # what if np.mean instead?

        # error = 0.0
        # for node_key, goal in goals.items():
        #     error += np.square(np.array(goal.target) - np.array(goal.reference))
