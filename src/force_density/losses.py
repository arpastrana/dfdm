#!/usr/bin/env python3

from abc import ABC
from abc import abstractmethod

import jax.numpy as np

from force_density.equilibrium import force_equilibrium
from force_density.equilibrium import ForceDensity


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


class MeanSquaredErrorGoals(Loss):
    """
    The mean squared error loss
    """
    def __init__(self):
        self.params = None

    def __call__(self, q, params):
        """
        Execute this.
        """
        # access stuff
        network = params["network"]
        goals = params["goals"]
        fd = params["fd"]

        # do fd
        xyz = fd(q)

        # update network
        network = network.copy()

        for i, node in enumerate(network.nodes()):
            for name, value in zip("xyz", xyz[i, :]):
                network.node_attribute(key=node, name=name, value=value)

        # network.nodes_xyz(xyz.tolist())

        # compute error
        error = 0.0
        for goal in goals.values():
            difference = np.array(goal.target()) - np.array(goal.reference(network))
            error += np.sum(np.square(difference))


        return error
        # return np.sum(np.square(references - targets))
