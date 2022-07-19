from abc import abstractmethod

import autograd.numpy as np


class Goal:
    """
    The base goal.

    All goal subclasses must inherit from this class.
    """
    def __init__(self, key, target, weight):
        self._key = key
        self._target = target
        self._weight = weight

    @abstractmethod
    def key(self):
        """
        The key of an element in a network.
        """
        raise NotImplementedError

    @abstractmethod
    def weight(self):
        """
        The importance of the goal.
        """
        raise NotImplementedError

    @abstractmethod
    def prediction(self, eq_state, index):
        """
        The current reference value in the equilibrium state.
        """
        raise NotImplementedError

    @abstractmethod
    def target(self, prediction):
        """
        The target to strive for.
        """
        raise NotImplementedError

    @abstractmethod
    def index(self, structure):
        """
        The index of the goal key in the canonical ordering of the equilibrium structure.
        """
        raise NotImplementedError


class ScalarGoal:
    """
    A goal that is expressed as a scalar quantity.
    """
    def weight(self):
        """
        The importance of the goal
        """
        return np.atleast_1d(self._weight)

    def target(self, prediction):
        """
        The target to strive for.
        """
        return np.atleast_1d(self._target)


class VectorGoal:
    """
    A goal that is expressed as a vector 3D quantity.
    """
    def weight(self):
        """
        The importance of the goal
        """
        return np.array([self._weight] * 3, dtype=np.float64)

    def target(self, prediction):
        """
        The target to strive for.
        """
        return np.array(self._target, dtype=np.float64)
