"""
A bunch of goals to strive for.
"""

from abc import abstractmethod

import autograd.numpy as np

from compas.geometry import closest_point_on_line
from compas.geometry import closest_point_on_plane


class Goal:
    """
    A base goal.
    All goal subclasses must inherit from this class.
    """
    def __init__(self, key, target, weight):
        self._key = key
        self._target = target
        self._weight = weight

    @property
    def key(self):
        """
        The key of an element in a network.
        """
        return self._key

    @abstractmethod
    def weight(self):
        """
        The importance of the goal.
        """
        raise NotImplementedError

    @abstractmethod
    def prediction(self, eq_state, structure):
        """
        Computes the current value of the prediction value.
        """
        raise NotImplementedError

    @abstractmethod
    def target(self, prediction):
        """
        What we want.
        """
        raise NotImplementedError

    @abstractmethod
    def index(self, structure):
        """
        The index of the goal key in the canonical ordering of the equilibrium structure.
        """
        raise NotImplementedError


class PointGoal(Goal):
    """
    Make a node of a network to reach target xyz coordinates.
    """
    def __init__(self, key, target, weight=1.0):
        super(PointGoal, self).__init__(key=key,
                                        target=target,
                                        weight=weight)

    def index(self, structure):
        """
        The index of the goal key in a structure.
        """
        return structure.node_index[self.key]

    def weight(self):
        """
        The importance of the goal.
        """
        return np.array([self._weight] * 3)

    def prediction(self, eq_state, structure):
        """
        The current xyz coordinates of the node in a network.
        """
        index = self.index(structure)
        return eq_state.xyz[index, :]

    def target(self, prediction):
        """
        The target xyz coordinates.
        """
        return self._target


class LineGoal(PointGoal):
    """
    Pulls the xyz position of a node to a target line ray.
    """
    def __init__(self, key, target, weight=1.0):
        super(LineGoal, self).__init__(key=key,
                                       target=target,
                                       weight=weight)

    def target(self, prediction):
        """
        """
        line = self._target
        point = closest_point_on_line(prediction, line)

        return np.array(point, type=np.float64)


class PlaneGoal(PointGoal):
    """
    Pulls the xyz position of a node to a target plane.
    """
    def __init__(self, key, target, weight=1.0):
        super(PlaneGoal, self).__init__(key=key,
                                        target=target,
                                        weight=weight)

    def target(self, prediction):
        """
        """
        point = prediction
        plane = self._target
        point = closest_point_on_plane(point, plane)

        return np.array(point, dtype=np.float64)


class LengthGoal(Goal):
    """
    Make an edge of a network to reach a certain length.
    """
    def __init__(self, key, target, weight=1.0):
        super(LengthGoal, self).__init__(key=key,
                                         target=target,
                                         weight=weight)

    def weight(self):
        """
        The importance of the goal.
        """
        return self._weight

    def index(self, structure):
        return structure.edge_index[self.key]

    def prediction(self, eq_state, structure):
        """
        The current edge length.
        """
        index = self.index(structure)
        return eq_state.lengths[index]

    def target(self, prediction):
        """
        The target to strive for.
        """
        return self._target


class ForceGoal(Goal):
    """
    Make an edge of a network to reach a certain force.
    """
    def __init__(self, key, target, weight=1.0):
        super(ForceGoal, self).__init__(key=key,
                                        target=target,
                                        weight=weight)

    def weight(self):
        """
        The importance of the goal.
        """
        return self._weight

    def index(self, structure):
        return structure.edge_index[self.key]

    def prediction(self, eq_state, structure):
        """
        The current edge length.
        """
        index = self.index(structure)
        return eq_state.forces[index]

    def target(self, prediction):
        """
        The target to strive for.
        """
        return self._target


class LoadPathGoal(Goal):
    """
    Make an edge of a network to reach a certain load path magnitude.

    The load path of an edge is the absolute value of the product of the
    the force on the edge time its length.
    """
    def __init__(self, key, target, weight=1.0):
        super(LoadPathGoal, self).__init__(key=key,
                                           target=target,
                                           weight=weight)

    def weight(self):
        """
        The importance of the goal.
        """
        return self._weight

    def index(self, structure):
        return structure.edge_index[self.key]

    def prediction(self, eq_state, structure):
        """
        The current edge length.
        """
        index = self.index(structure)
        return np.abs(eq_state.lengths[index] * eq_state.forces[index])

    def target(self, prediction):
        """
        The target to strive for.
        """
        return self._target


class ResidualVectorGoal(PointGoal):
    """
    Make the residual force in a network to match the magnitude and direction of a vector.
    """
    def __init__(self, key, target, weight=1.0):
        super(ResidualVectorGoal, self).__init__(key=key,
                                                 target=target,
                                                 weight=weight)

    def prediction(self, eq_state, structure):
        """
        The residual at the the predicted node of the network.
        """
        index = self.index(structure)
        return eq_state.residuals[index, :]


class ResidualForceGoal(ResidualVectorGoal):
    """
    Make the residual force in a network to match a non-negative magnitude.
    """
    def __init__(self, key, target, weight=1.0):
        assert target >= 0.0, "Only a non-negative target force is supported!"
        super(ResidualForceGoal, self).__init__(key=key,
                                                target=target,
                                                weight=weight)

    def weight(self):
        """
        The importance of the goal.
        """
        return self._weight

    def prediction(self, eq_state, structure):
        """
        The residual at the the predicted node of the network.
        """
        index = self.index(structure)
        residual = eq_state.residuals[index, :]
        return np.linalg.norm(residual)

    def target(self, prediction):
        """
        """
        return self._target


class ResidualDirectionGoal(ResidualVectorGoal):
    """
    Make the residual force in a network to match the direction of a vector.

    Another effective proxy for cosine distance can be obtained by
    L2 normalisation of the vectors, followed by the application of normal
    Euclidean distance. Using this technique each term in each vector is
    first divided by the magnitude of the vector, yielding a vector of unit
    length. Then, it is clear, the Euclidean distance over the end-points
    of any two vectors is a proper metric which gives the same ordering as
    the cosine distance (a monotonic transformation of Euclidean distance;
    see below) for any comparison of vectors, and furthermore avoids the
    potentially expensive trigonometric operations required to yield
    a proper metric.
    """
    def __init__(self, key, target, weight=1.0):
        super(ResidualDirectionGoal, self).__init__(key=key,
                                                    target=target,
                                                    weight=weight)

    def prediction(self, eq_state, structure):
        """
        The residual at the the predicted node of the network.
        """
        index = self.index(structure)
        residual = eq_state.residuals[index, :]

        return residual / np.linalg.norm(residual)  # unitized residual

    def target(self, prediction):
        """
        """
        return self._target / np.linalg.norm(self._target)
