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
    def __init__(self, key, target):
        self._key = key
        self._target = target

    @property
    def key(self):
        """
        The key of an element in a network.
        """
        return self._key

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
    def __init__(self, node_key, point):
        super(PointGoal, self).__init__(key=node_key, target=point)

    def index(self, structure):
        """
        """
        return structure.node_index[self.key]

    def prediction(self, eq_state, structure):
        """
        The current xyz coordinates of the node in a network.
        """
        index = self.index(structure)
        return eq_state.xyz[index, :]

    def target(self, prediction):
        """
        """
        return self._target


class LineGoal(Goal):
    """
    Pulls the xyz position of a node to a target line ray.
    """
    def __init__(self, node_key, line):
        super(LineGoal, self).__init__(key=node_key, target=line)

    def index(self, structure):
        """
        """
        return structure.node_index[self.key]

    def prediction(self, eq_state, structure):
        """
        The current xyz coordinates of the node in a network.
        """
        index = self.index(structure)
        return eq_state.xyz[index, :]

    def target(self, prediction):
        """
        """
        line = self._target
        return np.array(closest_point_on_line(prediction, line), dtype=np.float64)


class PlaneGoal(Goal):
    """
    Pulls the xyz position of a node to a target plane.
    """
    def __init__(self, node_key, plane):
        super(PlaneGoal, self).__init__(key=node_key, target=plane)

    def index(self, structure):
        """
        """
        return structure.node_index[self.key]

    def prediction(self, eq_state, structure):
        """
        The current xyz coordinates of the node in a network.
        """
        index = self.index(structure)
        return eq_state.xyz[index, :]

    def target(self, prediction):
        """
        """
        point = prediction
        plane = self._target
        return np.array(closest_point_on_plane(point, plane), dtype=np.float64)


class LengthGoal(Goal):
    """
    Make an edge of a network to reach a certain length.
    """
    def __init__(self, edge_key, length):
        super(LengthGoal, self).__init__(key=edge_key, target=length)

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


class ResidualVectorGoal(Goal):
    """
    Make the residual force in a network to match the magnitude and direction of a vector.
    """
    def __init__(self, node_key, vector):
        super(ResidualVectorGoal, self).__init__(key=node_key, target=vector)

    def index(self, structure):
        """
        """
        return structure.node_index[self.key]

    def prediction(self, eq_state, structure):
        """
        The residual at the the predicted node of the network.
        """
        index = self.index(structure)
        return eq_state.residuals[index, :]

    def target(self, prediction):
        """
        """
        return self._target


class ResidualForceGoal(Goal):
    """
    Make the residual force in a network to match a non-negative magnitude.
    """
    def __init__(self, node_key, force):
        assert force >= 0.0
        super(ResidualForceGoal, self).__init__(key=node_key, target=force)

    def index(self, structure):
        """
        """
        return structure.node_index[self.key]

    def prediction(self, eq_state, structure):
        """
        The residual at the the predicted node of the network.
        """
        residual = eq_state.residuals[self.key, :]
        return np.linalg.norm(residual)

    def target(self, prediction):
        """
        """
        return self._target


class ResidualDirectionGoal(Goal):
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
    def __init__(self, node_key, vector):
        super(ResidualDirectionGoal, self).__init__(key=node_key, target=vector)

    def index(self, structure):
        """
        """
        return structure.node_index[self.key]

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
