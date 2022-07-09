"""
A bunch of goals to strive for.
"""

from abc import abstractmethod
import autograd.numpy as np


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

    # @property
    # def target(self):
    #     """
    #     The target to strive for.
    #     """
    #     return self._target

    @abstractmethod
    def index(self, model):
        """
        The index of the goal key in the canonical ordering of the equilibrium model.
        """
        raise NotImplementedError

    @abstractmethod
    def reference(self, eq_state, model):
        """
        Computes the current value of the reference value.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self):
        """
        Update the value of target based on the current value of reference.
        """
        raise NotImplementedError


class LengthGoal(Goal):
    """
    Make an edge of a network to reach a certain length.
    """
    def __init__(self, edge_key, length):
        super(LengthGoal, self).__init__(key=edge_key,
                                         target=length)

    def index(self, model):
        return model.edge_index[self.key]

    def reference(self, eq_state, model):
        """
        The current edge length.
        """
        index = self.index(model)
        return eq_state.lengths[index]

    def update(self):
        """
        """
        pass

    def target(self):
        """
        The target to strive for.
        """
        return self._target


class PointGoal(Goal):
    """
    Make a node of a network to reach target xyz coordinates.
    """
    def __init__(self, node_key, point):
        super(PointGoal, self).__init__(key=node_key, target=point)

    def index(self, model):
        return model.node_index[self.key]

    def reference(self, eq_state, model):
        """
        The current xyz coordinates of the node in a network.
        """
        index = self.index(model)
        return eq_state.xyz[index, :]

    def target(self):
        """
        """
        return self._target

    def update(self):
        """
        """
        pass


class ResidualVectorGoal(Goal):
    """
    Make the residual force in a network to match the magnitude and direction of a vector.
    """
    def __init__(self, node_key, vector):
        super(ResidualVectorGoal, self).__init__(key=node_key, target=vector)

    def reference(self, eq_state):
        """
        The residual at the the reference node of the network.
        """
        return eq_state.residuals[self.key(), :]

    def update(self):
        """
        """
        pass


class ResidualForceGoal(Goal):
    """
    Make the residual force in a network to match a given magnitude.
    """
    def __init__(self, node_key, vector):
        super(ResidualVectorGoal, self).__init__(key=node_key, target=vector)

    def reference(self, eq_state):
        """
        The residual at the the reference node of the network.
        """
        residual = eq_state.residuals[self.key(), :]
        return np.linalg.norm(residual)

    def update(self):
        """
        """
        pass
