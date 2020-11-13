#!/usr/bin/env python3
"""
A bunch of goals to solve
"""

from abc import ABC
from abc import abstractmethod
from abc import abstractproperty


class Goal(ABC):
    """
    An abstract goal.
    """
    @abstractproperty
    def target(self):
        """
        """
        return

    @abstractproperty
    def key(self):
        """
        """
        return

    @abstractproperty
    def reference(self):
        """
        """
        return

    @abstractmethod
    def update(self, network):
        """
        """
        return


class PointGoal(Goal):
    """
    Make a node of a network to reach target xyz coordinates.
    """
    def __init__(self, key, target):
        """
        Let's get rollin'.
        """
        self._key = key
        self._target = target
        self._reference = None

    @property
    def key(self):
        """
        The key of a node in a network.
        """
        return self._key

    @property
    def target(self):
        """
        The xyz coordinates to reach.
        """
        return self._target

    @property
    def reference(self):
        """
        The current xyz coordinates of the node in a network.
        """
        return self._reference

    @reference.setter
    def reference(self, xyz):
        self._reference = xyz

    def update(self, network):
        """
        Extract and store the necessary geometric information from a network.
        """
        self.reference = network.node_coordinates(self.key)


if __name__ == "__main__":

    from compas.datastructures import Network

    net = Network()
    net.add_node(key=0, x=0.0, y=0.0, z=0.0)

    goal = PointGoal(key=0, target=[0.0, 1.0, 0.0])
    print(goal.target)
    print(goal.key)
    print(goal.reference)

    goal.update(net)

    print(goal.reference)
