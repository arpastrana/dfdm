"""
A bunch of goals to strive for.
"""

from abc import abstractmethod


class Goal:
    """
    An abstract goal.
    """
    @abstractmethod
    def target(self):
        """
        """
        raise NotImplementedError
        return

    @abstractmethod
    def key(self):
        """
        """
        raise NotImplementedError

    @abstractmethod
    def reference(self):
        """
        """
        raise NotImplementedError

    # @abstractmethod
    # def update(self):
    #     """
    #     """
    #     raise NotImplementedError


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

    def key(self):
        """
        The key of a node in a network.
        """
        return self._key

    def target(self):
        """
        The xyz coordinates to reach.
        """
        return self._target

    def reference(self, xyz):
        """
        The current xyz coordinates of the node in a network.
        """
        # return network.node_coordinates(self.key())
        return xyz[self.key()]


class LengthGoal(Goal):
    """
    Make an edge of a network to reach certain length.
    """
    def __init__(self, key, target):
        """
        Let's get rollin'.
        """
        self._key = key
        self._target = target

    def key(self):
        """
        The key of a node in a network.
        """
        return self._key

    def target(self):
        """
        The xyz coordinates to reach.
        """
        return self._target

    def reference(self):
        """
        The current xyz coordinates of the node in a network.
        """
        # return network.node_coordinates(self.key())
        return


if __name__ == "__main__":

    from compas.datastructures import Network

    net = Network()
    net.add_node(key=0, x=0.0, y=0.0, z=0.0)

    goal = PointGoal(key=0, target=[0.0, 1.0, 0.0])
    print(goal.target())
    print(goal.key())
    print(goal.reference(net))
