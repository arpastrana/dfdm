import autograd.numpy as np

from dfdm.goals import ScalarGoal
from dfdm.goals.edgegoal import EdgeGoal


class LengthGoal(ScalarGoal, EdgeGoal):
    """
    Make an edge of a network to reach a certain length.
    """
    def __init__(self, key, target, weight=1.0):
        super().__init__(key=key, target=target, weight=weight)

    def prediction(self, eq_state, index):
        """
        The current edge length.
        """
        length = eq_state.lengths[index]
        return np.atleast_1d(length)


class LengthMinGoal(ScalarGoal, EdgeGoal):
    """
    Make an edge of a network to reach a certain length.
    """
    def __init__(self, key, target, weight=1.0):
        super().__init__(key=key, target=target, weight=weight)

    def prediction(self, eq_state, index):
        """
        The current edge length.
        """
        length = eq_state.lengths[index]
        return np.atleast_1d(length)
