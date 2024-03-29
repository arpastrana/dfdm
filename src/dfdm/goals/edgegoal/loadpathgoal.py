import autograd.numpy as np

from dfdm.goals import ScalarGoal
from dfdm.goals.edgegoal import EdgeGoal


class EdgeLoadPathGoal(ScalarGoal, EdgeGoal):
    """
    Make an edge of a network to reach a certain load path magnitude.

    The load path of an edge is the absolute value of the product of the
    the force on the edge time its length.
    """
    def __init__(self, key, target, weight=1.0):
        super().__init__(key=key, target=target, weight=weight)

    def prediction(self, eq_state, index):
        """
        The current edge load path.
        """
        load_path = eq_state.lengths[index] * eq_state.forces[index]
        return np.atleast_1d(np.abs(load_path))
