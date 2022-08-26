import autograd.numpy as np

from dfdm.goals import ScalarGoal
from dfdm.goals.networkgoal import NetworkGoal


class NetworkLoadPathGoal(ScalarGoal, NetworkGoal):
    """
    Make the total load path of a network to reach a target magnitude.

    The load path of an edge is the absolute value of the product of the
    the force on the edge time its length.
    """
    def __init__(self, target, weight=1.0):
        super().__init__(key=None, target=target, weight=weight)

    def prediction(self, eq_state, *args, **kwargs):
        """
        The current load path of the network.
        """
        load_path = np.sum(np.abs(eq_state.lengths * eq_state.forces))

        return np.atleast_1d(load_path)
