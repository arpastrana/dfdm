import autograd.numpy as np

from dfdm.goals import VectorGoal
from dfdm.goals.edgegoal import EdgeGoal


class EdgeDirectionGoal(VectorGoal, EdgeGoal):
    """
    Make the direction of the edge of a network to be parallel to a target vector.
    """
    def __init__(self, key, target, weight=1.0):
        super().__init__(key=key, target=target, weight=weight)

    def prediction(self, eq_state, index):
        """
        The edge vector in the network.
        """
        vector = eq_state.vectors[index, :]
        return vector / np.linalg.norm(vector)

    def target(self, prediction):
        """
        The target vector.
        """
        return self._target / np.linalg.norm(self._target)
