from abc import abstractmethod

import autograd.numpy as np

from force_density.equilibrium import force_equilibrium
from force_density.equilibrium import ForceDensity
from force_density.goals import PointGoal
from force_density.goals import ResidualForceGoal
from force_density.goals import ResidualVectorGoal
from compas.datastructures import Network


__all__ = ["SquaredError"]


class Loss:
    """
    The base class for all loss functions.
    """
    @abstractmethod
    def __call__(self):
        """
        Callable loss object
        """
        raise NotImplementedError


class SquaredError(Loss):
    """
    The mean squared error loss
    """
    def __call__(self, q, params):
        """
        Execute this.
        """
        # access stuff
        network = params["network"]
        goals = params["goals"]
        # fd = params["fd"]

        node_goals = goals["node"]
        edge_goals = goals["edge"]

        # do fd
        fd_state = ForceDensity()(q, network)
        xyz = fd_state["xyz"]
        lengths = fd_state["lengths"]
        residuals = fd_state["residuals"]

        # indexing maps
        k_i = network.key_index()
        uv_i = network.uv_index()

        error = 0.0

        # do node goals
        for goal in node_goals:
            y = np.array(goal.target())

            if isinstance(goal, PointGoal):
                x = xyz[k_i[goal.key()], :]

            elif isinstance(goal, ResidualForceGoal):
                x = residuals[k_i[goal.key()], :]
                x = np.linalg.norm(x)

            elif isinstance(goal, ResidualVectorGoal):
                x = residuals[k_i[goal.key()], :]

            error += self.penalize(x, y)

        # do edge goals
        for goal in edge_goals:
            y = np.array(goal.target())
            x = lengths[uv_i[goal.key()]]
            error += self.penalize(x, y)

        return error

    def penalize(self, x, y):
        """
        """
        return np.sum(np.square(x - y))


# class MeanSquaredErrorGoals(Loss):
#     """
#     The mean squared error loss
#     """
#     def __call__(self, q, params):
#         """
#         Execute this.
#         """
#         # access stuff
#         network = params["network"]
#         goals = params["goals"]
#         fd = params["fd"]
#
#         # do fd
#         # network.fd(q)
#         # internally it should update all dictionary states
#         xyz = fd(q, network)
#         network = network.copy()
#         network.nodes_xyz(xyz)

#         references, targets = self.process_goals(network, goals)
#         return np.sum(np.square(references - targets))

#     def process_goals(self, network, goals):
#         """
#         """
#         references = []
#         targets = []

#         for key, goal in goals.items():
#             references.append(goal.reference(network))
#             targets.append(goal.target())

#         return np.array(references), np.array(targets)

# class MeanSquaredError(Loss):
#     """
#     The mean squared error loss
#     """
#     def __call__(self, q, params):
#         """
#         Execute this.
#         """
#         xyz = force_equilibrium(q, *params)
#         references = xyz[free, :]
#         return np.sum(np.square(references - targets))  # what if np.mean instead?
