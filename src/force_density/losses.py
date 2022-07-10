from abc import abstractmethod

import autograd.numpy as np

from force_density.equilibrium import ForceDensity
from force_density.equilibrium import EquilibriumModel
from force_density.equilibrium import EquilibriumSolver

from force_density.goals import PointGoal
from force_density.goals import ResidualForceGoal
from force_density.goals import ResidualVectorGoal
from compas.datastructures import Network


def squared_error(y, y_pred):
    """
    """
    return np.sum(np.square(y - y_pred))


def loss_another(q, loads, xyz_fixed, y):
    """
    """
    y_pred = model(q, loads, xyz_fixed)
    return mse(y_pred - y_target) + norm(model.parameters)


def loss_yet_another(predictions, targets, parameters):
    """
    """
    return mse(y_pred - y_target) + l2norm(parameters)


def loss_for_grad_api(q, model, goals, loss):
    """
    """
    # TODO: eqstate should output reindexed eqstate?
    # TODO: eqstate = reindex_equilibriumstate(eqstate, model)
    eqstate = model(q, loads, xyz_fixed)
    y_pred = predictor(eqstate)
    y = target(goals)

    return loss(y, y_pred)


# class Loss:
#     """
#     The base class for all loss functions.
#     """
#     @abstractmethod
#     def __call__(self):
#         """
#         Callable loss object
#         """
#         raise NotImplementedError


# class SquaredError(Loss):
#     """
#     The mean squared error loss
#     """
#     def __call__(self, q, loads, xyz, solver, goals):
#         """
#         Execute this.
#         """

#         # NOTE: loss function from here on
#         eqstate = solver(q, loads, xyz)
#         # indexing ma
#         # eqstate = reindex_equilibrium_state(eqstate, structure)
#         y_pred, y = collate_goals(goals, eqstate, solver.model)

#         return self.loss(y, y_pred)

#     def loss(self, x, y):
#         """
#         """
#         return np.sum(np.square(x - y))


# class SquaredError2(Loss):
#     """
#     The mean squared error loss
#     """
#     def __call__(self, q, data):
#         """
#         Execute this.
#         """
#         # access stuff
#         network = data["network"]
#         goals = data["goals"]

#         node_goals = data["node"]
#         edge_goals = data["edge"]

#         # do fd
#         fd_state = ForceDensity()(q, network)

#         # unpack
#         xyz = fd_state["xyz"]
#         lengths = fd_state["lengths"]
#         residuals = fd_state["residuals"]

#         # indexing maps
#         k_i = network.key_index()
#         uv_i = network.uv_index()

#         error = 0.0

#         # do node goals
#         for goal in node_goals:
#             y = np.array(goal.target())

#             if isinstance(goal, PointGoal):
#                 x = xyz[k_i[goal.key()], :]

#             elif isinstance(goal, ResidualForceGoal):
#                 x = residuals[k_i[goal.key()], :]
#                 x = np.linalg.norm(x)

#             elif isinstance(goal, ResidualVectorGoal):
#                 x = residuals[k_i[goal.key()], :]

#             error += self.penalize(x, y)

#         # do edge goals
#         for goal in edge_goals:
#             y = np.array(goal.target())
#             x = lengths[uv_i[goal.key()]]
#             error += self.penalize(x, y)

#         return error

#     def penalize(self, x, y):
#         """
#         """
#         return np.sum(np.square(x - y))


# class SquaredError3(Loss):
#     """
#     The mean squared error loss
#     """
#     def __call__(self, q, data):
#         """
#         Execute this.
#         """
#         # access stuff -- gotta move to optimizer
#         network = data["network"]
#         goals = data["goals"]
#         node_goals = goals["node"]
#         edge_goals = goals["edge"]

#         loads = np.array(list(network.node_loads()))
#         xyz = np.array(list(network.node_xyz()))
#         solver = EquilibriumSolver(network)  # model can be instantiated in solver
#         eqstate = solver(q, loads, xyz)

#         # indexing maps
#         # k_i = network.key_index()
#         # uv_i = network.uv_index()
#         k_i = solver.model.node_index
#         uv_i = solver.model.edge_index

#         error = 0.0

#         # do node goals
#         for goal in node_goals:
#             y = np.array(goal.target())

#             if isinstance(goal, PointGoal):
#                 x = eqstate.xyz[k_i[goal.key()], :]

#             elif isinstance(goal, ResidualForceGoal):
#                 x = eqstate.residuals[k_i[goal.key()], :]
#                 x = np.linalg.norm(x)

#             elif isinstance(goal, ResidualVectorGoal):
#                 x = eqstate.residuals[k_i[goal.key()], :]

#             error += self.penalize(x, y)

#         # do edge goals
#         for goal in edge_goals:
#             y = np.array(goal.target())
#             key = goal.key()
#             index = uv_i[key]
#             x = eqstate.lengths[index]
#             error += self.penalize(x, y)

#         return error

#     def penalize(self, x, y):
#         """
#         """
#         return np.sum(np.square(x - y))

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

#     def collate_goals(self, eqstate, goals):
#         """
#         """
#         references = []
#         targets = []

#         for key, goal in goals.items():
#             goal.update()  # update
#             references.append(goal.reference(eqstate))
#             targets.append(goal.target)

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
