"""
A gradient-based optimizer.
"""
from time import time

from functools import partial

import autograd.numpy as np
from autograd import grad

from scipy.optimize import minimize
from scipy.optimize import Bounds

from force_density.equilibrium import EquilibriumSolver


# ==========================================================================
# BaseOptimizer
# ==========================================================================

class BaseOptimizer():
    def __init__(self, name):
        self.name = name

    def minimize(self, network, loss, goals, bounds, maxiter, tol):
        # returns the optimization result: dataclass OptimizationResult
        """
        Perform gradient descent with Scipy.
        """
        name = self.name

        # array-ize parameters
        q = np.array(network.force_densities(), dtype=np.float64)
        loads = np.array(list(network.node_loads()), dtype=np.float64)
        xyz = np.array(list(network.node_xyz()), dtype=np.float64)  # probably should be xyz_fixed only

        # NOTE: It is immutable
        # access stuff -- gotta move to optimizer
        # TODO: Rename EquilibriumSolver to EquilibriumModel and EquilibriumModel to other
        solver = EquilibriumSolver(network)  # model can be instantiated in solver

        # loss matters
        loss_f = partial(loss_base,
                         solver=solver,
                         loads=loads,
                         xyz=xyz,
                         goals=goals,
                         loss=loss)

        grad_loss = grad(loss_f)  # grad w.r.t. first arg

        # parameter bounds
        # bounds makes a re-index from one count system to the other
        # bounds = optimization_bounds(model, bounds)
        lb=bounds[0] or -np.inf
        ub=bounds[1] or np.inf
        bounds = Bounds(lb=lb, ub=ub)

        # parameter constraints
        # constraints = optimization_constraints(model, constraints)

        # scipy optimization
        start_time = time()
        print("Optimization started...")

        # minimize
        res_q = minimize(fun=loss_f,
                         jac=grad_loss,
                         method=name,
                         x0=q,
                         tol=tol,
                         bounds=bounds,
                         options={"maxiter": maxiter})
        # print out
        print(res_q.message)
        print(f"Output loss in {res_q.nit} iterations: {res_q.fun}")
        print(f"Elapsed time: {time() - start_time} seconds")

        return res_q.x

# ==========================================================================
# Optimizers
# ==========================================================================

class SLSQP(BaseOptimizer):
    """
    The sequential least-squares programming optimizer.
    """
    def __init__(self):
        super(SLSQP, self).__init__(name="SLSQP")


class BFGS(BaseOptimizer):
    """
    The Boyd-Fletcher-Floyd-Shannon optimizer.
    """
    def __init__(self):
        super(BFGS, self).__init__(name="BFGS")

# ==========================================================================
# Utilities
# ==========================================================================

def collate_goals(goals, eqstate, model):
    """
    TODO: An optimizer / solver object should collate goals.
    """
    predictions = []
    targets = []

    for goal in goals:
        pred = goal.prediction(eqstate, model)
        target = goal.target(pred)

        predictions.append(np.atleast_1d(pred))
        targets.append(np.atleast_1d(target))

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    return predictions, targets


def loss_base(q, loads, xyz, solver, goals, loss):
    """
    The master loss to minimize.
    Takes user-defined loss as input.
    """
    eqstate = solver(q, loads, xyz)
    y_pred, y = collate_goals(goals, eqstate, solver.model)

    return loss(y, y_pred)
