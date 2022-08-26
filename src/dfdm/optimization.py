"""
A gradient-based optimizer.
"""
from time import time

from functools import partial

import autograd.numpy as np
from autograd import grad

from scipy.optimize import minimize
from scipy.optimize import Bounds

from dfdm.equilibrium import EquilibriumModel
from dfdm.losses import loss_base


# ==========================================================================
# Optimizer
# ==========================================================================


class Optimizer():
    def __init__(self, name):
        self.name = name

    def minimize(self, network, loss, goals, bounds, maxiter, tol):
        # returns the optimization result: dataclass OptimizationResult
        """
        Perform gradient descent with Scipy.
        """
        name = self.name

        # array-ize parameters
        q = np.asarray(network.edges_forcedensities(), dtype=np.float64)
        loads = np.asarray(list(network.nodes_loads()), dtype=np.float64)
        xyz = np.asarray(list(network.nodes_coordinates()), dtype=np.float64)

        # NOTE: model can be instantiated in solver?
        model = EquilibriumModel(network)

        # loss matters
        loss_f = partial(loss_base,
                         model=model,
                         loads=loads,
                         xyz=xyz,
                         goals=goals,
                         loss=loss)

        grad_loss = grad(loss_f)  # grad w.r.t. first arg

        # parameter bounds
        # bounds makes a re-index from one count system to the other
        # bounds = optimization_bounds(model, bounds)
        lb, ub = bounds
        if lb is None:
            lb = -np.inf
        if ub is None:
            ub = +np.inf

        bounds = Bounds(lb=lb, ub=ub)

        # TODO: support for scipy non-linear constraints
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
        print(f"Final loss in {res_q.nit} iterations: {res_q.fun}")
        print(f"Elapsed time: {time() - start_time} seconds")

        return res_q.x

# ==========================================================================
# Optimizers
# ==========================================================================


class SLSQP(Optimizer):
    """
    The sequential least-squares programming optimizer.
    """
    def __init__(self):
        super(SLSQP, self).__init__(name="SLSQP")


class BFGS(Optimizer):
    """
    The Boyd-Fletcher-Floyd-Shannon optimizer.
    """
    def __init__(self):
        super(BFGS, self).__init__(name="BFGS")
