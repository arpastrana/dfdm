"""
A gradient-based optimizer.
"""
from time import time

from functools import partial

import autograd.numpy as np
from autograd import grad

from scipy.optimize import minimize
from scipy.optimize import Bounds

from compas.data import Data

from dfdm.equilibrium import EquilibriumModel
from dfdm.losses import loss_base


# ==========================================================================
# Optimizer
# ==========================================================================


class Optimizer():
    def __init__(self, name, disp=False, **kwargs):
        self.name = name
        self.disp = disp

    def minimize(self, network, loss, bounds, maxiter, tol, verbose=True, callback=None):
        # returns the optimization result: dataclass OptimizationResult
        """
        Minimize a loss function via some flavor of gradient descent.
        """
        name = self.name

        # array-ize parameters
        q = np.asarray(network.edges_forcedensities(), dtype=np.float64)

        # create an equilibrium model from a network
        model = EquilibriumModel(network)

        # loss matters
        loss_f = partial(loss_base, model=model, loss=loss)

        # gradient of the loss function
        grad_loss = grad(loss_f)  # grad w.r.t. first function argument

        # TODO: parameter bounds
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

        if verbose:
            print("Optimization started...")

        # scipy optimization
        start_time = time()

        # minimize
        res_q = minimize(fun=loss_f,
                         jac=grad_loss,
                         method=name,
                         x0=q,
                         tol=tol,
                         bounds=bounds,
                         callback=callback,
                         options={"maxiter": maxiter, "disp": self.disp})
        # print out
        if verbose:
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
    def __init__(self, **kwargs):
        super().__init__(name="SLSQP", **kwargs)


class BFGS(Optimizer):
    """
    The Boyd-Fletcher-Floyd-Shannon optimizer.
    """
    def __init__(self, **kwargs):
        super().__init__(name="BFGS", **kwargs)


class TrustRegionConstrained(Optimizer):
    """
    A trust-region algorithm for constrained optimization.
    """
    def __init__(self, **kwargs):
        super().__init__(name="trust-constr", **kwargs)


# ==========================================================================
# Recorder
# ==========================================================================


class Recorder(Data):
    def __init__(self):
        self.history = []

    def record(self, value):
        self.history.append(value)

    def __call__(self, q, *args, **kwargs):
        self.record(q)

    @property
    def data(self):
        data = dict()
        data["history"] = self.history
        return data

    @data.setter
    def data(self, data):
        self.history = data["history"]
