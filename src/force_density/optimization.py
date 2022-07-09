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
from force_density.losses import loss

__all__ = ["Optimizer"]


class Optimizer():
    """
    A gradient-descent optimizer.
    """
    # def __init__(self, network, goals):
    #     """
    #     Get the ball rolling.
    #     """
    #     self.network = network
    #     self.goals = goals

    def solve_scipy(self, network, goals, loss_f, ub, method="SLSQP", maxiter=100, tol=None):
        """
        Perform gradient descent with Scipy.
        """
        # array-ize parameters
        q = np.array(network.force_densities(), dtype=np.float64)
        loads = np.array(list(network.node_loads()), dtype=np.float64)
        xyz = np.array(list(network.node_xyz()), dtype=np.float64)  # probably should be xyz_fixed only

        # NOTE: It is immutable
        # access stuff -- gotta move to optimizer
        # TODO: Rename EquilibriumSolver to EquilibriumModel and EquilibriumModel to other
        solver = EquilibriumSolver(network)  # model can be instantiated in solver

        # loss matters
        loss_f = partial(loss, loads=loads, xyz=xyz, solver=solver, goals=goals, myloss=loss_f)
        grad_loss = grad(loss_f)  # grad w.r.t. first arg

        # parameter bounds
        bounds = Bounds(lb=-np.inf, ub=ub)

        # scipy optimization
        start_time = time()
        print("Optimization started...")


        res_q = minimize(fun=loss_f,
                         x0=q,
                         method=method,  # SLSQP
                         tol=tol,
                         jac=grad_loss,
                         bounds=bounds,
                         options={"maxiter": maxiter})

        # print out
        print(res_q.message)
        print("Output error in {} iterations: {}".format(res_q.nit, res_q.fun))
        print("Elapsed time: {} seconds".format(time() - start_time))

        q = res_q.x

        return q
