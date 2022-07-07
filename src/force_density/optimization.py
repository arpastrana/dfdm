"""
A gradient-based optimizer.
"""
from time import time

from math import fabs

import autograd.numpy as anp
from autograd import grad

from scipy.optimize import minimize
from scipy.optimize import Bounds

from force_density.equilibrium import ForceDensity


__all__ = ["Optimizer"]


class Optimizer():
    """
    A gradient-descent optimizer.
    """
    def __init__(self, network, node_goals, edge_goals):
        """
        Get the ball rolling.
        """
        self.network = network.copy()
        self.goals = {"node": node_goals, "edge": edge_goals}

    def solve(self, lr, iters, loss_f, eps, reg=0.0, verbose=False):
        """
        Perform gradient descent
        """
        network = self.network
        goals = self.goals

        fd = ForceDensity()
        q = anp.array(network.force_densities())
        grad_loss = grad(loss_f)

        start_time = time()

        errors = []

        parameters = {"network": network, "goals": goals, "fd": fd, "regularizer": reg}

        print("Optimization started...")

        for k in range(iters):

            error = loss_f(q, parameters)
            errors.append(error)
            q_grad = grad_loss(q, parameters)

            q = q - lr * q_grad

            if verbose:
                print("Iteration: {} \t Loss: {}".format(k, error))

            if k < 2:
                continue

            rel_error = fabs((errors[-2] - errors[-1]) / errors[-1])
            print(f"Relative error: {rel_error}")

            if rel_error < eps:
                print("Early stopping at {}/{} iteration".format(k, iters))
                break

        # print out
        print("Output error in {} iterations: {}".format(iters, error))
        print("Elapsed time: {} seconds".format(time() - start_time))

        return q

    def solve_scipy(self, loss_f, ub, method="SLSQP", maxiter=100, tol=None):
        """
        Perform gradient descent with Scipy.
        """
        network = self.network
        goals = self.goals

        # fd = ForceDensity()
        q = anp.array(network.force_densities())
        grad_loss = grad(loss_f)
        # parameters = {"network": network, "goals": goals, "fd": fd}
        parameters = {"network": network, "goals": goals}

        bounds = Bounds(lb=-anp.inf, ub=ub)
        start_time = time()

        print("Optimization started...")

        res_q = minimize(fun=loss_f,
                         x0=q,
                         method=method,  # SLSQP
                         tol=tol,
                         args=(parameters),
                         jac=grad_loss,
                         bounds=bounds,
                         options={"maxiter": maxiter})

        # print out
        print(res_q.message)
        print("Output error in {} iterations: {}".format(res_q.nit, res_q.fun))
        print("Elapsed time: {} seconds".format(time() - start_time))

        q = res_q.x

        return q
