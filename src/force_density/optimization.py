#!/usr/bin/env python3
"""
A jax-based gradient descent optimization approach
"""
from time import time

import jax.numpy as np

from jax import grad

from force_density.equilibrium import force_equilibrium
from force_density.equilibrium import ForceDensity


__all__ = ["Optimizer"]


class Optimizer():
    """
    A gradient-descent optimizer.
    """
    def __init__(self, network, goals):
        """
        Get the ball rolling.
        """
        self.network = network
        self.goals = {goal.key(): goal for goal in goals}

    def solve(self, lr, iters, loss_f, verbose=False):
        """
        Perform gradient descent
        """
        network = self.network
        goals = self.goals

        fd = ForceDensity()
        q = np.array(network.force_densities())
        grad_loss = grad(loss_f)

        start_time = time()
        print("Optimization started...")

        for k in range(iters):

            parameters = {"network": network, "goals": goals, "fd": fd}

            error = loss_f(q, parameters)
            q_grad = grad_loss(q, parameters)

            q = q - lr * q_grad

            if verbose:
                print("Iteration: {} \t Loss: {}".format(k, error))

        # print out
        print("Output error in {} iterations: {}".format(iters, error))
        print("Elapsed time: {} seconds".format(time() - start_time))

        return q
