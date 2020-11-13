#!/usr/bin/env python3
"""
A jax-based gradient descent optimization approach
"""
from time import time

import jax.numpy as np

from jax import grad

from force_density.equilibrium import force_equilibrium


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
        self.goals = goals
        self.error_history = []

    def solve(self, lr, iters, loss_f, verbose=False):
        """
        Perform gradient descent
        """
        grad_loss = grad(loss_f)

        q, edges, xyz, free, fixed, loads = self._initialize_data()
        sn = self.goals

        start_time = time()

        print("Optimization started...")

        for k in range(iters):

            error = loss_f(q, sn, edges, xyz, free, fixed, loads)

            q_grad = grad_loss(q, sn, edges, xyz, free, fixed, loads)
            q = q - lr * q_grad

            # do fd and update network
            xyz = force_equilibrium(q, edges, xyz, free, fixed, loads)

            # store error
            self.error_history.append(error)

            if verbose:
                print("Iteration: {} \t Loss: {}".format(k, error))

        # print out
        print("Output error in {} iterations: {}".format(iters, error))
        print("Elapsed time: {} seconds".format(time() - start_time))

        return q, xyz

    def _initialize_data(self):
        """
        Prepare the initial data to carry out the force density method.
        """
        # node key: index mapping
        k_i = self.network.key_index()

        # find supports
        fixed = [k_i[key] for key in self.network.supports()]

        # find free nodes
        free = [k_i[key] for key in self.network.free_nodes()]

        # edges
        edges = [(k_i[u], k_i[v]) for u, v in self.network.edges()]

        # node coordinates
        xyz = np.array(self.network.nodes_xyz())

        # force densities
        q = np.array(self.network.force_densities())

        # forces
        loads = np.array(self.network.applied_load())

        return q, edges, xyz, free, fixed, loads
