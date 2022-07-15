from autograd import grad

from scipy.optimize import NonlinearConstraint


class Constraint:
    pass


class ForceConstraint(Constraint):
    """
    Set constraint bounds to the force in the edge of a network in equilibrium.
    """
    def __init__(self, key, force_min, force_max):
        """
        """
        self.key = key
        self.lb = force_min
        self.ub = force_max

    def __call__(self, model):
        """
        Below, a snapshot of scipy's documentation for NonlinearConstraint.

        Parameters
        ==========
        fun: callable
            The function defining the constraint.
            The signature is fun(x) -> array_like, shape (m,).

        lb, ub:  array_like
            Lower and upper bounds on the constraint.
            Each array must have the shape (m,) or be a scalar.
            in the latter case a bound will be the same for all components
            of the constraint.

            Use np.inf with an appropriate sign to specify a one-sided constraint.
            Set components of lb and ub equal to represent an equality constraint.
            Note that you can mix constraints of different types: interval, one-sided,
            or equality, by setting different components of lb and ub as necessary.

        jac: {callable, ‘2-point’, ‘3-point’, ‘cs’}, optional
            Method of computing the Jacobian matrix (an m-by-n matrix, where
            element (i, j) is the partial derivative of f[i] with respect to x[j]).
            The keywords {‘2-point’, ‘3-point’, ‘cs’} select a finite difference
            scheme for the numerical estimation. A callable must have the following
            signature: jac(x) -> {ndarray, sparse matrix}, shape (m, n). Default is ‘2-point’.
        """
        # fun = lambda q: model(q)
        index = model.structure.edge_index[self.key]

        def fun(q):
            # TODO: Can we compute the grad func of model() once and then just
            # pass that around?
            eqstate = model(q)
            return eqstate.forces[index]


        return NonlinearConstraint(fun=fun,
                                   lb=self.lb,
                                   ub=self.ub,
                                   jac=grad(fun))  # skip to use finite differences



if __name__ == "__main__":
    import autograd.numpy as np
    # compas
    from compas.colors import Color
    from compas.geometry import Line
    from compas.geometry import Point
    from compas.geometry import Polyline
    from compas.geometry import add_vectors
    from compas.geometry import length_vector

    # visualization
    from compas_view2.app import App

    # static equilibrium
    from dfdm.datastructures import ForceDensityNetwork

    from dfdm.equilibrium import EquilibriumModel

    # ==========================================================================
    # Initial parameters
    # ==========================================================================

    arch_length = 5.0
    num_segments = 10
    q_init = -1
    pz = -0.1

    # ==========================================================================
    # Create the geometry of an arch
    # ==========================================================================

    start = [0.0, 0.0, 0.0]
    end = add_vectors(start, [arch_length, 0.0, 0.0])
    curve = Polyline([start, end])
    points = curve.divide_polyline(num_segments)
    lines = Polyline(points).lines

    # ==========================================================================
    # Create arch
    # ==========================================================================

    network = ForceDensityNetwork.from_lines(lines)

    # ==========================================================================
    # Define structural system
    # ==========================================================================

    # assign supports
    network.node_support(key=0)
    network.node_support(key=len(points) - 1)

    # set initial q to all edges
    network.edges_forcedensities(q_init, keys=network.edges())

    # set initial point loads to all nodes of the network
    network.nodes_loads([0.0, 0.0, pz], keys=network.nodes_free())

    # ==========================================================================
    # Create constraints
    # ==========================================================================

    constraint = ForceConstraint(key=(0, 1), force_min=-np.inf, force_max=0.0)

    print(constraint)

    # ==========================================================================
    # Create constraints
    # ==========================================================================

    # get parameters
    q = np.array(network.edges_forcedensities(), dtype=np.float64)
    loads = np.array(network.nodes_loads(), dtype=np.float64)
    xyz = np.array(network.nodes_coordinates(), dtype=np.float64)

    # compute static equilibrium
    model = EquilibriumModel(network)  # model can be instantiated in solver
    eq_state = model(q, loads, xyz)

    from functools import partial
    model_partial = partial(model, loads=loads, xyz=xyz)
    eq_state = model_partial(q)

    # ==========================================================================
    # Create constraints
    # ==========================================================================

    constraint_scipy = constraint(model_partial)

    print(constraint_scipy)

    print("Done!")
