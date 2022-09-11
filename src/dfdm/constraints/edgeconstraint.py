import autograd.numpy as np

from dfdm.constraints import Constraint


class EdgeConstraint(Constraint):
    """
    Base class for all constraints that pertain to an edge of a network.
    """
    def __init__(self, key, bound_low, bound_up, **kwargs):
        super().__init__(bound_low=bound_low, bound_up=bound_up)
        self._key = key

    def index(self, model):
        """
        The index of the edge key in a structure.
        """
        return model.structure.edge_index[self.key()]

    def key(self):
        """
        The key of the edge in the network.
        """
        return self._key


class EdgeLengthConstraint(EdgeConstraint):
    """
    Constraints the length of an edge between a lower and an upper bound.
    """
    def constraint(self, eqstate, model):
        """
        Returns the length of an edge from an equilibrium state.
        """
        return eqstate.lengths[self.index(model)]


class EdgeVectorAngleConstraint(EdgeConstraint):
    """
    Constraints the angle formed by an edge and a vector between a lower and an upper bound.
    """
    def __init__(self, key, vector, bound_low, bound_up):
        super().__init__(key=key, bound_low=bound_low, bound_up=bound_up)
        self.vector_other = np.asarray(vector)

    def constraint(self, eqstate, model):
        """
        Returns the angle between an edge in an equilibrium state and a vector.
        """
        vector = eqstate.vectors[self.index(model)]
        return self._angle_vectors_numpy(vector, self.vector_other, deg=True)

    @staticmethod
    def _angle_vectors_numpy(u, v, deg=False):
        """
        Compute the smallest angle between two vectors.
        """
        a = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        a = max(min(a, 1), -1)
        if deg:
            return np.degrees(np.arccos(a))
        return np.arccos(a)


if __name__ == "__main__":
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
    from dfdm.datastructures import FDNetwork
    from dfdm.equilibrium import fdm
    from dfdm.equilibrium import constrained_fdm
    from dfdm.optimization import SLSQP
    from dfdm.optimization import TrustRegionConstrained
    from dfdm.goals import NetworkLoadPathGoal
    from dfdm.losses import Loss
    from dfdm.losses import PredictionError

    # ==========================================================================
    # Initial parameters
    # ==========================================================================

    arch_length = 5.0
    num_segments = 10
    q_init = -1.0
    pz = -0.1
    length_min = 0.5
    length_max = 1.0
    optimizer = SLSQP  # SLSQP

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

    network = FDNetwork.from_lines(lines)

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
    # Run the force density method
    # ==========================================================================

    print("\nForm found network")
    network = fdm(network)

    # ==========================================================================
    # Report stats
    # ==========================================================================

    q = list(network.edges_forcedensities())
    f = list(network.edges_forces())
    l = list(network.edges_lengths())

    print(f"Load path: {round(network.loadpath(), 3)}")
    for name, vals in zip(("FDs", "Forces", "Lengths"), (q, f, l)):

        minv = round(min(vals), 3)
        maxv = round(max(vals), 3)
        meanv = round(sum(vals) / len(vals), 3)
        print(f"{name}\t\tMin: {minv}\tMax: {maxv}\tMean: {meanv}")

    # ==========================================================================
    # Create loss funtion with total load path as the only goal
    # ==========================================================================

    goal = NetworkLoadPathGoal()
    loss = Loss(PredictionError(goals=[goal]))

    # ==========================================================================
    # Constrained form-finding
    # ==========================================================================

    print("\nInverse form found network. No constraints")
    ceq_network = constrained_fdm(network,
                                  optimizer=optimizer(),
                                  loss=loss)

    # ==========================================================================
    # Report stats
    # ==========================================================================

    q = list(ceq_network.edges_forcedensities())
    f = list(ceq_network.edges_forces())
    l = list(ceq_network.edges_lengths())

    print(f"Load path: {round(ceq_network.loadpath(), 3)}")
    for name, vals in zip(("FDs", "Forces", "Lengths"), (q, f, l)):

        minv = round(min(vals), 3)
        maxv = round(max(vals), 3)
        meanv = round(sum(vals) / len(vals), 3)
        print(f"{name}\t\tMin: {minv}\tMax: {maxv}\tMean: {meanv}")

    # ==========================================================================
    # Create constraint
    # ==========================================================================

    from dfdm.equilibrium import EquilibriumModel
    import autograd.numpy as np
    from autograd import jacobian
    from math import radians
    from functools import partial

    model = EquilibriumModel(network)
    q = np.array(network.edges_forcedensities())
    eqstate = model(q)

    vec = [0., 0., 1.]
    constraint= EdgeVectorAngleConstraint(key=(0, 1), vector=vec, bound_low=0., bound_up=30.)
    print("key", constraint.key())
    print("index", constraint.index(model))
    print("constraint val", constraint.constraint(eqstate, model))
    fun = partial(constraint, model=model)
    jac = jacobian(fun)
    print("fun", fun(q))
    print("jac", jac(q))

    constraints = [constraint]
    for edge in network.edges():
        c = EdgeLengthConstraint(edge, bound_low=length_min, bound_up=length_max)
        constraints.append(c)

    # constraint = NetworkLengthConstraint(bound_low=length_min, bound_up=length_max)
    # constraints = [constraint]

    # ==========================================================================
    # Constrained form-finding with constraint
    # ==========================================================================

    print("\nInverse form found network. With constraints")
    ceq_network = constrained_fdm(network,
                                  optimizer=optimizer(),
                                  loss=loss,
                                  constraints=constraints,
                                  maxiter=1000,
                                  tol=1e-6)

    q = np.array(ceq_network.edges_forcedensities())
    eqstate = model(q)
    print("constraint val", constraint.constraint(eqstate, model))

    # ==========================================================================
    # Report stats
    # ==========================================================================

    q = list(ceq_network.edges_forcedensities())
    f = list(ceq_network.edges_forces())
    l = list(ceq_network.edges_lengths())

    print(f"Load path: {round(ceq_network.loadpath(), 3)}")
    for name, vals in zip(("FDs", "Forces", "Lengths"), (q, f, l)):

        minv = round(min(vals), 3)
        maxv = round(max(vals), 3)
        meanv = round(sum(vals) / len(vals), 3)
        print(f"{name}\t\tMin: {minv}\tMax: {maxv}\tMean: {meanv}")

    # ==========================================================================
    # Visualization
    # ==========================================================================

    viewer = App(width=1600, height=900, show_grid=True)

    viewer.add(network, linewidth=1.0, linecolor=Color.grey().darkened())

    eq_network = ceq_network

    # equilibrated arch
    viewer.add(eq_network,
               show_vertices=True,
               pointsize=12.0,
               show_edges=True,
               linecolor=Color.teal(),
               linewidth=5.0)

    # reference arch
    viewer.add(network, show_points=False, linewidth=4.0)

    for node in eq_network.nodes():

        pt = eq_network.node_coordinates(node)

        # draw lines betwen subject and target nodes
        target_pt = network.node_coordinates(node)
        viewer.add(Line(target_pt, pt))

        # draw residual forces
        residual = eq_network.node_residual(node)

        if length_vector(residual) < 0.001:
            continue

        residual_line = Line(pt, add_vectors(pt, residual))
        viewer.add(residual_line,
                   linewidth=4.0,
                   color=Color.pink())

    # draw applied loads
    for node in eq_network.nodes():
        pt = eq_network.node_coordinates(node)
        load = network.node_load(node)
        viewer.add(Line(pt, add_vectors(pt, load)),
                   linewidth=4.0,
                   color=Color.green().darkened())

    # draw supports
    for node in eq_network.nodes_supports():
        x, y, z = eq_network.node_coordinates(node)
        viewer.add(Point(x, y, z), color=Color.green(), size=20)

    # show le crème
    viewer.show()
