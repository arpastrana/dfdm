import autograd.numpy as np

from dfdm.equilibrium.model import EquilibriumModel


# ==========================================================================
# Form-finding
# ==========================================================================


def fdm(network):
    """
    Compute a network in a state of static equilibrium using the force density method.
    """
    # get parameters
    q = np.asarray(network.edges_forcedensities(), dtype=np.float64)

    # compute static equilibrium
    model = EquilibriumModel(network)
    eq_state = model(q)

    # update equilibrium state in network copy
    return network_updated(network, eq_state)  # Network.update(eqstate)

# ==========================================================================
# Constrained form-finding
# ==========================================================================


def constrained_fdm(network, optimizer, loss, bounds, maxiter, tol, callback=None):

    # optimizer works
    q_opt = optimizer.minimize(network, loss, bounds, maxiter, tol, callback=callback)

    # compute static equilibrium
    model = EquilibriumModel(network)
    eq_state = model(q_opt)

    # update equilibrium state in network copy
    return network_updated(network, eq_state)

# ==========================================================================
# Helpers
# ==========================================================================


def network_updated(network, eq_state):
    """
    Return a copy of a network whose attributes are updated with an equilibrium state.
    """
    network = network.copy()
    network_update(network, eq_state)

    return network


def network_update(network, eq_state):
    """
    Update in-place the attributes of a network with an equilibrium state.
    TODO: to be extra sure, the node-index and edge-index mappings should be handled
    by EquilibriumModel/EquilibriumStructure
    """
    xyz = eq_state.xyz.tolist()
    lengths = eq_state.lengths.tolist()
    residuals = eq_state.residuals.tolist()
    forces = eq_state.forces.tolist()
    forcedensities = eq_state.force_densities.tolist()

    # update q values and lengths on edges
    for idx, edge in enumerate(network.edges()):
        network.edge_attribute(edge, name="length", value=lengths[idx])
        network.edge_attribute(edge, name="force", value=forces[idx])
        network.edge_attribute(edge, name="q", value=forcedensities[idx])

    # update residuals on nodes
    for idx, node in enumerate(network.nodes()):
        for name, value in zip("xyz", xyz[idx]):
            network.node_attribute(node, name=name, value=value)

        for name, value in zip(["rx", "ry", "rz"], residuals[idx]):
            network.node_attribute(node, name=name, value=value)
