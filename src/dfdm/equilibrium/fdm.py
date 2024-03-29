import autograd.numpy as np

from dfdm.equilibrium.model import EquilibriumModel


# ==========================================================================
# Form-finding
# ==========================================================================

def _fdm(network, q):
    """
    Compute a network in a state of static equilibrium using the force density method.
    """
    # compute static equilibrium
    model = EquilibriumModel(network)
    eq_state = model(q)

    # update equilibrium state in network copy
    return network_updated(network, eq_state)  # Network.update(eqstate)


def fdm(network):
    """
    Compute a network in a state of static equilibrium using the force density method.
    """
    # get parameters
    q = np.asarray(network.edges_forcedensities(), dtype=np.float64)

    return _fdm(network, q)

# ==========================================================================
# Constrained form-finding
# ==========================================================================


def constrained_fdm(network, optimizer, loss, bounds=(None, None), constraints=None, maxiter=100, tol=1e-6, callback=None):

    # optimizer works
    q_opt = optimizer.minimize(network, loss, bounds, constraints, maxiter, tol, callback=callback)

    return _fdm(network, q_opt)

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
    # for idx, edge in enumerate(network.edges()):
    for idx, edge in network.index_uv().items():
        network.edge_attribute(edge, name="length", value=lengths[idx])
        network.edge_attribute(edge, name="force", value=forces[idx])
        network.edge_attribute(edge, name="q", value=forcedensities[idx])

    # update residuals on nodes
    for idx, node in network.index_key().items():
    # for idx, node in enumerate(network.nodes()):
        for name, value in zip("xyz", xyz[idx]):
            network.node_attribute(node, name=name, value=value)

        for name, value in zip(["rx", "ry", "rz"], residuals[idx]):
            network.node_attribute(node, name=name, value=value)
