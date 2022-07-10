"""
Ideas notebook
"""


# ==========================================================================
#
# ==========================================================================

# optimizer = Optimizer("method", maxiter, tol)  # or SLSQP(parameters, goals, constraints)

optimizer = SLSQP(goals, parameters)
optimizer.add_parameter(Parameter)
optimizer.add_goal(Goal)
optimizer.add_constraint()
# optimizer = optimizer(goals, parameters, constraints)

network = fdm(network)

network = constrained_fdm(problem, optimizer, loss, maxiter, tol)

network = constrained_fdm(network, loss, optimizer, maxiter, tol)
network = constrained_fdm(network, optimizer, loss, maxiter, tol)
network = constrained_fdm(network, optimizer, loss, goals, parameters)
network = constrained_fdm(network, optimizer, loss, parameters, goals, constraints)


# ==========================================================================
#
# ==========================================================================

def loss(model, x, y):
    y_pred = model(x)
    return anp.mean((y - y_pred) ** 2)

def loss(y, y_pred):
    return np.sum(np.square(y - y_pred))

def loss_for_grad(x, y, loss):
    y_pred = model(x)  # equilibrium model
    return loss(y, y_pred)

def loss(model, q, y):
    loss = 0.0
    references = model(q)
    for reference, target in zip(references, targets):
        loss += np.square(np.linalg.norm(reference - target))

# ==========================================================================
#
# ==========================================================================

# NOTE sort optimizable parameters with a mask matrix
# TODO: how to guarantee ordering of nodes and edges?

q = np.array(network.force_densities())   # shape = (n_edges, )
P = np.array(network.loads())  # shape = (n_nodes, 3)

# store ordering of q and P

# combine into one long vector of optimization parameters, but preserve order
x = np.concatenate([q, np.ravel(P)])

# ==========================================================================
#
# ==========================================================================

# def static_equilibrium(force_densities, loads, structural_model):
#     """
#     Schek would be proud.
#     """
#     return EquilibriumState(xyz, lengths, forces, residuals)

# ==========================================================================
#
# ==========================================================================

# network = static_equilibrium(network)  # wrapper for force density eq calc + update network
# eqstate = force_density(network)


# def static_equilibrium(network):
#     structure = Structure(network)  # immutable stuff
#     eqstate = ForceDensityMethod(q, loads, structure)  # first 2 args reserved for opt params
#     return updated_network(eqstate)  # return new COMPAS network with updated attrs

# ==========================================================================
#
# ==========================================================================

# @dataclass
# class EquilibriumState():
#     """
#     An object that holds the results of a FDM run.
#     """
#     positions: node_positions
#     residuals: edge_residuals
#     lengths: edge_lengths
#     forces: edge_forces

# ==========================================================================
#
# ==========================================================================

# class Structure()   # Structural system, Equilibrium model, equilibrium structure
#     def __init__(self, network):
#         self.network = network

#         self._c_matrix = None
#         self._free_nodes = None
#         self._fixed_nodes = None
#         self._ordering_nodes = None
#         self._ordering_edges = None

#     @property
#     def c_matrix(self):
#         if not self._c_matrix:
#             self._c_matrix = self.network.edges()
#         return self._c_matrix

# ==========================================================================
#
# ==========================================================================


update(network, eqstate)
