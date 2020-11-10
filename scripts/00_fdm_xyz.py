import numpy as np
import matplotlib.pyplot as plt

from time import time

from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from compas.numerical import connectivity_matrix

from compas.datastructures import Network
from compas.datastructures import network_transformed

from compas.geometry import Scale
from compas.geometry import Line

from compas.numerical import fd_numpy

from compas.utilities import rgb_to_hex

from compas_viewers.objectviewer import ObjectViewer

from compas.geometry import dot_vectors


# ==============================================================================
# Functions
# ==============================================================================

def force_density(xyz, q, p, C, Cit, Ci, free, fixed):
    """
    """
    # force density
    Q = diags([q.flatten()], [0])
    A = Cit.dot(Q).dot(Ci)

    # z
    b = p[free] - Cit.dot(Q).dot(Cf).dot(xyz[fixed])

    x = spsolve(A, b)

    xyz[free] = x

    r = p - Ct.dot(Q).dot(C).dot(xyz)

    return xyz, r


def update_z_coordinates(network, z_vector):
    for key, attr in network.nodes(True):
        network.node_attribute(key, name="z", value=z_vector[key].item())


def update_xyz_coordinates(network, p_matrix):
    for key in network.nodes():
        network.node_attributes(key, names="xyz", values=p_matrix[key, :])


# ==============================================================================
# Input
# ==============================================================================

HERE = "/Users/arpj/princeton/phd/projects/light_vault/io/central_arch_light_vault.json"

SCALE = 1

gradient = True
verbose = False
plot_losses = False
reverse_grad_cantilevered_edges = False
verify_compas_fd = False

kmax = 50 # 5000
lr = 1.0  # 0.1, 1.0, 2.5, 5.0  # cross validation of lambda! sensitive here

pz = -0.01795 * SCALE  # netwons - TODO: update as position changes?

q_0 = -1.5  # -2.5
brick_length = 0.123  # m
q_0_cantilever = pz / brick_length
q_0_cantilever = q_0

extra_support = None

S = Scale.from_factors([SCALE]*3)

# ==============================================================================
# Network
# ==============================================================================

# load network
network = Network.from_json(HERE)
network = network_transformed(network, S)
print("n edges, vertices", network.number_of_edges(), network.number_of_nodes())

# load network
target_network = network.copy()

# ==============================================================================
# Boundary Conditions
# ==============================================================================

# the z lower-most two nodes
f_z_lowest = lambda x: network.node_attribute(x, "z")
fixed = sorted([node for node in network.nodes()], key=f_z_lowest, reverse=True)
fixed = fixed[-2:]

# find cantilevered edges
leaves = network.leaves()
cantilever_nodes = list(set(network.leaves()) - set(fixed))

cantilevered_edges = []
for node in cantilever_nodes:
    edges = network.connected_edges(node)
    if len(edges) == 1:
        cantilevered_edges.append(edges[0])

non_cantilevered_edges = list(set(network.edges()) - set(cantilevered_edges))

# add extra support
if extra_support:
    fixed = fixed + [extra_support]
network.nodes_attribute(name="is_anchor", value=True, keys=fixed)

# set initial force density - TODO: how to find optimal initial value?
network.edges_attribute(name="q", value=q_0, keys=non_cantilevered_edges)

# set force density for cantilevered bricks
network.edges_attribute(name="q", value=q_0_cantilever, keys=cantilevered_edges)

# set initial point loads
network.nodes_attributes(values=[0, 0, pz], names=["px", "py", "pz"])

# ==============================================================================
# Keys
# ==============================================================================

# prepare lists for fd - topology
sorted_vertices = sorted(list(network.nodes()))
fixed = [v for v in sorted_vertices if network.node[v].get("is_anchor")]
edges = list(network.edges())
v = len(sorted_vertices)

uv_index = network.uv_index()
cantilevered_indices = np.array([uv_index[edge] for edge in cantilevered_edges])

# ==============================================================================
# Target points
# ==============================================================================

target_points = [network.node_coordinates(node) for node in sorted_vertices if node not in fixed]
sn = np.array(target_points)[:, 2].reshape((-1, 1))

# ==============================================================================
# Force density - Starting Point
# ==============================================================================

# q
q = network.edges_attribute(name="q", keys=edges)
q = np.array(q, dtype=float).reshape((-1, 1)) # this is used further below

# z
# z_0 = [network.node_attribute(vkey, "z") for vkey in sorted_vertices]
# loads = network.nodes_attribute(name="pz", keys=sorted_vertices)
# p = np.array(loads, dtype=float).reshape((-1, 1))
# z = np.array(z_0, dtype=float).reshape((-1, 1))

xyz_0 = [network.node_attributes(v, "xyz") for v in sorted_vertices]
# print(f"xyz: {xyz_0}")
xyz = np.array(xyz_0, dtype=float)
xyz_copy = xyz.copy()


loads = [network.node_attributes(v, ("px", "py", "pz")) for v in sorted_vertices]
p = np.array(loads, dtype=float)

# force density - preliminaries
free = list(set(range(v)) - set(fixed))
C = connectivity_matrix(edges, 'csr')
Ci = C[:, free]
Cf = C[:, fixed]
Ct = C.transpose()
Cit = Ci.transpose()

# force density
xyz, r = force_density(xyz, q, p, C, Cit, Ci, free, fixed)

# print stuff
print("free, fixed", len(free), len(fixed))
print("C shape", C.shape)
print("Ci shape", Ci.shape)
print("Cf shape", Cf.shape)



# ==============================================================================
# Network update
# ==============================================================================

update_xyz_coordinates(network, xyz)

# ==============================================================================
# Store memory reference q new values for cantilevered edges
# ==============================================================================

q_ref = np.copy(q)

# ==============================================================================
# Gradient descent
# ==============================================================================

# TODO: Compute gradient with respect to XYZ! Autograd?
if gradient:

    losses = []

    start_time = time()

    # updates only for z coordinate
    for k in range(kmax):
        
        # 0. target updates with closest point, or always fixed?
        # points = [network.node_coordinates(node) for node in free]
        # closest, distance, _ = trimesh.nearest.on_surface(points)
        # sn = closest[:, 2].reshape((-1, 1))

        # 1. Fetch z values from network
        z = [network.node_attribute(node, "z") for node in sorted_vertices]
        z = np.array(z).reshape((-1, 1))
        
        zn = [network.node_attribute(node, "z") for node in free]
        zn = np.array(zn).reshape((-1, 1))

        # compute loss
        loss = np.sum(np.square(zn - sn))
        losses.append(loss)
        
        if verbose:
            print("Iteration: {} \t Loss: {}".format(k, loss))

        # 2. calculate gradient preliminaries
        Q = diags([q.flatten()], [0])
        Dn = Cit.dot(Q).dot(Ci)
        Cz = np.diag(C.dot(z).flatten())  # diagonal matrix of Cz
        CnTCz = Cit.dot(Cz)

        # 3. solve gradient
        x = spsolve(Dn, CnTCz)
        diff = (zn - sn).transpose()
        dq = -2 * diff.dot(x)  # 1 x m

        # 4. update force densities
        q -= lr * dq.transpose()  # m x 1

        # 4.5 annulate q update on cantilevered edges
        if reverse_grad_cantilevered_edges:
            q[cantilevered_indices, :] = q_ref[cantilevered_indices, :]

        # 5. do fd
        xyz = np.array([network.node_coordinates(n) for n in sorted_vertices])
        xyz_, r_ = force_density(xyz, q, p, C, Cit, Ci, free, fixed)

        # 6. update coordinates
        update_xyz_coordinates(network, xyz_)

        # TODO: check for early stopping?

    print("Output loss in {} iterations: {}".format(kmax, losses[-1]))
    print("Elapsed time: {} seconds".format(time() - start_time))

# ==============================================================================
# See losses
# ==============================================================================

    if plot_losses:
        plt.plot(losses)
        plt.show()

# ==============================================================================
# Verify with compas FD
# ==============================================================================

if verify_compas_fd:
    xyz, q, f, l, r = fd_numpy(xyz.tolist(), edges, fixed, q.tolist(), loads)
    update_xyz_coordinates(network, xyz)

# ==============================================================================
# Viewer
# ==============================================================================

viewer = ObjectViewer()
T = Scale.from_factors([1/SCALE]*3)
network_viz = network_transformed(network, T)
t_network_viz = network_transformed(target_network, T)

# blue is target, red is subject

viewer.add(network_viz, settings={
        'edges.color': rgb_to_hex((255, 0, 0)),
        'edges.width': 2,
        'opacity': 0.7,
        'vertices.size': 10,
        'vertices.on': False,
        'edges.on': True
        })

viewer.add(t_network_viz, settings={
    'edges.color': rgb_to_hex((0, 0, 255)),
    'edges.width': 1,
    'opacity': 0.5,
    'vertices.size': 10,
    'vertices.on': False,
    'edges.on': True
})

for node in network_viz.nodes():
    pt = network_viz.node_coordinates(node)
    target_pt = t_network_viz.node_coordinates(node)
    viewer.add(Line(target_pt, pt))

supports_network = Network()
for node in fixed:
    x, y, z = network_viz.node_coordinates(node)
    supports_network.add_node(node, x=x, y=y, z=z)

viewer.add(supports_network, settings={
    'vertices.size': 10,
    'vertices.on': True,
    'edges.on': False
})


# viewer.show()
