import autograd.numpy as np

from dfdm.constraints import Constraint


class NodeConstraint(Constraint):
    """
    Base class for all constraints that pertain to a node in a network.
    """
    def __init__(self, key, bound_low, bound_up, **kwargs):
        super().__init__(bound_low=bound_low, bound_up=bound_up)
        self._key = key

    def index(self, model):
        """
        The index of the edge key in a structure.
        """
        return model.structure.node_index[self.key()]

    def key(self):
        """
        The key of the edge in the network.
        """
        return self._key


class NodeNormalAngleConstraint(NodeConstraint):
    """
    Constraints the angle between the normal of the network at a node and a reference vector.
    """
    def __init__(self, key, polygon, vector, bound_low, bound_up):
        super().__init__(key, bound_low, bound_up)
        self.polygon = polygon
        self.vector_other = np.asarray(vector)

    def constraint(self, eqstate, model):
        """
        Returns the angle in radians between the the node normal and the reference vector.
        """
        normal = self._node_normal(eqstate, model)
        return self._angle_vectors(normal, self.vector_other)

    def _node_normal(self, eqstate, model):
        """
        Computes the unitized vector normal at a node in a network.
        """
        indices_polygon = [model.structure.node_index[nbr] for nbr in self.polygon]
        p = eqstate.xyz[indices_polygon, :]
        return self._normal_polygon(p)
        
    @staticmethod
    def _angle_vectors(u, v):
        """
        Compute the smallest angle between two vectors.
        """
        a = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        a = max(min(a, 1), -1)
        return np.arccos(a)

    @staticmethod
    def _normal_polygon(p):
        """
        Compute the normal of a polygon defined by a sequence of points (at least two points).
        Polygon is numpy array #points x 3.
        """
        o = np.mean(p, axis=0)
        op = p - o
        ns = np.array([np.cross(op[i - 1], op[i]) * 0.5 for i in range(len(op))])
        n = np.sum(ns, axis=0)
        return n / np.linalg.norm(n)
