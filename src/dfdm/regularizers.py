import autograd.numpy as np


class Regularizer:
    pass


class L2Regularizer(Regularizer):
    def __call__(self, eqstate, model):
        return np.sum(np.square(eqstate.force_densities))
