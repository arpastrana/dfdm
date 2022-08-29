import autograd.numpy as np


class Regularizer:
    pass


class L2Regularizer(Regularizer):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, eqstate, model):
        return self.alpha * np.sum(np.square(eqstate.force_densities))
