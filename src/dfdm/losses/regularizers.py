import autograd.numpy as np


class Regularizer:
    pass


class L2Regularizer(Regularizer):
    def __init__(self, alpha, name=None):
        self.alpha = alpha
        self._name = name

    @property
    def name(self):
        if not self._name:
            self._name = self.__class__.__name__
        return self._name

    def __call__(self, eqstate, model):
        return self.alpha * np.sum(np.square(eqstate.force_densities))
