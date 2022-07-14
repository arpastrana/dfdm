import autograd.numpy as np


def squared_loss(predictions, targets):
    """
    """
    return np.sum(np.square(predictions - targets))


def l2_loss(predictions, targets):
    return np.sqrt(np.sum(np.square(predictions - targets)))
