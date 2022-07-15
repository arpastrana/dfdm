import autograd.numpy as np


def squared_loss(predictions, targets, weights):
    """
    """
    return np.sum(weights * np.square(predictions - targets))


def l2_loss(predictions, targets, weights):
    """
    """
    return np.sqrt(squared_loss(predictions, targets, weights))
