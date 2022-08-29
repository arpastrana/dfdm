import autograd.numpy as np

from dfdm.goals import GoalManager


# ==========================================================================
# Precooked losses
# ==========================================================================

class Loss:
    def __init__(self, goals, *args, **kwargs):
        self.goals = goals
        self.goal_manager = GoalManager()

    def __call__(self, eqstate, model):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

# ==========================================================================
# Precooked losses
# ==========================================================================


class SquaredErrorLoss(Loss):
    """
    The canonical squared error.
    Measures the distance between the current and the target value of a goal.
    """
    @staticmethod
    def loss(gstate):
        return np.sum(gstate.weight * np.square(gstate.prediction - gstate.target))

    def __call__(self, eqstate, model):
        gstate = self.goal_manager.goals_state(self.goals, eqstate, model)
        return self.loss(gstate)


class MeanSquaredErrorLoss(SquaredErrorLoss):
    """
    The seminal mean squared error loss.

    Average out all errors because no single error is important enough.
    """
    def __call__(self, eqstate, model):
        squared_error = super().__call__(eqstate, model)
        error = squared_error / len(self.goals)

        return error


class PredictionLoss(Loss):
    """
    You lose when you predict too much of something.
    """
    def __call__(self, eqstate, model):
        gstate = self.goal_manager.goals_state(self.goals, eqstate, model)
        return gstate.prediction

# ==========================================================================
# Base loss
# ==========================================================================


def loss_base(q, model, loss):
    """
    The master loss to minimize.
    Takes user-defined loss as input.
    """
    eqstate = model(q)
    return loss(eqstate, model)
