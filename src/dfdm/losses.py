import autograd.numpy as np


# ==========================================================================
# Precooked losses
# ==========================================================================

class Loss:
    def __init__(self, goals, record=True, *args, **kwargs):
        self.goals = goals
        self.record = record
        self.recorder = Recorder()

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
        error = 0.
        for goal in self.goals:
            gstate = goal(eqstate, model)
            error += self.loss(gstate)

        if self.record:
            if isinstance(error, float):
                self.recorder.record(error)

        return error


class MeanSquaredErrorLoss(SquaredErrorLoss):
    """
    The seminal mean squared error loss.

    Average out all errors because no single error is important enough.
    """
    def __call__(self, eqstate, model):
        squared_error = super().__call__(eqstate, model)
        error = squared_error / len(self.goals)

        if self.record:
            self.recorder.record(error)

        return error


class PredictionLoss(Loss):
    """
    You lose when you predict too much of something.
    """
    def __call__(gstate):
        return gstate.prediction


# ==========================================================================
# Base loss
# ==========================================================================


def loss_base(q, loads, xyz, model, loss):
    """
    The master loss to minimize.
    Takes user-defined loss as input.
    """
    eqstate = model(q, loads, xyz)

    return loss(eqstate, model)

# ==========================================================================
# Goal manager
# ==========================================================================


class GoalManager:
    @staticmethod
    def goals_index(goals, model):
        """
        Compute the index of a goal based on its node or edge key.
        """
        for goal in goals:
            index = goals.index(model)
            goals._index = index

    @staticmethod
    def goals_collate(goals, eqstate, model):
        """
        Collate goals attributes into vectors.
        """
        predictions = []
        targets = []
        weights = []

        for goal in goals:
            gstate = goal(eqstate, model)
            predictions.append(gstate.prediction)
            targets.append(gstate.target)
            weights.append(gstate.weight)

        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        weights = np.concatenate(weights, axis=0)

        return predictions, targets, weights

# ==========================================================================
# Recorder
# ==========================================================================


class Recorder:
    def __init__(self):
        self.history = []

    def record(self, value):
        self.history.append(value)
