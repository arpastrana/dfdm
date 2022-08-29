import autograd.numpy as np

from dfdm.goals import GoalState


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
    def goals_state(goals, eqstate, model):
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

        return GoalState(prediction=predictions, target=targets, weight=weights)
