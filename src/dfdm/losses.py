import autograd.numpy as np


# ==========================================================================
# Precooked losses
# ==========================================================================

def squared_loss(predictions, targets, weights, q=None):
    """
    """
    return np.sum(weights * np.square(predictions - targets))


def l2_loss(predictions, targets, weights, q):
    """
    """
    return np.sqrt(squared_loss(predictions, targets, weights))


# ==========================================================================
# Loss stuff
# ==========================================================================


def loss_base(q, loads, xyz, model, goals, loss):
    """
    The master loss to minimize.
    Takes user-defined loss as input.
    """
    eqstate = model(q, loads, xyz)
    predictions, targets, weights = goals_collate(goals, eqstate, model)

    return loss(predictions, targets, weights, q)


# ==========================================================================
# Utilities
# ==========================================================================


def goals_index(goals, model):
    """
    Compute the index of a goal based on its node or edge key.
    """
    for goal in goals:
        index = goals.index(model)
        goals._index = index


def goals_collate(goals, eqstate, model):
    """
    TODO: An optimizer / solver object should collate goals.
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
# Main function
# ==========================================================================

if __name__ == "___main__":

    def loss_base_2(goals, model, q):

        eqstate = model(q)

        error = 0.0
        for goal in goals:
            # TODO: initialize goals to compute index before starting optimization
            # e.g. goal.index(model) (requires statefulness though!, may raise JIT issues)
            gstate = goal(model, eqstate)

            if isinstance(goal, LoadPathGoal):
                penalty = gstate.weight * gstate.prediction
            else:
                penalty = gstate.weight * (gstate.prediction - gstate.target) ** 2

            error += penalty

        regularizer = np.sum(eqstate.q ** 2)

        return error + alpha * regularizer


    def goals_filter(sentinel, goals, func):
        goals = filter(func, goals)
        return func(goals)


    # @goals_filter(func=lambda x: not isinstance(x, NetworkLoadPathGoal))
    def loss_a(gstate):
        return np.sum(gstate.weight * (gstate.prediction - gstate.target) ** 2)


    # @goals_filter(func=lambda goal: not isinstance(goal, LoadPathGoal))
    def loss_a(model, goals):
        gstate = goals(eqstate)
        return np.sum(gstate.weight * (gstate.prediction - gstate.target) ** 2)


    # @goals_filter(func=lambda x: not isinstance(x, NetworkLoadPathGoal))
    def loss_a(q, model, goal):
        eqstate = model(q)
        gstate = goal(eqstate)
        return (gstate.prediction - gstate.target) ** 2


    # @goals_filter(func=lambda x: isinstance(x, LoadPathGoal))
    def loss_b(q, model, goal):
        eqstate = model(q)
        gstate = goal(eqstate)
        return gstate.prediction


    def regularizer_l2(eqstate, goals):
        return np.sum(np.square(eqstate.q))


    def a(goals, func):
        goals = [goal for goal in goals if func(x)]
        return np.sum(goals.weights * (goals.predictions - goals.targets) ** 2)


    def loss_l2(goals, func):
        goals = [goal for goal in goals if func(x)]
        loss = 0.0
        for goal in goals:
            loss = loss + goal.weight * (goal.prediction - goal.target) ** 2

        return np.sum(goals.weights * (goals.predictions - goals.targets) ** 2)


    def loss_l2(goal):
        """
        """
        return goal.weight * (goal.prediction - goal.target) ** 2


    losses = [loss_a, loss_b, regularizer_l2]
