def trajectory_importance_max_min(states_importance):
    """ computes the importance of the trajectory, according to max-min approach: delta(max state, min state) """
    max, min = float("-inf"), float("inf")
    for i in range(len(states_importance)):
        state_importance = states_importance[i]
        if state_importance < min:
            min = state_importance
        if state_importance > max:
            max = state_importance
    return max - min

def trajectory_importance_max_avg(states_importance):
    """ computes the importance of the trajectory, according to max-avg approach """
    max, sum = float("-inf"), 0
    for i in range(len(states_importance)):
        state_importance = states_importance[i]
        # add to the curr sum for the avg in the future
        sum += state_importance
        if state_importance > max:
            max = state_importance
    avg = float(sum) / len(states_importance)
    return max - avg

def trajectory_importance_avg(states_importance):
    """ computes the importance of the trajectory, according to avg approach """
    sum = 0
    for i in range(len(states_importance)):
        state_importance = states_importance[i]
        # add to the curr sum for the avg in the future
        sum += state_importance
    avg = float(sum) / len(states_importance)
    return avg

def trajectory_importance_avg_delta(states_importance):
    """ computes the importance of the trajectory, according to the average delta approach """
    sum_delta = 0
    for i in range(len(states_importance)):
        sum_delta += states_importance[i] - states_importance[i - 1]
    avg_delta = sum_delta / len(states_importance)
    return avg_delta

