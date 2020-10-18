import numpy as np

def normalize_q_values(a, s):
    q = a.q[s]
    q_sum = np.sum(q)
    normalized = np.true_divide(q, q_sum)
    return normalized

def second_best_confidence(a1, a2, state):
    """compare best action to second-best action"""

    # normalize to get probabilities
    a1_actions_normalized = normalize_q_values(a1, state)
    a2_actions_normalized = normalize_q_values(a2, state)

    # get difference between best action and second best
    sorted_q1 = sorted(a1_actions_normalized,reverse=True)
    sorted_q2 = sorted(a2_actions_normalized,reverse=True)
    a1_diff = sorted_q1[0]-sorted_q1[1]
    a2_diff = sorted_q2[0]-sorted_q2[1]
    return a1_diff+a2_diff


def better_than_you_confidence(a1, a2, state):
    """compare best action to the action chosen by the opposing agent"""

    # normalize to get probabilities
    a1_actions_normalized = normalize_q_values(a1, state)
    a2_actions_normalized = normalize_q_values(a2, state)

    # get difference between best action and second best

    a1_diff = a1_actions_normalized.max() - a1_actions_normalized[np.argmax(a2_actions_normalized)]
    a2_diff = a2_actions_normalized.max() - a2_actions_normalized[np.argmax(a1_actions_normalized)]
    return a1_diff+a2_diff