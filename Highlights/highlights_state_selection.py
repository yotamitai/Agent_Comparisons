import numpy as np


def compute_states_importance(states_q_values_df, compare_to='worst'):
    if compare_to == 'worst':
        states_q_values_df['importance'] = states_q_values_df['q_values'].apply(lambda x: np.max(x) - np.min(x))
    elif compare_to == 'second':
        states_q_values_df['importance'] = states_q_values_df['q_values'].apply(
            lambda x: np.max(x) - np.partition(x.flatten(), -2)[-2])
    return states_q_values_df


def highlights(state_importance_df, exec_traces, budget, context_length, minimum_gap):
    ''' generate highlights summary
    :param state_importance_df: dataframe with 2 columns: state and importance score of the state
    :param budget: allowed length of summary - note this includes only the important states, it doesn't count context
    around them
    :param context_length: how many states to show around the chosen important state (e.g., if context_lenght=10, we
    will show 10 states before and 10 states after the important state
    :param minimum_gap: how many states should we skip after showing the context for an important state. For example, if
    we chose state 200, and the context length is 10, we will show states 189-211. If minimum_gap=10, we will not
    consider states 212-222 and states 178-198 because they are too close
    :return: a list with the indices of the important states, and a list with all summary states (includes the context)
    '''
    sorted_df = state_importance_df.sort_values(['importance'], ascending=False)
    summary_states = []
    summary_traces = []
    state_trajectories = {}
    """for each state by importance"""
    for index, row in sorted_df.iterrows():
        """get the state hash"""
        state = row['state']
        """find all traces where this state appears"""
        trajectories = {}
        for trace in exec_traces:
            if state in trace.states:
                state_index = trace.states.index(state)
                trace_index = exec_traces.index(trace)
                trace_len = len(trace.states)
                lower, upper = get_relevant_range(state_index, trace_len, context_length)
                """check if these states are not neighbours of previously seen states"""
                for seen_state in summary_states:
                    if [1 for x in trace.states[lower:upper] if x == seen_state]:
                        break
                    else:
                        trajectories[trace_index] = state_index
                if not summary_states:
                    trajectories[trace_index] = state_index

        """if no siutable trajectories found - try next state"""
        if not trajectories:
            continue
        else:
            state_trajectories[state] = trajectories

        """once a trace is obtained, get the state index in it"""
        summary_states.append(state)
        summary_traces.append(list(trajectories.keys()))
        if len(summary_states) == budget:
            break

    summary_states_with_context = {}
    for state in summary_states:
        t_i, s_i = list(state_trajectories[state].items())[0]
        t = exec_traces[t_i].states
        lower, upper = get_relevant_range(s_i, len(t), context_length)
        summary_states_with_context[state] = t[lower:upper]
    return summary_states_with_context


def get_relevant_range(indx, lst_len, range_len):
    lb, ub = indx - int(range_len/2), indx + int(range_len/2)
    if indx - range_len < 0:
        lb = 0
    if indx + range_len > lst_len:
        ub = lst_len - 1
    return lb, ub
