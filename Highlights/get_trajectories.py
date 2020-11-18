from os.path import join

from Highlights.HL_utils import make_clean_dirs, pickle_load, pickle_save


class Trajectory(object):
    def __init__(self, states, importance_dict):
        self.states = states
        self.state_importance = importance_dict
        self.importance = {
            'max_min': 0,
            'max_minus_avg': 0,
            'avg': 0,
            'sum': 0,
            'avg_delta': 0,
        }
        """calculate trajectory score"""
        self.trajectory_importance_max_avg()
        self.trajectory_importance_max_min()
        self.trajectory_importance_avg()
        self.trajectory_importance_avg_delta()

    def trajectory_importance_max_min(self):
        """ computes the importance of the trajectory, according to max-min approach """
        max, min = float("-inf"), float("inf")
        for state in self.states:
            state_importance = self.state_importance[state]
            if state_importance < min:
                min = state_importance
            if state_importance > max:
                max = state_importance
        self.importance['max_min'] = max - min

    def trajectory_importance_max_avg(self):
        """ computes the importance of the trajectory, according to max-avg approach """
        max, sum = float("-inf"), 0
        for state in self.states:
            state_importance = self.state_importance[state]
            # add to the curr sum for the avg in the future
            sum += state_importance
            if state_importance > max:
                max = state_importance
        avg = float(sum) / len(self.states)
        self.importance['max_minus_avg'] = max - avg

    def trajectory_importance_avg(self):
        """ computes the importance of the trajectory, according to avg approach """
        sum = 0
        for state in self.states:
            state_importance = self.state_importance[state]
            # add to the curr sum for the avg in the future
            sum += state_importance
        avg = float(sum) / len(self.states)
        self.importance['sum'] = sum
        self.importance['avg'] = avg

    def trajectory_importance_avg_delta(self):
        """ computes the importance of the trajectory, according to the average delta approach """
        sum_delta = 0
        for i in range(1, len(self.states)):
            sum_delta += self.state_importance[self.states[i]] - self.state_importance[self.states[i - 1]]
        avg_delta = sum_delta / len(self.states)
        self.importance['avg_delta'] = avg_delta


def trajectories_by_importance(execution_traces, state_importance, args):
    if args.load_trajectories:
        all_trajectories = pickle_load(join(args.results_dir,'Trajectories.pkl'))
    else:
        all_trajectories = get_all_trajectories(execution_traces, args.trajectory_length, state_importance)
        pickle_save(all_trajectories, join(args.results_dir,'Trajectories.pkl'))

    sorted_by_method = sorted([(x.importance[args.trajectory_importance], x) for x in all_trajectories],
                              key=lambda y: y[0], reverse=True)
    sorted_trajectories = [x[1] for x in sorted_by_method]
    summary_trajectories = trajectory_highlights(sorted_trajectories, args.allowed_similar_states,
                                                 args.num_trajectories)
    return all_trajectories, summary_trajectories


def get_all_trajectories(traces, length, importance):
    trajectories = []
    for trace in traces:
        for i in range(len(trace.states)):
            if (i + length) <= len(trace.states):
                trajectories.append(Trajectory(trace.states[i:i + length], importance))
    return trajectories


def trajectory_highlights(trajectories, similarity_limit, budget):
    # TODO check for similarity between chosen ones
    summary = [trajectories[0]]
    for i in range(1, len(trajectories)):
        for t in summary:
            if len(set(t.states).intersection(trajectories[i].states)) > similarity_limit:
                break
        else:
            summary.append(trajectories[i])
        if len(summary) == budget:
            break

    assert len(summary) == budget, "Not enough dis-similar trajectories found"
    return summary


def states_to_trajectories(states_list, importance_dict):
    trajectories = []
    for states in states_list.values():
        trajectories.append(Trajectory(states, importance_dict))
    return trajectories


def get_trajectory_images(trajectories, states, path):
    make_clean_dirs(path)
    trajectory_idx = 0
    for trajectory in trajectories:
        counter = 0
        for state in trajectory.states:
            trajectory_str = str(trajectory_idx) if trajectory_idx > 9 else "0" + str(trajectory_idx)
            counter_str = str(counter) if counter > 9 else "0" + str(counter)
            img_name = "_".join([trajectory_str, counter_str])
            counter += 1
            states[state].save_image(path, img_name)
        trajectory_idx += 1