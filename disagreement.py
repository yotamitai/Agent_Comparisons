from os.path import join
import gym
import imageio
import numpy as np
import matplotlib.pyplot as plt

from get_trajectories import trajectory_importance_max_min, \
    trajectory_importance_max_avg, trajectory_importance_avg, trajectory_importance_avg_delta
from utils import save_image, create_video, make_clean_dirs, \
    reload_agent


class DisagreementTrace(object):
    def __init__(self, horizon, a1_q_values, a2_q_values, agent_ratio):
        self.a1_q_values = a1_q_values
        self.a2_q_values = a2_q_values
        self.a1_states = []
        self.agent_ratio = agent_ratio
        self.a2_trajectories = []
        self.disagreement_indexes = []
        self.importance_scores = []
        self.trajectory_length = horizon
        self.disagreement_trajectories = []
        self.diverse_trajectories = []
        self.min_traj_len = (self.trajectory_length // 2) + 3

    def get_trajectories(self, importance_type, da_importance, t_importance):
        for i, a2_traj in enumerate(self.a2_trajectories):
            start = a2_traj[0].name
            end = start + len(a2_traj)
            if len(self.a1_states) <= end:
                a1_traj = self.a1_states[start:]
            else:
                a1_traj = self.a1_states[start:end]
            a2_traj = a2_traj[:len(a1_traj)]
            if len(a1_traj) < self.min_traj_len: continue
            dt = DisagreementTrajectory(a1_traj, a2_traj, importance_type, da_importance,
                                        t_importance, self.trajectory_length,
                                        self.a1_q_values, self.a2_q_values, self.agent_ratio)
            self.disagreement_trajectories.append(dt)


class State(object):
    def __init__(self, name, obs, state, action_values, img, agent_position):
        self.observation = obs
        self.image = img
        self.state = state
        self.action_values = action_values
        self.name = name
        self.agent_position = agent_position

    def plot_image(self):
        plt.imshow(self.image)
        plt.show()

    def save_image(self, path, name):
        imageio.imwrite(path + '/' + name + '.png', self.image)


class DisagreementTrajectory(object):
    def __init__(self, a1_states, a2_states, importance_type, disagreement_importance,
                 trajectory_importance, horizon, a1_q_vals, a2_q_vals, agent_ratio):
        self.a1_states = a1_states
        self.a2_states = a2_states
        self.importance_type = importance_type
        self.disagreement_state_importance = disagreement_importance
        self.trajectory_importance = trajectory_importance
        self.horizon = horizon
        da_index = horizon // 2
        self.disagreement_score = disagreement_score(a1_states[da_index], a2_states[da_index],
                                                     disagreement_importance)
        self.importance = 0
        self.state_importance_list = []

        """calculate trajectory score"""
        if importance_type == 'trajectory':
            self.state_importance_list = self.get_trajectory_importance(
                a1_states, a2_states, da_index, a1_q_vals, a2_q_vals, agent_ratio)
            if trajectory_importance == 'max_min':
                self.importance = trajectory_importance_max_min(self.state_importance_list)
            elif trajectory_importance == 'max_avg':
                self.importance = trajectory_importance_max_avg(self.state_importance_list)
            elif trajectory_importance == 'max_min':
                self.importance = trajectory_importance_avg(self.state_importance_list)
            elif trajectory_importance == 'max_min':
                self.importance = trajectory_importance_avg_delta(self.state_importance_list)
            elif trajectory_importance == 'last_state':
                self.importance = self.state_importance_list[-1]
            # self.importance += self.disagreement_score
        else:
            self.importance = self.disagreement_score

    def get_trajectory_importance(self, t1, t2, da_index, a1_q, a2_q, agent_ratio):
        state_importance_list = []
        for i in range(da_index + 1, len(t1)):
            if t1[i].state == t2[i].state:
                state_importance_list.append(0)
            else:
                s1_a1_vals, s1_a2_vals = a1_q[t1[i].state], a2_q[t1[i].state]
                s2_a1_vals, s2_a2_vals = a1_q[t2[i].state], a2_q[t2[i].state]
                """the value of the state is defined by the best available action from it"""
                s1_score = max(s1_a1_vals) * agent_ratio + max(s1_a2_vals)
                s2_score = max(s2_a1_vals) * agent_ratio + max(s2_a2_vals)
                state_importance_list.append(abs(s1_score - s2_score))
        return state_importance_list

    def get_frames(self):
        a1_frames = [x.image for x in self.a1_states]
        a2_frames = [x.image for x in self.a2_states]
        if len(a1_frames) != self.horizon:
            a1_frames = a1_frames + [a1_frames[-1] for _ in range(self.horizon - len(a1_frames))]
            a2_frames = a2_frames + [a2_frames[-1] for _ in range(self.horizon - len(a2_frames))]
        return a1_frames, a2_frames


def disagreement(episode, trace, a2_env, a2_agent, a2_helper, t, a1_old_s,
                 a1_old_obs, a2_config, a2_agent_dir,
                 agent_rng, prev_actions, args):
    trajectory_states = disagreement_states(episode, a2_env, a2_agent, a2_helper,
                                            trace.trajectory_length // 2, t, a1_old_s, a1_old_obs,
                                            trace.a1_states)
    trace.a2_trajectories.append(trajectory_states)
    trace.disagreement_indexes.append(t)
    a2_env.close()
    del gym.envs.registration.registry.env_specs[a2_env.spec.id]
    return reload_agent(a2_config, a2_agent_dir, args.a2_trial, a2_config.seed, agent_rng, t,
                        prev_actions, episode, args)


def normalize_q_values(s):
    q = s.action_values
    q_sum = np.sum(q)
    if not q_sum:
        # can happen in states where all q values are 0
        return np.array([0, 0, 0, 0])
    normalized = np.true_divide(q, q_sum)
    return normalized


def second_best_confidence(s1, s2):
    """compare best action to second-best action"""
    # normalize to get probabilities
    a1_actions_normalized = normalize_q_values(s1)
    a2_actions_normalized = normalize_q_values(s2)
    # get difference between best action and second best
    sorted_q1 = sorted(a1_actions_normalized, reverse=True)
    sorted_q2 = sorted(a2_actions_normalized, reverse=True)
    a1_diff = sorted_q1[0] - sorted_q1[1]
    a2_diff = sorted_q2[0] - sorted_q2[1]
    return a1_diff + a2_diff


def better_than_you_confidence(s1, s2):
    """compare best action to the action chosen by the opposing agent"""
    # normalize to get probabilities
    a1_actions_normalized = normalize_q_values(s1)
    a2_actions_normalized = normalize_q_values(s2)
    # get difference between best action and second best
    a1_diff = a1_actions_normalized.max() - a1_actions_normalized[
        np.argmax(a2_actions_normalized)]
    a2_diff = a2_actions_normalized.max() - a2_actions_normalized[
        np.argmax(a1_actions_normalized)]
    return a1_diff + a2_diff


# def compare_state_values(s1, s2, weight=1):
#     """return the value difference between agents"""
# a1_actions_normalized = normalize_q_values(s1) * w1
# a2_actions_normalized = normalize_q_values(s2) * w2
# if importance == 'worst':
#     score1 = np.max(a1_actions_normalized) - np.min(a1_actions_normalized)
#     score2 = np.max(a2_actions_normalized) - np.min(a2_actions_normalized)
# elif importance == 'second':
#     score1 = np.max(a1_actions_normalized) - \
#              np.partition(a1_actions_normalized.flatten(), -2)[-2]
#     score2 = np.max(a2_actions_normalized) - \
#              np.partition(a2_actions_normalized.flatten(), -2)[-2]
# return abs(score1 - score2)


def disagreement_score(s1, s2, importance):
    if importance == 'sb':
        return second_best_confidence(s1, s2)
    elif importance == 'bety':
        return better_than_you_confidence(s1, s2)


def get_disagreement_frames(a1_frames, a1_tracker, da_frames, da_indexes,
                            traces_a2, window, freeze_on_death):
    """get agent disagreement frames"""
    a1_hl, a2_hl, i = {}, {}, 0
    num_frames = len(a1_frames)
    for frame_i in da_frames:
        dis_len = len(traces_a2[da_indexes.index(frame_i)]) - window
        same_frames = []
        a2_hl[i] = traces_a2[da_indexes.index(frame_i)]
        start = frame_i - window
        if start < 0:
            same_frames = [a1_frames[0] for _ in range(abs(start))]
            start = 0
        a1_hl[i] = same_frames
        end = frame_i + dis_len if frame_i + dis_len <= num_frames - 1 else num_frames - 1

        if freeze_on_death:
            # check for death
            for j in range(frame_i, frame_i + dis_len):
                if a1_tracker[j] == 1295 or j == num_frames - 1:
                    end = j
                    break
            else:
                end = frame_i + dis_len

        a1_hl[i] += a1_frames[start:end]

        # set to same length
        if len(a1_hl[i]) < len(a2_hl[i]):
            a2_hl[i] = a2_hl[i][:len(a1_hl[i])]
        # add last frames to achieve HL desired length
        if len(a1_hl[i]) < 2 * window:
            a1_hl[i] += [a1_hl[i][-1] for _ in
                         range(2 * window - len(a1_hl[i]))]
            a2_hl[i] += [a2_hl[i][-1] for _ in
                         range(2 * window - len(a2_hl[i]))]

        i += 1
    return a1_hl, a2_hl


def save_disagreements(a1_DAs, a2_DAs, output_dir, fps):
    highlight_frames_dir = join(output_dir, "highlight_frames")
    video_dir = join(output_dir, "videos")
    make_clean_dirs(video_dir)
    make_clean_dirs(highlight_frames_dir)

    height, width, layers = a1_DAs[0][0].shape
    size = (width, height)
    trajectory_length = len(a1_DAs[0])
    for hl_i in range(len(a1_DAs)):
        for img_i in range(len(a1_DAs[hl_i])):
            save_image(highlight_frames_dir,
                       "a1_DA{}_Frame{}".format(str(hl_i), str(img_i)),
                       a1_DAs[hl_i][img_i])
            save_image(highlight_frames_dir,
                       "a2_DA{}_Frame{}".format(str(hl_i), str(img_i)),
                       a2_DAs[hl_i][img_i])

        create_video(highlight_frames_dir, video_dir, "a1_DA" + str(hl_i), size,
                     trajectory_length, fps)
        create_video(highlight_frames_dir, video_dir, "a2_DA" + str(hl_i), size,
                     trajectory_length, fps)

    return video_dir


def disagreement_states(e, env, agent, helper, window, time_step, old_s,
                        old_obs, previous_states):
    # obtain last pre-disagreement states
    same_states = []
    start = time_step - window
    if start < 0:
        same_states = [previous_states[0] for _ in range(abs(start))]
        start = 0
    da_states = same_states + previous_states[start:]
    # run for for frame_window frames
    done = False
    for step in range(time_step, time_step + window):
        if done or step == 300: break
        a = agent.act(old_s)
        obs, r, done, _ = env.step(a)
        s = helper.get_state_from_observation(obs, r, done)
        agent.update(old_s, a, r, s)
        # helper.update_stats(e, step, old_obs, obs, old_s, a, r, s)
        frame = env.render()
        agent_pos = [int(x) for x in env.env.game_state.game.frog.position]
        da_states.append(
            State(step, old_obs, old_s, agent.q[old_s], frame, agent_pos))
        old_s = s
        old_obs = obs
    return da_states


def non_similar_trajectories(trajectories, all_a1_states, args):
    diverse_trajectories, seen_indexes, seen_score = [], set(), []
    sorted_trajectories = sorted(trajectories, key=lambda x: x.importance, reverse=True)
    for trajectory in sorted_trajectories:
        # if trajectory.disagreement_score in seen_score: continue
        indexes = [x.name for x in trajectory.a1_states]
        if len(seen_indexes.intersection(set(indexes))) > args.similarity_limit: continue
        if args.similarity_context:
            start_i = 0 if indexes[0] < args.similarity_context else \
                indexes[0] - args.similarity_context
            end_i = len(all_a1_states) - 1 if indexes[-1] > (
                    len(all_a1_states) - 1) - args.similarity_context \
                else indexes[-1] + args.similarity_context
            indexes = list(range(start_i, end_i + 1))
        diverse_trajectories.append(trajectory)
        [seen_indexes.add(x) for x in indexes]
        # seen_score.append(trajectory.disagreement_score)
    return diverse_trajectories


def get_top_k_disagreements(traces, args):
    """"""
    top_k_diverse_trajectories = []
    """get diverse trajectories"""
    all_trajectories = []
    for trace in traces:
        all_trajectories += [t for t in trace.diverse_trajectories]
    sorted_trajectories = sorted(all_trajectories, key=lambda x: x.importance, reverse=True)

    for trajectory in sorted_trajectories:
        top_k_diverse_trajectories.append(trajectory)
        if len(top_k_diverse_trajectories) == args.n_disagreements:
            break

    return top_k_diverse_trajectories


def trajectory_distance(t1, t2):
    """get the distance between two trajectories"""
    pass
