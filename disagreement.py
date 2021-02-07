import logging
from copy import copy
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
    def __init__(self, horizon, a1_q_values, a2_q_values, agent_ratio, episode):
        self.a1_q_values = a1_q_values
        self.a2_q_values = a2_q_values
        self.episode = episode
        self.a1_states = []
        self.a1_scores = []
        self.a2_scores = []
        self.lilies_reached = 0
        self.agent_ratio = agent_ratio
        self.a2_trajectories = []
        self.disagreement_indexes = []
        self.importance_scores = []
        self.trajectory_length = horizon
        self.disagreement_trajectories = []
        self.diverse_trajectories = []
        self.min_traj_len = self.trajectory_length

    def get_trajectories(self, episode, importance_type, da_importance, t_importance):
        for i, a2_traj in enumerate(self.a2_trajectories):
            start = a2_traj[0].name[1]
            end = start + len(a2_traj)
            if len(self.a1_states) <= end:
                a1_traj = self.a1_states[start:]
            else:
                a1_traj = self.a1_states[start:end]
            a2_traj = a2_traj[:len(a1_traj)]
            if len(a1_traj) < self.min_traj_len: continue
            dt = DisagreementTrajectory(a1_traj, a2_traj, importance_type, da_importance,
                                        t_importance, self.trajectory_length, episode,
                                        self.a1_q_values, self.a2_q_values, self.agent_ratio)
            self.disagreement_trajectories.append(dt)


class State(object):
    def __init__(self, idx, episode, obs, state, action_values, img, agent_position):
        self.observation = obs
        self.image = img
        self.state = state
        self.action_values = action_values
        self.name = (episode, idx)
        self.agent_position = agent_position

    def plot_image(self):
        plt.imshow(self.image)
        plt.show()

    def save_image(self, path, name):
        imageio.imwrite(path + '/' + name + '.png', self.image)


class DisagreementTrajectory(object):
    def __init__(self, a1_states, a2_states, importance_type, disagreement_importance,
                 trajectory_importance, horizon, episode, a1_q_vals, a2_q_vals, agent_ratio):
        self.a1_states = a1_states
        self.a2_states = a2_states
        self.episode = episode
        self.importance_type = importance_type
        self.disagreement_state_importance = disagreement_importance
        self.trajectory_importance = trajectory_importance
        self.horizon = horizon
        da_index = (horizon // 2) -1
        self.disagreement_score = disagreement_score(a1_states[da_index], a2_states[da_index],
                                                     disagreement_importance, agent_ratio)
        self.importance = 0
        self.state_importance_list = []

        """calculate trajectory score"""
        if importance_type == 'state':
            self.importance = self.disagreement_score
        elif 'last_state' in trajectory_importance:
            if "loc" in trajectory_importance:
                height_diff = abs(
                    a1_states[-1].agent_position[1] - a2_states[-1].agent_position[1])
                self.importance = height_diff
            else:
                self.importance = self.get_trajectory_importance_last_state(
                    a1_states[-1].state, a2_states[-1].state, a1_q_vals, a2_q_vals, agent_ratio)
        else:
            self.state_importance_list = self.get_trajectory_importance(
                a1_states, a2_states, da_index, a1_q_vals, a2_q_vals, agent_ratio, weights=True)
            if trajectory_importance == 'max_min':
                self.importance = trajectory_importance_max_min(self.state_importance_list)
            elif trajectory_importance == 'max_avg':
                self.importance = trajectory_importance_max_avg(self.state_importance_list)
            elif trajectory_importance == 'max_min':
                self.importance = trajectory_importance_avg(self.state_importance_list)
            elif trajectory_importance == 'max_min':
                self.importance = trajectory_importance_avg_delta(self.state_importance_list)

    @staticmethod
    def get_trajectory_importance_last_state(s1, s2, a1_q, a2_q, agent_ratio):
        """state values"""
        if s1 == s2: return 0
        s1_a1_vals = a1_q[s1]
        s1_a2_vals = a2_q[s1]
        s2_a1_vals = a1_q[s2]
        s2_a2_vals = a2_q[s2]
        """the value of the state is defined by the best available action from it"""
        s1_score = max(s1_a1_vals) * agent_ratio + max(s1_a2_vals)
        s2_score = max(s2_a1_vals) * agent_ratio + max(s2_a2_vals)
        return abs(s1_score - s2_score)

    @staticmethod
    def get_trajectory_importance(t1, t2, da_index, a1_q, a2_q, agent_ratio, weights=False):
        state_importance_list = []
        a1_death_goal_weight, a2_death_goal_weight = 0, 0
        for i in range(da_index + 1, len(t1)):
            if weights:
                # penalty if death, bonus if lili's reached
                if t1[i].state == 1295:
                    a1_death_goal_weight -= 1000
                elif t1[i].state == 1036:
                    a1_death_goal_weight += 5000
                if t2[i].state == 1295:
                    a2_death_goal_weight -= 1000
                elif t2[i].state == 1036:
                    a2_death_goal_weight += 5000
            """state values"""
            s1_a1_vals = a1_q[t1[i].state] + a1_death_goal_weight
            s1_a2_vals = a2_q[t1[i].state] + a1_death_goal_weight
            s2_a1_vals = a1_q[t2[i].state] + a2_death_goal_weight
            s2_a2_vals = a2_q[t2[i].state] + a2_death_goal_weight
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


def disagreement(episode, trace, a2_env, a2_agent, a2_helper, t, a1_curr_s,
                 a2_config, a2_agent_dir, agent_rng, prev_actions, args):
    trajectory_states, trajectory_scores = disagreement_states(trace, a2_env, a2_agent, a2_helper,
                                                               t + 1, a1_curr_s)

    trace.a2_trajectories.append(trajectory_states)
    trace.a2_scores.append(trajectory_scores)
    trace.disagreement_indexes.append(t)
    a2_env.close()
    del gym.envs.registration.registry.env_specs[a2_env.spec.id]
    return reload_agent(a2_config, a2_agent_dir, 1, a2_config.seed, agent_rng, t,
                        prev_actions, episode, args)



def second_best_confidence(s1, s2, agent_ratio):
    """compare best action to second-best action"""
    a1_vals = (s1.action_values / abs(sum(s1.action_values)))
    a2_vals = (s2.action_values / abs(sum(s2.action_values)))
    sorted_q1 = sorted(a1_vals, reverse=True)
    sorted_q2 = sorted(a2_vals, reverse=True)
    a1_diff = sorted_q1[0] - sorted_q1[1] * agent_ratio
    a2_diff = sorted_q2[0] - sorted_q2[1]
    return a1_diff + a2_diff


def better_than_you_confidence(s1, s2, agent_ratio):
    a1_vals = (s1.action_values/abs(sum(s1.action_values)))
    a2_vals = (s2.action_values/abs(sum(s2.action_values)))
    a1_diff = (max(a1_vals) - a1_vals[np.argmax(a2_vals)]) * agent_ratio
    a2_diff = max(a2_vals) - a2_vals[np.argmax(a1_vals)]
    return a1_diff + a2_diff


def disagreement_score(s1, s2, importance, agent_ratio):
    if importance == 'sb':
        return second_best_confidence(s1, s2, agent_ratio)
    elif importance == 'bety':
        return better_than_you_confidence(s1, s2, agent_ratio)


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
            save_image(highlight_frames_dir, "a1_DA{}_Frame{}".format(str(hl_i), str(img_i)),
                       a1_DAs[hl_i][img_i])
            save_image(highlight_frames_dir, "a2_DA{}_Frame{}".format(str(hl_i), str(img_i)),
                       a2_DAs[hl_i][img_i])

        create_video(highlight_frames_dir, video_dir, "a1_DA" + str(hl_i), size,
                     trajectory_length, fps)
        create_video(highlight_frames_dir, video_dir, "a2_DA" + str(hl_i), size,
                     trajectory_length, fps)
    return video_dir


def disagreement_states(trace, env, agent, helper, time_step, curr_s):
    # obtain last pre-disagreement states
    same_scores = []
    same_states = []
    start = time_step - trace.trajectory_length // 2
    if start < 0:
        same_states = [trace.a1_states[0] for _ in range(abs(start))]
        same_scores = [trace.a1_scores[0] for _ in range(abs(start))]
        start = 0
    da_states = same_states + trace.a1_states[start:-1]
    the_da_state = copy(trace.a1_states[-1])
    the_da_state.action_values = agent.q[curr_s]
    da_states.append(the_da_state)
    da_scores = same_scores + trace.a1_scores[start:]
    # run for for frame_window frames
    done = False
    lilies_reached = trace.lilies_reached
    for step in range(time_step, time_step + trace.trajectory_length // 2):
        if done or step == 300: break
        a = agent.act(curr_s)
        obs, r, done, _ = env.step(a)
        s = helper.get_state_from_observation(obs, r, done)
        if s == 1036:
            lilies_reached += 1
            done = True if lilies_reached == 2 else False
        agent.update(curr_s, a, r, s)
        frame = env.render()
        agent_pos = [int(x) for x in env.env.game_state.game.frog.position]
        curr_s = s
        curr_obs = obs
        da_states.append(
            State(step, trace.episode, curr_obs, curr_s, agent.q[curr_s], frame, agent_pos))
        da_scores.append(env.env.previous_score)
    return da_states, da_scores

def get_top_k_disagreements(traces, args):
    """"""
    top_k_diverse_trajectories, discarded_context, discarded_importance = [], [], []
    """get diverse trajectories"""
    all_trajectories = []
    for trace in traces:
        all_trajectories += [t for t in trace.disagreement_trajectories]
    sorted_trajectories = sorted(all_trajectories, key=lambda x: x.importance, reverse=True)

    seen_importance = []
    seen_indexes = {i: [] for i in range(len(traces))}
    for t in sorted_trajectories:
        """unique score for frogger"""
        # if t.importance in seen_importance:
        #     discarded_importance.append(t); continue
        # else:
        #     seen_importance.append(t.importance)

        # if args.importance_type == "trajectory":
        """last_state_loc: specific distances"""
        if args.trajectory_importance == "last_state_loc":
            if t.importance == 429: continue
        """last_state_val"""
        if args.trajectory_importance == "last_state_val":
            if t.a1_states[-2].state == t.a2_states[-2].state:
                continue
            if t.a1_states[-1].state == 1036 or t.a2_states[-1].state == 1036:
                continue


        t_indexes = [s.name for s in t.a1_states]
        intersecting_indexes = set(seen_indexes[t.episode]).intersection(set(t_indexes))
        if len(intersecting_indexes) > args.similarity_limit:
            discarded_context.append(t)
            continue
        seen_indexes[t.episode] += t_indexes
        top_k_diverse_trajectories.append(t)
        if len(top_k_diverse_trajectories) == args.n_disagreements:
            break

    if not len(top_k_diverse_trajectories) == args.n_disagreements:
        top_k_diverse_trajectories += discarded_context
    if not len(top_k_diverse_trajectories) == args.n_disagreements:
        top_k_diverse_trajectories += discarded_importance
    for i in range(args.n_disagreements - len(top_k_diverse_trajectories)):
        top_k_diverse_trajectories.append(discarded_context[i])
    top_k_diverse_trajectories = top_k_diverse_trajectories[:args.n_disagreements]
    logging.info(f'Chosen disagreements:')
    for d in top_k_diverse_trajectories:
        logging.info(f'Name: {d.a1_states[(args.horizon//2)-1].name}, '
                     f'State: {d.a1_states[(args.horizon//2)-1].state} '
                     f'Position: {d.a1_states[(args.horizon//2)-1].agent_position}')

    """make all trajectories the same length"""
    for t in top_k_diverse_trajectories:
        if len(t.a1_states) < args.horizon:
            for _ in range(args.horizon - len(t.a1_states)):
                t.a1_states += [t.a1_states[-1]]
                t.a2_states += [t.a2_states[-1]]
    return top_k_diverse_trajectories
