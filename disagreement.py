from os.path import join

import numpy as np

from get_trajectories import State, DisagreementTrajectory
from utils import save_image, create_video, make_clean_dirs


def normalize_q_values(a, s):
    q = a.q[s]
    q_sum = np.sum(q)
    # TODO yotam: this is a stitch up because there can be states where all q values are 0
    if not q_sum:
        return np.array([0, 0, 0, 0])
    normalized = np.true_divide(q, q_sum)
    return normalized


def second_best_confidence(a1, a2, state):
    """compare best action to second-best action"""

    # normalize to get probabilities
    a1_actions_normalized = normalize_q_values(a1, state)
    a2_actions_normalized = normalize_q_values(a2, state)

    # get difference between best action and second best
    sorted_q1 = sorted(a1_actions_normalized, reverse=True)
    sorted_q2 = sorted(a2_actions_normalized, reverse=True)
    a1_diff = sorted_q1[0] - sorted_q1[1]
    a2_diff = sorted_q2[0] - sorted_q2[1]
    return a1_diff + a2_diff


def better_than_you_confidence(a1, a2, state):
    """compare best action to the action chosen by the opposing agent"""

    # normalize to get probabilities
    a1_actions_normalized = normalize_q_values(a1, state)
    a2_actions_normalized = normalize_q_values(a2, state)

    # get difference between best action and second best

    a1_diff = a1_actions_normalized.max() - a1_actions_normalized[np.argmax(a2_actions_normalized)]
    a2_diff = a2_actions_normalized.max() - a2_actions_normalized[np.argmax(a1_actions_normalized)]
    return a1_diff + a2_diff


def disagreement_frames(env, agent, helper, window, time_step, old_s, old_obs, previous_frames, freeze_on_death):
    # obtain last pre-disagreement frames
    same_frames = []
    start = time_step - window
    if start < 0:
        same_frames = [previous_frames[0] for _ in range(abs(start))]
        start = 0
    disagreement_frames = same_frames + previous_frames[start:]
    # run for for frame_window frames
    done = False
    for step in range(window):
        # TODO added the part : old_s == 1295 - this is the death state
        if done or (freeze_on_death and old_s == 1295):  # adds same frame if done so that all vids are same length
            # disagreement_frames.append(env.video_recorder.last_frame)
            # continue'
            break
        # record every step of the second agent
        a = agent.act(old_s)
        obs, r, done, _ = env.step(a)

        s = helper.get_state_from_observation(obs, r, done)
        agent.update(old_s, a, r, s)
        helper.update_stats(0, time_step, old_obs, obs, old_s, a, r, s)
        old_s = s
        old_obs = obs
        # save video scenes
        disagreement_frames.append(env.video_recorder.last_frame)
    return disagreement_frames


def disagreement_score(a1, a2, current_state, importance):
    if importance == 'sb':
        return second_best_confidence(a1, a2, current_state)
    elif importance == 'bety':
        return better_than_you_confidence(a1, a2, current_state)


def get_disagreement_frames(a1_frames, a1_tracker, da, traces_a2, window, freeze_on_death):
    """get agent disagreement frames"""
    a1_hl, a2_hl, i = {}, {}, 0
    num_frames = len(a1_frames)
    for d_i, frame_i in da:
        dis_len = len(traces_a2[d_i]) - window
        same_frames = []
        a2_hl[i] = traces_a2[d_i]
        start = frame_i - window
        if start < 0:
            same_frames = [a1_frames[0] for _ in range(abs(start))]
            start = 0
        a1_hl[i] = same_frames
        end = frame_i + dis_len if frame_i + dis_len <= num_frames-1 else num_frames-1

        if freeze_on_death:
            # check for death
            for j in range(frame_i, frame_i+dis_len):
                if a1_tracker[j] == 1295 or j == num_frames-1:
                    end = j
                    break
            else:
                end = frame_i+dis_len

        a1_hl[i] += a1_frames[start:end]

        # set to same length
        if len(a1_hl[i]) < len(a2_hl[i]):
            a2_hl[i] = a2_hl[i][:len(a1_hl[i])]
        # add last frames to achieve HL desired length
        if len(a1_hl[i]) < 2 * window:
            a1_hl[i] += [a1_hl[i][-1] for _ in range(2 * window - len(a1_hl[i]))]
            a2_hl[i] += [a2_hl[i][-1] for _ in range(2 * window - len(a2_hl[i]))]

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
            save_image(highlight_frames_dir, "a1_DA{}_Frame{}".format(str(hl_i), str(img_i)), a1_DAs[hl_i][img_i])
            save_image(highlight_frames_dir, "a2_DA{}_Frame{}".format(str(hl_i), str(img_i)), a2_DAs[hl_i][img_i])

        create_video(highlight_frames_dir, video_dir, "a1_DA" + str(hl_i), size, trajectory_length, fps)
        create_video(highlight_frames_dir, video_dir, "a2_DA" + str(hl_i), size, trajectory_length, fps)

    return video_dir


def get_top_k_disagreements(importance_scores, disagreement_indexes, behavior_tracker, args):
    importance_sorted_indexes = sorted(list(importance_scores.items()), key=lambda x: x[1], reverse=True)
    seen, top_k_diverse_state_indexes = [], []
    for idx, score in importance_sorted_indexes:
        disagreement_state = behavior_tracker.s_s[0][disagreement_indexes[idx]]
        if disagreement_state not in seen:
            seen.append(disagreement_state) # TODO is this a good diversity measure? mmm...
            top_k_diverse_state_indexes.append(idx)
        if len(top_k_diverse_state_indexes) == args.n_disagreements:
            break
    return sorted([(x, disagreement_indexes[x]) for x in top_k_diverse_state_indexes])


######## STATES

def disagreement_states(env, agent, helper, window, time_step, old_s, old_obs, previous_states, freeze_on_death):
    # obtain last pre-disagreement states
    same_states = []
    start = time_step - window
    if start < 0:
        same_states = [previous_states[0] for _ in range(abs(start))]
        start = 0
    da_states = same_states + previous_states[start:]
    # run for for frame_window frames
    done = False
    for step in range(time_step, time_step+window):
        # TODO added the part : old_s == 1295 - this is the death state
        if done or (freeze_on_death and (old_s == 1295 or old_s == 1036)):  # adds same frame if done so that all vids are same length
            break
        # record every step of the second agent
        a = agent.act(old_s)
        obs, r, done, _ = env.step(a)
        s = helper.get_state_from_observation(obs, r, done)
        agent.update(old_s, a, r, s)
        helper.update_stats(0, step, old_obs, obs, old_s, a, r, s)
        frame = env.video_recorder.last_frame
        da_states.append(State(step, old_obs, old_s, agent.q[old_s], frame))
        old_s = s
        old_obs = obs
        # save video scenes
    return da_states


def normalize_state_q_values(s):
    q = s.action_values
    q_sum = np.sum(q)
    # TODO yotam: this is a stitch up because there can be states where all q values are 0
    if not q_sum:
        return np.array([0, 0, 0, 0])
    normalized = np.true_divide(q, q_sum)
    return normalized


def second_best_state_confidence(state1, state2):
    """compare best action to second-best action"""
    # normalize to get probabilities
    s1_actions_normalized = normalize_state_q_values(state1)
    s2_actions_normalized = normalize_state_q_values(state2)
    # get difference between best action and second best
    sorted_q1 = sorted(s1_actions_normalized, reverse=True)
    sorted_q2 = sorted(s2_actions_normalized, reverse=True)
    s1_diff = sorted_q1[0] - sorted_q1[1]
    s2_diff = sorted_q2[0] - sorted_q2[1]
    return s1_diff + s2_diff


def better_than_you_state_confidence(state1, state2):
    """compare best action to the action chosen by the opposing agent"""
    # normalize to get probabilities
    s1_actions_normalized = normalize_state_q_values(state1)
    s2_actions_normalized = normalize_state_q_values(state2)
    # get difference between best action and second best
    s1_diff = s1_actions_normalized.max() - s1_actions_normalized[np.argmax(s2_actions_normalized)]
    s2_diff = s2_actions_normalized.max() - s2_actions_normalized[np.argmax(s1_actions_normalized)]
    return s1_diff + s2_diff


def get_trajectory_importance(t1, t2, kargs):
    trajectory_importance = []
    for i in range((kargs.horizon//2)-1, len(t1)):
        if kargs.state_importance == 'sb':
            trajectory_importance.append(second_best_state_confidence(t1[i], t2[i]))
        elif kargs.state_importance == 'bety':
            trajectory_importance.append(better_than_you_state_confidence(t1[i], t2[i]))

    return trajectory_importance


def get_top_k_disagreement_trajectories(a2_trajectories, a1_states, args):
    disagreement_trajectories = []
    for a2_traj in a2_trajectories:
        start = a2_traj[0].name
        end = start + len(a2_traj)
        if len(a1_states) <= end:
            end = len(a1_states) -1
        a1_traj = a1_states[start:end]
        a2_traj = a2_traj[:len(a1_traj)]
        trajectory_score = get_trajectory_importance(a2_traj, a1_traj, args)
        disagreement_trajectories.append(DisagreementTrajectory(a1_traj, a2_traj, trajectory_score))

    importance_sorted_trajectories = \
        sorted(disagreement_trajectories, key=lambda x: x.importance[args.trajectory_importance], reverse=True)
    seen, top_k_diverse_trajectories = [], []
    for trajectory in importance_sorted_trajectories:
        score = trajectory.importance[args.trajectory_importance]
        if score in seen:
            continue
        top_k_diverse_trajectories.append(trajectory)
        seen.append(trajectory.importance[args.trajectory_importance])
        if len(top_k_diverse_trajectories) == args.n_disagreements:
            break
    return top_k_diverse_trajectories

