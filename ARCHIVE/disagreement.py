from os.path import join
import gym
import numpy as np

from get_trajectories import State, DisagreementTrajectory
from utils import save_image, create_video, make_clean_dirs, load_agent_aux, reload_agent



def disagreement(da, a2_env, a1_agent, a2_agent, a2_helper, t, a1_old_s, a1_old_obs, a2_config, a2_agent_dir, agent_rng,
                 prev_actions, args):
    da.a2_trajectories.append(
        disagreement_states(a2_env, a2_agent, a2_helper, args.importance_type, da.frame_window, t, a1_old_s,
                            a1_old_obs, da.a1_states, args.freeze_on_death))
    if args.importance_type == 'state':
        a1_q = a1_agent.q[a1_old_s]
        a2_q = a2_agent.q[a1_old_s]
        da.importance_scores.append(disagreement_score(a1_q, a2_q, args.state_importance, by_state=True))
    da.disagreement_indexes.append(t)
    a2_env.close()
    del gym.envs.registration.registry.env_specs[a2_env.spec.id]
    return reload_agent(a2_config, a2_agent_dir, args.a2_trial, a2_config.seed, agent_rng, t, prev_actions, args)


def normalize_q_values(s, by_state=False):
    if by_state:
        q = s
    else:
        q = s.action_values
    q_sum = np.sum(q)
    if not q_sum:
        # can happen in states where all q values are 0
        return np.array([0, 0, 0, 0])
    normalized = np.true_divide(q, q_sum)
    return normalized


def second_best_confidence(s1, s2, by_state):
    """compare best action to second-best action"""
    # normalize to get probabilities
    a1_actions_normalized = normalize_q_values(s1, by_state)
    a2_actions_normalized = normalize_q_values(s2, by_state)
    # get difference between best action and second best
    sorted_q1 = sorted(a1_actions_normalized, reverse=True)
    sorted_q2 = sorted(a2_actions_normalized, reverse=True)
    a1_diff = sorted_q1[0] - sorted_q1[1]
    a2_diff = sorted_q2[0] - sorted_q2[1]
    return a1_diff + a2_diff


def better_than_you_confidence(s1, s2, by_state):
    """compare best action to the action chosen by the opposing agent"""
    # normalize to get probabilities
    a1_actions_normalized = normalize_q_values(s1, by_state)
    a2_actions_normalized = normalize_q_values(s2, by_state)
    # get difference between best action and second best
    a1_diff = a1_actions_normalized.max() - a1_actions_normalized[np.argmax(a2_actions_normalized)]
    a2_diff = a2_actions_normalized.max() - a2_actions_normalized[np.argmax(a1_actions_normalized)]
    return a1_diff + a2_diff


def trajectory_score(s1, s2, importance):
    a1_actions_normalized = normalize_q_values(s1)
    a2_actions_normalized = normalize_q_values(s2)
    if importance == 'worst':
        score1 = np.max(a1_actions_normalized) - np.min(a1_actions_normalized)
        score2 = np.max(a2_actions_normalized) - np.min(a2_actions_normalized)
    elif importance == 'second':
        score1 = np.max(a1_actions_normalized) - np.partition(a1_actions_normalized.flatten(), -2)[-2]
        score2 = np.max(a2_actions_normalized) - np.partition(a2_actions_normalized.flatten(), -2)[-2]
    return abs(score1 - score2)


def get_trajectory_importance(t1, t2, kargs):
    trajectory_importance = []
    da_index = kargs.horizon // 2
    # trajectory_importance.append(disagreement_score(t1[da_index], t2[da_index], kargs.disagreement_importance))
    da_importance = disagreement_score(t1[da_index], t2[da_index], kargs.disagreement_importance)
    for i in range((kargs.horizon // 2) + 1, len(t1)):
        trajectory_importance.append(trajectory_score(t1[i], t2[i], kargs.state_importance))
    return da_importance, trajectory_importance


def disagreement_score(s1, s2, importance, by_state=False):
    if importance == 'sb':
        return second_best_confidence(s1, s2, by_state)
    elif importance == 'bety':
        return better_than_you_confidence(s1, s2, by_state)


def get_disagreement_frames(a1_frames, a1_tracker, da_frames, da_indexes, traces_a2, window, freeze_on_death):
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


def disagreement_states(env, agent, helper, importance_type, window, time_step, old_s, old_obs, previous_states,
                        freeze_on_death):
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
        if done or (freeze_on_death and (old_s == 1295 or old_s == 1036)):
            # end if done or if freeze_on_death option is chosen and death/win occurs
            break
        # record every step of the second agent
        a = agent.act(old_s)
        obs, r, done, _ = env.step(a)
        s = helper.get_state_from_observation(obs, r, done)
        agent.update(old_s, a, r, s)
        helper.update_stats(0, step, old_obs, obs, old_s, a, r, s)
        # frame = env.video_recorder.last_frame
        frame = env.render()
        if importance_type == 'trajectory':
            da_states.append(State(step, old_obs, old_s, agent.q[old_s], frame))
        else:
            da_states.append(frame)
        old_s = s
        old_obs = obs
    return da_states


def get_diverse_trajectories(trajectories, all_a1_states, args):
    diverse_trajectories, seen_indexes, seen_score = [], set(), []
    sorted_trajectories = sorted(trajectories, key=lambda x: x.importance[args.trajectory_importance] +
                                                             x.disagreement_state_importance, reverse=True)
    for trajectory in sorted_trajectories:
        if trajectory.disagreement_state_importance in seen_score: continue
        indexes = [x.name for x in trajectory.a1_states]
        if args.similarity_context:
            start_i = 0 if indexes[0] < args.similarity_context else indexes[0] - args.similarity_context
            end_i = len(all_a1_states) - 1 if indexes[-1] > (len(all_a1_states) - 1) - args.similarity_context \
                else indexes[-1] + args.similarity_context
            indexes = list(range(start_i, end_i + 1))
        if len(seen_indexes.intersection(set(indexes))) > args.similarity_limit: continue
        diverse_trajectories.append(trajectory)
        [seen_indexes.add(x) for x in indexes]
        seen_score.append(trajectory.disagreement_state_importance)
        if len(diverse_trajectories) == args.n_disagreements:
            break
    return diverse_trajectories


def get_top_k_disagreements(a2_trajectories, a1_states, disagreement_indexes, importance_scores, args):
    seen, top_da_frame_indexes, top_k_diverse_trajectories = [], [], []
    if args.importance_type == 'trajectory':
        disagreement_trajectories = []
        for a2_traj in a2_trajectories:
            start = a2_traj[0].name
            end = start + len(a2_traj)
            if len(a1_states) <= end:
                continue
            a1_traj = a1_states[start:end]
            a2_traj = a2_traj[:len(a1_traj)]
            disagreement_importance, trajectory_score = get_trajectory_importance(a2_traj, a1_traj, args)
            disagreement_trajectories.append(
                DisagreementTrajectory(a1_traj, a2_traj, disagreement_importance, trajectory_score))

        """get diverse trajectories"""
        top_k_diverse_trajectories = get_diverse_trajectories(disagreement_trajectories, a1_states, args)
        top_da_frame_indexes = [x.disagreement_index for x in top_k_diverse_trajectories]

    else:  # importance by state
        importance_sorted_indexes = sorted(list(enumerate(importance_scores)), key=lambda x: x[1], reverse=True)
        for idx, score in importance_sorted_indexes:
            if score not in seen:
                seen.append(score)  # TODO is this a good diversity measure? mmm...
                top_da_frame_indexes.append(disagreement_indexes[idx])
            if len(top_da_frame_indexes) == args.n_disagreements:
                break

    return top_k_diverse_trajectories, top_da_frame_indexes