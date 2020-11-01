import numpy as np


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


def disagreement_frames(env, agent, helper, window, time_step, old_s, old_obs, previous_frames, episode, crop_out):
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
        if done or old_s == 1295:  # adds same frame if done so that all vids are same length
            # disagreement_frames.append(env.video_recorder.last_frame)
            # continue
            break
        # record every step of the second agent
        a = agent.act(old_s)
        obs, r, done, _ = env.step(a)

        s = helper.get_state_from_observation(obs, r, done)
        agent.update(old_s, a, r, s)
        helper.update_stats(episode, time_step, old_obs, obs, old_s, a, r, s)
        old_s = s
        old_obs = obs
        # save video scenes
        disagreement_frames.append(env.video_recorder.last_frame[:crop_out])
    return disagreement_frames


def disagreement_score(a1, a2, current_state, importance):
    if importance == 'sb':
        return second_best_confidence(a1, a2, current_state)
    elif importance == 'bety':
        return better_than_you_confidence(a1, a2, current_state)


def get_disagreement_frames(a1_frames, a1_tracker, hl, traces_a2, window):
    """get agent disagreement frames"""
    a1_hl, a2_hl, i = {}, {}, 0
    num_frames = len(a1_frames)
    for d_i, frame_i in hl:
        print(f'chosen disagreement frame: {frame_i}')
        dis_len = len(traces_a2[d_i]) - window
        same_frames = []
        a2_hl[i] = traces_a2[d_i]
        start = frame_i - window
        if start < 0:
            same_frames = [a1_frames[0] for _ in range(abs(start))]
            start = 0
        a1_hl[i] = same_frames

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