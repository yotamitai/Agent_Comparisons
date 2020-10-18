import argparse
import gym
import numpy as np
from os import makedirs
from os.path import exists, join, dirname
from gym.wrappers import Monitor

from interestingness_xrl.learning.behavior_tracker import BehaviorTracker
from interestingness_xrl.scenarios import create_helper

from agent_runner import video_schedule
from importance import better_than_you_confidence, second_best_confidence
from utils import get_agent_output_dir, create_agent, load_agent_config, create_video


def load_agent_aux(config, agent_dir, trial, seed, agent_rng, params, no_output=False):
    helper = create_helper(config)
    if not no_output:
        output_dir = params.output if params.output is not None else get_agent_output_dir(config, 5, trial)
    else:
        output_dir = join(dirname(dirname(agent_dir)), 'compare/temp')
    if not exists(output_dir):
        makedirs(output_dir)
    config.save_json(join(output_dir, 'config.json'))
    helper.save_state_features(join(output_dir, 'state_features.csv'))

    env_id = '{}-{}-v0'.format(config.gym_env_id, trial)
    helper.register_gym_environment(env_id, False, params.fps, params.show_score_bar)
    env = gym.make(env_id)
    config.num_episodes = params.num_episodes
    video_callable = video_schedule(config, params.record)
    env = Monitor(env, directory=output_dir, force=True, video_callable=video_callable)
    env.env.monitor = env
    env.seed(seed)
    agent, exploration_strategy = create_agent(helper, 5, agent_rng)
    agent.load(agent_dir)
    behavior_tracker = BehaviorTracker(config.num_episodes)
    return env, helper, agent, behavior_tracker, output_dir, video_callable


def disagreement_frames(env, agent, helper, window, time_step, old_s, old_obs, previous_frames, episode):
    # obtain last pre-disagreement frames
    same_frames = []
    start = time_step - window
    if start < 0:
        same_frames = [env.video_recorder.last_frame for _ in range(abs(start))]
        start = 0
    disagreement_frames = same_frames + previous_frames[start:]
    # run for for frame_window frames
    done = False
    for step in range(window):
        #TODO added the part : old_s == 1295 - this is the death state
        if done or old_s == 1295:  # adds same frame if done so that all vids are same length
            disagreement_frames.append(env.video_recorder.last_frame)
            continue
        # record every step of the second agent
        a = agent.act(old_s)
        obs, r, done, _ = env.step(a)

        # x = env.render(mode='human')
        # from PIL import Image
        # img = Image.fromarray(x, 'RGB')
        # img.show()
        # if env.game_state.game_over():
        #     pass

        s = helper.get_state_from_observation(obs, r, done)
        agent.update(old_s, a, r, s)
        helper.update_stats(episode, time_step, old_obs, obs, old_s, a, r, s)
        old_s = s
        old_obs = obs
        # save video scenes
        disagreement_frames.append(env.video_recorder.last_frame)
    return disagreement_frames


def reload_agent(config, agent_dir, trial, seed, rng, t, actions, params):
    env, helper, agent, behavior_tracker, output_dir, _ = \
        load_agent_aux(config, agent_dir, trial, seed, rng, params, no_output=True)
    old_obs = env.reset()
    old_s = helper.get_state_from_observation(old_obs, 0, False)
    i = 0
    for a in actions:
        obs, r, done, _ = env.step(a)
        s = helper.get_state_from_observation(obs, r, done)
        r = helper.get_reward(old_s, a, r, s, done)
        agent.update(old_s, a, r, s)
        behavior_tracker.add_sample(old_s, a)
        helper.update_stats(0, t, old_obs, obs, old_s, a, r, s)
        old_s = s
        old_obs = obs
        i += 1
    return env, helper, agent, behavior_tracker


def merge_frames(highlights_1, highlights_2, window):
    merged_frames = []
    buffer = list(highlights_1[0][0].shape)
    buffer[1] = 6
    separator_shape = np.concatenate((highlights_1[0][0], np.full(tuple(buffer), 0), highlights_2[0][0]), axis=1).shape
    separator = [np.full(separator_shape, 0) for _ in range(10)]

    for i in range(len(highlights_1)):
        for j in range(window):
            merged_frames.append(
                np.concatenate((highlights_1[i][j], np.full(tuple(buffer), 0), highlights_2[i][j]), axis=1))
        merged_frames += separator
    return merged_frames


# Archived
# def save_frames(frames, path):
#     frames_dir = join(path, "frames")
#     try:
#         makedirs(frames_dir)
#     except:
#         clean_dir(frames_dir)
#     for i, f in enumerate(frames):
#         img_name = str(i)
#         save_image(frames_dir, img_name, f)
#     return frames_dir


def divergence_score(a1, a2, current_state, importance):
    if importance == 'sb':
        return second_best_confidence(a1, a2, current_state)
    elif importance == 'bety':
        return better_than_you_confidence(a1, a2, current_state)
    # if importance == 'avg_q_value':
    #     return avg_q_value_score()


def get_disagreement_frames(a1_frames, hl, traces_a2, window):
    """get agent disagreement frames"""
    a1_hl, a2_hl, i = {}, {}, 0
    num_frames = len(a1_frames)
    for d_i, frame_i in hl:
        print(f'chosen disagreement frame: {frame_i}')
        same_frames = []
        a2_hl[i] = traces_a2[d_i]
        start = frame_i - window
        if start < 0:
            same_frames = [a1_frames[0] for _ in range(abs(start))]
            start = 0
        end = frame_i + window if frame_i + window < num_frames else num_frames - 1
        a1_hl[i] = same_frames + a1_frames[start:end]
        if len(a1_hl[i]) < 2 * window:
            a1_hl[i] += [a1_frames[-1] for _ in range(2 * window - len(a1_hl[i]))]
        i += 1
        return a1_hl, a2_hl


def online_comparison(args):
    """Compare two agents running online, search for disagreements"""

    """get agents"""
    a1_config, a1_agent_dir = load_agent_config(args.a1_results, args.a1_trial)
    a2_config, a2_agent_dir = load_agent_config(args.a2_results, args.a2_trial)
    seed = a1_config.seed
    agent_rng = np.random.RandomState(seed)
    a1_env, a1_helper, a1_agent, a1_behavior_tracker, a1_output_dir, video_callable = \
        load_agent_aux(a1_config, a1_agent_dir, args.a1_trial, seed, agent_rng, args)
    a2_env, a2_helper, a2_agent, _, a2_output_dir, _ = \
        load_agent_aux(a2_config, a2_agent_dir, args.a2_trial, seed, agent_rng, args, no_output=True)

    """Run"""
    disagreement_indexes, importance_scores, a2_traces, all_a1_frames = {}, {}, {}, []
    dis_i, frame_window = 0, int(args.horizon / 2)  # dis_i = disagreement index of frame
    for e in range(a1_config.num_episodes):
        if args.verbose:
            print(f'Episode: {e}')
        # set monitor
        a1_env.env.monitor = a1_env

        # reset environment
        a1_old_obs = a1_env.reset()
        _ = a2_env.reset()
        # this is a unique state that represents the elements surrounding the agent
        a1_old_s = a1_helper.get_state_from_observation(a1_old_obs, 0, False)
        t = 0
        a1_done = False
        while not a1_done:
            # select action
            a1_a = a1_agent.act(a1_old_s)
            a2_a = a2_agent.act(a1_old_s)
            # check for disagreement
            if a1_a != a2_a:  # and dis_i < args.k_highlights:
                print(f'Disagreement at frame {t}')
                disagreement_indexes[dis_i] = t
                preceding_actions = a1_behavior_tracker.s_a[0]
                a2_traces[dis_i] = disagreement_frames(a2_env, a2_agent, a2_helper, frame_window, t, a1_old_s,
                                                       a1_old_obs, all_a1_frames, e)
                # get score of diverged sequence to compare later
                importance_scores[dis_i] = divergence_score(a1_agent, a2_agent, a1_old_s, args.importance)
                # close diverged env and unregister it
                a2_env.close()
                del gym.envs.registration.registry.env_specs[a2_env.spec.id]
                a2_env, a2_helper, a2_agent, a2_behavior_tracker = \
                    reload_agent(a2_config, a2_agent_dir, args.a2_trial, seed, agent_rng, t, preceding_actions, args)
                dis_i += 1
            # observe transition
            a1_obs, a1_r, a1_done, _ = a1_env.step(a1_a)
            a2_obs, a2_r, a2_done, _ = a2_env.step(a1_a)
            a1_s = a1_helper.get_state_from_observation(a1_obs, a1_r, a1_done)
            #TODO save death states for frames in the videos. death state == 1295
            a1_r = a1_helper.get_reward(a1_old_s, a1_a, a1_r, a1_s, a1_done)

            # update agent and stats
            a1_agent.update(a1_old_s, a1_a, a1_r, a1_s)
            a2_agent.update(a1_old_s, a1_a, a1_r, a1_s)
            a1_helper.update_stats(e, t, a1_old_obs, a1_obs, a1_old_s, a1_a, a1_r, a1_s)
            a2_helper.update_stats(e, t, a1_old_obs, a1_obs, a1_old_s, a1_a, a1_r, a1_s)
            a1_behavior_tracker.add_sample(a1_old_s, a1_a)
            a1_old_s = a1_s
            a1_old_obs = a1_obs
            # save frames or video
            all_a1_frames.append(a1_env.video_recorder.last_frame)
            t += 1

    """top k disagreements"""
    top_k_indexes = sorted(list(importance_scores.items()), key=lambda x: x[1], reverse=True)[:args.k_highlights]
    highlights = sorted([(x[0], disagreement_indexes[x[0]]) for x in top_k_indexes])

    """get disagreement highlights"""
    a1_highlights, a2_highlights = get_disagreement_frames(all_a1_frames, highlights, a2_traces, frame_window)

    # writes results to files
    a1_agent.save(a1_output_dir)
    # a1_helper.save_stats(join(a1_output_dir, 'results'), args.clear_results)
    print('\nResults of trial {} written to:\n\t\'{}\''.format(args.a1_trial, a1_output_dir))
    a1_env.close()
    a2_env.close()
    comparison_frames = merge_frames(a1_highlights, a2_highlights, args.horizon)
    create_video(a1_output_dir, comparison_frames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Agent runner')
    parser.add_argument('-t1', '--a1_trial', help='agent 1 trial number', type=int, default=0)
    parser.add_argument('-t2', '--a2_trial', help='agent 2 trial number', type=int, default=1)
    parser.add_argument('-r1', '--a1_results', help='directory from which to load agent 1 results')
    parser.add_argument('-r2', '--a2_results', help='directory from which to load agent 2 results')
    parser.add_argument('-imp', '--importance', help='method for calculating divergence between agents', type=str,
                        default='bety')
    parser.add_argument('-n', '--num_episodes', help='number of episodes to run', type=int, default=1)
    parser.add_argument('-o', '--output', help='directory in which to store results')
    parser.add_argument('-c', '--config_file_path', help='path to config file')
    parser.add_argument('-rv', '--record', help='record videos according to linear schedule', action='store_true')
    parser.add_argument('-v', '--verbose', help='print information to the console', action='store_true')
    parser.add_argument('-hzn', '--horizon', help='number of frames to show per highlight', type=int,
                        default=20)
    args = parser.parse_args()

    """experiment parameters"""
    args.a1_trial = 0
    args.a2_trial = 1
    args.num_episodes = 1  # max 2000 (defined in configuration.py)
    args.fps = 20
    args.horizon = 20
    args.verbose = True
    args.record = True
    args.show_score_bar = True
    args.clear_results = True
    args.k_highlights = 5
    args.importance = "bety" # "sb" "bety"

    online_comparison(args)
