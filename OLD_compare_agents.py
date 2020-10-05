import argparse
from os import makedirs
from os.path import exists, join

import gym
import numpy as np
from gym.wrappers import Monitor
from gym.utils.ezpickle import EzPickle

from agent_runner import load_agent_config, video_schedule
from interestingness_xrl.learning import write_table_csv
from interestingness_xrl.learning.behavior_tracker import BehaviorTracker
from interestingness_xrl.scenarios import create_helper

from utils import get_agent_output_dir, create_agent, pickle_save, pickle_load


def load_agent_aux(config, agent_dir, trial, seed, agent_rng, params):
    helper = create_helper(config)
    output_dir = params.output if params.output is not None else get_agent_output_dir(config, 5, trial)
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


def disagreement(env, agent, helper, destination, window, time_step, old_s, old_obs, previous_frames, episode):
    # save pre-diverged environment so it can be loaded.

    from Agent_Comparisons.frogger import FroggerNoSurfaces
    x = env.unwrapped.game_state.game
    new_x = FroggerNoSurfaces(x)



    pickle_save(env, join(destination,'current_env.pkl'))
    pickle_save(agent, join(destination,'current_agent.pkl'))
    pickle_save(helper, join(destination,'current_helper.pkl'))
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
        if done:  # adds same frame if done so that all vids are same length
            disagreement_frames.append(env.video_recorder.last_frame)
            continue
        # record every step of the second agent
        a = agent.act(old_s)
        obs, r, done, _ = env.step(a)
        s = helper.get_state_from_observation(obs, r, done)
        agent.update(old_obs, a, r, s)
        helper.update_stats(episode, time_step, old_obs, obs, old_s, a, r, s)
        old_s = s
        old_obs = obs
        # save video scenes
        disagreement_frames.append(env.video_recorder.last_frame)
    # load pre-diverged environment for agent 1 and 2 and follow rest of code
    env = pickle_load(destination + 'current_env.pkl')
    agent = pickle_load(destination + 'current_agent.pkl')
    helper = pickle_load(destination + 'current_helper.pkl')
    return env, agent, helper


def online_comparison(args):
    """Compare two agents running online, search for disagreements"""

    """get agents"""
    a1_config, a1_agent_dir = load_agent_config(args.a1_results, args.a1_trial)
    a2_config, a2_agent_dir = load_agent_config(args.a2_results, args.a2_trial)
    seed = a1_config.seed
    agent_rng = np.random.RandomState(seed)
    a1_env, a1_helper, a1_agent, a1_behavior_tracker, a1_output_dir, video_callable = \
        load_agent_aux(a1_config, a1_agent_dir, args.a1_trial, seed, agent_rng, args)
    a2_env, a2_helper, a2_agent, a2_behavior_tracker, a2_output_dir, _ = \
        load_agent_aux(a2_config, a2_agent_dir, args.a2_trial, seed, agent_rng, args)

    """Run"""
    disagreement_indexes, a1_highlights, a2_highlights, all_a1_frames = {}, {}, {}, []
    dis_i, frame_window = 0, int(args.horizon / 2)  # dis_i = disagreement index of frame
    for e in range(a1_config.num_episodes):
        if args.verbose:
            print(f'Episode: {e}')
        # set monitor
        a1_env.env.monitor = a1_env
        # a2_env.env.monitor = a2_env

        # reset environment
        a1_old_obs = a1_env.reset()
        _ = a2_env.reset()
        # this is a unique state that represents the elements surrounding the agent
        a1_old_s = a1_helper.get_state_from_observation(a1_old_obs, 0, False)
        t = 0
        a1_done = False
        while not a1_done:
            # select action
            # sample an action based on the softmax probabilities as determined by the q values of the available actions
            a1_a = a1_agent.act(a1_old_s)
            a2_a = a2_agent.act(a1_old_s)

            # check for disagreement
            if a1_a != a2_a:
                disagreement_indexes[dis_i] = t
                dis_i += 1
                disagreement(a2_env, a2_agent, a2_helper, a1_output_dir, frame_window, t, a1_old_s, a1_old_obs, all_a1_frames, e)

            # observe transition
            a1_obs, a1_r, a1_done, _ = a1_env.step(a1_a)
            a2_obs, a2_r, a2_done, _ = a2_env.step(a1_a)
            a1_s = a1_helper.get_state_from_observation(a1_obs, a1_r, a1_done)
            # a2_s = a2_helper.get_state_from_observation(a2_obs, a2_r, a2_done)
            # s is a unique state defined by the elements around the element. combinations of ELEM_LABELS
            a1_r = a1_helper.get_reward(a1_old_s, a1_a, a1_r, a1_s, a1_done)
            # a2_r = a2_helper.get_reward(a2_old_s, a2_a, a2_r, a2_s, a2_done)

            # update agent and stats
            a1_agent.update(a1_old_s, a1_a, a1_r, a1_s)
            a2_agent.update(a1_old_s, a1_a, a1_r, a1_s)
            a1_helper.update_stats(e, t, a1_old_obs, a1_obs, a1_old_s, a1_a, a1_r, a1_s)
            a2_helper.update_stats(e, t, a1_old_obs, a1_obs, a1_old_s, a1_a, a1_r, a1_s)
            a1_old_s = a1_s
            a1_old_obs = a1_obs
            # save frames or video
            all_a1_frames.append(a1_env.video_recorder.last_frame)
            t += 1

    """get agent 1 disagreement frames"""
    num_frames = len(all_a1_frames)
    for d_i, frame_i in disagreement_indexes.items():
        same_frames = []
        start = frame_i - frame_window
        if start < 0:
            same_frames = [all_a1_frames[0] for _ in range(abs(start))]
            start = 0
        end = frame_i + frame_window if frame_i + frame_window < num_frames else num_frames - 1
        a1_highlights[d_i] = same_frames + all_a1_frames[start:end]
        if len(a1_highlights[d_i]) < 2 * frame_window:
            a1_highlights[d_i] += [all_a1_frames[-1] for _ in range(2 * frame_window - len(a1_highlights[d_i]))]

    # writes results to files
    a1_agent.save(a1_output_dir)
    a1_helper.save_stats(join(a1_output_dir, 'results'), args.clear_results)
    print('\nResults of trial {} written to:\n\t\'{}\''.format(args.a1_trial, a1_output_dir))
    a1_env.close()
    a2_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Agent runner')
    parser.add_argument('-t1', '--a1_trial', help='agent 1 trial number', type=int, default=0)
    parser.add_argument('-t2', '--a2_trial', help='agent 2 trial number', type=int, default=1)
    parser.add_argument('-r1', '--a1_results', help='directory from which to load agent 1 results')
    parser.add_argument('-r2', '--a2_results', help='directory from which to load agent 2 results')
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

    online_comparison(args)
