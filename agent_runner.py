import os

import gym
import numpy as np
import argparse
import logging
from os import makedirs
from os.path import join, exists
from gym.wrappers import Monitor
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from Agent_Comparisons.utils import FROGGER_CONFIG_DICT, AgentType, load_agent_config
from interestingness_xrl.scenarios.configurations import EnvironmentConfiguration
from interestingness_xrl.learning import write_table_csv
from interestingness_xrl.learning.behavior_tracker import BehaviorTracker
from interestingness_xrl.scenarios import DEFAULT_CONFIG, create_helper, get_agent_output_dir, create_agent


def video_schedule(config, videos):
    # linear capture schedule
    return lambda e: videos and \
                     (e == config.num_episodes - 1 or e % int(config.num_episodes / config.num_recorded_videos) == 0)


# def load_agent_config(results_dir, trial=0):
#     results_dir = results_dir if results_dir else get_agent_output_dir(DEFAULT_CONFIG, AgentType.Learning, trial)
#     config_file = os.path.join(results_dir, 'config.json')
#     if not os.path.exists(results_dir) or not os.path.exists(config_file):
#         raise ValueError(f'Could not load configuration from: {config_file}.')
#     configuration = EnvironmentConfiguration.load_json(config_file)
#     # if testing, we want to force a seed different than training (diff. test environments)
#     #     configuration.seed += 1
#     return configuration, results_dir


def run_trial(args):
    # tries to get agent type
    agent_t = args.agent

    if agent_t == AgentType.Testing:
        # tries to load a pre-trained agent configuration file
        config, results_dir = load_agent_config(args.results, args.trial)
    else:
        # tries to load env config from provided file path
        config_file = args.config_file_path
        config = args.default_frogger_config if config_file is None or not exists(config_file) \
            else EnvironmentConfiguration.load_json(config_file)

    # creates env helper
    helper = create_helper(config)

    # checks for provided output dir
    output_dir = args.output if args.output is not None else get_agent_output_dir(config, agent_t, args.trial)
    if not exists(output_dir):
        makedirs(output_dir)

    # saves / copies configs to file
    config.save_json(join(output_dir, 'config.json'))
    helper.save_state_features(join(output_dir, 'state_features.csv'))

    # register environment in Gym according to env config
    env_id = '{}-{}-v0'.format(config.gym_env_id, args.trial)
    helper.register_gym_environment(env_id, False, args.fps, args.show_score_bar)

    # create environment and monitor
    env = gym.make(env_id)
    # todo
    config.num_episodes = args.num_episodes # TODO Yotam: why is this 100?
    video_callable = video_schedule(config, args.record)
    env = Monitor(env, directory=output_dir, force=True, video_callable=video_callable)

    # adds reference to monitor to allow for gym environments to update video frames
    # TODO Yotam: not sure what this does
    if video_callable(0):
        env.env.monitor = env

    # initialize seeds (one for the environment, another for the agent)
    env.seed(config.seed + args.trial)
    agent_rng = np.random.RandomState(config.seed + args.trial)

    # creates the agent
    agent, exploration_strategy = create_agent(helper, agent_t, agent_rng)

    # if testing, loads tables from file (some will be filled by the agent during the interaction)
    if agent_t == AgentType.Testing:
        agent.load(results_dir)

    # runs episodes
    behavior_tracker = BehaviorTracker(config.num_episodes)
    recorded_episodes = []
    for e in range(config.num_episodes):

        # checks whether to activate video monitoring
        env.env.monitor = env if video_callable(e) else None

        # reset environment
        old_obs = env.reset()
        old_s = helper.get_state_from_observation(old_obs, 0, False)

        if args.verbose:
            print(f'Episode: {e}')
            # helper.update_stats_episode(e)
        exploration_strategy.update(e)  # update for learning agent

        t = 0
        done = False
        while not done:
            # select action
            # sample an action based on the softmax probabilities as determined by the q values of the available actions
            a = agent.act(old_s)

            # observe transition
            obs, r, done, _ = env.step(a)
            s = helper.get_state_from_observation(obs, r, done)
            # s is a unique state defined by the elements around the element. combinations of ELEM_LABELS
            r = helper.get_reward(old_s, a, r, s, done)

            # update agent and stats
            agent.update(old_s, a, r, s)
            behavior_tracker.add_sample(old_s, a)
            helper.update_stats(e, t, old_obs, obs, old_s, a, r, s)

            old_s = s
            old_obs = obs
            t += 1

        # adds to recorded episodes list
        if video_callable(e):
            recorded_episodes.append(e)

        # signals new episode to tracker
        behavior_tracker.new_episode()

    # writes results to files
    agent.save(output_dir)
    behavior_tracker.save(output_dir)
    write_table_csv(recorded_episodes, join(output_dir, 'rec_episodes.csv'))
    helper.save_stats(join(output_dir, 'results'), args.clear_results)
    print('\nResults of trial {} written to:\n\t\'{}\''.format(args.trial, output_dir))

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Agent runner')
    parser.add_argument('-a', '--agent', help='agent type', type=int, default=0)
    parser.add_argument('-n', '--num_episodes', help='number of episodes to run', type=int, default=100)
    parser.add_argument('-o', '--output', help='directory in which to store results')
    parser.add_argument('-r', '--results', help='directory from which to load results')
    parser.add_argument('-c', '--config_file_path', help='path to config file')
    parser.add_argument('-t', '--trial', help='trial number to run', type=int, default=0)
    parser.add_argument('-rv', '--record', help='record videos according to linear schedule', action='store_true')
    parser.add_argument('-v', '--verbose', help='print information to the console', action='store_true')
    args = parser.parse_args()

    """experiment parameters"""
    args.agent = 0
    args.trial = 0
    args.num_episodes = 2000 # max 2000 (defined in configuration.py)
    args.fps = 2
    args.verbose = True
    args.record = True
    args.show_score_bar = True
    args.clear_results = True
    args.default_frogger_config = FROGGER_CONFIG_DICT['DEFAULT']

    # for agent_type in FROGGER_CONFIG_DICT:
    #     if agent_type in ['EXPERT', 'DEFAULT']:
    #         continue
    #     args.default_frogger_config = FROGGER_CONFIG_DICT[agent_type]
    #     args.num_episodes = args.default_frogger_config.num_episodes
    #     args.trial+=1
    run_trial(args)
