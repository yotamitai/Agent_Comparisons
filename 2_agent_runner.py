__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import gym
import numpy as np
import argparse
from os import makedirs
from os.path import join, exists
from gym.wrappers import Monitor
from interestingness_xrl.scenarios.configurations import EnvironmentConfiguration
from interestingness_xrl.learning import write_table_csv
from interestingness_xrl.learning.behavior_tracker import BehaviorTracker
from interestingness_xrl.scenarios import DEFAULT_CONFIG, create_helper, AgentType, get_agent_output_dir, create_agent

DEF_AGENT_TYPE = AgentType.Learning
DEF_TRIAL = 0

FPS = 20
SHOW_SCORE_BAR = True
CLEAR_RESULTS = True


def video_schedule(config, videos):
    # linear capture schedule
    return lambda e: videos and \
                     (e == config.num_episodes - 1 or e % int(config.num_episodes / config.num_recorded_videos) == 0)


def run_trial(args):
    # tries to get agent type
    agent_t = args.agent
    results_dir = ''
    if agent_t == AgentType.Testing:

        # tries to load config from provided results dir path
        results_dir_1 = args.results if args.results is not None else \
            get_agent_output_dir(DEFAULT_CONFIG, AgentType.Learning, trial_num=DEF_TRIAL)
        config_file_1 = join(results_dir_1, 'config.json')
        if not exists(results_dir_1) or not exists(config_file_1):
            raise ValueError('Could not load configuration from: {}.'.format(config_file_1))
        config_1 = EnvironmentConfiguration.load_json(config_file_1)

        # if testing, we want to force a seed different than training (diff. test environments)
        config_1.seed += 1

        # tries to load config from provided results dir path
        results_dir_2 = args.results if args.results is not None else \
            get_agent_output_dir(DEFAULT_CONFIG, AgentType.Learning, trial_num=999)
        config_file_2 = join(results_dir_2, 'config.json')
        if not exists(results_dir_2) or not exists(config_file_2):
            raise ValueError('Could not load configuration from: {}.'.format(config_file_2))
        config_2 = EnvironmentConfiguration.load_json(config_file_2)

        # if testing, we want to force a seed different than training (diff. test environments)
        config_2.seed += 1

    # else:
    #     # tries to load env config from provided file path
    #     config_file = args.config
    #     config = DEFAULT_CONFIG if config_file is None or not exists(config_file) \
    #         else EnvironmentConfiguration.load_json(config_file)

    # creates env helper
    helper_1 = create_helper(config_1)
    helper_2 = create_helper(config_2)

    # checks for provided output dir
    output_dir_1 = args.output if args.output is not None else get_agent_output_dir(config_1, agent_t, 0)
    if not exists(output_dir_1):
        makedirs(output_dir_1)
    output_dir_2 = args.output if args.output is not None else get_agent_output_dir(config_2, agent_t, 1)
    if not exists(output_dir_2):
        makedirs(output_dir_2)

    # saves / copies configs to file
    config_1.save_json(join(output_dir_1, 'config.json'))
    helper_1.save_state_features(join(output_dir_1, 'state_features.csv'))
    config_2.save_json(join(output_dir_2, 'config.json'))
    helper_2.save_state_features(join(output_dir_2, 'state_features.csv'))

    # register environment in Gym according to env config
    env_id_1 = '{}-{}-v0'.format(config_1.gym_env_id, 0)
    env_id_2 = '{}-{}-v0'.format(config_1.gym_env_id, 1)
    helper_1.register_gym_environment(env_id_1, False, FPS, SHOW_SCORE_BAR)
    helper_2.register_gym_environment(env_id_2, False, FPS, SHOW_SCORE_BAR)

    # create environment and monitor
    env_1 = gym.make(env_id_1)
    env_2 = gym.make(env_id_2)
    # todo
    config_1.num_episodes = 100
    config_2.num_episodes = 100
    video_callable_1 = video_schedule(config_1, args.record)
    video_callable_2 = video_schedule(config_2, args.record)
    env_1 = Monitor(env_1, directory=output_dir_1, force=True, video_callable=video_callable_1)
    env_2 = Monitor(env_2, directory=output_dir_2, force=True, video_callable=video_callable_2)

    # adds reference to monitor to allow for gym environments to update video frames
    if video_callable_1(0):
        env_1.env.monitor = env_1
    if video_callable_2(0):
        env_2.env.monitor = env_2

    # initialize seeds (one for the environment, another for the agent)
    env_1.seed(config_1.seed + 0)
    env_2.seed(config_2.seed + 1)
    agent_rng_1 = np.random.RandomState(config_1.seed + 0)
    agent_rng_2 = np.random.RandomState(config_2.seed + 1)

    # creates the agent
    agent_1, exploration_strategy_1 = create_agent(helper_1, agent_t, agent_rng_1)
    agent_2, exploration_strategy_2 = create_agent(helper_2, agent_t, agent_rng_2)

    # if testing, loads tables from file (some will be filled by the agent during the interaction)
    if agent_t == AgentType.Testing:
        agent_1.load(results_dir, )
        agent_2.load(results_dir, )

    # runs episodes
    behavior_tracker_1 = BehaviorTracker(config_1.num_episodes)
    behavior_tracker_2 = BehaviorTracker(config_2.num_episodes)
    recorded_episodes_1 = []
    recorded_episodes_2 = []
    for e in range(config_1.num_episodes):

        # checks whether to activate video monitoring
        env_1.env.monitor = env_1 if video_callable_1(e) else None

        # reset environment
        old_obs_1 = env_1.reset()
        old_s_1 = helper_1.get_state_from_observation(old_obs_1, 0, False)
        if e: env_2.close()  # this line terminates env_2 when env_1 is done.
        old_obs_2 = env_2.reset()
        old_s_2 = helper_2.get_state_from_observation(old_obs_2, 0, False)

        if args.verbose:
            helper_1.update_stats_episode(e)
            helper_2.update_stats_episode(e)
        exploration_strategy_1.update(e)
        exploration_strategy_2.update(e)

        t = 0
        done1 = False
        while not done1:
            # select action
            a1 = agent_1.act(old_s_1)
            a2 = agent_2.act(old_s_2)

            # observe transition
            obs1, r1, done1, _ = env_1.step(a1)
            s1 = helper_1.get_state_from_observation(obs1, r1, done1)
            r1 = helper_1.get_reward(old_s_1, a1, r1, s1, done1)
            # observe transition
            obs2, r2, done2, _ = env_2.step(a2)
            s2 = helper_2.get_state_from_observation(obs2, r2, done1) #TODO done1 or done2?
            r2 = helper_2.get_reward(old_s_2, a2, r2, s2, done1)

            # update agent and stats
            agent_1.update(old_s_1, a1, r1, s1)
            behavior_tracker_1.add_sample(old_s_1, a1)
            helper_1.update_stats(e, t, old_obs_1, obs1, old_s_1, a1, r1, s1)
            # update agent and stats
            agent_2.update(old_s_2, a2, r2, s2)
            behavior_tracker_2.add_sample(old_s_2, a2)
            helper_2.update_stats(e, t, old_obs_2, obs2, old_s_2, a2, r2, s2)

            old_s_1 = s1
            old_obs_1 = obs1
            t += 1

        # adds to recorded episodes list
        if video_callable_1(e):
            recorded_episodes_1.append(e)
            recorded_episodes_2.append(e)

        # signals new episode to tracker
        behavior_tracker_1.new_episode()
        behavior_tracker_2.new_episode()

    # writes results to files
    agent_1.save(output_dir_1)
    agent_2.save(output_dir_2)
    behavior_tracker_1.save(output_dir_1)
    behavior_tracker_2.save(output_dir_2)
    write_table_csv(recorded_episodes_1, join(output_dir_1, 'rec_episodes.csv'))
    write_table_csv(recorded_episodes_2, join(output_dir_2, 'rec_episodes.csv'))
    helper_1.save_stats(join(output_dir_1, 'results'), CLEAR_RESULTS)
    helper_2.save_stats(join(output_dir_2, 'results'), CLEAR_RESULTS)
    print('\nResults of trial {} written to:\n\t\'{}\''.format(0, output_dir_1))
    print('\nResults of trial {} written to:\n\t\'{}\''.format(1, output_dir_2))

    env_1.close()
    env_2.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Agent runner')
    parser.add_argument('-a', '--agent', help='agent type', type=int, default=1)
    parser.add_argument('-o', '--output', help='directory in which to store results')
    parser.add_argument('-r', '--results', help='directory from which to load results')
    parser.add_argument('-c', '--config', help='path to config file')
    parser.add_argument('-t', '--trial', help='trial number to run', type=int, default=DEF_TRIAL)
    parser.add_argument('-rv', '--record', help='record videos according to linear schedule', action='store_true')
    parser.add_argument('-v', '--verbose', help='print information to the console', action='store_true')
    args = parser.parse_args()

    # runs trial
    run_trial(args)
