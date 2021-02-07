import glob
import os
import shutil
import pickle
from os.path import join, dirname, exists

import cv2
import gym
from gym.wrappers import Monitor
from skimage import img_as_ubyte

import imageio

from interestingness_xrl.learning.agents import QValueBasedAgent
from interestingness_xrl.learning.behavior_tracker import BehaviorTracker
from interestingness_xrl.scenarios import _get_base_dir, DEFAULT_CONFIG, create_helper
from interestingness_xrl.scenarios.configurations import EnvironmentConfiguration
from interestingness_xrl.scenarios.frogger.configurations import FROGGER_CONFIG, \
    FAST_FROGGER_CONFIG, \
    FROGGER_LIMITED_CONFIG, FROGGER_FEAR_WATER_CONFIG, FROGGER_HIGH_VISION_CONFIG
from configurations.configurations import EXPERT_CONFIG

from interestingness_xrl.scenarios import create_agent as original_create_agent

from Agent_Comparisons.explorations import GreedyExploration

FROGGER_CONFIG_DICT = {
    'DEFAULT': FROGGER_CONFIG,
    'FAST': FAST_FROGGER_CONFIG,
    'LIMITED': FROGGER_LIMITED_CONFIG,
    'FEAR_WATER': FROGGER_FEAR_WATER_CONFIG,
    'HIGH_VISION': FROGGER_HIGH_VISION_CONFIG,
    'EXPERT': EXPERT_CONFIG,
}


class AgentType(object):
    """
    Contains definitions for all types of agent that can be run in the simulations.
    """
    Learning = 0
    Testing = 1
    Random = 2
    Reactive = 3
    Manual = 4
    Compare = 5

    @staticmethod
    def get_name(agent_t):
        if agent_t == AgentType.Learning:
            return 'learn'
        elif agent_t == AgentType.Testing:
            return 'test'
        elif agent_t == AgentType.Random:
            return 'random'
        elif agent_t == AgentType.Reactive:
            return 'reactive'
        elif agent_t == AgentType.Manual:
            return 'manual'
        elif agent_t == AgentType.Compare:
            return 'compare'
        else:
            return 'Unknown'


def load_agent_config(agent_dir, trial=0):
    agent_dir = agent_dir if agent_dir else get_agent_output_dir(DEFAULT_CONFIG,
                                                                 AgentType.Learning, trial)
    config_file = os.path.join(agent_dir, 'config.json')
    if not os.path.exists(agent_dir) or not os.path.exists(config_file):
        raise ValueError(f'Could not load configuration from: {config_file}.')
    configuration = EnvironmentConfiguration.load_json(config_file)
    # if testing, we want to force a seed different than training (diff. test environments)
    #     configuration.seed += 1
    return configuration, agent_dir


def get_agent_output_dir(config, agent_t, trial_num=0):
    return join(_get_base_dir(config), AgentType.get_name(agent_t), str(trial_num))


def create_agent(helper, agent_t, rng):
    """
    Creates an agent and exploration strategy according to the given parameters.
    :param ScenarioHelper helper: the helper containing all necessary methods to run a simulation scenario.
    :param int agent_t: the type of agent to be created.
    :param np.random.RandomState rng: the random number generator to be used by the action selection strategy.
    :rtype: tuple
    :return: a tuple (agent, exploration_strat) containing the created agent and respective exploration strategy.
    """
    config = helper.config
    # compare: Q-agent (table loaded from learning) with fixed (greedy) SoftMax
    if agent_t == AgentType.Compare:
        exploration_strategy = GreedyExploration(config.min_temp, rng)
        agent = QValueBasedAgent(config.num_states, config.num_actions,
                                 action_names=config.get_action_names(),
                                 exploration_strategy=exploration_strategy)
        # assigns agent to helper for collecting stats
        helper.agent = agent
    else:
        agent, exploration_strategy = original_create_agent(helper, agent_t, rng)

    return agent, exploration_strategy


def pickle_load(filename):
    return pickle.load(open(filename, "rb"))


def pickle_save(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def create_video(frame_dir, video_dir, agent_hl, size, length, fps):
    img_array = []
    for i in range(length):
        img = cv2.imread(os.path.join(frame_dir, agent_hl + f'_Frame{i}.png'))
        img_array.append(img)
    out = cv2.VideoWriter(os.path.join(video_dir, agent_hl) + '.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def save_image(path, name, img):
    imageio.imsave(path + '/' + name + '.png', img_as_ubyte(img))


def clean_dir(path, file_type='', hard=False):
    if not hard:
        files = glob.glob(path + "/*" + file_type)
        for f in files:
            os.remove(f)
    else:
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def make_clean_dirs(path, no_clean=False, file_type='', hard=False):
    try:
        os.makedirs(path)
    except:  # if exists
        if not no_clean: clean_dir(path, file_type, hard)


def load_agent_aux(config, agent_type, agent_dir, trial, seed, agent_rng, args, no_output=False):
    helper = create_helper(config)
    if not no_output:
        output_dir = args.output if args.output is not None else get_agent_output_dir(config,
                                                                                      agent_type,
                                                                                      trial)
    else:
        output_dir = join(dirname(dirname(agent_dir)), 'compare/temp')
    make_clean_dirs(output_dir, hard=True)
    config.save_json(join(output_dir, 'config.json'))
    helper.save_state_features(join(output_dir, 'state_features.csv'))
    env_id = '{}-{}-v0'.format(config.gym_env_id, trial)
    helper.register_gym_environment(env_id, False, args.fps, args.show_score_bar)
    env = gym.make(env_id, level=1)  # .env
    config.num_episodes = args.num_episodes
    video_callable = video_schedule(config, True)
    env.seed(seed)
    agent, exploration_strategy = create_agent(helper, agent_type, agent_rng)
    agent.load(agent_dir)
    behavior_tracker = BehaviorTracker(config.num_episodes)
    return env, helper, agent, behavior_tracker, output_dir, video_callable


def reload_agent(config, agent_dir, trial, seed, rng, t, actions, e, params):
    env, helper, agent, behavior_tracker, _, _ = \
        load_agent_aux(config, 5, agent_dir, trial, seed, rng, params, no_output=True)
    # each episode resets the environment.
    # each such env is different in randomly generated parts (logs, cars...)
    # so -- to load the same game env there is need to do the same number of resets
    [env.reset() for _ in range(e)]
    # [env.reset() for _ in range(5+e)]
    old_obs = env.reset()
    old_s = helper.get_state_from_observation(old_obs, 0, False)
    i = 0
    for a in actions:
        obs, r, done, _ = env.step(a)
        s = helper.get_state_from_observation(obs, r, done)
        r = helper.get_reward(old_s, a, r, s, done)
        agent.update(old_s, a, r, s)
        behavior_tracker.add_sample(old_s, a)
        helper.update_stats(1, t, old_obs, obs, old_s, a, r, s)
        old_s = s
        old_obs = obs
        i += 1
    return env, helper, agent, behavior_tracker


def video_schedule(config, videos):
    # linear capture schedule
    return (lambda e: True) if videos else \
        (lambda e: videos and (e == config.num_episodes - 1 or
                               e % int(config.num_episodes / config.num_recorded_videos) == 0))


def mark_agent(img, position=None, color=255, thickness=2):
    img2 = img.copy()
    top_left = (position[0], position[1])
    bottom_right = (position[0] + 30, position[1] + 30)
    cv2.rectangle(img2, top_left, bottom_right, color, thickness)
    return img2
