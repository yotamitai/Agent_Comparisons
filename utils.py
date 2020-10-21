import glob
import os
import pickle
from os.path import join
from skimage import img_as_ubyte

import imageio

from interestingness_xrl.learning.agents import QValueBasedAgent
from interestingness_xrl.scenarios import _get_base_dir, DEFAULT_CONFIG
from interestingness_xrl.scenarios.configurations import EnvironmentConfiguration
from interestingness_xrl.scenarios.frogger.configurations import FROGGER_CONFIG, FAST_FROGGER_CONFIG, \
    FROGGER_LIMITED_CONFIG, FROGGER_FEAR_WATER_CONFIG, FROGGER_HIGH_VISION_CONFIG
from configurations.configurations import EXPERT_CONFIG



from Agent_Comparisons.explorations import GreedyExploration

FROGGER_CONFIG_DICT = {
         'DEFAULT': FROGGER_CONFIG,
         'FAST': FAST_FROGGER_CONFIG,
         'LIMITED': FROGGER_LIMITED_CONFIG,
         'FEAR_WATER': FROGGER_FEAR_WATER_CONFIG,
         'HIGH_VISION': FROGGER_HIGH_VISION_CONFIG,
         'EXPERT': EXPERT_CONFIG
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


def load_agent_config(results_dir, trial=0):
    results_dir = results_dir if results_dir else get_agent_output_dir(DEFAULT_CONFIG, AgentType.Learning, trial)
    config_file = os.path.join(results_dir, 'config.json')
    if not os.path.exists(results_dir) or not os.path.exists(config_file):
        raise ValueError(f'Could not load configuration from: {config_file}.')
    configuration = EnvironmentConfiguration.load_json(config_file)
    # if testing, we want to force a seed different than training (diff. test environments)
    #     configuration.seed += 1
    return configuration, results_dir


# class Trace(object):
#     def __init__(self):
#         self.obs = []
#         self.actions = []
#         self.rewards = []
#         self.dones = []
#         self.infos = []
#         self.reward_sum = 0
#         self.game_score = None
#         self.length = 0
#         self.states = []
#
#
# class State(object):
#     def __init__(self, name, obs, action_vector, feature_vector, img):
#         self.observation = obs
#         self.image = img
#         self.observed_actions = action_vector
#         self.name = name
#         self.features = feature_vector



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
    agent = None
    exploration_strategy = None

    # compare: Q-agent (table loaded from learning) with fixed (greedy) SoftMax
    if agent_t == AgentType.Compare:
        exploration_strategy = GreedyExploration(config.min_temp, rng)
        agent = QValueBasedAgent(config.num_states, config.num_actions,
                                 action_names=config.get_action_names(), exploration_strategy=exploration_strategy)

    # assigns agent to helper for collecting stats
    helper.agent = agent

    return agent, exploration_strategy


def clean_dir(path, file_type=''):
    files = glob.glob(path + "/*" + file_type)
    for f in files:
        os.remove(f)


def pickle_load(filename):
    return pickle.load(open(filename, "rb"))


def pickle_save(obj, path):
    try:
        os.makedirs(os.path.dirname(path))
    except:
        clean_dir(path)
    with open(path, "wb") as file:
        pickle.dump(obj, file)

def create_video(output_dir, frames, name='video'):
    imgs = []
    kargs = {'macro_block_size': None, 'fps': 3}
    for f in frames:
        imgs.append(img_as_ubyte(f))
    imageio.mimwrite(output_dir + "/" + name +'.mp4', imgs, 'MP4', **kargs)


def save_image(path, name, img):
    imageio.imsave(path + '/' + name + '.png', img_as_ubyte(img))


def clean_dir(path, file_type=''):
    files = glob.glob(path + "/*" + file_type)
    for f in files:
        os.remove(f)


def make_or_clean_dir(path):
    try:
        os.makedirs(path)
    except:
        clean_dir(path)

def save_highlights(a1_hl, a2_hl, output_dir):
    highlight_frames_dir = os.path.join(output_dir, "highlight_frames")
    video_dir = os.path.join(output_dir, "videos")
    make_or_clean_dir(video_dir)
    make_or_clean_dir(highlight_frames_dir)

    for hl_i in range(len(a1_hl)):
        for img_i in range(len(a1_hl[hl_i])):
            save_image(highlight_frames_dir, "a1_HL{}_FRAME{}".format(str(hl_i),str(img_i)), a1_hl[hl_i][img_i])
            save_image(highlight_frames_dir, "a2_HL{}_FRAME{}".format(str(hl_i),str(img_i)), a2_hl[hl_i][img_i])

        create_video(video_dir, a1_hl[hl_i], "a1_HL"+str(hl_i))
        create_video(video_dir, a2_hl[hl_i], "a2_HL"+str(hl_i))