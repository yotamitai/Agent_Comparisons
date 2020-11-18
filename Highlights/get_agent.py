import numpy as np
from utils import load_agent_config, load_agent_aux


def get_agent(args):
    """Implement here for specific agent and environment loading scheme"""
    config, agent_dir = load_agent_config(args.agent_dir, 0)
    seed = config.seed
    agent_rng = np.random.RandomState(seed)
    env, helper, agent, behavior_tracker, _, video_callable = \
        load_agent_aux(config, 1, agent_dir, 0, seed, agent_rng, args)
    return env, agent, {'helper': helper, 'behavior_tracker':behavior_tracker, 'video_callable':video_callable}
