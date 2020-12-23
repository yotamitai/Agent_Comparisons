import argparse

import gym
import numpy as np

from ARCHIVE.utils import load_agent_config

from interestingness_xrl.learning.behavior_tracker import BehaviorTracker
from interestingness_xrl.scenarios import create_helper, create_agent


def agent_score(path):
    config, results_dir = load_agent_config(path)
    helper = create_helper(config)
    env_id = '{}-{}-v0'.format(config.gym_env_id, 0)
    helper.register_gym_environment(env_id, False)
    env = gym.make(env_id)  # .env
    config.num_episodes = 1
    env.seed(config.seed)
    agent_rng = np.random.RandomState(config.seed)
    agent, exploration_strategy = create_agent(helper, 1, agent_rng)
    agent.load(path)
    behavior_tracker = BehaviorTracker(config.num_episodes)
    old_obs = env.reset()
    old_s = helper.get_state_from_observation(old_obs, 0, False)

    t = 0
    done = False
    while not done:
        a = agent.act(old_s)
        obs, r, done, _ = env.step(a)
        s = helper.get_state_from_observation(obs, r, done)
        r = helper.get_reward(old_s, a, r, s, done)
        agent.update(old_s, a, r, s)
        behavior_tracker.add_sample(old_s, a)
        helper.update_stats(0, t, old_obs, obs, old_s, a, r, s)
        old_s = s
        old_obs = obs
        t += 1

    overall_score = env.env.previous_score
    env.close()
    del gym.envs.registration.registry.env_specs[env.spec.id]
    return overall_score, agent.r_sa.max()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='agent path', type=str)
    args = parser.parse_args()
    agent_score(args.path)