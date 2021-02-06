import argparse

import gym
import numpy as np

from utils import load_agent_config

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

    scores =[]
    for k in range(5):
        curr_obs = env.reset()
        curr_s = helper.get_state_from_observation(curr_obs, 0, False)
        t = 0
        done = False
        while not done:
            a = agent.act(curr_s)
            obs, r, done, _ = env.step(a)
            s = helper.get_state_from_observation(obs, r, done)
            r = helper.get_reward(curr_s, a, r, s, done)
            agent.update(curr_s, a, r, s)
            behavior_tracker.add_sample(curr_s, a)
            helper.update_stats(k, t, curr_obs, obs, curr_s, a, r, s)
            curr_s = s
            curr_obs = obs
            t += 1
        scores.append(env.env.previous_score)

    env.close()
    del gym.envs.registration.registry.env_specs[env.spec.id]
    return sum(scores)/len(scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='agent path', type=str)
    args = parser.parse_args()
    agent_score(args.path)