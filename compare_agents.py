import argparse
import logging
from datetime import datetime
import gym
import numpy as np

from disagreement import get_disagreement_frames, disagreement_frames, disagreement_score, save_disagreements, \
    get_top_k_disagreements
from merge_and_fade import merge_and_fade
from utils import load_agent_config, load_agent_aux


def reload_agent(config, agent_dir, trial, seed, rng, t, actions, params):
    env, helper, agent, behavior_tracker, output_dir, _ = \
        load_agent_aux(config, 5, agent_dir, trial, seed, rng, params, no_output=True)
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


def get_logging(params):
    name = '_'.join([params.a1_config.split('/')[-1], params.a2_config.split('/')[-1]])
    log_name = 'logs/' + '_'.join([name, datetime.now().strftime("%d-%m %H:%M").replace(' ', '_')])
    logging.basicConfig(filename=log_name + '.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    return name


def online_comparison(args):
    """Compare two agents running online, search for disagreements"""
    """logging"""
    name = get_logging(args)
    if args.verbose: print(name)
    logging.info(name)

    """get agents"""
    a1_config, a1_agent_dir = load_agent_config(args.a1_config, args.a1_trial)
    a2_config, a2_agent_dir = load_agent_config(args.a2_config, args.a2_trial)
    seed = a1_config.seed
    agent_rng = np.random.RandomState(seed)
    a1_env, a1_helper, a1_agent, a1_behavior_tracker, a1_output_dir, video_callable = \
        load_agent_aux(a1_config, 5, a1_agent_dir, args.a1_trial, seed, agent_rng, args)
    a2_env, a2_helper, a2_agent, _, a2_output_dir, _ = \
        load_agent_aux(a2_config, 5, a2_agent_dir, args.a2_trial, seed, agent_rng, args, no_output=True)

    """Run"""
    disagreement_indexes, importance_scores, a2_traces, all_a1_frames = {}, {}, {}, []
    dis_i, frame_window = 0, int(args.horizon / 2)  # dis_i = disagreement index of frame
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
    # for i in range(20):  # for testing
        # select action
        a1_a = a1_agent.act(a1_old_s)
        a2_a = a2_agent.act(a1_old_s)
        # check for disagreement
        if a1_a != a2_a:  # and dis_i < args.n_highlights:
            if args.verbose: print(f'Disagreement at frame {t}')
            logging.info(f'Disagreement at frame {t}')
            disagreement_indexes[dis_i] = t
            preceding_actions = a1_behavior_tracker.s_a[0]
            a2_traces[dis_i] = disagreement_frames(a2_env, a2_agent, a2_helper, frame_window, t, a1_old_s,
                                                   a1_old_obs, all_a1_frames, args.freeze_on_death)
            # get score of diverged sequence to compare later
            importance_scores[dis_i] = disagreement_score(a1_agent, a2_agent, a1_old_s, args.importance)
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
        # TODO save death states for frames in the videos. death state == 1295
        a1_r = a1_helper.get_reward(a1_old_s, a1_a, a1_r, a1_s, a1_done)

        # update agent and stats
        a1_agent.update(a1_old_s, a1_a, a1_r, a1_s)
        a2_agent.update(a1_old_s, a1_a, a1_r, a1_s)
        a1_helper.update_stats(0, t, a1_old_obs, a1_obs, a1_old_s, a1_a, a1_r, a1_s)
        a2_helper.update_stats(0, t, a1_old_obs, a1_obs, a1_old_s, a1_a, a1_r, a1_s)
        a1_behavior_tracker.add_sample(a1_old_s, a1_a)
        a1_old_s = a1_s
        a1_old_obs = a1_obs
        # save frames or video
        all_a1_frames.append(a1_env.video_recorder.last_frame)
        t += 1

    """top k diverse disagreements"""
    disagreements = get_top_k_disagreements(importance_scores, disagreement_indexes, a1_behavior_tracker, args)
    if args.verbose: print(f'chosen disagreement frames: {[x[1] for x in disagreements]}')
    logging.info(f'chosen disagreement frames: {[x[1] for x in disagreements]}')

    """get disagreement frames"""
    a1_disagreements, a2_disagreements = get_disagreement_frames(all_a1_frames, a1_behavior_tracker.s_s[0],
                                                                 disagreements, a2_traces, frame_window,
                                                                 args.freeze_on_death)

    """save disagreements"""
    video_dir = save_disagreements(a1_disagreements, a2_disagreements, a1_output_dir, args.fps)
    if args.verbose: print(f'Disagreements saved')
    logging.info(f'Disagreements saved')

    """generate video"""
    fade_duration = 2
    fade_out_frame = args.horizon - fade_duration
    merge_and_fade(video_dir, args.n_disagreements, fade_out_frame, fade_duration=fade_duration)
    if args.verbose: print(f'DAs Video Generated')
    logging.info(f'DAs Video Generated')

    """ writes results to files"""
    a1_agent.save(a1_output_dir)
    if args.verbose: print('\nResults of trial {} written to:\n\t\'{}\''.format(args.a1_trial, a1_output_dir))
    logging.info('\nResults of trial {} written to:\n\t\'{}\''.format(args.a1_trial, a1_output_dir))
    a1_env.close()
    a2_env.close()
    del gym.envs.registration.registry.env_specs[a1_env.spec.id]
    del gym.envs.registration.registry.env_specs[a2_env.spec.id]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Agent runner')
    parser.add_argument('-t1', '--a1_trial', help='agent 1 trial number', type=int, default=0)
    parser.add_argument('-t2', '--a2_trial', help='agent 2 trial number', type=int, default=1)
    parser.add_argument('-r1', '--a1_config', help='directory from which to load agent 1 configuration file')
    parser.add_argument('-r2', '--a2_config', help='directory from which to load agent 2 configuration file')
    parser.add_argument('-imp', '--importance', help='method for calculating divergence between agents', type=str,
                        default='bety')
    parser.add_argument('-n', '--num_episodes', help='number of episodes to run', type=int, default=1)
    parser.add_argument('-o', '--output', help='directory in which to store results')
    parser.add_argument('-c', '--config_file_path', help='path to config file')
    parser.add_argument('-rv', '--record', help='record videos according to linear schedule', default=True)
    parser.add_argument('-v', '--verbose', help='print information to the console', default=True)
    parser.add_argument('-hzn', '--horizon', help='number of frames to show per highlight', type=int, default=20)
    args = parser.parse_args()

    """experiment parameters"""
    args.a1_config = '/home/yotama/Local_Git/InterestingnessXRL/Agent_Comparisons/agents/Expert'
    args.a2_config = '/home/yotama/Local_Git/InterestingnessXRL/Agent_Comparisons/agents/FearWater'
    args.fps = 2
    args.horizon = 10
    args.show_score_bar = False
    args.n_disagreements = 5
    args.importance = "bety"  # "sb" "bety"
    args.freeze_on_death = False  # when an agent dies, keep getting frames or freeze

    """RUN"""
    online_comparison(args)
