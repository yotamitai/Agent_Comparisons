import argparse
import logging
from datetime import datetime
import gym
import numpy as np

from disagreement import get_disagreement_frames, disagreement_score, save_disagreements, \
    get_top_k_disagreements, disagreement_states
from get_trajectories import State
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
    logging.info(f'Comparing Agents: {name}')
    logging.info(f'Disagreement importance by: {args.importance_type}')
    if args.verbose:
        print(f'Comparing Agents: {name}')
        print(f'Disagreement importance by: {args.importance_type}')

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
    a1_states, a2_trajectories = [], []
    disagreement_indexes, importance_scores = [], []
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
    # for t in range(50):  # for testing
        # select action
        a1_a = a1_agent.act(a1_old_s)
        a2_a = a2_agent.act(a1_old_s)
        # check for disagreement
        if a1_a != a2_a and t != 0:  # dont allow da on first state
            if args.verbose: print(f'Disagreement at frame {t}')
            logging.info(f'Disagreement at frame {t}')
            preceding_actions = a1_behavior_tracker.s_a[0]
            a2_trajectories.append(disagreement_states(a2_env, a2_agent, a2_helper, args.importance_type,
                                                       frame_window, t, a1_old_s, a1_old_obs, a1_states,
                                                       args.freeze_on_death))
            if args.importance_type == 'state':
                importance_scores.append(disagreement_score(a1_agent.q[a1_old_s], a2_agent.q[a1_old_s],
                                                            args.state_importance, by_state=True))
                disagreement_indexes.append(t)
            a2_env.close()
            del gym.envs.registration.registry.env_specs[a2_env.spec.id]
            a2_env, a2_helper, a2_agent, a2_behavior_tracker = \
                reload_agent(a2_config, a2_agent_dir, args.a2_trial, seed, agent_rng, t, preceding_actions, args)

        # observe transition
        a1_obs, a1_r, a1_done, _ = a1_env.step(a1_a)
        a2_obs, a2_r, a2_done, _ = a2_env.step(a1_a)
        a1_s = a1_helper.get_state_from_observation(a1_obs, a1_r, a1_done)
        a1_r = a1_helper.get_reward(a1_old_s, a1_a, a1_r, a1_s, a1_done)

        # update agent and stats
        a1_agent.update(a1_old_s, a1_a, a1_r, a1_s)
        a2_agent.update(a1_old_s, a1_a, a1_r, a1_s)
        a1_helper.update_stats(0, t, a1_old_obs, a1_obs, a1_old_s, a1_a, a1_r, a1_s)
        a2_helper.update_stats(0, t, a1_old_obs, a1_obs, a1_old_s, a1_a, a1_r, a1_s)
        a1_behavior_tracker.add_sample(a1_old_s, a1_a)
        a1_env.video_recorder.capture_frame()
        frame = a1_env.video_recorder.last_frame

        # save frames or state
        if args.importance_type == 'state':
            a1_states.append(frame)
        else:
            a1_states.append(State(t, a1_old_obs, a1_old_s, a1_agent.q[a1_old_s], frame))
        a1_old_s = a1_s
        a1_old_obs = a1_obs
        t += 1

    """top k diverse disagreements"""
    disagreements, da_frames = get_top_k_disagreements(a2_trajectories, a1_states, disagreement_indexes,
                                                       importance_scores, args)
    if args.verbose: print(f'chosen disagreement frames: {da_frames}')
    logging.info(f'chosen disagreement frames: {da_frames}')

    """get disagreement frames"""
    if args.importance_type == 'trajectory':
        a1_disagreement_frames, a2_disagreement_frames = [], []
        for d in disagreements:
            a1_frames, a2_frames = d.get_frames()
            if len(a1_frames) != args.horizon:
                a1_frames = a1_frames + [a1_frames[-1] for _ in range(args.horizon - len(a1_frames))]
                a2_frames = a2_frames + [a2_frames[-1] for _ in range(args.horizon - len(a2_frames))]
            a1_disagreement_frames.append(a1_frames)
            a2_disagreement_frames.append(a2_frames)
    else:
        a1_disagreement_frames, a2_disagreement_frames = \
            get_disagreement_frames(a1_states, a1_behavior_tracker.s_s[0], da_frames, disagreement_indexes,
                                    a2_trajectories, frame_window, args.freeze_on_death)

    """save disagreement frames"""
    video_dir = save_disagreements(a1_disagreement_frames, a2_disagreement_frames, a1_output_dir, args.fps)
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
    parser.add_argument('-imp', '--disagreement_importance', help='method for calculating divergence between agents', type=str,
                        default='bety')
    parser.add_argument('-n', '--num_episodes', help='number of episodes to run', type=int, default=1)
    parser.add_argument('-o', '--output', help='directory in which to store results')
    parser.add_argument('-c', '--config_file_path', help='path to config file')
    parser.add_argument('-rv', '--record', help='record videos according to linear schedule', default=True)
    parser.add_argument('-v', '--verbose', help='print information to the console', default=True)
    parser.add_argument('-hzn', '--horizon', help='number of frames to show per highlight', type=int, default=20)
    parser.add_argument('-frz_dth', '--freeze_on_death', help='number of frames to show per highlight', default=False)
    args = parser.parse_args()

    """experiment parameters"""
    args.a1_config = '/home/yotama/Local_Git/InterestingnessXRL/Agent_Comparisons/agents/Expert'
    args.a2_config = '/home/yotama/Local_Git/InterestingnessXRL/Agent_Comparisons/agents/LimitedVision_Mid'
    args.fps = 2
    args.horizon = 10
    args.show_score_bar = False
    args.n_disagreements = 5
    args.similarity_limit = 4
    args.similarity_context = 5
    args.disagreement_importance = "bety"  # "sb" "bety"
    args.state_importance = "second"  # worst, second
    args.trajectory_importance = "max_min"  # max_min, max_avg, avg, avg_delta, last_state
    args.importance_type = 'trajectory'  # state/trajectory
    """RUN"""
    online_comparison(args)
