import argparse
import logging
import numpy as np
from datetime import datetime
from os.path import join, basename
from agent_score import agent_score
from disagreement import save_disagreements, get_top_k_disagreements, disagreement, \
    DisagreementTrace, State, non_similar_trajectories
from merge_and_fade import merge_and_fade
from utils import load_agent_config, load_agent_aux, mark_agent


def get_logging(args):
    name = '_'.join([basename(args.a1_config), basename(args.a2_config)])
    file_name = '_'.join([name, datetime.now().strftime("%d-%m %H:%M").replace(' ', '_')])
    log_name = join('logs', file_name)
    args.output = join('results', file_name)
    logging.basicConfig(filename=log_name + '.log', filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s',
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

    """agent overall score"""
    a1_overall, a1_Rmax = agent_score(args.a1_config)
    a2_overall, a2_Rmax = agent_score(args.a2_config)

    if a1_overall < 0 and a2_overall > 0:
        a2_overall += abs(a1_overall)
        a1_overall = 1
    if a2_overall < 0 and a1_overall > 0:
        a1_overall += abs(a2_overall)
        a2_overall = 1

    agent_ratio = a1_overall / a2_overall
    # TODO use this to weigh agent value function of state

    """get agents"""
    a1_config, a1_agent_dir = load_agent_config(args.a1_config, args.a1_trial)
    a2_config, a2_agent_dir = load_agent_config(args.a2_config, args.a2_trial)
    seed = a1_config.seed
    agent_rng = np.random.RandomState(seed)
    a1_env, a1_helper, a1_agent, a1_behavior_tracker, a1_output_dir, video_callable = \
        load_agent_aux(a1_config, 5, a1_agent_dir, args.a1_trial, seed, agent_rng, args)
    a2_env, a2_helper, a2_agent, _, _, _ = \
        load_agent_aux(a2_config, 5, a2_agent_dir, args.a2_trial, seed, agent_rng, args,
                       no_output=True)

    """Run"""
    traces, da_index = [], args.horizon // 2
    for e in range(args.num_episodes):
        logging.info(f'Running Episode number: {e}')
        if args.verbose: print(f'Running Episode number: {e}')
        t = 0
        trace = DisagreementTrace(args.horizon, a1_agent.q, a2_agent.q, agent_ratio)
        a1_done = False

        a1_old_obs = a1_env.reset()
        _ = a2_env.reset()
        a1_old_s = a1_helper.get_state_from_observation(a1_old_obs, 0, False)

        # while not a1_done:
        for t in range(100):  # for testing
            # select action
            a1_a = a1_agent.act(a1_old_s)
            a2_a = a2_agent.act(a1_old_s)
            """check for disagreement"""
            if a1_a != a2_a and t != 0:  # dont allow da on first state
                if args.verbose: print(f'\tDisagreement at frame {t}')
                logging.info(f'\tDisagreement at frame {t}')
                preceding_actions = a1_behavior_tracker.s_a[e]
                a2_env, a2_helper, a2_agent, _ = \
                    disagreement(e, trace, a2_env, a2_agent, a2_helper, t, a1_old_s,
                                 a1_old_obs, a2_config, a2_agent_dir, agent_rng,
                                 preceding_actions, args)
            """observe transition"""
            a1_obs, a1_r, a1_done, _ = a1_env.step(a1_a)
            a2_obs, a2_r, a2_done, _ = a2_env.step(a1_a)
            a1_s = a1_helper.get_state_from_observation(a1_obs, a1_r, a1_done)
            a1_r = a1_helper.get_reward(a1_old_s, a1_a, a1_r, a1_s, a1_done)
            """update agent and stats"""
            a1_agent.update(a1_old_s, a1_a, a1_r, a1_s)
            a2_agent.update(a1_old_s, a1_a, a1_r, a1_s)
            a1_helper.update_stats(e, t, a1_old_obs, a1_obs, a1_old_s, a1_a, a1_r, a1_s)
            # a2_helper.update_stats(e, t, a1_old_obs, a1_obs, a1_old_s, a1_a, a1_r, a1_s)
            a1_behavior_tracker.add_sample(a1_old_s, a1_a)
            frame = a1_env.render()

            # save state
            a1_position = [int(x) for x in a1_env.env.game_state.game.frog.position]
            trace.a1_states.append(
                State(t, a1_old_obs, a1_old_s, a1_agent.q[a1_old_s], frame, a1_position))
            a1_old_s = a1_s
            a1_old_obs = a1_obs
            t += 1

        """end of episode"""
        trace.get_trajectories(args.importance_type, args.disagreement_importance,
                               args.trajectory_importance)
        trace.diverse_trajectories = non_similar_trajectories(trace.disagreement_trajectories,
                                                              trace.a1_states, args)
        traces.append(trace)
        a1_behavior_tracker.new_episode()

    """top k diverse disagreements"""
    disagreements = get_top_k_disagreements(traces, args)
    if args.verbose: print(f'Obtained {len(disagreements)} disagreements')
    logging.info(f'Obtained {len(disagreements)} disagreements')

    """mark disagreement frames"""
    a1_disagreement_frames, a2_disagreement_frames = [], []
    for d in disagreements:
        d_state = d.a1_states[da_index - 1]
        d_state.image = mark_agent(d_state.image, d_state.agent_position)
        a1_frames, a2_frames = d.get_frames()
        for i in range(args.horizon // 2, args.horizon):
            a1_position = d.a1_states[i].agent_position
            a2_position = d.a2_states[i].agent_position
            a1_frames[i] = mark_agent(a1_frames[i], position=a1_position, color=255, thickness=1)
            a2_frames[i] = mark_agent(a2_frames[i], position=a2_position, color=0, thickness=1)
        a1_disagreement_frames.append(a1_frames)
        a2_disagreement_frames.append(a2_frames)

    """save disagreement frames"""
    video_dir = save_disagreements(a1_disagreement_frames, a2_disagreement_frames, a1_output_dir,
                                   args.fps)
    if args.verbose: print(f'Disagreements saved')
    logging.info(f'Disagreements saved')

    """generate video"""
    fade_duration = 2
    fade_out_frame = args.horizon - fade_duration
    merge_and_fade(video_dir, args.n_disagreements, fade_out_frame)
    if args.verbose: print(f'DAs Video Generated')
    logging.info(f'DAs Video Generated')

    """ writes results to files"""
    a1_agent.save(a1_output_dir)
    if args.verbose: print(f'\nResults written to:\n\t\'{a1_output_dir}\'')
    logging.info(f'\nResults written to:\n\t\'{a1_output_dir}\'')
    # a1_env.close()
    # a2_env.close()
    # del gym.envs.registration.registry.env_specs[a1_env.spec.id]
    # del gym.envs.registration.registry.env_specs[a2_env.spec.id]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Agent runner')
    parser.add_argument('-t1', '--a1_trial', help='agent 1 trial number', type=int, default=0)
    parser.add_argument('-t2', '--a2_trial', help='agent 2 trial number', type=int, default=1)
    parser.add_argument('-r1', '--a1_config',
                        help='directory from which to load agent 1 configuration file')
    parser.add_argument('-r2', '--a2_config',
                        help='directory from which to load agent 2 configuration file')
    parser.add_argument('-imp', '--disagreement_importance',
                        help='method for calculating divergence between agents',
                        type=str,
                        default='bety')
    parser.add_argument('-n', '--num_episodes', help='number of episodes to run', type=int,
                        default=1)
    parser.add_argument('-o', '--output', help='directory in which to store results')
    parser.add_argument('-c', '--config_file_path', help='path to config file')
    parser.add_argument('-rv', '--record', help='record videos according to linear schedule',
                        default=True)
    parser.add_argument('-v', '--verbose', help='print information to the console', default=True)
    parser.add_argument('-hzn', '--horizon', help='number of frames to show per highlight',
                        type=int, default=20)
    parser.add_argument('-frz_dth', '--freeze_on_death',
                        help='number of frames to show per highlight', default=False)
    args = parser.parse_args()

    """experiment parameters"""
    args.a1_config = '/home/yotama/Local_Git/InterestingnessXRL/Agent_Comparisons/agents/HighVision'
    args.a2_config = '/home/yotama/Local_Git/InterestingnessXRL/Agent_Comparisons/agents/Expert'
    args.fps = 2
    args.horizon = 10
    args.show_score_bar = False
    args.n_disagreements = 5
    args.num_episodes = 10

    """get more/less trajectories"""
    args.similarity_context = 0  # window of indexes around trajectory for reducing similarity
    args.similarity_limit = int(args.horizon * 0.66)

    """importance measures"""
    args.disagreement_importance = "bety"  # "sb" "bety"
    args.state_importance = "worst"  # worst, second,
    args.trajectory_importance = "last_state"  # max_min, max_avg, avg, avg_delta, last_state
    args.importance_type = 'trajectory'  # state/trajectory

    """RUN"""
    online_comparison(args)
