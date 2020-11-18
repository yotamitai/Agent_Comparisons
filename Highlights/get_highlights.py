import argparse
from datetime import datetime

import pandas as pd
from os.path import join

from Highlights.get_agent import get_agent
from Highlights.get_traces import get_traces
from Highlights.HL_utils import create_video, make_clean_dirs, pickle_save
from Highlights.highlights_state_selection import compute_states_importance, highlights
from Highlights.get_trajectories import states_to_trajectories, trajectories_by_importance, \
    get_trajectory_images
from Highlights.ffmpeg import merge_and_fade
from utils import FROGGER_CONFIG_DICT


def get_highlights(args):
    env, agent, agent_args = get_agent(args)
    traces, states = get_traces(env, agent, agent_args, args)

    """highlights algorithm"""
    data = {
        'state': list(states.keys()),
        'q_values': [x.observed_actions for x in states.values()]
    }
    q_values_df = pd.DataFrame(data)

    """importance by state"""
    q_values_df = compute_states_importance(q_values_df, compare_to=args.state_importance)
    highlights_df = q_values_df
    state_importance_dict = dict(zip(highlights_df["state"], highlights_df["importance"]))

    """get highlights"""
    if args.trajectory_importance == "single_state":
        """highlights importance by single state importance"""
        summary_states = highlights(highlights_df, traces, args.num_trajectories, args.trajectory_length,
                                    args.minimum_gap)
        all_trajectories = states_to_trajectories(summary_states, state_importance_dict)
        summary_trajectories = all_trajectories
    else:
        """highlights importance by trajectory"""
        all_trajectories, summary_trajectories = trajectories_by_importance(traces, state_importance_dict, args)
    if args.verbose: print(f"HIGHLIGHTS {15 * '-' + '>'} obtained")

    """Save data used for this run"""
    pickle_save(traces, join(args.output_dir, 'Traces.pkl'))
    pickle_save(states, join(args.output_dir, 'States.pkl'))
    pickle_save(all_trajectories, join(args.output_dir, 'Trajectories.pkl'))

    """Save Highlight videos"""
    frames_dir = join(args.output_dir, 'Highlight_Frames')
    height, width, layers = list(states.values())[0].image.shape
    img_size = (width, height)
    get_trajectory_images(summary_trajectories, states, frames_dir)
    create_video(frames_dir, join(args.output_dir, "Highlight_Videos"), args.num_trajectories, img_size)
    if args.verbose: print(f"HIGHLIGHTS {15 * '-' + '>'} Video Obtained")

    """Merge Highlights to a single video with fade in/ fade out effects"""
    # TODO is this s good frame to start fading out?
    merge_and_fade(args.output_dir, args.num_trajectories, fade_out_frame=10)

    """Save data used for this run"""
    pickle_save(traces, join(args.output, 'Traces.pkl'))
    pickle_save(states, join(args.output, 'States.pkl'))
    pickle_save(all_trajectories, join(args.output_dir, 'Trajectories.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Agent runner')
    parser.add_argument('-n', '--num_episodes', help='number of episodes to run', type=int, default=100)
    parser.add_argument('-o', '--output', help='directory in which to store results')
    parser.add_argument('-a', '--agent_dir', help='directory from which to load the agent')
    parser.add_argument('-c', '--config_file_path', help='path to config file')
    parser.add_argument('-rv', '--record', help='record videos according to linear schedule', action='store_true')
    parser.add_argument('-v', '--verbose', help='print information to the console', action='store_true')
    args = parser.parse_args()

    """agent parameters"""
    args.agent_dir = '/home/yotama/Local_Git/InterestingnessXRL/Agent_Comparisons/agents/Expert'
    args.agent_name = 'Expert'
    args.results_dir = '/home/yotama/Local_Git/InterestingnessXRL/Agent_Comparisons/Highlights/results'
    args.num_episodes = 3  # max 2000 (defined in configuration.py)
    args.fps = 2
    args.verbose = True
    args.record = 'all'
    args.show_score_bar = False
    args.clear_results = True
    args.default_frogger_config = FROGGER_CONFIG_DICT['DEFAULT']

    """Highlight parameters"""
    args.n_traces = 20
    args.trajectory_importance = "max_min"
    args.state_importance = "second"
    args.num_trajectories = 7
    args.trajectory_length = 2 * args.num_trajectories
    args.minimum_gap = 10
    args.allowed_similar_states = 5
    args.load_traces = True
    args.load_trajectories = True

    args.output_dir = join(args.results_dir, datetime.now().strftime("%H:%M:%S_%d-%m-%Y"))
    make_clean_dirs(args.output_dir)
    # RUN
    get_highlights(args)
