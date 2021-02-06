import argparse
from itertools import permutations
from os.path import join

from compare_agents import online_comparison
from utils import make_clean_dirs

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
    parser.add_argument('-hzn', '--horizon', help='number of frames to show per highlight', type=int,
                        default=20)
    parser.add_argument('-frz_dth', '--freeze_on_death', help='number of frames to show per highlight', default=False)

    args = parser.parse_args()

    """experiment parameters"""
    args.fps = 1
    args.horizon = 10
    args.show_score_bar = False
    args.n_disagreements = 5
    args.num_episodes = 10
    args.randomized = True

    """get more/less trajectories"""
    args.similarity_context = 0  # window of indexes around trajectory for reducing similarity
    args.similarity_limit = 3 #int(args.horizon * 0.66)

    """importance measures"""
    args.disagreement_importance = "bety"  # "sb" "bety"
    args.trajectory_importance = "last_state_val"  # last_state_val, max_min, max_avg, avg, avg_delta, last_state
    args.importance_type = 'trajectory'  # state/trajectory
    """Experiments"""
    for a1, a2 in permutations(['Expert', 'LimitedVision', 'HighVision'], 2):
        args.name = '_'.join([a1, a2])
        args.a1_config = '/home/yotama/Local_Git/InterestingnessXRL/Agent_Comparisons/agents/' + a1
        args.a2_config = '/home/yotama/Local_Git/InterestingnessXRL/Agent_Comparisons/agents/' + a2

        """run"""
        online_comparison(args)

