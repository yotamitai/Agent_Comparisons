import argparse
from itertools import permutations
from os.path import abspath

from compare_agents import online_comparison

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Agent Comparisons')
    parser.add_argument('-r1', '--a1_dir', help='agent 1 configuration file directory')
    parser.add_argument('-r2', '--a2_dir', help='agent 2 configuration file directory')
    parser.add_argument('-n', '--num_episodes', help='number of episodes to run', type=int,
                        default=3)
    parser.add_argument('-fps', '--fps', help='summary video fps', type=int, default=1)
    parser.add_argument('-l', '--horizon', help='number of frames to show per highlight',
                        type=int, default=10)
    parser.add_argument('-sb', '--show_score_bar', help='score bar', type=bool, default=False)
    parser.add_argument('-rand', '--randomized', help='randomize order of summary trajectories',
                        type=bool, default=True)
    parser.add_argument('-k', '--n_disagreements', help='# of disagreements in the summary',
                        type=int, default=5)
    parser.add_argument('-overlaplim', '--similarity_limit', help='# overlaping',
                        type=int, default=3)
    parser.add_argument('-impMeth', '--importance_type',
                        help='importance by state or trajectory', default='trajectory')
    parser.add_argument('-impTraj', '--trajectory_importance',
                        help='method calculating trajectory importance', default='last_state_val')
    parser.add_argument('-impState', '--state_importance',
                        help='method calculating state importance', default='bety')
    parser.add_argument('-v', '--verbose', help='print information to the console', default=True)
    args = parser.parse_args()

    """experiment parameters"""
    args.fps = 1
    args.horizon = 10
    args.show_score_bar = False
    args.n_disagreements = 5
    args.num_episodes = 10
    args.randomized = True

    """get more/less trajectories"""
    args.similarity_limit = 3 #int(args.horizon * 0.66)

    """importance measures"""
    args.state_importance = "bety"  # "sb" "bety"
    args.trajectory_importance = "last_state_val"  # last_state_val, max_min, max_avg, avg, avg_delta, last_state
    args.importance_type = 'trajectory'  # state/trajectory
    """Experiments"""
    for a1, a2 in permutations(['Expert', 'LimitedVision', 'HighVision'], 2):
        args.name = '_'.join([a1, a2])
        args.a1_config = abspath('agents/'+a1)
        args.a2_config = abspath('agents/'+a2)

        """run"""
        online_comparison(args)

