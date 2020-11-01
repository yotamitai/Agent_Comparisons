import argparse
import gym
import numpy as np
from disagreement import get_disagreement_frames, disagreement_frames, disagreement_score
from utils import load_agent_config, save_highlights, load_agent_aux


def reload_agent(config, agent_dir, trial, seed, rng, t, actions, params):
    env, helper, agent, behavior_tracker, output_dir, _ = \
        load_agent_aux(config, agent_dir, trial, seed, rng, params, no_output=True)
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


# def get_overlaying_video(n_highlights, n_frames, a1_highlights, a2_highlights, output_dir):
#     """overlay a1 and a2 HL over each other"""
#     video_dir = os.path.join(output_dir, "videos")
#     overlay_hls = {}
#     for hl_i in range(n_highlights):
#         # get similar frames - up until disagreement
#         overlay_hls[hl_i] = a1_highlights[hl_i][:n_frames//2]
#         # for disagreements - find difference in pixels
#         for f_i in range(n_frames//2,n_frames):
#             diff_mat = a1_highlights[hl_i][f_i]-a2_highlights[hl_i][f_i]
#             # [[(i,j) for i in range(a1_highlights[0][0].shape[1]) if diff_mat[j][i].sum()] for j in range(a1_highlights[0][0].shape[0]-20)]
#             new_frame = copy.copy(a1_highlights[hl_i][f_i])
#             for r in range(a1_highlights[0][0].shape[0]-20):
#                 for c in range(a1_highlights[0][0].shape[1]):
#                     if diff_mat[r][c].sum():
#                         if a1_highlights[hl_i][f_i][r][c] ==
#
#
#                     if not (a1_highlights[hl_i][f_i][r][c] == a2_highlights[hl_i][f_i][r][c]).all():
#                         # add the change of a2 to a1 but change the color a bit (* 1.1)
#                         #TODO how to change only one agent color?
#                         # get diff matrix (subtraction) and only look at non zero pixels
#                         # create box for areas of change and color those
#                         new_frame[r][c] = (a2_highlights[hl_i][f_i][r][c] * 1.1).astype(int)
#             overlay_hls[hl_i].append(new_frame)
#
#         create_video(video_dir, overlay_hls[hl_i], "overlayed_" + str(hl_i))

def online_comparison(args):
    """Compare two agents running online, search for disagreements"""
    """get agents"""
    a1_config, a1_agent_dir = load_agent_config(args.a1_config, args.a1_trial)
    a2_config, a2_agent_dir = load_agent_config(args.a2_config, args.a2_trial)
    seed = a1_config.seed
    agent_rng = np.random.RandomState(seed)
    a1_env, a1_helper, a1_agent, a1_behavior_tracker, a1_output_dir, video_callable = \
        load_agent_aux(a1_config, a1_agent_dir, args.a1_trial, seed, agent_rng, args)
    a2_env, a2_helper, a2_agent, _, a2_output_dir, _ = \
        load_agent_aux(a2_config, a2_agent_dir, args.a2_trial, seed, agent_rng, args, no_output=True)

    """Run"""
    disagreement_indexes, importance_scores, a2_traces, all_a1_frames = {}, {}, {}, []
    dis_i, frame_window = 0, int(args.horizon / 2)  # dis_i = disagreement index of frame
    for e in range(a1_config.num_episodes):
        if args.verbose:
            print(f'Episode: {e}')
        # set monitor
        a1_env.env.monitor = a1_env

        # reset environment
        a1_old_obs = a1_env.reset()
        _ = a2_env.reset()
        # this is a unique state that represents the elements surrounding the agent
        a1_old_s = a1_helper.get_state_from_observation(a1_old_obs, 0, False)
        t = 0
        a1_done = False
        # while not a1_done:
        for i in range(10):  # for testing
            # select action
            a1_a = a1_agent.act(a1_old_s)
            a2_a = a2_agent.act(a1_old_s)
            # check for disagreement
            if a1_a != a2_a:  # and dis_i < args.n_highlights:
                print(f'Disagreement at frame {t}')
                disagreement_indexes[dis_i] = t
                preceding_actions = a1_behavior_tracker.s_a[0]
                a2_traces[dis_i] = disagreement_frames(a2_env, a2_agent, a2_helper, frame_window, t, a1_old_s,
                                                       a1_old_obs, all_a1_frames, e, args.crop_out)
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
            a1_helper.update_stats(e, t, a1_old_obs, a1_obs, a1_old_s, a1_a, a1_r, a1_s)
            a2_helper.update_stats(e, t, a1_old_obs, a1_obs, a1_old_s, a1_a, a1_r, a1_s)
            a1_behavior_tracker.add_sample(a1_old_s, a1_a)
            a1_old_s = a1_s
            a1_old_obs = a1_obs
            # save frames or video
            all_a1_frames.append(a1_env.video_recorder.last_frame[:args.crop_out])
            t += 1

    """top k diverse disagreements"""
    importance_sorted_indexes = sorted(list(importance_scores.items()), key=lambda x: x[1], reverse=True)
    seen, top_k_diverse_state_indexes = [], []

    for idx, score in importance_sorted_indexes:
        disagreement_state = a1_behavior_tracker.s_s[0][disagreement_indexes[idx]]
        if disagreement_state not in seen:
            seen.append(disagreement_state)
            top_k_diverse_state_indexes.append(idx)
        if len(top_k_diverse_state_indexes) == args.n_highlights:
            break
    highlights = sorted([(x, disagreement_indexes[x]) for x in top_k_diverse_state_indexes])

    """get disagreement frames"""
    a1_highlights, a2_highlights = get_disagreement_frames(all_a1_frames, a1_behavior_tracker.s_s[0], highlights,
                                                           a2_traces, frame_window)

    """overlay video"""
    # get_overlaying_video(args.n_highlights, args.horizon, a1_highlights, a2_highlights, a1_output_dir)

    """save highlights"""
    save_highlights(a1_highlights, a2_highlights, a1_output_dir)


    # comparison_frames = merge_frames(a1_highlights, a2_highlights, args.horizon)
    # create_video(a1_output_dir, comparison_frames)

    """ writes results to files"""
    a1_agent.save(a1_output_dir)
    print('\nResults of trial {} written to:\n\t\'{}\''.format(args.a1_trial, a1_output_dir))
    a1_env.close()
    a2_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Agent runner')
    parser.add_argument('-t1', '--a1_trial', help='agent 1 trial number', type=int, default=0)
    parser.add_argument('-t2', '--a2_trial', help='agent 2 trial number', type=int, default=1)
    parser.add_argument('-r1', '--a1_config', help='directory from which to load agent 1 configuration file')
    parser.add_argument('-r2', '--a2_config', help='directory from which to load agent 2 configuration file')
    parser.add_argument('-imp', '--importance', help='method for calculating divergence between agents', type=str,
                        default='bety')
    parser.add_argument('-n', '--num_episodes', help='number of episodes to run', type=int, default=1)
    parser.add_argument('-crop', '--crop_out', help='number of rows to crop to', type=int, default=None)
    parser.add_argument('-o', '--output', help='directory in which to store results')
    parser.add_argument('-c', '--config_file_path', help='path to config file')
    parser.add_argument('-rv', '--record', help='record videos according to linear schedule', action='store_true')
    parser.add_argument('-v', '--verbose', help='print information to the console', action='store_true')
    parser.add_argument('-hzn', '--horizon', help='number of frames to show per highlight', type=int,
                        default=20)
    args = parser.parse_args()

    """experiment parameters"""
    args.a1_trial = 0
    args.a2_trial = 1
    args.a1_config = '/home/yotama/Local_Git/InterestingnessXRL/Agent_Comparisons/agents/Default'
    args.a2_config = '/home/yotama/Local_Git/InterestingnessXRL/Agent_Comparisons/agents/Fast'
    args.num_episodes = 1  # max 2000 (defined in configuration.py)
    args.fps = 20
    args.horizon = 20
    args.verbose = True
    args.record = True
    args.show_score_bar = True
    args.clear_results = True
    args.n_highlights = 5
    args.importance = "bety"  # "sb" "bety"
    args.crop_out = -40

    online_comparison(args)