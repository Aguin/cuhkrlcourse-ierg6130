"""
This file defines the API of your agent.

Usages:
1. Get your policy through: `student_compute_action_function(num_envs)`
2. Test your policy through: `from this_is_my_agent import test; test()`
3. Visualize builtin agents and your agent through:
    `python this_is_my_agent.py --left MY_AGENT --right ALPHA_PONG`

You need to finish student_compute_action_function and make sure passing
the tests.

-----
2019-2020 2nd term, IERG 6130: Reinforcement Learning and Beyond. Department
of Information Engineering, The Chinese University of Hong Kong. Course
Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.
"""
import argparse
import os.path as osp

import numpy as np
from competitive_rl import evaluate_two_policies, make_envs, \
    get_builtin_agent_names, get_compute_action_function

from load_agents import PolicyAPI


def student_compute_action_function(num_envs=1):
    """We will use this function to load your agent and then testing.

    Make sure this function can run bug-free, when the working directory is
    "ierg6130-assignment/assignment4/"

    You can rewrite this function completely if you have custom agents, but you
    need to make sure the codes is bug-free and add necessary description on
    report_SID.md

    Run this file directly to make sure everything is fine.
    """
    # [TODO] rewrite this function
    my_agent_log_dir = "data/YOUR-LOG-DIR/PPO"
    my_agent_suffix = "iter0"

    checkpoint_path = osp.join(my_agent_log_dir,
                               "checkpoint-{}.pkl".format(my_agent_suffix))
    if not osp.exists(checkpoint_path):
        print("Can't find anything at {}!".format(checkpoint_path))
    else:
        print("Found your checkpoint at {}!".format(checkpoint_path))

    return PolicyAPI(
        num_envs=num_envs,
        log_dir=my_agent_log_dir,
        suffix=my_agent_suffix
    )


def test():
    # Run this function to make sure your API is runnable
    policy = student_compute_action_function()

    for i in range(1000):
        act = policy(np.random.random([1, 42, 42]))
        assert act in [0, 1, 2], act

    for i in range(1000):
        act = policy(np.random.random([1, 1, 42, 42]))
        assert act in [0, 1, 2], act

    policy = student_compute_action_function(4)

    for i in range(1000):
        act = policy(np.random.random([4, 1, 42, 42]))
        assert act.shape[0] == 4, act.shape
        assert np.all(
            np.logical_or(np.logical_or(act == 0, act == 1), act == 2)), act

    print("Test passed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", default="RULE_BASED", type=str,
                        help="Left agent names, must in {}.".format(
                            get_builtin_agent_names()))
    parser.add_argument("--right", default="RULE_BASED", type=str,
                        help="Right agent names, must in {}.".format(
                            get_builtin_agent_names()))
    parser.add_argument("--num-episodes", "-N", default=10, type=int,
                        help="Number of episodes to run.")
    args = parser.parse_args()

    agent_names = get_builtin_agent_names() + ["MY_AGENT"]

    print("Agent names: ", agent_names)
    print("Your chosen agents: left - {}, right - {}".format(
        args.left, args.right))

    assert args.left in agent_names, agent_names
    assert args.right in agent_names, agent_names

    env = make_envs(
        "cPongDouble-v0", num_envs=1, asynchronous=False).envs[0]

    if args.left != "MY_AGENT":
        left = get_compute_action_function(args.left)
    else:
        left = student_compute_action_function()
    if args.right != "MY_AGENT":
        right = get_compute_action_function(args.right)
    else:
        right = student_compute_action_function()

    result = evaluate_two_policies(
        left, right, env=env, render=False,
        num_episode=args.num_episodes, render_interval=0.05  # 20 FPS rendering
    )
    print(result)

    env.close()
