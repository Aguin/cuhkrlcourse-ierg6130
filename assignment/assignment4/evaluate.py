"""
This file defines the evaluation function to see the performance when they
competes with each others.

Usages:
    1. Run this file directly
    2. Use launch function in this_is_what_we_will_do.py

Notes:
    1. You need to finish this_is_my_agent.py first!

Nothing you need to implement in this file.

-----
2019-2020 2nd term, IERG 6130: Reinforcement Learning and Beyond. Department
of Information Engineering, The Chinese University of Hong Kong. Course
Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.
"""

import argparse

import pandas as pd
import tabulate
from competitive_rl import PrintConsole, make_envs, get_builtin_agent_names, \
    get_compute_action_function
from competitive_rl.pong.evaluate import evaluate_two_policies_in_batch

from this_is_my_agent import student_compute_action_function


def launch(my_policy_name, my_policy, agents, envs, num_episodes):
    console = PrintConsole(num_episodes)
    results = []
    print("\nStart evaluating agent ", my_policy_name)
    for k2, a2 in agents.items():
        console.start()
        a1_result, a2_result = evaluate_two_policies_in_batch(
            my_policy, a2, envs, num_episodes)
        print("\n===== {} VS {} result =====".format(my_policy_name, k2))
        console.printResultInfo(my_policy_name, a1_result, print_time=True)
        console.printResultInfo(k2, a2_result, print_time=True)
        assert a1_result[0] == a2_result[2]
        assert a1_result[1] == a2_result[1]
        assert sum(a1_result[:3]) == sum(a2_result[:3])
        results.append(dict(
            agent0=my_policy_name,
            agent1=k2,
            agent0_win=a1_result[0],
            agent1_win=a2_result[0],
            draw=a1_result[1],
            agent0_reward=a1_result[3],
            agent1_reward=a2_result[3],
            num_matches=sum(a1_result[:3])
        ))
    return pd.DataFrame(results)


def build_matrix(result, single_line=False):
    """
    Build the wining rate matrix.

    matrix[a0, a1] represent the wining rate of agent a0 when competing
    with agent1.
    """
    assert isinstance(result, pd.DataFrame)

    agent_names = result.agent0.unique()
    if single_line:
        win_rate_matrix = pd.DataFrame(columns=agent_names)
        reward_matrix = pd.DataFrame(columns=agent_names)
    else:
        win_rate_matrix = pd.DataFrame(index=agent_names, columns=agent_names)
        reward_matrix = pd.DataFrame(index=agent_names, columns=agent_names)

    for _, record in result.iterrows():
        # agent0 win rate against agent1
        win_rate_matrix.loc[record.agent0, record.agent1] = \
            record.agent0_win / record.num_matches
        reward_matrix.loc[record.agent0, record.agent1] = \
            record.agent0_reward / record.num_matches

        if single_line:
            continue

        # agent1 win rate against agent0
        win_rate_matrix.loc[record.agent1, record.agent0] = \
            record.agent1_win / record.num_matches
        reward_matrix.loc[record.agent1, record.agent0] = \
            record.agent1_reward / record.num_matches

    return win_rate_matrix, reward_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-envs",
        default=10,
        type=int,
        help="The number of parallel environments. Default: 10"
    )
    parser.add_argument(
        "--num-episodes",
        "-N",
        default=100,
        type=int,
        help="The number of episodes to run. Default: 100"
    )
    args = parser.parse_args()

    num_episodes = args.num_episodes
    num_envs = args.num_envs

    agents = {
        l: get_compute_action_function(l, num_envs)
        for l in get_builtin_agent_names()
    }
    agents["MY_AGENT"] = student_compute_action_function(num_envs)

    print("All agents ready: ", agents.keys())

    envs = make_envs(
        "cPongDouble-v0", num_envs=num_envs, asynchronous=True)
    print("Environment ready")

    result = launch("MY_AGENT", student_compute_action_function(num_envs),
                    agents, envs, num_episodes)

    winning_rate_matrix, reward_matrix = build_matrix(result, single_line=True)
    print("\n===== Winning Rate Matrix (row vs column) =====")
    print(winning_rate_matrix)
    print("\n===== Reward Matrix (row vs column) =====")
    print(reward_matrix)

    with open("data/evaluate_result.md", "w") as f:
        f.write("winning rate matrix:\n\n")
        f.write(tabulate.tabulate(winning_rate_matrix,
                                  winning_rate_matrix.keys(),
                                  tablefmt="pipe"))
        f.write("\n\n\n\n\n")
        f.write("reward matrix\n\n")
        f.write(tabulate.tabulate(reward_matrix, reward_matrix.keys(),
                                  tablefmt="pipe"))

    envs.close()
    result.to_csv("data/evaluate_result.csv")

    print("\nEvaluate result is saved at:\n{}\n{}".format(
        "data/evaluate_result.md",
        "data/evaluate_result.csv"
    ))
