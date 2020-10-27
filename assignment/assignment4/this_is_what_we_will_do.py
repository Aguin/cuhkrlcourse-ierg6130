"""
This file shows what we will do when we have collected all students assignments.

First, we merge all students' submitted checkpoints into "data" directory. (
So you need to make sure your submitted checkpoint, i.e. pkl file, have unique
name.)

Then, we rename the function "student_compute_action_function" in
"this_is_my_agent.py" to "student_YOUR-STUDENT-ID". But we do not change the
content in your function.

Third, we run this file "this_is_what_we_will_do.py", which automatically
gather all functions in "this_is_my_agent.py" and launch a series matches.

Finally, we summary the match results of all your agents against others to a
winning-rate matrix.

You do not need to modify anything in this file. If you want to make sure
your codes in "this_is_my_agent.py" works well, you can run this file
directly via:
    python this_is_what_we_will_do.py

It may takes a long time since we need to launch N*N/2 matches where N is the
number of existing agents, including those builtin agents.

-----
2019-2020 2nd term, IERG 6130: Reinforcement Learning and Beyond. Department
of Information Engineering, The Chinese University of Hong Kong. Course
Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.
"""
import argparse

import pandas as pd
import tabulate
from competitive_rl import make_envs, get_builtin_agent_names, \
    get_compute_action_function

import this_is_my_agent
from evaluate import launch, build_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", "-N", default=100, type=int,
                        help="Number of episodes to run.")
    parser.add_argument("--num-envs", default=10, type=int,
                        help="Number of parallel environments.")
    args = parser.parse_args()
    num_envs = args.num_envs
    num_episodes = args.num_episodes

    # ===== Load student policies =====
    student_function_names = [
        function_name for function_name in dir(this_is_my_agent)
        if function_name.startswith("student")
    ]
    student_functions = {}
    for f in student_function_names:
        studnet_policy_creator = this_is_my_agent.__dict__[f]
        studnet_id = f.split("student_")[-1]
        student_functions[studnet_id] = studnet_policy_creator(num_envs)
    print("Collected policies: ", student_functions.keys())

    # Merge builtin agent with students' agents
    for name in get_builtin_agent_names():
        student_functions[name] = get_compute_action_function(name, num_envs)

    # ===== Setup environment =====
    envs = make_envs(
        "cPongDouble-v0", num_envs=num_envs, asynchronous=True)
    print("Environment ready")

    # ===== Run Matches =====
    visited_agent = set()
    result_list = []
    for name, policy in student_functions.items():
        # Remove repeat agents
        opponent_functions = student_functions.copy()
        for opponent in visited_agent:
            opponent_functions.pop(opponent)

        print("Start match between agent {} with {}.".format(
            name, opponent_functions.keys()
        ))

        result = launch(name, policy, opponent_functions, envs, num_episodes)
        result_list.append(result)
        visited_agent.add(name)

    result_list = pd.concat(result_list)
    winning_rate_matrix, reward_matrix = build_matrix(result_list)
    print("===== Winning Rate Matrix (row vs column) =====")
    print(winning_rate_matrix)
    print("===== Reward Matrix (row vs column) =====")
    print(reward_matrix)

    with open("data/full_evaluate_result.md", "w") as f:
        f.write("winning rate matrix:\n\n")
        f.write(tabulate.tabulate(winning_rate_matrix,
                                  winning_rate_matrix.keys(),
                                  tablefmt="pipe"))
        f.write("\n\n\n\n\n")
        f.write("reward matrix\n\n")
        f.write(tabulate.tabulate(reward_matrix, reward_matrix.keys(),
                                  tablefmt="pipe"))

    result_list.to_csv("data/full_evaluate_result.csv")
    envs.close()

    print("\nEvaluate result is saved at:\n{}\n{}".format(
        "data/full_evaluate_result.md",
        "data/full_evaluate_result.csv"
    ))
