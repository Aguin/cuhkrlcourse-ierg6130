"""
A suits of blackbox tests of PPO algorithm. Run this file to make sure your
computer can run the system.

Usages:
    python test_ppo.py
or
    pytest test_ppo.py

Nothing to do in this file.

-----
2019-2020 2nd term, IERG 6130: Reinforcement Learning and Beyond. Department
of Information Engineering, The Chinese University of Hong Kong. Course
Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.
"""
import unittest

from train import train, parser


class PPOTest(unittest.TestCase):
    def test_cartpole_single_env(self):
        args = parser.parse_args([
            "--env-id", "CartPole-v0",
            "--max-steps", "2000",
            "--num-envs", "1",
            "--algo", "PPO",
        ])
        train(args)

    def test_cartpole_multiple_env(self):
        args = parser.parse_args([
            "--env-id", "CartPole-v0",
            "--max-steps", "6000",
            "--num-envs", "3",
            "--algo", "PPO",
        ])
        train(args)

    def test_pong_single_env(self):
        args = parser.parse_args([
            "--env-id", "cPong-v0",
            "--max-steps", "2000",
            "--num-envs", "1",
            "--algo", "PPO",
        ])
        train(args)

    def test_pong_multiple_env(self):
        args = parser.parse_args([
            "--env-id", "cPong-v0",
            "--max-steps", "6000",
            "--num-envs", "3",
            "--algo", "PPO",
        ])
        train(args)

    def test_pong_tournament_single(self):
        args = parser.parse_args([
            "--env-id", "cPongTournament-v0",
            "--max-steps", "2000",
            "--num-envs", "1",
            "--algo", "PPO",
        ])
        train(args)

    def test_pong_tournament_multiple(self):
        args = parser.parse_args([
            "--env-id", "cPongTournament-v0",
            "--max-steps", "6000",
            "--num-envs", "3",
            "--algo", "PPO",
        ])
        train(args)


class A2CTest(unittest.TestCase):
    def test_cartpole_single_env(self):
        args = parser.parse_args([
            "--env-id", "CartPole-v0",
            "--max-steps", "200",
            "--num-envs", "1",
            "--algo", "A2C",
        ])
        train(args)

    def test_cartpole_multiple_env(self):
        args = parser.parse_args([
            "--env-id", "CartPole-v0",
            "--max-steps", "600",
            "--num-envs", "3",
            "--algo", "A2C",
        ])
        train(args)

    def test_pong_single_env(self):
        args = parser.parse_args([
            "--env-id", "cPong-v0",
            "--max-steps", "200",
            "--num-envs", "1",
            "--algo", "A2C",
        ])
        train(args)

    def test_pong_multiple_env(self):
        args = parser.parse_args([
            "--env-id", "cPong-v0",
            "--max-steps", "600",
            "--num-envs", "3",
            "--algo", "A2C",
        ])
        train(args)


if __name__ == '__main__':
    unittest.main(verbosity=2)
