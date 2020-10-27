"""
This file implement a base trainer class for both A2C and PPO trainers.

You should finish `evaluate_actions` and `compute_action`

-----
2019-2020 2nd term, IERG 6130: Reinforcement Learning and Beyond. Department
of Information Engineering, The Chinese University of Hong Kong. Course
Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.
"""
import os

import gym
import numpy as np
import torch
from torch.distributions import Categorical

from .network import ActorCritic, MLP


class BaseTrainer:
    def __init__(self, env, config, frame_stack=4, _test=False):
        self.device = config.device
        self.config = config
        self.lr = config.LR
        self.num_envs = config.num_envs
        self.value_loss_weight = config.value_loss_weight
        self.entropy_loss_weight = config.entropy_loss_weight
        self.num_steps = config.num_steps
        self.grad_norm_max = config.grad_norm_max

        if isinstance(env.observation_space, gym.spaces.Tuple):
            num_feats = env.observation_space[0].shape
            self.num_actions = env.action_space[0].n
        else:
            num_feats = env.observation_space.shape
            self.num_actions = env.action_space.n
        self.num_feats = (num_feats[0] * frame_stack, *num_feats[1:])

        if _test:
            self.model = MLP(num_feats[0], self.num_actions)
        else:
            self.model = ActorCritic(self.num_feats, self.num_actions)
        self.model = self.model.to(self.device)
        self.model.train()

        self.setup_optimizer()
        self.setup_rollouts()

    def setup_optimizer(self):
        raise NotImplementedError()

    def setup_rollouts(self):
        raise NotImplementedError()

    def compute_loss(self, rollouts):
        raise NotImplementedError()

    def update(self, rollout):
        raise NotImplementedError()

    def compute_action(self, obs, deterministic=False):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device)
        logits, values = self.model(obs)

        # [TODO] Get the action and action's log probabilities based on the
        #  output logits
        # Hint:
        #   1. Use torch.distributions to help you build a distribution
        #   2. Remember to check the shape of action and log prob.
        #   3. When deterministic is True, return the action with maximum
        #    probability
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            actions = dist.probs.argmax(dim=1, keepdim=True)
        else:
            actions = dist.sample().view(-1, 1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, actions)

        return values.view(-1, 1), actions.view(-1, 1), action_log_probs.view(
            -1, 1)

    def evaluate_actions(self, obs, act):
        """Run models to get the values, log probability and action
        distribution entropy of the action in current state"""
        logits, values = self.model(obs)
        # [TODO] Get the log probability of specified action, and the entropy of
        #  current distribution w.r.t. the output logits.
        # Hint: Use proper distribution to help you
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, act)
        dist = torch.distributions.Categorical(logits=logits)
        dist_entropy = dist.entropy().mean()

        assert dist_entropy.shape == ()
        return values.view(-1, 1), action_log_probs.view(-1, 1), dist_entropy

    def compute_values(self, obs):
        """Compute the values corresponding to current policy at current
        state"""
        _, values = self.model(obs)
        return values

    def save_w(self, log_dir="", suffix=""):
        os.makedirs(log_dir, exist_ok=True)
        save_path = os.path.join(log_dir, "checkpoint-{}.pkl".format(suffix))
        torch.save(dict(
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict()
        ), save_path)
        return save_path

    def load_w(self, log_dir="", suffix=""):
        save_path = os.path.join(log_dir, "checkpoint-{}.pkl".format(suffix))
        if os.path.isfile(save_path):
            state_dict = torch.load(
                save_path,
                torch.device('cpu') if not torch.cuda.is_available() else None
            )
            self.model.load_state_dict(state_dict["model"])
            self.optimizer.load_state_dict(state_dict["optimizer"])
