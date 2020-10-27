"""
This file implement A2C algorithm.

You need to implement `update` and `compute_loss` functions.

-----
2019-2020 2nd term, IERG 6130: Reinforcement Learning and Beyond. Department
of Information Engineering, The Chinese University of Hong Kong. Course
Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.
"""
import torch
from torch import optim

from .base_trainer import BaseTrainer
from .buffer import A2CRolloutStorage


class A2CConfig(object):
    def __init__(self):
        # Common
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.eval_freq = 100
        self.save_freq = 200
        self.log_freq = 10
        self.num_envs = 1

        # Sample
        self.num_steps = 20  # num_steps * num_envs = sample_batch_size
        self.resized_dim = 42

        # Learning
        self.GAMMA = 0.99
        self.LR = 7e-4
        self.grad_norm_max = 10.0
        self.entropy_loss_weight = 0.01
        self.value_loss_weight = 0.5


a2c_config = A2CConfig()


class A2CTrainer(BaseTrainer):
    def __init__(self, env, config, frame_stack=4, _test=False):
        super(A2CTrainer, self).__init__(env, config, frame_stack, _test=_test)

    def setup_optimizer(self):
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr,
                                       alpha=0.99, eps=1e-5)

    def setup_rollouts(self):
        self.rollouts = A2CRolloutStorage(self.num_steps, self.num_envs,
                                          self.num_feats, self.device)

    def compute_loss(self, rollouts):
        obs_shape = rollouts.observations.size()[2:]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy = self.evaluate_actions(
            rollouts.observations[:-1].view(-1, *obs_shape),
            rollouts.actions.view(-1, 1))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        # [TODO] Get the unnormalized advantages
        advantages = rollouts.returns[:-1] - values

        # [TODO] Get the value loss
        value_loss = advantages.pow(2).mean()

        # [TODO] Normalize the advantages
        advantages = (advantages - advantages.mean()) / advantages.std()

        # [TODO] Get the policy loss
        policy_loss = -(advantages.detach() * action_log_probs).mean()

        # Get the total loss
        loss = policy_loss + self.value_loss_weight * value_loss - \
               self.entropy_loss_weight * dist_entropy

        return loss, policy_loss, value_loss, dist_entropy

    def update(self, rollout):
        total_loss, action_loss, value_loss, dist_entropy = self.compute_loss(
            rollout)
        # [TODO] Step self.optimizer by computing the gradient of total loss
        # Hint: remember to clip the gradient to self.grad_norm_max
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_max)
        self.optimizer.step()

        return action_loss.item(), value_loss.item(), dist_entropy.item(), \
               total_loss.item()
