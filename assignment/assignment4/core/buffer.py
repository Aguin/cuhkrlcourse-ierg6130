"""
This file defines A2C and PPO rollout buffer.

You need to implement both A2C and PPO algorithms which compute the expected
return. Concretely, you need to implement "compute_returns" function in both
classes of storage.

-----
2019-2020 2nd term, IERG 6130: Reinforcement Learning and Beyond. Department
of Information Engineering, The Chinese University of Hong Kong. Course
Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.
"""
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class A2CRolloutStorage:
    def __init__(self, num_steps, num_processes, obs_shape, device):
        def zeros(*shapes):
            return torch.zeros(*shapes).to(device)

        self.observations = zeros(num_steps + 1, num_processes, *obs_shape)
        self.rewards = zeros(num_steps, num_processes, 1)
        self.value_preds = zeros(num_steps + 1, num_processes, 1)
        self.returns = zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = zeros(num_steps, num_processes, 1)
        self.actions = zeros(num_steps, num_processes, 1).to(torch.long)
        self.masks = torch.ones(num_steps + 1, num_processes, 1).to(device)

        self.num_steps = num_steps
        self.step = 0

    def insert(self, current_obs, action, action_log_prob, value_pred, reward,
               mask):
        self.observations[self.step + 1].copy_(current_obs)
        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_prob)
        self.value_preds[self.step].copy_(value_pred)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, gamma):
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.size(0))):
            # [TODO] Compute the expected return of current timestep `step`
            # Hint:
            #  1. self.returns[step] is the expected return at timestep `step`
            #  2. You need to use self.masks to help you remove bootstrapping
            #   when the t=`step` state is the terminal state, at which time
            #   the mask is 0 and otherwise 1.
            #  3. self.rewards stores the rewards at each timestep.
            self.returns[step] = self.rewards[step] + gamma * self.masks[step + 1] * self.returns[step + 1]


class PPORolloutStorage(A2CRolloutStorage):
    def __init__(self, num_steps, num_processes, obs_shape, device,
                 use_gae=True, gae_lambda=0.95):
        super().__init__(num_steps, num_processes, obs_shape, device)
        self.gae = use_gae
        self.gae_lambda = gae_lambda

    def feed_forward_generator(self, advantages, mini_batch_size):
        """A generator to provide samples for PPO. PPO run SGD for multiple
        times so we need more efforts to prepare data for it."""
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size, drop_last=True)
        for indices in sampler:
            observations_batch = self.observations[:-1].view(
                -1, *self.observations.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[
                indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[
                indices]
            adv_targ = advantages.view(-1, 1)[indices]

            yield observations_batch, actions_batch, return_batch, \
                  masks_batch, old_action_log_probs_batch, adv_targ

    def compute_returns(self, next_value, gamma):
        if self.gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                # [TODO] Implement GAE advantage computing here.
                # Hint:
                #  1. Return at timestep t should be (advantage_t + value_t)
                #  2. You should use reward, values, mask to compute TD error
                #   delta. Then combine TD error of timestep t with advantage
                #   of timestep t+1 to get the advantage of timestep t.
                #  3. The variable `gae` represents the advantage
                #  4. The for-loop is in a reverse order, so the varable
                #   `step` is started from `num_steps`
                #  5. Check the notebook for more information.
                delta = self.rewards[step] + gamma * self.masks[step + 1] * self.value_preds[step + 1] - self.value_preds[step]
                gae = delta + gamma * self.gae_lambda * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                # [TODO] Copy your A2C returns computing codes here.
                self.returns[step] = self.rewards[step] + gamma * self.masks[step + 1] * self.returns[step + 1]
