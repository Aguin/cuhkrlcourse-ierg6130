"""
This file defines a set of built-in agents that can be used as opponent in
training and evaluation.

Usages:
Get policy (compute_action_function) by get_compute_action_function(agent_name)

Nothing you need to implement in this file, unless you wish to implement a
custom policy or introduce a custom agent as opponent in training and
evaluation.

-----
2019-2020 2nd term, IERG 6130: Reinforcement Learning and Beyond. Department
of Information Engineering, The Chinese University of Hong Kong. Course
Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.
"""
from competitive_rl import make_envs

from core.ppo_trainer import PPOTrainer, ppo_config
from core.utils import FrameStackTensor


class PolicyAPI:
    """
    This class wrap an agent into a callable function that return action given
    an raw observation or a batch of raw observations from environment.

    This function maintain a frame stacker so that the user can securely use it.
    A reset function is provided so user can refresh the frame stacker when
    an episode is ended.

    Note that if you have implement other arbitrary custom agent, you are
    welcomed to implement a function-like API by yourself. You can write
    another API function or class and replace this one used in evaluation or
    even training.

    Your custom agent may have different network structure and different
    preprocess techniques. Remember that the API take the raw observation with
    shape (num_envs, 1, 42, 42) as input and return an single or a batch of
    integer(s) as action in [0, 1, 2]. Custom agent worth plenty of extra
    credits!
    """

    def __init__(self, num_envs=1, log_dir="", suffix=""):
        self.resized_dim = 42
        env = make_envs(num_envs=1, resized_dim=self.resized_dim)
        self.obs_shape = env.observation_space.shape
        self.agent = PPOTrainer(env, ppo_config)
        if log_dir:  # log_dir is None only in testing
            self.agent.load_w(log_dir, suffix)
        self.num_envs = num_envs
        self.frame_stack = FrameStackTensor(
            self.num_envs, self.obs_shape, 4, self.agent.device
        )

    def reset(self):
        # A potential bug is that, the frame stack is not properly reset in
        # a vectorized environment. We assume this will not impact the
        # performance significantly.
        self.frame_stack.reset()

    def __call__(self, obs):
        self.frame_stack.update(obs)
        action = self.agent.compute_action(self.frame_stack.get(), True)[1]
        if self.num_envs == 1:
            action = action.item()
        else:
            action = action.cpu().numpy()
        return action
