"""
This file implements the some helper functions.

You need to finish `step_envs` function.

Note that many codes in this file is not written by us. The credits go to the
original writers.

-----
2019-2020 2nd term, IERG 6130: Reinforcement Learning and Beyond. Department
of Information Engineering, The Chinese University of Hong Kong. Course
Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.
"""
import copy
import glob
import json
import os
import time
from collections import deque

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.signal import medfilt


def step_envs(cpu_actions, envs, episode_rewards, frame_stack_tensor,
              reward_recorder, length_recorder, total_steps, total_episodes,
              device, test):
    """Step the vectorized environments for one step. Process the reward
    recording and terminal states."""
    obs, reward, done, info = envs.step(cpu_actions)
    episode_rewards += reward.reshape(episode_rewards.shape)
    episode_rewards_old_shape = episode_rewards.shape
    if not np.isscalar(done[0]):
        done = np.all(done, axis=1)
    for idx, d in enumerate(done):
        if d:  # the episode is done
            # Record the reward of the terminated episode to
            reward_recorder.append(episode_rewards[idx].copy())

            # For CartPole-v0 environment, the length of episodes is not
            # recorded.
            if "num_steps" in info[idx]:
                length_recorder.append(info[idx]["num_steps"])
            total_episodes += 1
    masks = 1. - done.astype(np.float32)

    # [TODO] Multiply the episode_rewards (a vector to record the episode
    #  rewards of those episodes that are running in multiple concurrent
    #  environments) with masks, in order to refresh the reward accumulating
    #  when some episodes is terminated.
    # Hint: Pay attention to the shape of `masks` and `episode_rewards`
    episode_rewards *= masks.reshape(episode_rewards.shape)

    assert episode_rewards.shape == episode_rewards_old_shape

    total_steps += obs[0].shape[0] if isinstance(obs, tuple) else obs.shape[0]
    masks = torch.from_numpy(masks).to(device).view(-1, 1)
    # frame_stack_tensor is refreshed in-place if done happen.
    if test:
        frame_stack_masks = masks.view(-1, 1)
    else:
        frame_stack_masks = masks.view(-1, 1, 1, 1)
    # If in multiple pong mode, we suppose only the first observation is used to
    # train agent.
    frame_stack_tensor.update(obs[0] if isinstance(obs, tuple) else obs,
                              frame_stack_masks)
    return obs, reward, done, info, masks, total_episodes, total_steps, \
           episode_rewards


def save_progress(log_dir, progress):
    path = os.path.join(log_dir, "progress.pkl")
    torch.save(progress, path)
    return path


def load_progress(log_dir):
    progress = torch.load(os.path.join(log_dir, "progress.pkl"))
    progress = [flatten_dict(d) for d in progress]
    return pd.DataFrame(progress)


def flatten_dict(dt, delimiter="/"):
    dt = copy.deepcopy(dt)
    while any(isinstance(v, dict) for v in dt.values()):
        remove = []
        add = {}
        for key, value in dt.items():
            if isinstance(value, dict):
                for subkey, v in value.items():
                    add[delimiter.join([key, subkey])] = v
                remove.append(key)
        dt.update(add)
        for k in remove:
            del dt[k]
    return dt


def summary(array, name, extra_dict=None):
    ret = {
        "{}_mean".format(name): float(np.mean(array)) if len(array) else np.nan,
        "{}_min".format(name): float(np.min(array)) if len(array) else np.nan,
        "{}_max".format(name): float(np.max(array)) if len(array) else np.nan,
    }
    if extra_dict:
        ret.update(extra_dict)
    return ret


def evaluate(trainer, eval_envs, frame_stack, num_episodes=10, seed=0):
    """This function evaluate the given policy and return the mean episode
    reward.
    :param policy: a function whose input is the observation
    :param env: an environment instance
    :param num_episodes: number of episodes you wish to run
    :param seed: the random seed
    :return: the averaged episode reward of the given policy.
    """

    frame_stack_tensor = FrameStackTensor(
        eval_envs.num_envs, eval_envs.observation_space.shape, frame_stack,
        trainer.device
    )

    def get_action(frame_stack_tensor):
        obs = frame_stack_tensor.get()
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(trainer.device)
        with torch.no_grad():
            act = trainer.compute_action(obs, deterministic=True)[1]
        act = act.view(-1).cpu().numpy()
        return act

    reward_recorder = []
    episode_length_recorder = []
    episode_rewards = np.zeros([eval_envs.num_envs, 1], dtype=np.float)
    total_steps = 0
    total_episodes = 0
    eval_envs.seed(seed)
    obs = eval_envs.reset()
    frame_stack_tensor.update(obs)
    while True:
        obs, reward, done, info, masks, total_episodes, total_steps, \
        episode_rewards = step_envs(
            get_action(frame_stack_tensor), eval_envs, episode_rewards,
            frame_stack_tensor, reward_recorder, episode_length_recorder,
            total_steps, total_episodes, trainer.device, frame_stack == 1)
        if total_episodes >= num_episodes:
            break
    return reward_recorder, episode_length_recorder


class FrameStackTensor:
    def __init__(self, num_envs, obs_shape, frame_stack, device):
        self.num_channels = obs_shape[0]
        self.obs_shape = (obs_shape[0] * frame_stack, *obs_shape[1:])
        self.current_obs = torch.zeros(num_envs, *self.obs_shape, device=device,
                                       dtype=torch.float)
        self.mask_shape = [1] * self.current_obs.dim()
        self.mask_shape[0] = -1
        self.device = device

    def update(self, obs, mask=None):
        """current_obs is a tensor with shape [num_envs, num_stacks, 84, 84].
        It keeps rolling at second dimension in order to stack the latest
        num_stacks frames.
        """
        if mask is not None:
            mask = mask.reshape(self.mask_shape)
            self.current_obs *= mask
        self.current_obs = self.current_obs.roll(
            shifts=-self.num_channels, dims=1)
        obs = torch.from_numpy(obs.astype(np.float32)).to(self.device)
        self.current_obs[:, -self.num_channels:] = obs
        return self.current_obs

    def reset(self):
        self.current_obs.fill_(0)

    def get(self):
        return self.current_obs


class Timer:
    def __init__(self, interval=10):
        self.value = 0.0
        self.start = time.time()
        self.buffer = deque(maxlen=interval)

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.value = time.time() - self.start
        self.buffer.append(self.value)

    @property
    def now(self):
        """Return the seconds elapsed since initializing this class"""
        return time.time() - self.start

    @property
    def avg(self):
        return np.mean(self.buffer, dtype=float)


def pretty_print(result):
    result = result.copy()
    out = {}
    for k, v in result.items():
        if v is not None:
            out[k] = v
    cleaned = json.dumps(out)
    print(yaml.safe_dump(json.loads(cleaned), default_flow_style=False))


def verify_log_dir(log_dir, algo=None):
    if algo:
        log_dir = os.path.join(log_dir, algo)
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
    return os.path.abspath(log_dir)


def load_data(indir, smooth, bin_size):
    datas = []
    infiles = glob.glob(os.path.join(indir, '*.monitor.csv'))

    for inf in infiles:
        with open(inf, 'r') as f:
            f.readline()
            f.readline()
            for line in f:
                line = line.replace('\x00', '').rstrip()
                tmp = line.split(',')
                t_time = float(tmp[2])
                tmp = [t_time, int(tmp[1]), float(tmp[0])]
                datas.append(tmp)

    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = []
    timesteps = 0
    for i in range(len(datas)):
        result.append([timesteps, datas[i][-1]])
        timesteps += datas[i][1]

    if len(result) < bin_size:
        return [None, None]

    x, y = np.array(result)[:, 0], np.array(result)[:, 1]

    if smooth == 1:
        x, y = smooth_reward_curve(x, y)

    if smooth == 2:
        y = medfilt(y, kernel_size=9)

    x, y = fix_point(x, y, bin_size)
    return [x, y]


def smooth_reward_curve(x, y):
    # Halfwidth of our smoothing convolution
    halfwidth = min(31, int(np.ceil(len(x) / 30)))
    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
            np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]


def fix_point(x, y, interval):
    np.insert(x, 0, 0)
    np.insert(y, 0, 0)

    fx, fy = [], []
    pointer = 0

    ninterval = int(max(x) / interval + 1)

    for i in range(ninterval):
        tmpx = interval * i

        while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
            pointer += 1

        if pointer + 1 < len(x):
            alpha = (y[pointer + 1] - y[pointer]) / \
                    (x[pointer + 1] - x[pointer])
            tmpy = y[pointer] + alpha * (tmpx - x[pointer])
            fx.append(tmpx)
            fy.append(tmpy)

    return fx, fy


def plot(folder, game, name, num_steps, bin_size=100, smooth=1):
    matplotlib.rcParams.update({'font.size': 20})
    tx, ty = load_data(folder, smooth, bin_size)

    if tx is None or ty is None:
        return

    fig = plt.figure(figsize=(20, 5))
    plt.plot(tx, ty, label="{}".format(name))

    tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = tick_fractions * num_steps
    tick_names = ["{:.0e}".format(tick) for tick in ticks]
    plt.xticks(ticks, tick_names)
    plt.xlim(0, num_steps * 1.01)

    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')

    plt.title(game)
    plt.legend(loc=4)
    plt.show()
