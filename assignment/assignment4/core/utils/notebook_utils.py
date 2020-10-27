import copy
import json
import tempfile
import time

import IPython
import PIL
import numpy as np
import yaml

assert_almost_equal = np.testing.assert_almost_equal


def pretty_print(result):
    result = result.copy()
    out = {}
    for k, v in result.items():
        if v is not None:
            out[k] = v
    cleaned = json.dumps(out)
    print(yaml.safe_dump(json.loads(cleaned), default_flow_style=False))


def merge_config(new_config, old_config):
    """Merge the user-defined config with default config"""
    config = copy.deepcopy(old_config)
    if new_config is not None:
        config.update(new_config)
    return config


def check_and_merge_config(user_config, default_config):
    if user_config.get("checked", False):
        return user_config
    for k in user_config.keys():
        assert k in default_config, \
            "The key {} is not in default config domain: {}".format(
                k, default_config.keys())
    config = merge_config(user_config, default_config)
    config["checked"] = True
    return config


def evaluate_agent(pg_agent, env, num_episodes=1, render=False):
    """This function evaluate the given policy and return the mean episode
    reward.
    :param policy: a function whose input is the observation
    :param num_episodes: number of episodes you wish to run
    :param seed: the random seed
    :param env_name: the name of the environment
    :param render: a boolean flag indicating whether to render policy
    :return: the averaged episode reward of the given policy.
    """
    rewards = []
    if render: num_episodes = 1
    for i in range(num_episodes):
        obs = env.reset()
        act = pg_agent.compute_action(obs)
        ep_reward = 0
        while True:
            obs, reward, done, info = env.step(act)

            # Query the agent to get action
            act = pg_agent.compute_action(obs)

            ep_reward += reward
            if render:
                env.render()
                wait(sleep=0.05)
            if done:
                break
        rewards.append(ep_reward)
    if render:
        env.close()
    return np.mean(rewards)


def wait(sleep=0.2):
    time.sleep(sleep)


def animate(img_array):
    path = tempfile.mkstemp(suffix=".gif")[1]
    images = [PIL.Image.fromarray(frame) for frame in img_array]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=0.05,
        loop=0
    )
    with open(path, "rb") as f:
        IPython.display.display(
            IPython.display.Image(data=f.read(), format='png'))
