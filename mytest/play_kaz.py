# https://github.com/Farama-Foundation/SuperSuit/pull/131
# fix by this https://github.com/Farama-Foundation/PettingZoo/pull/629

import argparse
import json
import os
import sys
from copy import deepcopy
import time

import supersuit as ss
from pettingzoo.butterfly import (
    cooperative_pong_v3,
    pistonball_v4,
    knights_archers_zombies_v8,
    prospector_v4,
)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecMonitor
from torch import nn as nn

from utils import (
    image_transpose,
    AgentIndicatorWrapper,
    BinaryIndicator,
    GeometricPatternIndicator,
    InvertColorIndicator,
)

class Args:
    def __init__(self) -> None:
        self.env_name = "knights-archers-zombies-v8"
        self.n_runs =  1
        self.n_evaluations = 100
        self.timesteps = 1e7
        #self.num_cpus = 8
        self.num_cpus = 1
        self.num_eval_cpus = 1
        self.num_vec_envs = 1
args = Args()

params = {
    "net_arch": "small",
    "activation_fn": "relu",
    "agent_indicator": "invert",
    #"batch_size": 64,
    "batch_size": 128,
    "n_steps": 1024,
    "gamma": 0.995,
    "learning_rate": 1.30299e-05,
    "ent_coef": 1.85841e-06,
    "clip_range": 0.1,
    "n_epochs": 10,
    "gae_lambda": 0.95,
    "max_grad_norm": 0.7,
    "vf_coef": 0.95971
}


print("Hyperparameters:")
print(params)
muesli_obs_size = 96
muesli_frame_size = 4
evaluations = args.n_evaluations
timesteps = args.timesteps

net_arch = {
    "small": [dict(pi=[64, 64], vf=[64, 64])],
    "medium": [dict(pi=[256, 256], vf=[256, 256])],
}[params["net_arch"]]

activation_fn = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}[params["activation_fn"]]

policy_kwargs = dict(
    net_arch=net_arch,
    activation_fn=activation_fn,
    ortho_init=False,
)
agent_indicator_name = params["agent_indicator"]

del params["net_arch"]
del params["activation_fn"]
del params["agent_indicator"]
params["policy_kwargs"] = policy_kwargs
params["policy"] = "CnnPolicy"

# Generate env
if args.env_name == "knights-archers-zombies-v8":
    env = knights_archers_zombies_v8.parallel_env()
    agent_type = "archer"


env.reset()
num_agents = env.num_agents
env = ss.color_reduction_v0(env)
env = ss.pad_action_space_v0(env)
env = ss.pad_observations_v0(env)
env = ss.resize_v0(
    env, x_size=muesli_obs_size, y_size=muesli_obs_size, linear_interp=True
)
env = ss.frame_stack_v1(env, stack_size=muesli_frame_size)

# Enable black death
if args.env_name == "knights-archers-zombies-v8":
    print("black_death_v2")
    env = ss.black_death_v2(env)

# Agent indicator wrapper
if agent_indicator_name == "invert":
    print("invert")
    agent_indicator = InvertColorIndicator(env, agent_type)
    agent_indicator_wrapper = AgentIndicatorWrapper(agent_indicator)
elif agent_indicator_name == "invert-replace":
    agent_indicator = InvertColorIndicator(env, agent_type)
    agent_indicator_wrapper = AgentIndicatorWrapper(agent_indicator, False)
elif agent_indicator_name == "binary":
    agent_indicator = BinaryIndicator(env, agent_type)
    agent_indicator_wrapper = AgentIndicatorWrapper(agent_indicator)
elif agent_indicator_name == "geometric":
    agent_indicator = GeometricPatternIndicator(env, agent_type)
    agent_indicator_wrapper = AgentIndicatorWrapper(agent_indicator)


if agent_indicator_name != "identity":
    env = ss.observation_lambda_v0(
        env, agent_indicator_wrapper.apply, agent_indicator_wrapper.apply_space
    )

env = ss.pettingzoo_env_to_vec_env_v1(env)
eval_env = deepcopy(env)
env = ss.concat_vec_envs_v1(
    env,
    num_vec_envs=args.num_vec_envs,
    num_cpus=args.num_cpus,
    base_class="stable_baselines3",
)
env = VecMonitor(env)
env = image_transpose(env)

eval_env = ss.concat_vec_envs_v1(
    eval_env,
    num_vec_envs=args.num_vec_envs,
    num_cpus=args.num_eval_cpus,
    base_class="stable_baselines3",
)
eval_env = VecMonitor(eval_env)
eval_env = image_transpose(eval_env)

all_mean_rewards = []


log_dir = None

ckpt = "./models/PPO_Kaz/10000000"
model= PPO.load(ckpt, env=env, verbose=1, tensorboard_log=log_dir)


def policy(agent, observation):
    action, _states = model.predict(observation)
    #print(agent)
    #return env.action_space(agent).sample()
    return action

#for agent in env.agent_iter():
ep = 10
for i in range(ep):
    observation = env.reset()
    done = [0, 0, 0, 0]
    while not 1 in done:
        action = policy(None, observation)
        observation, reward, done, info = env.step(action)
        env.render()
    print(done)
env.close()