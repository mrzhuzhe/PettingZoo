# https://github.com/Farama-Foundation/SuperSuit/pull/131
# fix by this https://github.com/Farama-Foundation/PettingZoo/pull/629
# https://github.com/jkterry1/Butterfly-Baselines/blob/main/train_all.sh
import argparse
import json
import os
import sys
from copy import deepcopy
import time

import supersuit as ss

"""
from pettingzoo.butterfly import (
    cooperative_pong_v3,
    pistonball_v4,
    knights_archers_zombies_v8,
    prospector_v4,
)
"""
import knights_archers_zombies.knights_archers_zombies as knights_archers_zombies_v8

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
"""
parser = argparse.ArgumentParser()
parser.add_argument(
    "--env-name",
    help="Butterfly Environment to use from PettingZoo",
    type=str,
    default="pistonball_v4",
    choices=[
        "pistonball_v4",
        "cooperative_pong_v3",
        "knights_archers_zombies_v8",
        "prospector_v4",
    ],
)
parser.add_argument("--n-runs", type=int, default=5)
parser.add_argument("--n-evaluations", type=int, default=100)
parser.add_argument("--timesteps", type=int, default=0)
parser.add_argument("--num-cpus", type=int, default=8)
parser.add_argument("--num-eval-cpus", type=int, default=4)
parser.add_argument("--num-vec-envs", type=int, default=4)
args = parser.parse_args()

param_file = "./config/" + str(args.env_name) + ".json"
with open(param_file) as f:
    params = json.load(f)
"""
class Args:
    def __init__(self) -> None:
        self.env_name = "knights-archers-zombies-v8"
        #self.env_name = "pistonball_v4"
        self.n_runs =  1
        self.n_evaluations = 100
        self.timesteps = 1e7
        #self.timesteps = 2e6
        #self.num_cpus = 8
        self.num_cpus = 16
        self.num_eval_cpus = 16
        self.num_vec_envs = 4
args = Args()
params = {
    "net_arch": "small",
    "activation_fn": "relu",
    "agent_indicator": "invert",
    #"batch_size": 64,
    "batch_size": 64,
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
muesli_obs_size = 320
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
if args.env_name == "prospector_v4":
    env = prospector_v4.parallel_env()
    agent_type = "prospector"
elif args.env_name == "knights-archers-zombies-v8":
    env = knights_archers_zombies_v8.parallel_env()
    agent_type = "archer"
elif args.env_name == "cooperative_pong_v3":
    env = cooperative_pong_v3.parallel_env()
    agent_type = "paddle_0"
elif args.env_name == "pistonball_v4":
    env = pistonball_v4.parallel_env()

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
log_dir = "./data/" + args.env_name + "/"
os.makedirs(log_dir, exist_ok=True)



models_dir = f"models/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)


for i in range(args.n_runs):
    model = PPO(
        env=env,
        tensorboard_log=log_dir,
        # We do not seed the trial
        seed=None,
        verbose=2,
        **params
    )

    run_log_dir = log_dir + "run_" + str(i)

    n_eval_episodes = 5
    eval_freq = timesteps // evaluations // model.get_env().num_envs

    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=n_eval_episodes,
        log_path=run_log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False, callback=eval_callback)
    model.save(f"{models_dir}/ori-{i}-{timesteps}")