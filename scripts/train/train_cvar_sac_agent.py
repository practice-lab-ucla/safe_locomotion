import torch
import torch.nn as nn

# import the skrl components to build the RL system
# from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from networks import C51, StochasticActorMLP
from agents import CVaRSAC, CVAR_SAC_DEFAULT_CONFIG


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# load isaac lab environment
# launch the simulation app
from utils import make_isaac_environment

env = make_isaac_environment(
    task="Safe-Locomotion-Base-v0", num_envs=100, headless=True
)
env = wrap_env(env)
device = env.device

# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=80000, num_envs=env.num_envs, device=device)


# define a bounded action space so skrl can sample from it
import numpy as np
from gymnasium.spaces import Box

low = np.zeros(env.action_space.shape)
high = np.zeros_like(low)

low[0:4] = -2
high[0:4] = 2
low[4:8] = -2
high[4:8] = 2
low[8:12] = -2
high[8:12] = 2

action_space = Box(low=low, high=high)

# instantiate the agent's models (function approximators).
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
models = {}
models["policy"] = StochasticActorMLP(
    env.observation_space, action_space, device, clip_actions=False
)
models["critic_1"] = C51(env.observation_space, action_space, device)
models["critic_2"] = C51(env.observation_space, action_space, device)
models["target_critic_1"] = C51(env.observation_space, action_space, device)
models["target_critic_2"] = C51(env.observation_space, action_space, device)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
cfg = CVAR_SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 1
cfg["batch_size"] = 4096
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 5e-4
cfg["critic_learning_rate"] = 5e-4
cfg["random_timesteps"] = 2000
cfg["learning_starts"] = 100
cfg["grad_norm_clip"] = 1.0
cfg["learn_entropy"] = True
cfg["entropy_learning_rate"] = 5e-3
cfg["initial_entropy_value"] = 1.0
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["cvar_alpha"] = 1 - 0.01
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 800
cfg["experiment"]["checkpoint_interval"] = 8000
cfg["experiment"]["directory"] = "runs/finetune_sac"
cfg["experiment"]["wandb"] = True
cfg["experiment"]["wandb_kwargs"] = {}

agent = CVaRSAC(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 80000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()
