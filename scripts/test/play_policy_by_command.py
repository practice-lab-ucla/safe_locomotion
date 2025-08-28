import os
import torch
import json

import numpy as np
from gymnasium.spaces import Box
from itertools import cycle

from skrl.envs.wrappers.torch import wrap_env
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils import set_seed

from networks import StochasticActorMLP

from utils import make_isaac_environment


set_seed(42)


def get_command_from_npz(npz_path):
    with np.load(npz_path, allow_pickle=False) as f:
        obs = np.array(f["obs"])  # shape [T, 48], float32
        commands = obs[:, 9:12]
    return commands


def get_actions_from_npz(npz_path):
    with np.load(npz_path, allow_pickle=False) as f:
        obs = np.array(f["obs"])  # shape [T, 48], float32
        actions = obs[:, -12:]
    return actions


def play_single_policy(env, app, policy, npz_path):
    obs = env.reset()[0]
    commands = get_command_from_npz(npz_path)
    obs_list = []
    with torch.inference_mode():
        # while app.app.is_running():
        for idx in range(len(commands)):
            next_cmd = commands[idx]
            obs[:, 9:12] = torch.from_numpy(next_cmd).cuda()
            actions = policy(obs)
            obs_list.append(obs.tolist())
            obs, *_ = env.step(actions)
            env.render()

    np.save("/home/danny/Documents/safe_locomotion/logs/ppo_sim_gain", np.array(obs_list))


def play_single_policy_action(env, app, policy, npz_path):
    obs = env.reset()[0]
    actions = get_actions_from_npz(npz_path)
    obs_list = []
    with torch.inference_mode():
        # while app.app.is_running():
        for idx in range(len(actions)):
            next_action = torch.from_numpy(np.broadcast_to(actions[idx], (len(obs), 12)).copy()).cuda().float()
            obs_list.append(obs.tolist())
            obs, *_ = env.step(next_action)
            env.render()

    np.save("/home/danny/Documents/safe_locomotion/logs/ppo_sim_action_gain", np.array(obs_list))


def build_policy(policy_checkpoint, env):
    state_dict = torch.load(
        policy_checkpoint,
        map_location="cuda",
        weights_only=True,
    )
    use_affine = any("8" in key for key in state_dict["policy"].keys())
    action_space = Box(
        low=-2 * np.pi,
        high=2 * np.pi,
        shape=env.action_space.shape,
        dtype=env.action_space.dtype,
    )
    actor = StochasticActorMLP(
        env.observation_space,
        action_space,
        "cuda",
        clip_actions=False,
        elementwise_affine=use_affine,
    )
    actor.load_state_dict(state_dict["policy"])
    actor.to("cuda")

    state_preprocessor = lambda x: x
    if "state_preprocessor" in state_dict:
        state_preprocessor = RunningStandardScaler(size=(48,), device="cuda")
        state_preprocessor.load_state_dict(state_dict["state_preprocessor"])
        state_preprocessor.to("cuda")

    def policy(obs):
        action = actor.act({"states": state_preprocessor(obs)}, "policy")[-1].get(
            "mean_actions"
        )
        return action

    return policy


def run_one(env_id, exp, checkpoint_dir, num_envs, agent_name, npz_path):
    env, simulation_app = make_isaac_environment(env_id, num_envs, headless=False)
    env = wrap_env(env)
    ckpt = os.path.join(checkpoint_dir, exp, "checkpoints", agent_name)
    policy = build_policy(ckpt, env)
    play_single_policy(env, simulation_app, policy, npz_path)
    # play_single_policy_action(env, simulation_app, policy, npz_path)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--env_id", required=True)
    p.add_argument("--exp", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--num_envs", required=False, default=3)
    p.add_argument("--agent_name", required=False, default="best_agent.pt")
    p.add_argument(
        "--npz_path", default="/home/danny/Documents/safe_locomotion/logs/ppo.npz"
    )
    args = p.parse_args()
    run_one(
        args.env_id,
        args.exp,
        args.checkpoint_dir,
        args.num_envs,
        args.agent_name,
        args.npz_path,
    )
