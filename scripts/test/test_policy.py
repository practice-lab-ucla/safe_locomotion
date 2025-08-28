import os
import torch

import numpy as np
from gymnasium.spaces import Box

from tqdm import tqdm

from skrl.envs.wrappers.torch import wrap_env
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils import set_seed

from networks import StochasticActorMLP

from utils import make_isaac_environment


def test_single_policy(env, policy, save_dir, num_envs=1000, timesteps=6000):
    reward_save_dir = os.path.join(save_dir, "reward")
    alive_save_dir = os.path.join(save_dir, "time_alive")

    if os.path.exists(reward_save_dir) and os.path.exists(alive_save_dir):
        print(f"[Skip] Results already exist in {save_dir}")
        return

    os.makedirs(save_dir, exist_ok=True)

    reward_history = np.zeros((num_envs, timesteps))
    time_alive = np.zeros(num_envs, dtype=int)
    alive = np.ones(num_envs, dtype=bool)

    obs = env.reset()[0]

    with torch.inference_mode():
        for t in tqdm(range(timesteps)):
            actions = policy(obs)
            obs, rewards, terminated, *_ = env.step(actions)

            alive = alive & np.logical_not(terminated.squeeze().cpu().numpy())
            reward_history[:, t] = rewards.squeeze().cpu().numpy()
            time_alive += alive.astype(int)

    np.save(reward_save_dir, reward_history)
    np.save(alive_save_dir, time_alive)


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


def run_one(env_id, exp, checkpoint_dir, result_dir, agent_name):
    env, simulation_app = make_isaac_environment(env_id, 1000, headless=True)
    env = wrap_env(env)
    ckpt = os.path.join(checkpoint_dir, exp, "checkpoints", agent_name)
    policy = build_policy(ckpt, env)
    outdir = os.path.join(result_dir, env_id, exp)
    test_single_policy(env, policy, outdir)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--env_id", required=True)
    p.add_argument("--exp", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--result_dir", required=True)
    p.add_argument("--agent_name", required=False, default="best_agent.pt")
    p.add_argument("--seed", required=False, default=42)
    args = p.parse_args()
    set_seed(int(args.seed))
    run_one(
        args.env_id, args.exp, args.checkpoint_dir, args.result_dir, args.agent_name
    )
