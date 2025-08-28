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
from bandit import *


set_seed(42)


def test_bandit(
        env, 
        policies, 
        bandit, 
        save_dir, 
        num_envs=500, 
        timesteps=6000,
        select_arm_every=15
    ):
    reward_save_dir = os.path.join(save_dir, "reward")
    selection_save_dir = os.path.join(save_dir, "selection")

    os.makedirs(save_dir, exist_ok=True)

    decision_steps = timesteps // select_arm_every

    reward_history = np.zeros((num_envs, decision_steps))
    selection_history = np.zeros((num_envs, decision_steps), dtype=int)


    obs = env.reset()[0]

    with torch.inference_mode():
        env_idx = torch.arange(num_envs, device="cuda")
        for t in tqdm(range(decision_steps)):
            arms = bandit.select_arm()
            sum_reward = np.zeros((num_envs,))

            for _ in range(select_arm_every):
                actions = policies(obs)[torch.as_tensor(arms, device="cuda"), env_idx]
                obs, rewards, terminated, *_ = env.step(actions)

                # alive = alive & np.logical_not(terminated.squeeze().cpu().numpy())
                rewards_np = rewards.squeeze().cpu().numpy()
                sum_reward += rewards_np

            reward_history[:, t] = sum_reward
            selection_history[:, t] = arms
            bandit.update(arms, sum_reward)

    np.save(reward_save_dir, reward_history)
    np.save(selection_save_dir, selection_history)


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


def build_policies(policy_checkpoints, env):
    policy_list = [build_policy(pc, env) for pc in policy_checkpoints]

    def policies(obs):
        actions = [policy(obs) for policy in policy_list]
        return torch.stack(actions, dim=0).to("cuda")  # n_policy, n_env

    return policies


def run_one(
        env_id,   
        exps, 
        ckpt, 
        result_dir,
        bandit, 
        n_envs=1000, 
        timesteps=12000, 
        select_arm_every=15
    ):
    bandit = eval(bandit)(len(exps), n_envs)
    env, simulation_app = make_isaac_environment(env_id, n_envs, headless=True)
    env = wrap_env(env)
    ckpts_dirs = [os.path.join(ckpt, exp, "checkpoints", "best_agent.pt") for exp in exps]
    policy = build_policies(ckpts_dirs, env)
    outdir = os.path.join(result_dir, env_id, "+".join(exps))
    test_bandit(env, policy, bandit, outdir, n_envs, timesteps, select_arm_every)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--env_id", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument(
        "--exps",
        nargs="+",
        required=True,
        help="One or more experiment directories"
    )
    p.add_argument("--result_dir", required=True)
    p.add_argument("--bandit", type=str, default="UCB")
    p.add_argument("--num_envs", type=int, default=1000)
    p.add_argument("--select_arm_every", type=int, default=30)
    p.add_argument("--timesteps", type=int, default=12000)
    args = p.parse_args()
    run_one(
        args.env_id, 
        args.exps,
        args.checkpoint_dir, 
        args.result_dir,
        args.bandit,
        args.num_envs,
        args.timesteps,
        args.select_arm_every
    )


# pythonp scripts/test/test_bandit.py --env_id Safe-Locomotion-Friction-v1 --checkpoint_dir runs/cvar_ppo/Safe-Locomotion-Base-v0 --exps alpha0.1_adaptive_lammax0.25_True_warmup5400 alpha0.25_adaptive_lammax0.25_True_warmup5400 ppo2 --result_dir results_bandit/