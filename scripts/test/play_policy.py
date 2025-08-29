import os
import torch
import argparse
from datetime import datetime

import numpy as np
from gymnasium.spaces import Box

from skrl.envs.wrappers.torch import wrap_env
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils import set_seed

from networks import StochasticActorMLP
from utils import make_isaac_environment


set_seed(42)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def resolve_save_path(save_path, checkpoint_dir, exp, agent_name):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"rollout_{exp}_{agent_name}_{ts}.npz"
    if save_path is None:
        # default: put the file next to checkpoints under the experiment folder
        target_dir = os.path.join(checkpoint_dir, exp, "rollouts")
        os.makedirs(target_dir, exist_ok=True)
        return os.path.join(target_dir, default_name)

    # If user passed a directory or a path without .npz, make a filename inside it
    if (os.path.isdir(save_path)) or (not save_path.lower().endswith(".npz")):
        os.makedirs(save_path, exist_ok=True)
        return os.path.join(save_path, default_name)

    # User passed a full file path
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    return save_path


def save_rollout_npz(path, obs_list, next_obs_list, action_list, meta):
    # Shapes: T x num_envs x ...
    obs_arr = np.stack(obs_list, axis=0)
    next_obs_arr = np.stack(next_obs_list, axis=0)
    action_arr = np.stack(action_list, axis=0)
    np.savez_compressed(
        path,
        obs=obs_arr,
        next_obs=next_obs_arr,
        actions=action_arr,
        **meta,
    )
    print(f"\nSaved rollout to: {path}")
    print(
        f"  obs: {obs_arr.shape} | next_obs: {next_obs_arr.shape} | actions: {action_arr.shape}"
    )


def play_single_policy(env, app, policy, headless, save_path, meta):
    """Run until window closes or Ctrl-C. Save rollout on exit."""
    obs = env.reset()[0]

    obs_list = []
    next_obs_list = []
    action_list = []
    i = 0
    with torch.inference_mode():
        while app.app.is_running():
            # record obs_t
            i += 1
            obs[:, 9:12] = torch.clamp(obs[:, 9:12], max=0.5, min=-0.5)

            obs_list.append(to_numpy(obs))

            # a_t
            actions = policy(obs)
            action_list.append(to_numpy(actions))

            # step -> s_{t+1}
            obs, *_ = env.step(actions)
            next_obs_list.append(to_numpy(obs))

            if i > 5000:
                break

            if not headless:
                env.render()

        if action_list:  # only save if we actually stepped at least once
            save_rollout_npz(save_path, obs_list, next_obs_list, action_list, meta)


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
    actor.eval()

    state_preprocessor = lambda x: x
    if "state_preprocessor" in state_dict:
        state_preprocessor = RunningStandardScaler(size=(48,), device="cuda")
        state_preprocessor.load_state_dict(state_dict["state_preprocessor"])
        state_preprocessor.to("cuda")

    def policy(obs):
        action = actor.act({"states": state_preprocessor(obs)}, "policy")[-1].get(
            "mean_actions"
        )
        return action  # torch tensor on cuda

    return policy


def run_one(env_id, exp, checkpoint_dir, num_envs, agent_name, headless, save_path):
    env, simulation_app = make_isaac_environment(env_id, num_envs, headless=headless)
    env = wrap_env(env)

    ckpt = os.path.join(checkpoint_dir, exp, "checkpoints", agent_name)
    policy = build_policy(ckpt, env)

    resolved_path = resolve_save_path(save_path, checkpoint_dir, exp, agent_name)
    meta = {
        "env_id": str(env_id),
        "exp": str(exp),
        "agent_name": str(agent_name),
        "num_envs": int(num_envs),
        "headless": bool(headless),
        "seed": 42,
    }

    play_single_policy(env, simulation_app, policy, headless, resolved_path, meta)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", default="Safe-Locomotion-Base-v0")
    p.add_argument("--exp", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--num_envs", type=int, default=3)
    p.add_argument("--agent_name", default="best_agent.pt")
    p.add_argument("--headless", action="store_true")
    p.add_argument(
        "--save_path",
        type=str,
        default=None,
        help=(
            "Where to save the rollout. "
            "If a directory or a path without .npz, a timestamped file is created inside it. "
            "If a file ending with .npz, it will be used directly."
        ),
    )
    args = p.parse_args()
    run_one(
        args.env_id,
        args.exp,
        args.checkpoint_dir,
        args.num_envs,
        args.agent_name,
        args.headless,
        args.save_path,
    )
