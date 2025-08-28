import os
import torch
import numpy as np
from typing import List, Sequence, Callable

from gymnasium.spaces import Box
from skrl.envs.wrappers.torch import wrap_env
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils import set_seed

from networks import StochasticActorMLP
from utils import make_isaac_environment


set_seed(0)


def build_policy(policy_checkpoint: str, env, device: str = "cuda") -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Load a policy from checkpoint and return a callable:
        policy(obs_batch: torch.Tensor[B, obs_dim]) -> torch.Tensor[B, act_dim]
    """
    state_dict = torch.load(policy_checkpoint, map_location=device, weights_only=True)

    # Heuristic from your original code: use elementwise_affine if any '8' in keys
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
        device,
        clip_actions=False,
        elementwise_affine=use_affine,
    )
    actor.load_state_dict(state_dict["policy"])
    actor.to(device)

    # Optional state preprocessor from checkpoint; size matches obs space
    state_preprocessor = (lambda x: x)
    if "state_preprocessor" in state_dict:
        obs_shape = env.observation_space.shape
        state_preprocessor = RunningStandardScaler(size=obs_shape, device=device)
        state_preprocessor.load_state_dict(state_dict["state_preprocessor"])
        state_preprocessor.to(device)

    def policy(obs_batch: torch.Tensor) -> torch.Tensor:
        # obs_batch: [B, obs_dim] on correct device
        out = actor.act({"states": state_preprocessor(obs_batch)}, "policy")[-1]
        return out["mean_actions"]  # [B, act_dim]

    return policy


def play_multi_policies(env, app, policies: Sequence[Callable[[torch.Tensor], torch.Tensor]]):
    """
    Run N policies on an Isaac env with N sub-envs.
    Each policy i receives obs[i:i+1] and returns actions[i:i+1].
    """
    obs = env.reset()[0]  # torch.Tensor [N, obs_dim]
    num_envs = obs.shape[0]
    assert num_envs == len(policies), f"Env count ({num_envs}) must match #policies ({len(policies)})"

    device = obs.device
    with torch.inference_mode():
        while app.app.is_running():
            # Compute actions per-env, then concatenate
            actions_per_env = []
            # hard code command into observation vector
            # obs[:, 9:12] = torch.tensor([0.6, 0, 0], device=obs.device)
            for i, pol in enumerate(policies):
                o_i = obs[i:i+1]                          # keep batch dim
                a_i = pol(o_i)                            # [1, act_dim]
                actions_per_env.append(a_i)
            actions = torch.cat(actions_per_env, dim=0).to(device)  # [N, act_dim]

            obs, *_ = env.step(actions)
            env.render()


def ckpt_path(checkpoint_dir: str, exp: str, agent_name: str) -> str:
    return os.path.join(checkpoint_dir, exp, "checkpoints", agent_name)


def run_many_with_exps(env_id: str, exps: List[str], checkpoint_dir: str, agent_name: str = "best_agent.pt"):
    """
    Create a single Isaac env with num_envs=len(exps). Load one policy per exp.
    """
    num_envs = len(exps)
    env, simulation_app = make_isaac_environment(env_id, num_envs, headless=False)
    env = wrap_env(env)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policies = []
    for exp in exps:
        path = ckpt_path(checkpoint_dir, exp, agent_name)
        policies.append(build_policy(path, env, device=device))

    play_multi_policies(env, simulation_app, policies)


def run_many_with_ckpts(env_id: str, ckpts: List[str]):
    """
    Same as above, but you pass explicit checkpoint file paths.
    """
    num_envs = len(ckpts)
    env, simulation_app = make_isaac_environment(env_id, num_envs, headless=False)
    env = wrap_env(env)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policies = [build_policy(p, env, device=device) for p in ckpts]
    play_multi_policies(env, simulation_app, policies)


# Backward-compatible single-policy entry (kept for convenience)
def run_one(env_id, exp, checkpoint_dir, num_envs, agent_name):
    env, simulation_app = make_isaac_environment(env_id, num_envs, headless=False)
    env = wrap_env(env)
    ckpt = ckpt_path(checkpoint_dir, exp, agent_name)
    policy = build_policy(ckpt, env, device="cuda" if torch.cuda.is_available() else "cpu")
    # Reuse single-policy runner by wrapping in a list and requiring num_envs==1
    if num_envs != 1:
        raise ValueError("run_one is for a single sub-env. Use run_many_* for multiple policies.")
    play_multi_policies(env, simulation_app, [policy])


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--env_id", required=True)

    sub = p.add_subparsers(dest="mode", required=True)

    # Option A: pass experiment names under a common checkpoint_dir
    a = sub.add_parser("exps", help="Load policies from checkpoint_dir/<exp>/checkpoints/<agent_name>")
    a.add_argument("--exps", nargs="+", required=True, help="Experiment names")
    a.add_argument("--checkpoint_dir", required=True, help="Root with experiment subfolders")
    a.add_argument("--agent_name", default="best_agent.pt")

    # Option B: pass explicit checkpoint paths
    b = sub.add_parser("ckpts", help="Load policies from explicit checkpoint paths")
    b.add_argument("--ckpts", nargs="+", required=True, help="Paths to checkpoint .pt files")

    # Legacy single-policy path (kept just in case)
    c = sub.add_parser("one", help="(Legacy) single policy in one sub-env")
    c.add_argument("--exp", required=True)
    c.add_argument("--checkpoint_dir", required=True)
    c.add_argument("--agent_name", default="best_agent.pt")

    args = p.parse_args()

    if args.mode == "exps":
        run_many_with_exps(args.env_id, args.exps, args.checkpoint_dir, args.agent_name)
    elif args.mode == "ckpts":
        run_many_with_ckpts(args.env_id, args.ckpts)
    elif args.mode == "one":
        run_one(args.env_id, args.exp, args.checkpoint_dir, num_envs=1, agent_name=args.agent_name)
