from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaacsim.core.utils.extensions import enable_extension

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def push_by_setting_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    magnitude: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset by setting the root velocity to a random value within the given ranges.

    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
    It samples the root velocity from the given ranges and sets the velocity into the physics simulation.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the velocity is set to zero for that axis.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    B = env_ids.shape[0]
    headings = torch.rand(B, device=asset.device) * (2 * torch.pi)

    vx = magnitude * torch.cos(headings)
    vy = magnitude * torch.sin(headings)

    vel_w = torch.zeros((B, 6), device=asset.device)
    vel_w[:, 0] = vx
    vel_w[:, 1] = vy

    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


def reset_root_state_fixed(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose: dict[str, float] | None = None,
    velocity: dict[str, float] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    *,
    pose_is_delta: bool = True,
    vel_is_delta: bool = True,
    add_env_origins: bool = True,
):
    """
    Set the asset root pose/velocity for all `env_ids` to the SAME values.

    Args:
        env: Isaac Lab environment.
        env_ids: tensor of environment indices to modify.
        pose: dict with keys in {"x","y","z","roll","pitch","yaw"}; omitted keys default to 0.0.
              - If `pose_is_delta=True` (default), these are deltas added to the default root pose.
              - If `pose_is_delta=False`, these are absolute (w.r.t. each env's origin if `add_env_origins=True`).
        velocity: dict with keys in {"x","y","z","roll","pitch","yaw"}; omitted keys default to 0.0.
                  - If `vel_is_delta=True` (default), added to default root velocity.
                  - If `vel_is_delta=False`, set as absolute root velocity.
        asset_cfg: which scene entity to reset (default "robot").
        pose_is_delta: see above.
        vel_is_delta: see above.
        add_env_origins: if True, positions are offset by `env.scene.env_origins[env_ids]`
                         so the same *relative* pose is used in each sub-env (typical multi-env setup).

    Notes:
        - Orientation is specified via Euler (roll, pitch, yaw) in radians and converted to a quaternion.
        - Velocity uses ordering [vx, vy, vz, wx, wy, wz].
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    device = asset.device

    # Default dicts
    pose = pose or {}
    velocity = velocity or {}

    # Gather constants (x,y,z, roll,pitch,yaw)
    keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    pose_vals = torch.tensor([pose.get(k, 0.0) for k in keys], device=device, dtype=torch.float32)
    vel_vals  = torch.tensor([velocity.get(k, 0.0) for k in keys], device=device, dtype=torch.float32)

    # Repeat for all env_ids
    n = len(env_ids)
    pose_batch = pose_vals.unsqueeze(0).repeat(n, 1)   # [N, 6]
    vel_batch  = vel_vals.unsqueeze(0).repeat(n, 1)    # [N, 6]

    # Base (default) states for selected envs
    root_states = asset.data.default_root_state[env_ids].clone()  # [N, 13]
    base_pos = root_states[:, 0:3] if pose_is_delta else torch.zeros_like(root_states[:, 0:3])
    base_quat = root_states[:, 3:7] if pose_is_delta else torch.tensor(
        [1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32
    ).repeat(n, 1)  # identity quaternion

    # Position: optionally add per-env origins so pose is "same relative pose" across envs
    pos = base_pos + pose_batch[:, 0:3]
    if add_env_origins:
        pos = pos + env.scene.env_origins[env_ids]

    # Orientation: delta or absolute from euler
    dq = math_utils.quat_from_euler_xyz(pose_batch[:, 3], pose_batch[:, 4], pose_batch[:, 5])  # [N, 4]
    quat = math_utils.quat_mul(base_quat, dq) if pose_is_delta else dq

    # Velocity: delta or absolute
    base_vel = root_states[:, 7:13] if vel_is_delta else torch.zeros_like(root_states[:, 7:13])
    vel = base_vel + vel_batch  # [N, 6] -> [vx, vy, vz, wx, wy, wz]

    # Write into sim
    asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(vel, env_ids=env_ids)

def unif_rand_in_range(shape, low, high, device="cuda:0"):
    print(low, high)
    return (high - low) * torch.rand(shape, device=device) + low


def randomizing_stiffness_and_gain(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gain_range=(25, 40),
    damp_range=(0.5, 1)
):
    asset: Articulation = env.scene[asset_cfg.name]
    new_gains = unif_rand_in_range((len(env_ids), 12), *gain_range, env_ids.device)
    new_damps = unif_rand_in_range((len(env_ids), 12), *damp_range, env_ids.device)



    print(asset.actuators["base_legs"])

    asset.write_joint_stiffness_to_sim(new_gains, asset_cfg.joint_ids, env_ids)
    asset.write_joint_damping_to_sim(new_damps, asset_cfg.joint_ids, env_ids)

