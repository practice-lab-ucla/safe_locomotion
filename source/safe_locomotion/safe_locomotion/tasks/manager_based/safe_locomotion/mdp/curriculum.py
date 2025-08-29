from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def switch_command_mode(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    switch_every=1500,
):
    if "base_velocity" in env.command_manager.active_terms:
        d = env.common_step_counter // switch_every
        command_term = env.command_manager.get_term("base_velocity").cfg

        if d % 2 == 0:
            command_term.ranges.lin_vel_x = (0.3, 1.0)
            command_term.ranges.lin_vel_y = (-0.2, 0.2)
        else:
            command_term.ranges.lin_vel_x = (-1.0, 1.0)
            command_term.ranges.lin_vel_y = (-1.0, 1.0)

        env.command_manager._terms["base_velocity"].cfg = command_term
        # print("hi")