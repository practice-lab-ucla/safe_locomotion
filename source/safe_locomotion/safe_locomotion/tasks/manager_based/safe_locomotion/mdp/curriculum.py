from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv




def update_joint_gain_range(
        env: ManagerBasedRLEnv, 
        env_ids: Sequence[int], 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        num_steps=5000,
        base_ratio=1.25,
        gain_max_start=26,
        damp_max_start=0.5
):
    if "randomize_joint_gain" in env.event_manager.active_terms["reset"]:
        ratio = base_ratio ** (env.common_step_counter // num_steps)
        env_term = env.event_manager.get_term_cfg("randomize_joint_gain")
        past_gain_range = env_term.params["stiffness_distribution_params"]
        past_damp_range = env_term.params["damping_distribution_params"]
        env_term.params["stiffness_distribution_params"] = (past_gain_range[0], gain_max_start*ratio)
        env_term.params["damping_distribution_params"] = (past_damp_range[0], damp_max_start*ratio)
        env.event_manager.set_term_cfg("randomize_joint_gain", env_term)
