from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.markers import VisualizationMarkers

from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class StraightLineVelocityCommand(CommandTerm):
    """
    Drive forward at constant vx and regulate yaw to a heading target.

    Command is in the base frame: [vx, 0, w_z], shape (num_envs, 3).
    w_z = clamp(gain * wrap_to_pi(heading_target - heading_current), [-w_max, w_max]).
    """

    cfg: "StraightLineVelocityCommandCfg"

    def __init__(self, cfg: "StraightLineVelocityCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        base_env = getattr(env, "unwrapped", env)
        self.robot = base_env.scene[cfg.asset_name]

        self.vx = float(cfg.vx)
        self.gain = float(cfg.heading_gain)
        self.w_max = float(cfg.w_max)
        if cfg.target_mode not in ("initial", "world_x"):
            raise ValueError("target_mode must be 'initial' or 'world_x'")
        self.target_mode = cfg.target_mode

        # Buffers
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)  # (N, 3)
        self.vel_command_b[:, 0] = self.vx                                     # vx constant, vy=0 init
        self.heading_target = torch.zeros(self.num_envs, device=self.device)    # (N,)

    # -------------------------------------------------------------------------
    # Isaac Lab command API
    # -------------------------------------------------------------------------
    @property
    def command(self) -> torch.Tensor:
        """Desired base velocity in base frame, shape (num_envs, 3)."""
        return self.vel_command_b

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """(Re)define heading targets for the selected envs."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if self.target_mode == "initial":
            # keep straight along current heading
            self.heading_target[env_ids] = self.robot.data.heading_w[env_ids]
        else:
            # keep straight along world +X
            self.heading_target[env_ids] = 0.0

        # ensure forward speed and 0 lateral for those envs
        self.vel_command_b[env_ids, 0] = self.vx
        self.vel_command_b[env_ids, 1] = 0.0  # vy=0

    def _update_command(self):
        """Update w_z from heading error and write into command buffer."""
        heading_now = self.robot.data.heading_w                      # (N,)
        err = math_utils.wrap_to_pi(self.heading_target - heading_now)
        w_z = torch.clamp(self.gain * err, min=-self.w_max, max=self.w_max)

        self.vel_command_b[:, 0] = self.vx
        self.vel_command_b[:, 1] = 0.0
        self.vel_command_b[:, 2] = w_z

    def _update_metrics(self):
        pass


    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


@configclass
class StraightLineVelocityCommandCfg(CommandTermCfg):
    """Configuration for straight-line velocity command."""

    class_type: type = StraightLineVelocityCommand

    asset_name: str = MISSING
    vx: float = 0.8
    heading_gain: float = 1
    w_max: float = 0.5
    target_mode: str = "initial"  # "initial" or "world_x"

    # Optional visualization configs
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    # Scale the arrows a bit smaller
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
