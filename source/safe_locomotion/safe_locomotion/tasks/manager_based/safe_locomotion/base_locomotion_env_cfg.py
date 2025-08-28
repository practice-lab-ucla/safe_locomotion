import math
import numpy as np
import torch

from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .mdp.rewards import orientation_penalty
from .mdp.events import reset_root_state_fixed, randomizing_stiffness_and_gain
from .mdp.commands import StraightLineVelocityCommandCfg
from .mdp.curriculum import update_joint_gain_range


from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
import isaaclab.envs.mdp as env_mdp


@configclass
class BaseCfg(UnitreeGo2FlatEnvCfg):
    """
    Base environment
    """
    def __post_init__(self):
        super().__post_init__()


@configclass
class BaseCfg1(BaseCfg):
    """
    Base environment but with domain randomization
    curriculum for joint gain and 
    """
    def __post_init__(self):
        super().__post_init__()
        # self.events.randomize_joint_gain = EventTerm(
        #     func=randomizing_stiffness_and_gain,
        #     params={ "gain_range": (25, 26), "damp_range": (0.5, 0.6) },
        #     mode="reset"
        # )

        self.events.randomize_joint_gain = EventTerm(
            func=env_mdp.randomize_actuator_gains,
            min_step_count_between_reset=720,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (25, 26),
                "damping_distribution_params": (0.5, 0.6),
                "operation": "abs",
                "distribution": "log_uniform",
            },
        )
        self.curriculum.joint_gain_curriculum = CurrTerm(
            func=update_joint_gain_range,
            params={ "num_steps": 5000, "gain_max_start": 26, "damp_max_start": 0.6 }
        )


@configclass
class BaseCfgTest(UnitreeGo2FlatEnvCfg):
    """
    Base environment
    """
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 15.0
        self.terminations.base_contact = None
        self.rewards.flat_orientation_l2.func = orientation_penalty


@configclass
class BaseCfgVisualize(BaseCfgTest):
    """
    Base environment
    """
    def __post_init__(self):
        super().__post_init__()
        self.scene.env_spacing = 2.5
        self.episode_length_s = 30.0
        self.commands.base_velocity.goal_vel_visualizer_cfg.markers["arrow"].scale = (0.01, 0.01, 0.01)

        self.events.reset_base = EventTerm(
            func=reset_root_state_fixed,
            mode="reset",
            params={
                "pose": {"x": 0.2, "yaw": np.deg2rad(90)},
                "velocity": {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                "pose_is_delta": True,
                "vel_is_delta": False,
                "add_env_origins": True
            },
        )

        self.commands.base_velocity = StraightLineVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            debug_vis=True
        )


@configclass
class MoreRewardCfg(BaseCfg):
    """
    Base environment, whereas the reward signal is dependent on magnitude of command
    """
    def __post_init__(self):

        def track_lin_vel_xy_exp_norm(
            env: ManagerBasedRLEnv, 
            std: float, 
            command_name: str, 
            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
        ) -> torch.Tensor:
            reward = mdp.track_lin_vel_xy_exp(env, std, command_name, asset_cfg)
            vel = env.command_manager.get_command(command_name)[:, :2]
            return 0.5 * (torch.norm(vel) + 1) * reward


        def track_ang_vel_z_exp_norm(
            env: ManagerBasedRLEnv, 
            std: float, 
            command_name: str, 
            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
        ) -> torch.Tensor:
            reward = mdp.track_ang_vel_z_exp(env, std, command_name, asset_cfg)
            vel = env.command_manager.get_command(command_name)[:, 2]
            return 0.5 * (torch.norm(vel) + 1) * reward
    
        super().__post_init__()
        self.rewards.track_lin_vel_xy_exp.func = track_lin_vel_xy_exp_norm
        self.rewards.track_ang_vel_z_exp.func = track_ang_vel_z_exp_norm


@configclass
class FasterCfg(BaseCfg):
    """
    Base environment with faster velocity command
    """
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x=(-1.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y=(-1.5, 1.5)