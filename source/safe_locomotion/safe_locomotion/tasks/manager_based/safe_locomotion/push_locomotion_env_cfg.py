import numpy as np
from isaaclab.utils import configclass
from .base_locomotion_env_cfg import BaseCfgTest
from isaaclab.managers import EventTermCfg as EventTerm
from .mdp import push_by_setting_velocity
from .mdp.events import reset_root_state_fixed
from .mdp.commands import StraightLineVelocityCommandCfg

@configclass
class PushLevel0(BaseCfgTest):
    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = EventTerm(
            func=push_by_setting_velocity,
            mode="interval",
            interval_range_s=(3, 5),
            params={"magnitude": 0.75},
        )

@configclass
class PushLevel1(PushLevel0):
    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot.interval_range_s = (1, 3)


@configclass
class PushLevel2(PushLevel0):
    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot.interval_range_s = (0.5, 1.5)


@configclass
class PushLevel2Visualize(PushLevel2):
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