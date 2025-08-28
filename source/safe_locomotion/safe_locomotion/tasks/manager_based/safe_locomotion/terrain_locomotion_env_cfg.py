import math
import numpy as np

import isaaclab.terrains as terrain_gen
from .base_locomotion_env_cfg import BaseCfgTest
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .mdp.events import reset_root_state_fixed
from .mdp.commands import StraightLineVelocityCommandCfg
from isaaclab.managers import EventTermCfg as EventTerm


class FrictionLevel0(BaseCfgTest):
    def __post_init__(self):
        super().__post_init__()
        # asphalt
        self.scene.terrain.physics_material.static_friction = 0.9
        self.scene.terrain.physics_material.dynamic_friction = 0.6


class FrictionLevel1(BaseCfgTest):
    def __post_init__(self):
        super().__post_init__()
        # wet concrete
        self.scene.terrain.physics_material.static_friction = 0.45
        self.scene.terrain.physics_material.dynamic_friction = 0.3


class FrictionLevel2(BaseCfgTest):
    def __post_init__(self):
        super().__post_init__()
        # ice
        self.scene.terrain.physics_material.static_friction = 0.1
        self.scene.terrain.physics_material.dynamic_friction = 0.02


class RestitutionLevel0(BaseCfgTest):
    def __post_init__(self):
        super().__post_init__()
        # concrete
        self.scene.terrain.physics_material.restitution = 0.2


class RestitutionLevel1(BaseCfgTest):
    def __post_init__(self):
        super().__post_init__()
        # rubber
        self.scene.terrain.physics_material.restitution = 0.5


class InclineLevel0(BaseCfgTest):
    def __post_init__(self):
        super().__post_init__()
        angle = math.radians(5)

        terrain_generator_cfg = ROUGH_TERRAINS_CFG
        terrain_generator_cfg.sub_terrains = {
            "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=1.0, 
                slope_range=(angle, angle),
                size=(75, 75), 
                platform_width=1, 
                border_width=1,
                inverted=True
            ),
        }

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = terrain_generator_cfg


class InclineLevel1(InclineLevel0):
    def __post_init__(self):
        super().__post_init__()
        angle = math.radians(10)

        self.scene.terrain.terrain_generator.sub_terrains["slope"].slope_range = (angle, angle)

class InclineLevel1Visualize(InclineLevel1):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator.num_cols = 3
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.size = (16, 8)

        self.scene.env_spacing = 2.5
        self.episode_length_s = 30.0
        self.commands.base_velocity.goal_vel_visualizer_cfg.markers["arrow"].scale = (0.01, 0.01, 0.01)

        self.events.reset_base = EventTerm(
            func=reset_root_state_fixed,
            mode="reset",
            params={
                "pose": {"x": 0.2, "yaw": np.deg2rad(0)},
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


class RoughLevel0(BaseCfgTest):
    def __post_init__(self):
        super().__post_init__()

        terrain_generator_cfg = ROUGH_TERRAINS_CFG
        terrain_generator_cfg.sub_terrains = {
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=1.0, 
                noise_range=(0.02, 0.04), 
                noise_step=0.015, 
                border_width=0.1
            )
        }

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = terrain_generator_cfg


class RoughLevel0Visualize(RoughLevel0):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_cols = 3
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.size = (30, 5)

        self.scene.env_spacing = 2.5
        self.episode_length_s = 30.0
        self.commands.base_velocity.goal_vel_visualizer_cfg.markers["arrow"].scale = (0.01, 0.01, 0.01)

        self.events.reset_base = EventTerm(
            func=reset_root_state_fixed,
            mode="reset",
            params={
                "pose": {"x": 0.2, "yaw": np.deg2rad(0)},
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