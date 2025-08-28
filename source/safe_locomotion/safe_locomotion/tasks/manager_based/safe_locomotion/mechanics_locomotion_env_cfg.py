import math

from .base_locomotion_env_cfg import BaseCfgTest
from .mdp import JointPositionActionJitterCfg



class GainLevel0(BaseCfgTest):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators["base_legs"].stiffness = 15.0


class GainLevel1(BaseCfgTest):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators["base_legs"].stiffness = 50.0
        self.scene.robot.actuators["base_legs"].damping = 2.0


class JitterLevel0(BaseCfgTest):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos = JointPositionActionJitterCfg(
            asset_name="robot", 
            joint_names=[".*"], 
            scale=0.25, 
            use_default_offset=True,
            jitter_prob=0.8
        )

