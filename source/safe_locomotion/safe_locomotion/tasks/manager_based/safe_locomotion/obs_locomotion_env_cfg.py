import math
import torch

from isaaclab.utils import configclass
from .base_locomotion_env_cfg import BaseCfgTest

from .mdp.observations import BrownianMotionModifierCfg, DelayModifierCfg


@configclass
class BrownianLevel0(BaseCfgTest):
    """
    Base environment, but observation now has random noise
    """
    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.base_ang_vel.noise = None
        self.observations.policy.base_ang_vel.modifiers = [BrownianMotionModifierCfg()]

        self.observations.policy.base_lin_vel.noise = None
        self.observations.policy.base_lin_vel.modifiers = [BrownianMotionModifierCfg()]

        self.observations.policy.base_lin_vel.modifiers[0].params["std"] = 1.0
        self.observations.policy.base_ang_vel.modifiers[0].params["std"] = 1.0


@configclass
class BrownianLevel1(BrownianLevel0):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.base_lin_vel.modifiers[0].params["std"] = 2.0
        self.observations.policy.base_ang_vel.modifiers[0].params["std"] = 2.0


@configclass
class DelayLevel0(BaseCfgTest):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.joint_pos.modifiers = [DelayModifierCfg(delay_period=5)]
        self.observations.policy.joint_vel.modifiers = [DelayModifierCfg(delay_period=5)]


@configclass
class DelayLevel1(BaseCfgTest):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.joint_pos.modifiers = [DelayModifierCfg(delay_period=10)]
        self.observations.policy.joint_vel.modifiers = [DelayModifierCfg(delay_period=10)]


@configclass
class DelayLevel2(BaseCfgTest):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.joint_pos.modifiers = [DelayModifierCfg(delay_period=15)]
        self.observations.policy.joint_vel.modifiers = [DelayModifierCfg(delay_period=15)]


@configclass
class CommandLevel0(BaseCfgTest):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x=(-2.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y=(-1.0, 1.0)


@configclass
class CommandLevel1(BaseCfgTest):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x=(-2.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y=(-1.5, 1.5)

