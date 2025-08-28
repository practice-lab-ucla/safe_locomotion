import torch
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import (
    ActionTerm,
    JointPositionAction, 
    JointPositionActionCfg
)


class JointPositionActionJitter(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.jitter_prob = cfg.jitter_prob
        self.past_action = None

    def apply_actions(self):
        if self.past_action is not None and torch.rand(1).item() < self.jitter_prob:
            self._asset.set_joint_position_target(self.past_action, joint_ids=self._joint_ids)
            self.past_action = self.past_action
        else:
            super().apply_actions()
            self.past_action = self.processed_actions


@configclass
class JointPositionActionJitterCfg(JointPositionActionCfg):
    jitter_prob = 0.1

    class_type: type[ActionTerm] = JointPositionActionJitter
