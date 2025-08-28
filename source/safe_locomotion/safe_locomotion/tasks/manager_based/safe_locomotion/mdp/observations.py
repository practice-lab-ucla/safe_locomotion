import math
import torch
import collections
from isaaclab.utils import configclass

from isaaclab.utils.modifiers.modifier_cfg import ModifierCfg
from isaaclab.utils.modifiers.modifier_base import ModifierBase


class BrownianNoiseModifier(ModifierBase):
    def __init__(self, cfg, data_dim, device):
        super().__init__(cfg, data_dim, device)
        self.num_envs = data_dim[0]
        self.prev_noise = torch.zeros(data_dim, device=self._device)
    
    def reset(self, env_ids = None):
        if env_ids is not None:
            self.prev_noise[env_ids] = 0
        else:
            self.prev_noise = torch.zeros(self._data_dim, device=self._device)
    
    def __call__(self, data, bias, std, dt):
        noise_inc = torch.randn(self._data_dim, device=self._device) * std * math.sqrt(dt) + bias
        noise_applied = self.prev_noise + noise_inc
        self.prev_noise = noise_applied
        return data + noise_applied 


@configclass
class BrownianMotionModifierCfg(ModifierCfg):
    func = BrownianNoiseModifier
    params = {
        "bias": 0,
        "std": 1,
        "dt": 0.005
    }


class DelayModifier(ModifierBase):
    def __init__(self, cfg, data_dim, device):
        super().__init__(cfg, data_dim, device)
        self.num_envs = data_dim[0]
        self.delay_period = cfg.delay_period
        self.observation_buffer = collections.deque([
            torch.zeros(*self._data_dim, device=self._device)
        ] * self.delay_period, maxlen=self.delay_period)
    
    def reset(self, env_ids = None):
        for i in range(self.delay_period):
            self.observation_buffer[i][env_ids] = 0
    
    def __call__(self, data):
        observation = self.observation_buffer[0]
        self.observation_buffer.append(data)
        return observation


@configclass
class DelayModifierCfg(ModifierCfg):
    func = DelayModifier
    delay_period = 5