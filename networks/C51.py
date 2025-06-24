import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, Model


class C51(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        num_atoms=51,
        v_min=-15,
        v_max=15,
        hidden_dim=128,
    ):
        DeterministicMixin.__init__(self, clip_actions=False)
        Model.__init__(self, observation_space, action_space, device)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.register_buffer(
            "support", torch.linspace(v_min, v_max, num_atoms).to(device)
        )

        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, num_atoms),
        )

    def compute(self, inputs, role):
        x = torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)
        logits = self.net(x)
        return logits, {}
