import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, DeterministicMixin, Model


class ElementwiseAffine(nn.Module):
    def __init__(self, dim, use_bias=False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.use_bias = use_bias

        if self.use_bias:
            self.bias  = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = x * self.scale
        if self.use_bias:
            x = x + self.bias
        return x


class StochasticActorMLP(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        hidden_dim=128,
        activation=nn.ELU,
        elementwise_affine=False
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self, clip_actions, clip_log_std, min_log_std, max_log_std
        )
        if elementwise_affine:
            self.net = nn.Sequential(
                nn.Linear(self.num_observations, hidden_dim),
                activation(),
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
                nn.Linear(hidden_dim, self.num_actions),
                nn.Tanh(),
                ElementwiseAffine(self.num_actions, use_bias=False)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(self.num_observations, hidden_dim),
                activation(),
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
                nn.Linear(hidden_dim, self.num_actions)
            )

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}


class StateCriticMLP(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        hidden_dim=128,
        activation=nn.ELU,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


class ActionStateCriticMLP(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        hidden_dim=128,
        activation=nn.ELU,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        
        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, 1),
        )

    def compute(self, inputs, role):
        return (
            self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)),
            {},
        )
