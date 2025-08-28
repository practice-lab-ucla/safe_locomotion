import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG


CVAR_PPO_DEFAULT_CONFIG = PPO_DEFAULT_CONFIG | {
    "cvar_alpha": 0.5,
    "cvar_beta": "adaptive",
    "v_learning_rate": 1e-2,
    "lam_learning_rate": 1e-2,
    "v_gradient_steps": 1,
    "lam_gradient_steps": 1,
}


class CVARPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cvar_alpha = self.cfg.get("cvar_alpha", 0.25)
        assert self._cvar_alpha > 0 and self._cvar_alpha < 1, "0 < alpha < 1"
        self._cvar_beta = self.cfg.get("cvar_beta", "adaptive")  # cvar constraint
        self._form = self.cfg.get("form", "lagrangian")
        self._lam_max = self.cfg.get("lam_max", self._cvar_alpha)
        self._warmup = self.cfg.get("warmup", 0)

        if self._cvar_beta == "adaptive":
            self._moving_beta = 0

        # value-at-risk optimizer
        self._v = nn.Parameter(
            torch.tensor(0.0, device=self.device), requires_grad=True
        )
        self._v_optimizer = torch.optim.Adam(
            [self._v], lr=self.cfg.get("v_learning_rate", self._learning_rate)
        )
        self._v_gradient_steps = self.cfg.get("v_gradient_steps", 1)

        if self._form == "lagrangian":
            # multiplier optimizer
            self._lam = torch.tensor(0.0, device=self.device)
            self._lam_learning_rate = self.cfg.get(
                "lam_learning_rate", self._learning_rate
            )
            self._lam_gradient_steps = self.cfg.get("lam_gradient_steps", 1)
        elif self._form == "epigraph":
            pass
        else:
            raise NotImplementedError

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)

            :param rewards: Rewards obtained by the agent
            :type rewards: torch.Tensor
            :param dones: Signals to indicate that episodes have ended
            :type dones: torch.Tensor
            :param values: Values obtained by the agent
            :type values: torch.Tensor
            :param next_values: Next values obtained by the agent
            :type next_values: torch.Tensor
            :param discount_factor: Discount factor
            :type discount_factor: float
            :param lambda_coefficient: Lambda coefficient
            :type lambda_coefficient: float

            :return: Generalized Advantage Estimator
            :rtype: torch.Tensor
            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor
                    * not_dones[i]
                    * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(
            device_type=self._device_type, enabled=self._mixed_precision
        ):
            self.value.train(False)
            last_values, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float())},
                role="value",
            )
            self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated")
            | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        returns = returns.detach()
        advantages = advantages.detach()

        self.memory.set_tensor_by_name(
            "values", self._value_preprocessor(values, train=True)
        )
        self.memory.set_tensor_by_name(
            "returns", self._value_preprocessor(returns, train=True)
        )
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(
            names=self._tensors_names, mini_batches=self._mini_batches
        )

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_cvar_loss = 0

        cumulative_v = 0
        cumulative_lambda = 0
        cumulative_cvar = 0
        cumulative_cvar_p1 = 0

        # update var and dual variables
        def cvar_left(var_estimates, alpha, returns):
            return (
                var_estimates
                - (1 / alpha) * torch.clamp_min(var_estimates - returns, 0).mean()
            )

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for (
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in sampled_batches:

                with torch.autocast(
                    device_type=self._device_type, enabled=self._mixed_precision
                ):

                    sampled_states = self._state_preprocessor(
                        sampled_states, train=not epoch
                    )

                    _, next_log_prob, _ = self.policy.act(
                        {"states": sampled_states, "taken_actions": sampled_actions},
                        role="policy",
                    )

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence
                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        break

                    # compute entropy loss
                    if self._entropy_loss_scale:
                        entropy_loss = (
                            -self._entropy_loss_scale
                            * self.policy.get_entropy(role="policy").mean()
                        )
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # compute value loss
                    predicted_values, _, _ = self.value.act(
                        {"states": sampled_states}, role="value"
                    )

                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values,
                            min=-self._value_clip,
                            max=self._value_clip,
                        )
                    value_loss = self._value_loss_scale * F.mse_loss(
                        sampled_returns, predicted_values
                    )

                if self._form == "lagrangian":
                    # optimization step
                    cvar_surrogate = (
                        self._lam
                        / self._cvar_alpha
                        * (ratio * torch.clamp_min(self._v - sampled_returns, 0))
                    )

                    cvar_surrogate_clipped = (
                        self._lam
                        / self._cvar_alpha
                        * (
                            torch.clip(
                                ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                            )
                            * torch.clamp_min(self._v - sampled_returns, 0)
                        )
                    )

                    cvar_loss = torch.min(cvar_surrogate, cvar_surrogate_clipped).mean()

                    self.optimizer.zero_grad()
                    self.scaler.scale(
                        policy_loss + entropy_loss + value_loss + cvar_loss
                    ).backward()
                elif self._form == "epigraph":
                    cvar_loss = (
                        1
                        / self._cvar_alpha
                        * (
                            next_log_prob
                            * torch.clamp_min(self._v - sampled_returns, 0)
                        ).mean()
                    )
                    with torch.no_grad():
                        cvar_return = cvar_left(
                            self._v, self._cvar_alpha, sampled_returns
                        )
                    self.optimizer.zero_grad()
                    if self._cvar_beta == "adaptive":
                        if cvar_return.item() < self._moving_beta:
                            self.scaler.scale(cvar_loss + value_loss).backward()
                        else:
                            self.scaler.scale(
                                policy_loss + entropy_loss + value_loss
                            ).backward()
                        self._moving_beta = 0.3 * self._moving_beta + 0.7 * self._v
                    else:
                        if cvar_return.item() < self._cvar_beta:
                            self.scaler.scale(cvar_loss + value_loss).backward()
                        else:
                            self.scaler.scale(
                                policy_loss + entropy_loss + value_loss
                            ).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(
                            self.policy.parameters(), self._grad_norm_clip
                        )
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(
                                self.policy.parameters(), self.value.parameters()
                            ),
                            self._grad_norm_clip,
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                detached_returns = sampled_returns.detach()
                cvar = cvar_left(self._v, self._cvar_alpha, sampled_returns.detach())

                for _ in range(self._v_gradient_steps):
                    cvar = cvar_left(self._v, self._cvar_alpha, detached_returns)
                    negative_cvar = -cvar
                    self._v_optimizer.zero_grad()
                    negative_cvar.backward()
                    nn.utils.clip_grad_norm_([self._v], self._grad_norm_clip)
                    self._v_optimizer.step()

                varp1 = torch.quantile(detached_returns, 0.1, dim=0)
                cvarp1 = detached_returns[detached_returns < varp1].mean()

                if self._form == "lagrangian":
                    with torch.no_grad():
                        for _ in range(self._lam_gradient_steps):
                            if self._cvar_beta == "adaptive":
                                self._lam = self._lam + self._lam_learning_rate * (
                                    self._moving_beta
                                    - cvar_left(
                                        self._v, self._cvar_alpha, sampled_returns
                                    )
                                )
                                self._moving_beta = (
                                    0.3 * self._moving_beta + 0.7 * self._v
                                )
                            else:
                                self._lam = self._lam + self._lam_learning_rate * (
                                    self._cvar_beta
                                    - cvar_left(
                                        self._v, self._cvar_alpha, sampled_returns
                                    )
                                )
                            if self._lam_max == "adaptive":
                                self._lam = torch.clamp(
                                    self._lam, 0, 0.25 * (timestep / timesteps) ** 2
                                )
                            else:
                                self._lam = torch.clamp(
                                    self._lam,
                                    0,
                                    self._lam_max * float(timestep > self._warmup),
                                )

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()
                cumulative_cvar_loss += cvar_loss.item()

                cumulative_v += self._v.item()
                if self._form == "lagrangian":
                    cumulative_lambda += self._lam.item()
                cumulative_cvar += cvar.item()
                cumulative_cvar_p1 += cvarp1.item()

            # update learning rate
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(
                            kl, op=torch.distributed.ReduceOp.SUM
                        )
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        # record data
        self.track_data(
            "Loss / Policy loss",
            cumulative_policy_loss / (self._learning_epochs * self._mini_batches),
        )
        self.track_data(
            "Loss / Value loss",
            cumulative_value_loss / (self._learning_epochs * self._mini_batches),
        )
        if self._entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss",
                cumulative_entropy_loss / (self._learning_epochs * self._mini_batches),
            )
        if self._form == "lagrangian":
            self.track_data(
                "Loss / CVaR loss",
                cumulative_cvar_loss / (self._learning_epochs * self._mini_batches),
            )

        self.track_data(
            "Policy / Standard deviation",
            self.policy.distribution(role="policy").stddev.mean().item(),
        )

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

        self.track_data(
            f"Coefficient / VaR {self._cvar_alpha}",
            cumulative_v / (self._learning_epochs * self._mini_batches),
        )
        self.track_data(
            f"Coefficient / CVaR {self._cvar_alpha}",
            cumulative_cvar / (self._learning_epochs * self._mini_batches),
        )
        self.track_data(
            "Coefficient / CVaR 0.1",
            cumulative_cvar_p1 / (self._learning_epochs * self._mini_batches),
        )
        self.track_data(
            "Coefficient / Lagrangian multiplier",
            cumulative_lambda / (self._learning_epochs * self._mini_batches),
        )
