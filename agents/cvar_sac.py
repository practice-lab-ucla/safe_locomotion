import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG

from networks import C51


CVAR_SAC_DEFAULT_CONFIG = SAC_DEFAULT_CONFIG | {"cvar_alpha": 0.4}


class CVaRSAC(SAC):
    """CVaR Soft Actor-Critic

    :param models: Models used by the agent
    :type models: C51 model
    :param memory: Memory to storage the transitions.
                    If it is a tuple, the first element will be used for training and
                    for the rest only the environment transitions will be added
    :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
    :param observation_space: Observation/state space or shape (default: ``None``)
    :type observation_space: int, tuple or list of int, gymnasium.Space or None, optional
    :param action_space: Action space or shape (default: ``None``)
    :type action_space: int, tuple or list of int, gymnasium.Space or None, optional
    :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                    If None, the device will be either ``"cuda"`` if available or ``"cpu"``
    :type device: str or torch.device, optional
    :param cfg: Configuration dictionary
    :type cfg: dict

    :raises KeyError: If the models dictionary is missing a required key
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cvar_alpha = self.cfg.get("cvar_alpha", 0.4)
        for key in ["critic_1", "critic_2", "target_critic_1", "target_critic_2"]:
            assert isinstance(
                self.models[key], C51
            ), "CVaR SAC only works with C51 network"

        assert cvar_alpha < 1, "alpha < 1"
        print("Using CVaR alpha", "{0:0.2f}".format(cvar_alpha))
        self._cvar_alpha = cvar_alpha
        self._support = self.models["critic_1"].support
        self._v_min = self._support[0]
        self._v_max = self._support[-1]
        self.num_atoms = len(self._support)
        self.delta = (self._v_max - self._v_min) / self.num_atoms

        # freeze actor network for finetuning
        self.freeze_actor_until = self.cfg.get("freeze_actor_until", 0)
        self.actor_frozen = False

        if self.freeze_actor_until > 0:
            for p in self.models["policy"].parameters():
                p.requires_grad_(False)

            self.models["policy"].eval()
            self.actor_frozen = True

    def _compute_cvar(self, probs):
        """
        probs: (batch, num_atoms)
        """
        alpha = self._cvar_alpha
        cdf = torch.cumsum(probs, dim=1)
        mask = cdf < alpha
        weighted_sum = (probs * mask * self._support.unsqueeze(0)).sum(
            dim=1
        )  # (batch,)
        excess = alpha - (probs * mask).sum(dim=1)  # (batch,)
        weighted_sum = weighted_sum + excess * self._support[mask.sum(dim=1).long()]
        return weighted_sum / alpha

    def _categorical_projection(self, probs_next, log_probs_next, rewards, dones):
        """
        logits: (batch, num_atoms)
        rewards, dones: (batch, 1) tensors
        returns prob shifted by reward projected on support
        """
        target_raw = (
            rewards
            + self._discount_factor * (1 - dones.float()) * self._support
            - self._entropy_coefficient * log_probs_next
        )  # (B, 1, )
        target_raw = target_raw.clamp(self._v_min, self._v_max)

        bin_num = (target_raw - self._v_min) / self.delta
        lower_bin_num = bin_num.floor().long().clamp(0, self.num_atoms - 1)
        upper_bin_num = bin_num.ceil().long().clamp(0, self.num_atoms - 1)

        dist_to_upper = upper_bin_num.float() - bin_num
        dist_to_lower = bin_num - lower_bin_num.float()

        target = torch.zeros_like(probs_next)
        target.scatter_add_(1, upper_bin_num, probs_next * dist_to_lower)
        target.scatter_add_(1, lower_bin_num, probs_next * dist_to_upper)

        return target

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        if self.actor_frozen and timestep >= self.freeze_actor_until:
            # unfreeze actor
            for param in self.models["policy"].parameters():
                param.requires_grad_(True)

            self.models["policy"].train()
            self.actor_frozen = False

        # gradient steps
        for gradient_step in range(self._gradient_steps):

            # sample a batch from memory
            (
                sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_states,
                sampled_terminated,
                sampled_truncated,
            ) = self.memory.sample(
                names=self._tensors_names, batch_size=self._batch_size
            )[
                0
            ]

            with torch.autocast(
                device_type=self._device_type, enabled=self._mixed_precision
            ):

                sampled_states = self._state_preprocessor(sampled_states, train=True)
                sampled_next_states = self._state_preprocessor(
                    sampled_next_states, train=True
                )

                # compute target values
                with torch.no_grad():
                    next_actions, next_log_prob, _ = self.policy.act(
                        {"states": sampled_next_states}, role="policy"
                    )

                    target_q1_logits, _, _ = self.target_critic_1.act(
                        {"states": sampled_next_states, "taken_actions": next_actions},
                        role="target_critic_1",
                    )
                    target_q2_logits, _, _ = self.target_critic_2.act(
                        {"states": sampled_next_states, "taken_actions": next_actions},
                        role="target_critic_2",
                    )
                    target_q1_probs = torch.softmax(target_q1_logits, dim=-1)
                    target_q2_probs = torch.softmax(target_q2_logits, dim=-1)

                    target_q1_expected = (target_q1_probs * self._support).sum(dim=1)
                    target_q2_expected = (target_q2_probs * self._support).sum(dim=1)

                    mask = (target_q1_expected <= target_q2_expected).unsqueeze(1)
                    chosen_probs = torch.where(mask, target_q1_probs, target_q2_probs)

                    target_probs = self._categorical_projection(
                        chosen_probs,
                        next_log_prob,
                        sampled_rewards,
                        sampled_terminated | sampled_truncated,
                    )  # (B, 51)

                # compute critic loss
                critic_1_logits, _, _ = self.critic_1.act(
                    {"states": sampled_states, "taken_actions": sampled_actions},
                    role="critic_1",
                )
                critic_2_logits, _, _ = self.critic_2.act(
                    {"states": sampled_states, "taken_actions": sampled_actions},
                    role="critic_2",
                )

                loss_1 = -(
                    target_probs * torch.log_softmax(critic_1_logits, dim=-1)
                ).sum()
                loss_2 = -(
                    target_probs * torch.log_softmax(critic_2_logits, dim=-1)
                ).sum()

                critic_loss = 0.5 * (loss_1 + loss_2) / self._batch_size

            # optimization step (critic)
            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()

            if config.torch.is_distributed:
                self.critic_1.reduce_parameters()
                self.critic_2.reduce_parameters()

            if self._grad_norm_clip > 0:
                self.scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(
                    itertools.chain(
                        self.critic_1.parameters(), self.critic_2.parameters()
                    ),
                    self._grad_norm_clip,
                )

            self.scaler.step(self.critic_optimizer)

            with torch.autocast(
                device_type=self._device_type, enabled=self._mixed_precision
            ):
                # compute policy (actor) loss
                actions, log_prob, _ = self.policy.act(
                    {"states": sampled_states}, role="policy"
                )
                critic_1_logits, _, _ = self.critic_1.act(
                    {"states": sampled_states, "taken_actions": actions},
                    role="critic_1",
                )
                critic_2_logits, _, _ = self.critic_2.act(
                    {"states": sampled_states, "taken_actions": actions},
                    role="critic_2",
                )

                critic_1_cvar = self._compute_cvar(
                    torch.softmax(critic_1_logits, dim=-1)
                )
                critic_2_cvar = self._compute_cvar(
                    torch.softmax(critic_2_logits, dim=-1)
                )

                policy_loss = (
                    self._entropy_coefficient * log_prob
                    - torch.min(critic_1_cvar, critic_2_cvar)
                ).mean()

            # optimization step (policy)
            self.policy_optimizer.zero_grad()
            self.scaler.scale(policy_loss).backward()

            if config.torch.is_distributed:
                self.policy.reduce_parameters()

            if self._grad_norm_clip > 0:
                self.scaler.unscale_(self.policy_optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)

            self.scaler.step(self.policy_optimizer)

            # entropy learning
            if self._learn_entropy:
                with torch.autocast(
                    device_type=self._device_type, enabled=self._mixed_precision
                ):
                    # compute entropy loss
                    entropy_loss = -(
                        self.log_entropy_coefficient
                        * (log_prob + self._target_entropy).detach()
                    ).mean()

                # optimization step (entropy)
                self.entropy_optimizer.zero_grad()
                self.scaler.scale(entropy_loss).backward()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.entropy_optimizer)
                    # clip the gradient on log_entropy_coefficient
                    nn.utils.clip_grad_norm_(
                        [self.log_entropy_coefficient], max_norm=0.5
                    )

                self.scaler.step(self.entropy_optimizer)

                # compute entropy coefficient
                self._entropy_coefficient = torch.exp(
                    self.log_entropy_coefficient.detach()
                )

            self.scaler.update()  # called once, after optimizers have been stepped

            # update target networks
            self.target_critic_1.update_parameters(self.critic_1, polyak=self._polyak)
            self.target_critic_2.update_parameters(self.critic_2, polyak=self._polyak)

            # update learning rate
            if self._learning_rate_scheduler:
                self.policy_scheduler.step()
                self.critic_scheduler.step()

            # record data
            if self.write_interval > 0:
                self.track_data("Loss / Policy loss", policy_loss.item())
                self.track_data("Loss / Critic loss", critic_loss.item())

                self.track_data(
                    "Q-network cvar / Q1 (max)", torch.max(critic_1_cvar).item()
                )
                self.track_data(
                    "Q-network cvar / Q1 (min)", torch.min(critic_1_cvar).item()
                )
                self.track_data(
                    "Q-network cvar / Q1 (mean)", torch.mean(critic_1_cvar).item()
                )

                self.track_data(
                    "Q-network cvar / Q2 (max)", torch.max(critic_2_cvar).item()
                )
                self.track_data(
                    "Q-network cvar / Q2 (min)", torch.min(critic_2_cvar).item()
                )
                self.track_data(
                    "Q-network cvar / Q2 (mean)", torch.mean(critic_2_cvar).item()
                )

                if self._learn_entropy:
                    self.track_data("Loss / Entropy loss", entropy_loss.item())
                    self.track_data(
                        "Coefficient / Entropy coefficient",
                        self._entropy_coefficient.item(),
                    )

                if self._learning_rate_scheduler:
                    self.track_data(
                        "Learning / Policy learning rate",
                        self.policy_scheduler.get_last_lr()[0],
                    )
                    self.track_data(
                        "Learning / Critic learning rate",
                        self.critic_scheduler.get_last_lr()[0],
                    )
