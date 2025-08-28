from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG


SAC_FINETUNE_DEFAULT_CONFIG = SAC_DEFAULT_CONFIG | {"freeze_actor_until": 10000}


class SACFinetune(SAC):
    """CVaR Soft Actor-Critic

    :param models: Models used by the agent
    :type models: Gaussian Mixin
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

        # freeze actor network for finetuning
        self.freeze_actor_until = self.cfg.get("freeze_actor_until", 0)
        self.actor_frozen = False

        if self.freeze_actor_until > 0:
            for p in self.models["policy"].parameters():
                p.requires_grad_(False)

            self.models["policy"].eval()
            self.actor_frozen = True

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

        super()._update(timestep, timesteps)
